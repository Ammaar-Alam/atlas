from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional
from threading import Event

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from atlas.broker.alpaca_broker import assert_market_open, submit_market_order, trading_client, wait_for_fill
from atlas.config import AlpacaSettings
from atlas.strategies.base import Strategy
from atlas.utils.time import NY_TZ, now_ny

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperConfig:
    symbols: list[str]
    lookback_bars: int
    poll_seconds: int
    max_position_notional_usd: float
    allow_trading_when_closed: bool
    dry_run: bool


def _bars_client(settings: AlpacaSettings) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        settings.api_key, settings.secret_key, url_override=settings.data_url_override
    )


def _fetch_recent_minute_bars(
    *,
    client: StockHistoricalDataClient,
    symbol: str,
    lookback_bars: int,
) -> pd.DataFrame:
    end = now_ny()
    start = end - timedelta(minutes=max(lookback_bars * 2, 10))
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=max(lookback_bars, 10),
    )
    res = client.get_stock_bars(req).df
    if isinstance(res.index, pd.MultiIndex):
        res = res.xs(symbol)
    if res.index.tz is None:
        res.index = res.index.tz_localize("UTC")
    res.index = res.index.tz_convert(NY_TZ)
    res = res.sort_index()
    res = res[["open", "high", "low", "close", "volume"]].copy()
    if len(res) > lookback_bars:
        res = res.iloc[-lookback_bars:]
    return res


def _align_to_next_minute(now: pd.Timestamp) -> float:
    next_minute = (now + pd.Timedelta(minutes=1)).floor("min")
    sleep_s = (next_minute - now).total_seconds()
    return max(float(sleep_s), 0.0)


def run_paper_loop(
    *,
    settings: AlpacaSettings,
    strategy: Strategy,
    cfg: PaperConfig,
    run_dir: Path,
    max_loops: Optional[int],
    stop_event: Optional[Event] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    orders_path = run_dir / "orders.csv"
    orders_jsonl_path = run_dir / "orders.jsonl"
    fills_path = run_dir / "fills.csv"
    fills_jsonl_path = run_dir / "fills.jsonl"
    equity_path = run_dir / "equity_curve.csv"

    trade_client = trading_client(settings)
    data_client = _bars_client(settings)

    with (
        orders_path.open("w", newline="") as f_orders,
        orders_jsonl_path.open("w") as f_orders_jsonl,
        fills_path.open("w", newline="") as f_fills,
        fills_jsonl_path.open("w") as f_fills_jsonl,
    ):
        orders_writer = csv.DictWriter(
            f_orders,
            fieldnames=[
                "timestamp",
                "symbol",
                "side",
                "qty",
                "order_id",
                "dry_run",
                "strategy_reason",
            ],
        )
        fills_writer = csv.DictWriter(
            f_fills,
            fieldnames=[
                "timestamp",
                "symbol",
                "side",
                "status",
                "filled_qty",
                "filled_avg_price",
                "order_id",
            ],
        )
        orders_writer.writeheader()
        fills_writer.writeheader()

        last_target: dict[str, float] = {s: 0.0 for s in cfg.symbols}
        loops = 0

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("stop requested, exiting paper loop")
                return
            if max_loops is not None and loops >= max_loops:
                logger.info("max loops reached, stopping")
                return

            if not cfg.allow_trading_when_closed:
                assert_market_open(trade_client)

            for symbol in cfg.symbols:
                if stop_event is not None and stop_event.is_set():
                    logger.info("stop requested, exiting paper loop")
                    return
                bars = _fetch_recent_minute_bars(
                    client=data_client, symbol=symbol, lookback_bars=cfg.lookback_bars
                )
                decision = strategy.target_exposure(bars)
                target_exposure = float(decision.target_exposure)

                if abs(target_exposure - last_target.get(symbol, 0.0)) < 1e-8:
                    continue

                target_notional = target_exposure * float(cfg.max_position_notional_usd)
                last_price = float(bars["close"].iloc[-1])
                target_qty = target_notional / last_price if last_price > 0 else 0.0

                try:
                    pos = trade_client.get_open_position(symbol_or_asset_id=symbol)
                    current_qty = float(pos.qty)
                except Exception:
                    current_qty = 0.0

                if target_exposure <= 1e-8:
                    if current_qty <= 1e-8:
                        last_target[symbol] = target_exposure
                        continue
                    side = "SELL"
                    qty = current_qty
                else:
                    delta_qty = target_qty - current_qty
                    if delta_qty <= 1e-6:
                        last_target[symbol] = target_exposure
                        continue
                    side = "BUY"
                    qty = abs(delta_qty)

                ts = now_ny().isoformat()

                if cfg.dry_run:
                    order_id = "dry_run"
                else:
                    order_id = submit_market_order(
                        client=trade_client, symbol=symbol, qty=qty, side=side
                    )

                order_row = {
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "order_id": order_id,
                    "dry_run": cfg.dry_run,
                    "strategy_reason": decision.reason,
                }
                orders_writer.writerow(order_row)
                f_orders.flush()
                f_orders_jsonl.write(json.dumps(order_row) + "\n")
                f_orders_jsonl.flush()

                if not cfg.dry_run:
                    fill = wait_for_fill(
                        client=trade_client, order_id=order_id, timeout_s=60, poll_s=2.0
                    )
                    fill_row = {
                        "timestamp": now_ny().isoformat(),
                        "symbol": fill.symbol,
                        "side": fill.side,
                        "status": fill.status,
                        "filled_qty": fill.filled_qty,
                        "filled_avg_price": fill.filled_avg_price,
                        "order_id": fill.order_id,
                    }
                    fills_writer.writerow(fill_row)
                    f_fills.flush()
                    f_fills_jsonl.write(json.dumps(fill_row) + "\n")
                    f_fills_jsonl.flush()

                last_target[symbol] = target_exposure

            equity = float(trade_client.get_account().equity)
            pd.DataFrame(
                [
                    {
                        "timestamp": now_ny().isoformat(),
                        "equity": equity,
                    }
                ]
            ).to_csv(equity_path, mode="a", header=not equity_path.exists(), index=False)

            loops += 1
            sleep_s = _align_to_next_minute(pd.Timestamp.now(tz=NY_TZ))
            sleep_s = max(sleep_s, float(cfg.poll_seconds))
            logger.info("sleeping %.1fs", sleep_s)
            if stop_event is not None:
                if stop_event.wait(sleep_s):
                    logger.info("stop requested, exiting paper loop")
                    return
            else:
                time.sleep(sleep_s)

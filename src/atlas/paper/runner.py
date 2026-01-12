from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional
from threading import Event

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from atlas.broker.alpaca_broker import (
    submit_limit_order,
    submit_market_order,
    trading_client,
    wait_for_fill,
)
from atlas.config import AlpacaSettings
from atlas.data.alpaca_data import parse_alpaca_feed
from atlas.data.bars import filter_regular_hours, parse_bar_timeframe
from atlas.strategies.base import Strategy, StrategyState
from atlas.utils.time import NY_TZ, now_ny

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperConfig:
    symbols: list[str]
    bar_timeframe: str
    alpaca_feed: str
    lookback_bars: int
    poll_seconds: int
    max_position_notional_usd: float
    allow_short: bool
    regular_hours_only: bool
    allow_trading_when_closed: bool
    limit_offset_bps: float
    dry_run: bool


def _bars_client(settings: AlpacaSettings) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        settings.api_key, settings.secret_key, url_override=settings.data_url_override
    )


def _fetch_recent_bars(
    *,
    client: StockHistoricalDataClient,
    symbols: list[str],
    lookback_bars: int,
    timeframe: str,
    feed: str,
) -> pd.DataFrame:
    tf = parse_bar_timeframe(timeframe)
    feed_cfg = parse_alpaca_feed(feed)
    end = now_ny() - timedelta(minutes=feed_cfg.min_end_delay_minutes)
    start = end - timedelta(minutes=max(lookback_bars * tf.minutes * 2, 10))
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(amount=tf.minutes, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
        limit=max(lookback_bars, 10),
        feed=feed_cfg.api_feed,
    )
    res = client.get_stock_bars(req).df
    if res.index.tz is None:
        res.index = res.index.tz_localize("UTC")
    res.index = res.index.tz_convert(NY_TZ)
    res = res.sort_index()
    res = res[["open", "high", "low", "close", "volume"]].copy()
    return res


def _align_to_next_bar_open(now: pd.Timestamp, *, timeframe_minutes: int) -> float:
    timeframe_minutes = int(timeframe_minutes) if timeframe_minutes > 0 else 1
    current = now.floor("min")
    minutes = int(current.hour) * 60 + int(current.minute)
    next_minutes = ((minutes // timeframe_minutes) + 1) * timeframe_minutes
    next_open = current.normalize() + pd.Timedelta(minutes=next_minutes)
    sleep_s = (next_open - now).total_seconds()
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
    decisions_jsonl_path = run_dir / "decisions.jsonl"
    equity_path = run_dir / "equity_curve.csv"

    trade_client = trading_client(settings)
    data_client = _bars_client(settings)
    cfg_symbols = [s.strip().upper() for s in cfg.symbols if s.strip()]
    if not cfg_symbols:
        raise ValueError("cfg.symbols must be non-empty")
    tf = parse_bar_timeframe(cfg.bar_timeframe)

    with (
        orders_path.open("w", newline="") as f_orders,
        orders_jsonl_path.open("w") as f_orders_jsonl,
        fills_path.open("w", newline="") as f_fills,
        fills_jsonl_path.open("w") as f_fills_jsonl,
        decisions_jsonl_path.open("w") as f_decisions_jsonl,
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

        last_target: dict[str, float] = {s: 0.0 for s in cfg_symbols}
        holding_bars: dict[str, int] = {s: 0 for s in cfg_symbols}
        day_key: Optional[object] = None
        day_start_equity: Optional[float] = None
        loops = 0
        last_handled_bar_open: Optional[pd.Timestamp] = None

        initial_sleep = _align_to_next_bar_open(
            pd.Timestamp.now(tz=NY_TZ), timeframe_minutes=tf.minutes
        )
        if initial_sleep >= 1.0:
            logger.info("aligning to next bar open in %.1fs", initial_sleep)
            if stop_event is not None:
                if stop_event.wait(initial_sleep):
                    logger.info("stop requested, exiting paper loop")
                    return
            else:
                time.sleep(initial_sleep)

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("stop requested, exiting paper loop")
                return
            if max_loops is not None and loops >= max_loops:
                logger.info("max loops reached, stopping")
                return

            clock = trade_client.get_clock()
            market_open = bool(clock.is_open)
            if (not market_open) and (not cfg.allow_trading_when_closed):
                decision_ts = now_ny()
                f_decisions_jsonl.write(
                    json.dumps(
                        {
                            "timestamp": decision_ts.isoformat(),
                            "targets": {},
                            "reason": f"market closed: next_open={clock.next_open} next_close={clock.next_close}",
                            "debug": {"market_open": market_open},
                            "positions": {},
                            "equity": float(trade_client.get_account().equity),
                            "cash": float(trade_client.get_account().cash),
                        }
                    )
                    + "\n"
                )
                f_decisions_jsonl.flush()

                next_open = getattr(clock, "next_open", None)
                sleep_s = float(cfg.poll_seconds)
                if next_open is not None:
                    try:
                        sleep_s = max((pd.Timestamp(next_open) - pd.Timestamp(decision_ts)).total_seconds(), sleep_s)
                    except Exception:
                        sleep_s = float(cfg.poll_seconds)

                logger.info("market closed, sleeping %.1fs until next open", sleep_s)
                if stop_event is not None:
                    if stop_event.wait(sleep_s):
                        logger.info("stop requested, exiting paper loop")
                        return
                else:
                    time.sleep(sleep_s)
                continue

            now = pd.Timestamp.now(tz=NY_TZ)
            bar_open = now.floor(f"{int(tf.minutes)}min")
            if last_handled_bar_open is not None and bar_open <= last_handled_bar_open:
                sleep_s = _align_to_next_bar_open(now, timeframe_minutes=tf.minutes)
                logger.info("waiting for next bar open in %.1fs", sleep_s)
                if stop_event is not None:
                    if stop_event.wait(sleep_s):
                        logger.info("stop requested, exiting paper loop")
                        return
                else:
                    time.sleep(sleep_s)
                continue

            last_handled_bar_open = bar_open

            bars_df = _fetch_recent_bars(
                client=data_client,
                symbols=cfg_symbols,
                lookback_bars=cfg.lookback_bars,
                timeframe=cfg.bar_timeframe,
                feed=cfg.alpaca_feed,
            )
            if not isinstance(bars_df.index, pd.MultiIndex):
                raise RuntimeError("expected multi-index bars response from alpaca")

            bars_by_symbol: dict[str, pd.DataFrame] = {}
            for symbol in cfg_symbols:
                df = bars_df.xs(symbol)
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df.sort_index()
                if cfg.regular_hours_only:
                    df = filter_regular_hours(df)
                if len(df) > cfg.lookback_bars:
                    df = df.iloc[-cfg.lookback_bars :]
                bars_by_symbol[symbol] = df

            equity = float(trade_client.get_account().equity)
            cash_balance = float(trade_client.get_account().cash)
            decision_ts = bar_open

            for symbol in cfg_symbols:
                df = bars_by_symbol[symbol]
                if not len(df):
                    continue
                last_open = pd.Timestamp(df.index[-1])
                if last_open + pd.Timedelta(minutes=tf.minutes) > decision_ts:
                    df = df.iloc[:-1]
                    bars_by_symbol[symbol] = df

            if day_key != decision_ts.date():
                day_key = decision_ts.date()
                day_start_equity = equity
                holding_bars = {s: 0 for s in cfg_symbols}
            if day_start_equity is None:
                day_start_equity = equity

            positions: dict[str, float] = {}
            for symbol in cfg_symbols:
                try:
                    pos = trade_client.get_open_position(symbol_or_asset_id=symbol)
                    positions[symbol] = float(pos.qty)
                except Exception:
                    positions[symbol] = 0.0

            day_pnl = float(equity - float(day_start_equity))
            day_return = (
                float(day_pnl / float(day_start_equity))
                if float(day_start_equity) > 0
                else 0.0
            )
            for symbol in cfg_symbols:
                if abs(positions[symbol]) > 1e-8:
                    holding_bars[symbol] = holding_bars.get(symbol, 0) + 1
                else:
                    holding_bars[symbol] = 0

            state = StrategyState(
                timestamp=decision_ts,
                allow_short=cfg.allow_short,
                cash=cash_balance,
                positions=positions,
                equity=equity,
                day_start_equity=float(day_start_equity),
                day_pnl=day_pnl,
                day_return=day_return,
                holding_bars={s: int(holding_bars[s]) for s in cfg_symbols},
            )
            decision = strategy.target_exposures(bars_by_symbol, state)
            targets = {s: float(decision.target_exposures.get(s, 0.0)) for s in cfg_symbols}
            f_decisions_jsonl.write(
                json.dumps(
                    {
                        "timestamp": decision_ts.isoformat(),
                        "targets": targets,
                        "reason": decision.reason,
                        "debug": decision.debug,
                        "positions": positions,
                        "equity": equity,
                        "cash": cash_balance,
                    }
                )
                + "\n"
            )
            f_decisions_jsonl.flush()

            orders: list[dict[str, object]] = []
            for symbol in cfg_symbols:
                target_exposure = float(targets.get(symbol, 0.0))
                if not cfg.allow_short and target_exposure < 0:
                    target_exposure = 0.0

                if symbol not in bars_by_symbol or not len(bars_by_symbol[symbol]):
                    continue

                last_price = float(bars_by_symbol[symbol]["close"].iloc[-1])
                target_notional = target_exposure * float(cfg.max_position_notional_usd)
                target_qty = target_notional / last_price if last_price > 0 else 0.0

                current_qty = float(positions.get(symbol, 0.0))
                delta_qty = float(target_qty - current_qty)
                if abs(delta_qty) <= 1e-6:
                    last_target[symbol] = target_exposure
                    continue

                reducing_risk = (current_qty > 1e-8 and delta_qty < 0) or (
                    current_qty < -1e-8 and delta_qty > 0
                )
                priority = 0 if reducing_risk else 1
                side = "BUY" if delta_qty > 0 else "SELL"
                orders.append(
                    {
                        "priority": int(priority),
                        "symbol": symbol,
                        "side": side,
                        "qty": float(abs(delta_qty)),
                        "target_exposure": float(target_exposure),
                    }
                )

            orders.sort(key=lambda o: (int(o["priority"]), str(o["symbol"])))

            for order in orders:
                if stop_event is not None and stop_event.is_set():
                    logger.info("stop requested, exiting paper loop")
                    return

                symbol = str(order["symbol"])
                side = str(order["side"])
                qty = float(order["qty"])
                target_exposure = float(order["target_exposure"])

                if abs(target_exposure - last_target.get(symbol, 0.0)) < 1e-8:
                    continue

                ts = now_ny().isoformat()
                if cfg.dry_run:
                    order_id = "dry_run"
                else:
                    if market_open:
                        order_id = submit_market_order(
                            client=trade_client, symbol=symbol, qty=qty, side=side
                        )
                    else:
                        last_price = float(bars_by_symbol[symbol]["close"].iloc[-1])
                        offset = float(cfg.limit_offset_bps) / 10_000.0
                        px = last_price * (1.0 + offset) if side.upper() == "BUY" else last_price * (1.0 - offset)
                        order_id = submit_limit_order(
                            client=trade_client,
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            limit_price=round(float(px), 2),
                            extended_hours=True,
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

            pd.DataFrame(
                [
                    {
                        "timestamp": now_ny().isoformat(),
                        "equity": equity,
                        "day_return": day_return,
                    }
                ]
            ).to_csv(equity_path, mode="a", header=not equity_path.exists(), index=False)

            loops += 1
            sleep_s = _align_to_next_bar_open(
                pd.Timestamp.now(tz=NY_TZ), timeframe_minutes=tf.minutes
            )
            logger.info("sleeping until next bar open in %.1fs", sleep_s)
            if stop_event is not None:
                if stop_event.wait(sleep_s):
                    logger.info("stop requested, exiting paper loop")
                    return
            else:
                time.sleep(sleep_s)

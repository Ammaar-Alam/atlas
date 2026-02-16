from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from atlas.broker.alpaca_broker import (
    submit_limit_order as submit_alpaca_limit_order,
    submit_market_order as submit_alpaca_market_order,
    trading_client,
    wait_for_fill as wait_for_alpaca_fill,
)
from atlas.broker.coinbase_broker import (
    client as coinbase_client,
    submit_market_order as submit_coinbase_market_order,
    wait_for_fill as wait_for_coinbase_fill,
)
from atlas.config import AlpacaSettings
from atlas.data.alpaca_data import parse_alpaca_feed, to_alpaca_timeframe
from atlas.data.bars import filter_regular_hours, parse_bar_timeframe, resample_ohlcv
from atlas.market import Market, coerce_symbols_for_market, parse_market
from atlas.strategies.base import Strategy, StrategyState
from atlas.utils.time import NY_TZ, now_ny

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperConfig:
    symbols: list[str]
    bar_timeframe: str
    data_source: str
    execution_venue: str
    alpaca_feed: str
    lookback_bars: int
    poll_seconds: int
    initial_cash_usd: float
    max_position_notional_usd: float
    slippage_bps: float
    taker_fee_bps: float
    fixed_fee_per_contract_usd: float
    contract_size_units: float
    allow_short: bool
    regular_hours_only: bool
    allow_trading_when_closed: bool
    limit_offset_bps: float
    dry_run: bool
    market: str = "equity"


def _fixed_fee_from_fill_qty(
    *,
    fill_qty: float,
    fixed_fee_per_contract_usd: float,
    contract_size_units: float,
) -> float:
    if float(fixed_fee_per_contract_usd) <= 0.0:
        return 0.0
    contract_size = float(contract_size_units or 1.0)
    if contract_size <= 0.0:
        contract_size = 1.0
    contracts = math.ceil((abs(float(fill_qty)) - 1e-12) / contract_size)
    if contracts <= 0:
        return 0.0
    return float(contracts * float(fixed_fee_per_contract_usd))


def _stock_bars_client(settings: AlpacaSettings):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Paper trading requires the optional 'alpaca-py' dependency. "
            "Install it (e.g. `pip install alpaca-py`) to use `atlas paper`."
        ) from exc
    return StockHistoricalDataClient(settings.api_key, settings.secret_key, url_override=settings.data_url_override)


def _make_crypto_bars_client(settings: AlpacaSettings):
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Paper trading requires the optional 'alpaca-py' dependency. "
            "Install it (e.g. `pip install alpaca-py`) to use `atlas paper`."
        ) from exc
    kwargs: dict[str, object] = {}
    if settings.api_key and settings.secret_key:
        kwargs["api_key"] = settings.api_key
        kwargs["secret_key"] = settings.secret_key
    if settings.data_url_override:
        kwargs["url_override"] = settings.data_url_override
    try:
        return CryptoHistoricalDataClient(**kwargs)
    except TypeError:
        kwargs.pop("url_override", None)
        return CryptoHistoricalDataClient(**kwargs)


def _to_utc(dt: pd.Timestamp) -> pd.Timestamp:
    if dt.tzinfo is None:
        dt = dt.tz_localize(NY_TZ)
    return dt.tz_convert(ZoneInfo("UTC"))


def _normalize_bars_index_to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        symbols = df.index.get_level_values(0)
        ts = pd.DatetimeIndex(df.index.get_level_values(1))
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        ts = ts.tz_convert(NY_TZ)
        out = df.copy()
        out.index = pd.MultiIndex.from_arrays([symbols, ts], names=df.index.names)
        return out
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        out = df.copy()
        out.index = ts.tz_convert(NY_TZ)
        return out
    raise RuntimeError("unexpected bars index type (expected DatetimeIndex or MultiIndex)")


def _fetch_recent_bars(
    *,
    settings: AlpacaSettings,
    symbols: list[str],
    lookback_bars: int,
    timeframe: str,
    feed: str,
    market: Market,
) -> pd.DataFrame:
    try:
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Paper trading requires the optional 'alpaca-py' dependency. "
            "Install it (e.g. `pip install alpaca-py`) to use `atlas paper`."
        ) from exc
    tf = parse_bar_timeframe(timeframe)
    end = now_ny()
    start = end - timedelta(minutes=max(lookback_bars * tf.minutes * 2, 10))
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )

    if market == Market.CRYPTO:
        client = _make_crypto_bars_client(settings)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=to_alpaca_timeframe(tf),
            start=_to_utc(pd.Timestamp(start)),
            end=_to_utc(pd.Timestamp(end)),
        )
        res = client.get_crypto_bars(req).df
        if res is None or len(res) == 0:
            return empty
    else:
        client = _stock_bars_client(settings)
        feed_cfg = parse_alpaca_feed(feed)
        end = end - timedelta(minutes=feed_cfg.min_end_delay_minutes)
        start = end - timedelta(minutes=max(lookback_bars * tf.minutes * 2, 10))
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=to_alpaca_timeframe(tf),
            start=start,
            end=end,
            feed=feed_cfg.api_feed,
        )
        res = client.get_stock_bars(req).df
        if res is None or len(res) == 0:
            return empty

    res = _normalize_bars_index_to_ny(res)
    res = res.sort_index()
    res = res[["open", "high", "low", "close", "volume"]].copy()
    return res


def _coinbase_granularity_for_fetch(fetch_minutes: int) -> str:
    if fetch_minutes <= 1:
        return "ONE_MINUTE"
    if fetch_minutes == 5:
        return "FIVE_MINUTE"
    if fetch_minutes in {15, 30}:
        # Coinbase does not provide native 30-minute candles.
        return "FIFTEEN_MINUTE"
    if fetch_minutes == 60:
        return "ONE_HOUR"
    if fetch_minutes == 360:
        return "SIX_HOUR"
    if fetch_minutes == 1440:
        return "ONE_DAY"
    return "ONE_MINUTE"


def _fetch_recent_bars_coinbase(
    *,
    client,
    symbols: list[str],
    lookback_bars: int,
    timeframe: str,
) -> pd.DataFrame:
    tf = parse_bar_timeframe(timeframe)
    fetch_minutes = int(tf.minutes)
    granularity = _coinbase_granularity_for_fetch(fetch_minutes)
    end = pd.Timestamp.now(tz=ZoneInfo("UTC"))
    start = end - timedelta(minutes=max(lookback_bars * fetch_minutes * 2, 10))

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = client.get_product_candles(
            product_id=symbol,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            granularity=granularity,
        )
        if df is None or df.empty:
            continue
        out = df[["open", "high", "low", "close", "volume"]].copy()
        out = out.sort_index()
        out["symbol"] = symbol
        out = out.reset_index(names="timestamp").set_index(["symbol", "timestamp"])
        frames.append(out)

    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
    )
    if not frames:
        return empty

    return pd.concat(frames).sort_index()


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
    settings: Optional[AlpacaSettings],
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

    data_source = str(cfg.data_source or "alpaca").strip().lower()
    execution_venue = str(cfg.execution_venue or "alpaca").strip().lower()
    if data_source not in {"alpaca", "coinbase"}:
        raise ValueError(f"unsupported paper data_source: {cfg.data_source!r}")
    if execution_venue not in {"alpaca", "coinbase"}:
        raise ValueError(f"unsupported paper execution_venue: {cfg.execution_venue!r}")

    trade_client = None
    if execution_venue == "alpaca":
        if settings is None:
            raise ValueError("alpaca settings are required for execution_venue=alpaca")
        trade_client = trading_client(settings)

    cb_client = None
    if data_source == "coinbase" or execution_venue == "coinbase":
        cb_client = coinbase_client()

    mkt = parse_market(cfg.market)
    if mkt == Market.DERIVATIVES and data_source != "coinbase":
        raise ValueError("market=derivatives requires data_source=coinbase")
    if mkt == Market.DERIVATIVES and execution_venue != "coinbase":
        raise ValueError("market=derivatives requires execution_venue=coinbase")
    if mkt == Market.EQUITY and execution_venue == "coinbase":
        raise ValueError("execution_venue=coinbase supports crypto/derivatives only")
    if execution_venue == "coinbase" and data_source != "coinbase":
        raise ValueError("execution_venue=coinbase currently requires data_source=coinbase")
    if execution_venue == "coinbase" and (not cfg.dry_run):
        if cb_client is None:
            raise ValueError("coinbase client is required for execution_venue=coinbase")
        if not (cb_client.settings.api_key and cb_client.settings.api_secret):
            raise ValueError(
                "missing coinbase api credentials: set COINBASE_API_KEY and COINBASE_API_SECRET in .env"
            )

    cfg_symbols = coerce_symbols_for_market(cfg.symbols, mkt)
    if not cfg_symbols:
        raise ValueError("cfg.symbols must be non-empty")
    tf = parse_bar_timeframe(cfg.bar_timeframe)

    fetch_timeframe = cfg.bar_timeframe
    fetch_lookback_bars = int(cfg.lookback_bars)
    if (
        mkt in {Market.CRYPTO, Market.DERIVATIVES}
        and int(tf.minutes) >= 120
        and int(tf.minutes) % 60 == 0
    ):
        # For multi-hour crypto/derivatives, fetch 1H bars and resample locally to keep candle boundaries
        # deterministic (matches `load_universe_bars` behavior used in backtests).
        fetch_timeframe = "1H"
        fetch_lookback_bars = int(cfg.lookback_bars * (int(tf.minutes) // 60))
    elif data_source == "coinbase" and int(tf.minutes) == 30:
        # Coinbase has no native 30-minute bars; fetch 15-minute and resample locally.
        fetch_timeframe = "15Min"
        fetch_lookback_bars = int(cfg.lookback_bars * 2)

    synthetic_cash = float(cfg.initial_cash_usd) if float(cfg.initial_cash_usd) > 0 else float(
        cfg.max_position_notional_usd
    )
    synthetic_positions: dict[str, float] = {s: 0.0 for s in cfg_symbols}
    synthetic_entry_prices: dict[str, float] = {s: 0.0 for s in cfg_symbols}

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

        now_for_bins = pd.Timestamp.now(tz=NY_TZ)
        if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
            now_for_bins = now_for_bins.tz_convert(ZoneInfo("UTC"))
        initial_sleep = _align_to_next_bar_open(now_for_bins, timeframe_minutes=tf.minutes)
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

            market_open = True
            clock = None
            if execution_venue == "alpaca" and mkt != Market.CRYPTO:
                if trade_client is None:
                    raise RuntimeError("alpaca trade client is not initialized")
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
                                "debug": {"market_open": market_open, "market": mkt.value},
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
                            sleep_s = max(
                                (pd.Timestamp(next_open) - pd.Timestamp(decision_ts)).total_seconds(),
                                sleep_s,
                            )
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
            now_bins = now.tz_convert(ZoneInfo("UTC")) if mkt in {Market.CRYPTO, Market.DERIVATIVES} else now
            bar_open = now_bins.floor(f"{int(tf.minutes)}min")
            if last_handled_bar_open is not None and bar_open <= last_handled_bar_open:
                sleep_s = _align_to_next_bar_open(now_bins, timeframe_minutes=tf.minutes)
                logger.info("waiting for next bar open in %.1fs", sleep_s)
                if stop_event is not None:
                    if stop_event.wait(sleep_s):
                        logger.info("stop requested, exiting paper loop")
                        return
                else:
                    time.sleep(sleep_s)
                continue

            last_handled_bar_open = bar_open

            if data_source == "alpaca":
                if settings is None:
                    raise RuntimeError("alpaca settings are required for data_source=alpaca")
                bars_df = _fetch_recent_bars(
                    settings=settings,
                    symbols=cfg_symbols,
                    lookback_bars=fetch_lookback_bars,
                    timeframe=fetch_timeframe,
                    feed=cfg.alpaca_feed,
                    market=mkt,
                )
            else:
                if cb_client is None:
                    raise RuntimeError("coinbase client is not initialized")
                bars_df = _fetch_recent_bars_coinbase(
                    client=cb_client,
                    symbols=cfg_symbols,
                    lookback_bars=fetch_lookback_bars,
                    timeframe=fetch_timeframe,
                )

            if not isinstance(bars_df.index, pd.MultiIndex):
                raise RuntimeError("expected multi-index bars response")

            bars_by_symbol: dict[str, pd.DataFrame] = {}
            symbols_present = set(bars_df.index.get_level_values(0).unique())
            for symbol in cfg_symbols:
                if symbol not in symbols_present:
                    logger.warning(
                        "%s returned no bars for %s (market=%s)",
                        data_source,
                        symbol,
                        mkt.value,
                    )
                    bars_by_symbol[symbol] = pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex([], tz=NY_TZ),
                    )
                    continue

                df = bars_df.xs(symbol)
                df = df[["open", "high", "low", "close", "volume"]].copy()
                df = df.sort_index()
                if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
                    df.index = df.index.tz_convert(ZoneInfo("UTC"))
                if cfg.regular_hours_only and mkt == Market.EQUITY:
                    df = filter_regular_hours(df)
                if len(df) > fetch_lookback_bars:
                    df = df.iloc[-fetch_lookback_bars :]
                if fetch_timeframe != cfg.bar_timeframe:
                    df = resample_ohlcv(df, minutes=int(tf.minutes), drop_zero_volume=False)
                if len(df) > cfg.lookback_bars:
                    df = df.iloc[-cfg.lookback_bars :]
                bars_by_symbol[symbol] = df

            decision_ts = bar_open

            for symbol in cfg_symbols:
                df = bars_by_symbol[symbol]
                if not len(df):
                    continue
                last_open = pd.Timestamp(df.index[-1])
                if last_open + pd.Timedelta(minutes=tf.minutes) > bar_open:
                    df = df.iloc[:-1]
                    bars_by_symbol[symbol] = df

            last_prices: dict[str, float] = {}
            for symbol in cfg_symbols:
                df = bars_by_symbol.get(symbol)
                if df is not None and len(df):
                    last_prices[symbol] = float(df["close"].iloc[-1])

            if execution_venue == "alpaca":
                if trade_client is None:
                    raise RuntimeError("alpaca trade client is not initialized")
                equity = float(trade_client.get_account().equity)
                cash_balance = float(trade_client.get_account().cash)
                positions: dict[str, float] = {}
                for symbol in cfg_symbols:
                    try:
                        pos = trade_client.get_open_position(symbol_or_asset_id=symbol)
                        positions[symbol] = float(pos.qty)
                    except Exception:
                        positions[symbol] = 0.0
            else:
                cash_balance = float(synthetic_cash)
                positions = {s: float(synthetic_positions.get(s, 0.0)) for s in cfg_symbols}
                equity = float(cash_balance)
                if mkt == Market.DERIVATIVES:
                    for symbol, qty in positions.items():
                        last_px = float(last_prices.get(symbol, 0.0))
                        entry_px = float(synthetic_entry_prices.get(symbol, 0.0))
                        if abs(qty) <= 1e-12 or last_px <= 0 or entry_px <= 0:
                            continue
                        if qty > 0:
                            equity += float((last_px - entry_px) * qty)
                        else:
                            equity += float((entry_px - last_px) * abs(qty))
                else:
                    for symbol, qty in positions.items():
                        last_px = float(last_prices.get(symbol, 0.0))
                        equity += float(qty) * last_px

            if day_key != decision_ts.date():
                day_key = decision_ts.date()
                day_start_equity = equity
                holding_bars = {s: 0 for s in cfg_symbols}
            if day_start_equity is None:
                day_start_equity = equity

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
                extra={
                    "max_position_notional_usd": float(cfg.max_position_notional_usd),
                    "slippage_bps": float(cfg.slippage_bps),
                    "taker_fee_bps": float(cfg.taker_fee_bps),
                    "fixed_fee_per_contract_usd": float(cfg.fixed_fee_per_contract_usd),
                    "contract_size_units": float(cfg.contract_size_units),
                },
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
                        "market": mkt.value,
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
                    if execution_venue == "alpaca":
                        if trade_client is None:
                            raise RuntimeError("alpaca trade client is not initialized")
                        if market_open:
                            order_id = submit_alpaca_market_order(
                                client=trade_client, symbol=symbol, qty=qty, side=side, market=mkt.value
                            )
                        else:
                            last_price = float(bars_by_symbol[symbol]["close"].iloc[-1])
                            offset = float(cfg.limit_offset_bps) / 10_000.0
                            px = (
                                last_price * (1.0 + offset)
                                if side.upper() == "BUY"
                                else last_price * (1.0 - offset)
                            )
                            order_id = submit_alpaca_limit_order(
                                client=trade_client,
                                symbol=symbol,
                                qty=qty,
                                side=side,
                                limit_price=round(float(px), 2),
                                extended_hours=True,
                                market=mkt.value,
                            )
                    else:
                        if cb_client is None:
                            raise RuntimeError("coinbase client is not initialized")
                        order_id = submit_coinbase_market_order(
                            client=cb_client,
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            market=mkt.value,
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
                    if execution_venue == "alpaca":
                        if trade_client is None:
                            raise RuntimeError("alpaca trade client is not initialized")
                        fill = wait_for_alpaca_fill(
                            client=trade_client, order_id=order_id, timeout_s=60, poll_s=2.0
                        )
                    else:
                        if cb_client is None:
                            raise RuntimeError("coinbase client is not initialized")
                        fill = wait_for_coinbase_fill(
                            client=cb_client, order_id=order_id, timeout_s=60, poll_s=2.0
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

                    if execution_venue == "coinbase":
                        fill_qty = float(fill.filled_qty or 0.0)
                        fill_px = (
                            float(fill.filled_avg_price)
                            if fill.filled_avg_price not in (None, "")
                            else float(last_prices.get(symbol, 0.0))
                        )
                        if fill_qty > 0 and fill_px > 0:
                            signed = fill_qty if side.upper() == "BUY" else -fill_qty
                            prev_qty = float(synthetic_positions.get(symbol, 0.0))
                            prev_entry = float(synthetic_entry_prices.get(symbol, 0.0))

                            if mkt == Market.DERIVATIVES:
                                reducing_or_closing = (
                                    (prev_qty > 1e-12 and signed < 0)
                                    or (prev_qty < -1e-12 and signed > 0)
                                )
                                if reducing_or_closing and prev_entry > 0:
                                    close_qty = float(min(abs(prev_qty), abs(signed)))
                                    if close_qty > 0:
                                        if prev_qty > 0:
                                            realized = float((fill_px - prev_entry) * close_qty)
                                        else:
                                            realized = float((prev_entry - fill_px) * close_qty)
                                        synthetic_cash = float(synthetic_cash + realized)

                                percent_fee = float(abs(fill_qty * fill_px) * (float(cfg.taker_fee_bps) / 10_000.0))
                                contract_size = float(fill.contract_size or cfg.contract_size_units or 1.0)
                                fixed_fee = _fixed_fee_from_fill_qty(
                                    fill_qty=fill_qty,
                                    fixed_fee_per_contract_usd=float(cfg.fixed_fee_per_contract_usd),
                                    contract_size_units=contract_size,
                                )
                                fee_usd = float(percent_fee + fixed_fee)
                                synthetic_cash = float(synthetic_cash - fee_usd)

                                new_qty = float(prev_qty + signed)
                                if abs(new_qty) <= 1e-12:
                                    new_qty = 0.0
                                    synthetic_entry_prices[symbol] = 0.0
                                elif abs(prev_qty) <= 1e-12:
                                    synthetic_entry_prices[symbol] = float(fill_px)
                                elif (prev_qty > 0 and new_qty > 0 and signed > 0) or (
                                    prev_qty < 0 and new_qty < 0 and signed < 0
                                ):
                                    weighted_entry = (
                                        abs(prev_qty) * prev_entry + abs(signed) * float(fill_px)
                                    ) / abs(new_qty)
                                    synthetic_entry_prices[symbol] = float(weighted_entry)
                                elif (prev_qty > 0 and new_qty < 0) or (prev_qty < 0 and new_qty > 0):
                                    synthetic_entry_prices[symbol] = float(fill_px)
                                # If reducing without flipping, keep prior entry.
                                synthetic_positions[symbol] = float(new_qty)
                            else:
                                synthetic_positions[symbol] = float(prev_qty + signed)
                                synthetic_cash = float(synthetic_cash - (signed * fill_px))
                                percent_fee = float(abs(fill_qty * fill_px) * (float(cfg.taker_fee_bps) / 10_000.0))
                                contract_size = float(fill.contract_size or cfg.contract_size_units or 1.0)
                                fixed_fee = _fixed_fee_from_fill_qty(
                                    fill_qty=fill_qty,
                                    fixed_fee_per_contract_usd=float(cfg.fixed_fee_per_contract_usd),
                                    contract_size_units=contract_size,
                                )
                                fee_usd = float(percent_fee + fixed_fee)
                                synthetic_cash = float(synthetic_cash - fee_usd)

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
            sleep_now = pd.Timestamp.now(tz=NY_TZ)
            if mkt in {Market.CRYPTO, Market.DERIVATIVES}:
                sleep_now = sleep_now.tz_convert(ZoneInfo("UTC"))
            sleep_s = _align_to_next_bar_open(sleep_now, timeframe_minutes=tf.minutes)
            logger.info("sleeping until next bar open in %.1fs", sleep_s)
            if stop_event is not None:
                if stop_event.wait(sleep_s):
                    logger.info("stop requested, exiting paper loop")
                    return
            else:
                time.sleep(sleep_s)

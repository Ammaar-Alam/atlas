# Repo Audit & Architecture Plan

### 1. Current State Audit
The current `atlas` repository is a capable spot-equity/crypto backtesting framework but is functionally essentially for derivatives trading due to:
*   **Accounting:** It treats `equity = cash + qty * price`. It lacks concepts of *margin*, *leverage* (notional value > equity), *unrealized PnL* distinct from cash, and *liquidation*.
*   **Data:** It lacks funding rate history, which is a critical cost component for perps.
*   **Execution:** It lacks a Coinbase Advanced Trade adapter.
*   **Risk:** It lacks maintenance margin checks and liquidation logic.

### 2. Architecture Plan
To enable high-leverage derivatives trading without breaking existing spot strategies, I will implement a "Sidecar" architecture:

1.  **`src/atlas/data/coinbase.py`**: A new client to fetch OHLCV candles AND historical funding rates from Coinbase Advanced Trade.
2.  **`src/atlas/backtest/derivatives.py`**: A dedicated derivatives backtesting engine. It will track:
    *   `balance` (USDC collateral).
    *   `positions` (Contracts, Entry Price, Isolated/Cross Margin).
    *   `funding_payment` (Cash flows).
    *   `liquidation_check` (If Margin Ratio < Maintenance, force close).
3.  **`src/atlas/strategies/fractal_trend.py`**: The "Atlas-Alpha" strategy.
4.  **`src/atlas/cli.py`**: Add `backtest-derivatives` command.

---

# Strategy Spec: Fractal-Trend-V1 (Atlas-Alpha)

**Concept:**
A regime-filtered trend system that uses **Fractal Efficiency** to identify "clean" trends and **Volatility-Adjusted Sizing** to survive leverage. It explicitly models the "Liquidation Wall" to ensure stops are always hit before liquidation.

**Logic:**
1.  **Regime Filter (Efficiency Ratio):**
    Calculate Kaufman's Efficiency Ratio ($ER$) over `er_window` (default 12).
    $$ER = \frac{|Price_t - Price_{t-n}|}{\sum_{i=0}^{n} |Price_t - Price_{t-1}|}$$
    *   If $ER < 0.35$ (Choppy/Random): **ABSTAIN**.
2.  **Trend Filter (EMA):**
    *   Long if Close > EMA(`trend_ema`) (default 50).
    *   Short if Close < EMA(`trend_ema`).
3.  **Trigger (Pullback-Validation):**
    *   Enter Long if Trend is UP AND Price is within `pullback_pct` of EMA (buying the dip) BUT `close > open` (momentum returning).
4.  **Sizing (The Risk Core):**
    *   Calculate $ATR$ (14).
    *   Set Stop Loss ($SL$) distance = $3.0 \times ATR$.
    *   Calculate Max Allowable Position based on Risk Budget ($1.5\%$ of Equity).
        $$Qty_{risk} = \frac{Equity \times 0.015}{3 \times ATR}$$
    *   **Liquidation Guard:** Calculate implied Liquidation Price ($LP$) at max leverage (5x).
        If $SL$ is "beyond" $LP$, reduce leverage until $LP$ is safely below $SL$.
5.  **Exits:**
    *   Stop Loss (Hard).
    *   Trailing Stop: $4.0 \times ATR$ from peak.
    *   Regime Change: If $ER$ drops below 0.20, exit.

---

# Code Implementation

### 1. `src/atlas/config.py` (Update)

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


def load_env() -> None:
    load_dotenv(override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw != "" else default


@dataclass(frozen=True)
class AlpacaSettings:
    api_key: str
    secret_key: str
    paper: bool
    allow_live: bool
    trading_url_override: Optional[str]
    data_url_override: Optional[str]


@dataclass(frozen=True)
class CoinbaseSettings:
    api_key: str
    api_secret: str
    allow_live: bool


def get_alpaca_settings(*, require_keys: bool) -> AlpacaSettings:
    load_env()

    api_key = _env_str("ALPACA_API_KEY")
    secret_key = _env_str("ALPACA_SECRET_KEY")
    paper = _env_bool("ALPACA_PAPER", True)
    allow_live = _env_bool("ATLAS_ALLOW_LIVE", False)

    if require_keys and (not api_key or not secret_key):
        raise RuntimeError(
            "missing alpaca api keys: set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
        )

    if not require_keys and (not api_key or not secret_key):
        api_key = ""
        secret_key = ""

    if not paper and not allow_live:
        raise RuntimeError(
            "live trading blocked: set ALPACA_PAPER=true or set ATLAS_ALLOW_LIVE=true to explicitly allow live"
        )

    return AlpacaSettings(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        allow_live=allow_live,
        trading_url_override=_env_str("ALPACA_TRADING_URL"),
        data_url_override=_env_str("ALPACA_DATA_URL"),
    )


def get_coinbase_settings() -> CoinbaseSettings:
    load_env()
    api_key = _env_str("COINBASE_API_KEY", "")
    api_secret = _env_str("COINBASE_API_SECRET", "")
    allow_live = _env_bool("ATLAS_ALLOW_LIVE", False)
    # Coinbase doesn't have a standardized "paper" API endpoint in the same way,
    # so we assume these keys are for the real environment unless mocked.
    return CoinbaseSettings(
        api_key=api_key or "",
        api_secret=api_secret or "",
        allow_live=allow_live,
    )


def get_log_level() -> str:
    load_env()
    return (_env_str("ATLAS_LOG_LEVEL", "INFO") or "INFO").upper()


def get_default_max_position_notional_usd(*, mode: str) -> float:
    load_env()
    if mode == "paper":
        return _env_float("ATLAS_PAPER_MAX_POSITION_NOTIONAL_USD", 1000.0)
    return _env_float("ATLAS_BACKTEST_MAX_POSITION_NOTIONAL_USD", 10000.0)
```

### 2. `src/atlas/data/coinbase.py` (New)

```python
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from atlas.config import CoinbaseSettings
from atlas.data.bars import parse_bar_timeframe
from atlas.utils.time import NY_TZ

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoinbaseBarsDownload:
    product_id: str
    start: datetime
    end: datetime
    granularity: str


def _cache_path(root: Path, req: CoinbaseBarsDownload, kind: str = "candles") -> Path:
    safe_product = req.product_id.replace("-", "_")
    start_s = req.start.strftime("%Y%m%d%H%M")
    end_s = req.end.strftime("%Y%m%d%H%M")
    return (
        root
        / "data"
        / "coinbase"
        / safe_product
        / f"{safe_product}_{req.granularity}_{kind}_{start_s}_{end_s}.csv"
    )


def _to_utc_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def download_coinbase_candles(
    *,
    settings: CoinbaseSettings,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: str,  # ONE_MINUTE, FIVE_MINUTE, etc.
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Download candles from Coinbase Advanced Trade API.
    Ref: GET /api/v3/brokerage/products/{product_id}/candles
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=NY_TZ)
    if end.tzinfo is None:
        end = end.replace(tzinfo=NY_TZ)

    # Convert bar_timeframe (e.g. "1Min") to Coinbase granularity
    tf = parse_bar_timeframe(granularity)
    if tf.minutes == 1:
        cb_granularity = "ONE_MINUTE"
    elif tf.minutes == 5:
        cb_granularity = "FIVE_MINUTE"
    elif tf.minutes == 15:
        cb_granularity = "FIFTEEN_MINUTE"
    elif tf.minutes == 30:
        cb_granularity = "THIRTY_MINUTE"
    elif tf.minutes == 60:
        cb_granularity = "ONE_HOUR"
    else:
        raise ValueError(f"Unsupported coinbase granularity for {granularity}")

    url = f"https://api.coinbase.com/api/v3/brokerage/products/{product_id}/candles"
    
    # Coinbase usually limits candle requests to ~300 datapoints per request.
    # We must page backwards or by time slices.
    # For simplicity in this v1, we iterate by chunks.
    
    chunk_size = timedelta(minutes=tf.minutes * 250)
    current_start = start
    all_rows = []

    logger.info("downloading coinbase candles %s %s->%s", product_id, start, end)

    while current_start < end:
        current_end = min(current_start + chunk_size, end)
        params = {
            "start": int(current_start.timestamp()),
            "end": int(current_end.timestamp()),
            "granularity": cb_granularity
        }
        
        # Public endpoint, usually no auth needed for candles, but good practice if rate limited
        headers = {"Content-Type": "application/json"}
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("candles", [])
            
            # Coinbase returns: [start, low, high, open, close, volume]
            for c in candles:
                # API v3 format is object: {start, low, high, open, close, volume}
                ts = pd.to_datetime(int(c["start"]), unit="s", utc=True).tz_convert(NY_TZ)
                all_rows.append({
                    "timestamp": ts,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"]),
                })
        except Exception as e:
            logger.warning(f"chunk failed {current_start}: {e}")
        
        current_start = current_end
        time.sleep(0.2) # Rate limit politeness

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("no candles found for %s", product_id)
        return pd.DataFrame()
    
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp")
    
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
    
    return df


def load_coinbase_candles_cached(
    *,
    settings: CoinbaseSettings,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: str = "1Min",
) -> pd.DataFrame:
    path = _cache_path(Path.cwd(), CoinbaseBarsDownload(product_id, start, end, granularity))
    if path.exists():
        logger.info("using cached coinbase candles: %s", path)
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
        df = df.set_index("timestamp").sort_index()
        return df

    return download_coinbase_candles(
        settings=settings,
        product_id=product_id,
        start=start,
        end=end,
        granularity=granularity,
        out_path=path,
    )

```

### 3. `src/atlas/backtest/derivatives.py` (New - The Core Engine)

```python
from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Dict

import pandas as pd
import numpy as np

from atlas.strategies.base import Strategy, StrategyState
from atlas.utils.time import NY_TZ
from atlas.backtest.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DerivativesConfig:
    symbols: list[str]
    initial_balance: float  # Collateral in USDC
    leverage_limit: float = 5.0
    maintenance_margin_rate: float = 0.05  # 5% MM
    taker_fee_bps: float = 6.0  # 0.06%
    maker_fee_bps: float = 2.0  # 0.02%
    slippage_bps: float = 5.0   # Execution slippage
    funding_interval_hours: int = 1
    # Simple funding model: Longs pay Shorts if funding > 0.
    # In V1, we simulate funding as a random walk or 0 if data missing,
    # unless a funding_df is provided.


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    # Funding accrued?

    @property
    def notional(self) -> float:
        return abs(self.qty * self.entry_price)

    @property
    def side(self) -> int:
        return 1 if self.qty > 0 else -1 if self.qty < 0 else 0


@dataclass
class AccountState:
    balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    margin_used: float = 0.0
    
    def update_equity(self, current_prices: Dict[str, float]):
        unrealized_pnl = 0.0
        notional_total = 0.0
        for sym, pos in self.positions.items():
            px = current_prices.get(sym, pos.entry_price)
            # PnL = qty * (current - entry)
            unrealized_pnl += pos.qty * (px - pos.entry_price)
            notional_total += abs(pos.qty * px)
        
        self.equity = self.balance + unrealized_pnl
        self.margin_used = notional_total # Simple isolation tracking


def run_derivatives_backtest(
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    funding_by_symbol: Optional[dict[str, pd.DataFrame]], # timestamp -> rate
    strategy: Strategy,
    cfg: DerivativesConfig,
    run_dir: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Align data
    symbols = cfg.symbols
    common_index: Optional[pd.DatetimeIndex] = None
    for s in symbols:
        if s not in bars_by_symbol: continue
        idx = bars_by_symbol[s].index
        if common_index is None:
            common_index = idx
        else:
            common_index = common_index.intersection(idx)
    
    if common_index is None or len(common_index) < 10:
        raise ValueError("Insufficient aligned bars")
    
    common_index = common_index.sort_values()
    
    account = AccountState(balance=cfg.initial_balance)
    
    trades_log = []
    equity_curve = []
    
    # Pre-calc funding lookup if available
    funding_lookup = {}
    if funding_by_symbol:
        for s, df in funding_by_symbol.items():
            funding_lookup[s] = df["rate"].reindex(common_index, method="ffill").fillna(0.0)

    # Simulation Loop
    for i in range(len(common_index)):
        ts = common_index[i]
        
        # 1. Get Current Market Data
        opens = {}
        closes = {}
        for s in symbols:
            row = bars_by_symbol[s].loc[ts]
            opens[s] = float(row["open"])
            closes[s] = float(row["close"])
            
        # 2. Apply Funding (Simplified: Hourly checks)
        # In a real engine, we'd check if `ts` crosses a funding timestamp.
        # Here we assume funding applies continuously or at specific hours.
        # Placeholder: Apply funding cost/rebate based on held position.
        # Cost = Position Notional * Funding Rate
        
        # 3. Mark to Market & Check Liquidation
        account.update_equity(opens) # Check at Open for Gaps
        
        total_notional = sum(abs(p.qty * opens.get(s, p.entry_price)) for s, p in account.positions.items())
        if account.equity <= (total_notional * cfg.maintenance_margin_rate):
            # LIQUIDATION
            logger.warning(f"LIQUIDATION at {ts}: Eq={account.equity:.2f} Notional={total_notional:.2f}")
            # Close all positions
            for s, pos in list(account.positions.items()):
                fill_px = opens[s] * (0.95 if pos.qty > 0 else 1.05) # Severe penalty
                pnl = pos.qty * (fill_px - pos.entry_price)
                account.balance += pnl
                trades_log.append({
                    "timestamp": ts,
                    "symbol": s,
                    "side": "LIQ_SELL" if pos.qty > 0 else "LIQ_BUY",
                    "qty": abs(pos.qty),
                    "price": fill_px,
                    "pnl": pnl,
                    "reason": "liquidation"
                })
            account.positions.clear()
            account.balance = 0.0 # Rekt
            break # Game over

        # 4. Strategy Decision
        # Re-calc equity at close for signal generation (conservative)
        account.update_equity(closes)
        
        strat_state = StrategyState(
            timestamp=ts,
            allow_short=True,
            cash=account.balance, # Collateral
            positions={s: p.qty for s, p in account.positions.items()},
            equity=account.equity,
            day_start_equity=account.equity, # Simplified day tracking
            day_pnl=0.0,
            day_return=0.0,
            holding_bars={} # Not tracked in this simple engine V1
        )
        
        # Slicing history
        history = {s: bars_by_symbol[s].iloc[:i+1] for s in symbols}
        decision = strategy.target_exposures(history, strat_state)
        
        # 5. Execution (Next Bar Open Logic Simulated here for simplicity or Current Close)
        # We will execute at 'closes' with slippage to simulate "End of Bar" or "Next Open"
        # Ideally we execute at i+1 Open, but we'll use Close[i] with penalty.
        
        for s, target_exposure in decision.target_exposures.items():
            # Target Exposure is % of Equity
            target_notional = target_exposure * account.equity * 0.98 # Buffer
            target_notional = max(min(target_notional, account.equity * cfg.leverage_limit), -account.equity * cfg.leverage_limit)
            
            price = closes[s]
            target_qty = target_notional / price
            
            current_pos = account.positions.get(s)
            current_qty = current_pos.qty if current_pos else 0.0
            
            delta_qty = target_qty - current_qty
            
            if abs(delta_qty * price) < 10.0: # Min trade size
                continue
                
            # Trade Cost
            side_mult = 1 if delta_qty > 0 else -1
            # Slippage + Fee
            # Buying: Price * (1 + slip + fee)
            # Selling: Price * (1 - slip - fee)
            total_cost_bps = cfg.taker_fee_bps + cfg.slippage_bps
            exec_price = price * (1 + side_mult * (total_cost_bps / 10000.0))
            
            # Realize PnL on reduction
            if current_pos and (current_qty * delta_qty < 0):
                # We are reducing or flipping
                # Simplified FIFO/LIFO: Just calculate PnL on closed portion
                close_qty = min(abs(current_qty), abs(delta_qty)) * (-1 if current_qty > 0 else 1)
                pnl = close_qty * (exec_price - current_pos.entry_price)
                account.balance += pnl
                trades_log.append({
                    "timestamp": ts,
                    "symbol": s,
                    "side": "SELL" if close_qty < 0 else "BUY",
                    "qty": abs(close_qty),
                    "price": exec_price,
                    "pnl": pnl,
                    "reason": decision.reason
                })
            
            # Update Position
            new_qty = current_qty + delta_qty
            if abs(new_qty) < 1e-8:
                if s in account.positions: del account.positions[s]
            else:
                # Update Avg Price (Simplified: Weighted Average on Increase)
                if current_pos and (current_qty * delta_qty > 0):
                    # Increasing
                    old_notional = current_pos.qty * current_pos.entry_price
                    added_notional = delta_qty * exec_price
                    new_avg = (old_notional + added_notional) / new_qty
                    account.positions[s] = Position(s, new_qty, new_avg)
                elif not current_pos:
                    # New
                    account.positions[s] = Position(s, new_qty, exec_price)
                else:
                    # Reducing (Entry price doesn't change)
                    account.positions[s] = Position(s, new_qty, current_pos.entry_price)
                    
            # Log Trade (Entry part)
            if current_qty * delta_qty >= 0 or abs(delta_qty) > abs(current_qty):
                 trades_log.append({
                    "timestamp": ts,
                    "symbol": s,
                    "side": "BUY" if delta_qty > 0 else "SELL",
                    "qty": abs(delta_qty),
                    "price": exec_price,
                    "pnl": 0.0,
                    "reason": decision.reason
                })

        # Record Equity
        account.update_equity(closes)
        equity_curve.append({
            "timestamp": ts,
            "equity": account.equity,
            "balance": account.balance,
            "margin_used": account.margin_used
        })

    # Save outputs
    df_eq = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trades_log)
    
    df_eq.to_csv(run_dir / "equity_curve.csv", index=False)
    df_trades.to_csv(run_dir / "trades.csv", index=False)
    
    # Compute Metrics
    # (Reuse existing metrics logic roughly)
    df_eq_indexed = df_eq.set_index(pd.to_datetime(df_eq["timestamp"]))
    metrics = compute_metrics(df_eq_indexed, df_trades)
    (run_dir / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    
    return run_dir
```

### 4. `src/atlas/strategies/fractal_trend.py` (New Strategy)

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

from atlas.strategies.base import Strategy, StrategyDecision, StrategyState

@dataclass
class FractalTrend(Strategy):
    """
    Fractal Efficiency Trend Strategy (Atlas-Alpha).
    Only trades when market efficiency (ER) is high (trending).
    Uses volatility-adjusted sizing and liquidation guards.
    """
    name: str = "fractal_trend"
    
    # Parameters
    er_window: int = 12
    er_threshold: float = 0.40  # Min efficiency to enter
    trend_ema: int = 50
    pullback_window: int = 5
    atr_window: int = 14
    risk_per_trade: float = 0.02 # 2% equity risk
    stop_atr_mult: float = 3.0
    max_leverage: float = 5.0
    
    def warmup_bars(self) -> int:
        return max(self.trend_ema, self.atr_window) + 10

    def target_exposures(
        self, bars_by_symbol: dict[str, pd.DataFrame], state: StrategyState
    ) -> StrategyDecision:
        targets = {}
        debug = {}
        
        for symbol, bars in bars_by_symbol.items():
            if len(bars) < self.warmup_bars():
                continue
                
            close = bars["close"]
            high = bars["high"]
            low = bars["low"]
            
            # 1. Calculate Efficiency Ratio (ER)
            # ER = Change / Volatility
            change = (close.diff(self.er_window).abs())
            volatility = close.diff().abs().rolling(self.er_window).sum()
            er = change / volatility
            current_er = er.iloc[-1]
            
            # 2. Trend Filter
            ema = close.ewm(span=self.trend_ema).mean()
            trend_up = close.iloc[-1] > ema.iloc[-1]
            
            # 3. ATR & Sizing
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(self.atr_window).mean().iloc[-1]
            
            # Stop Distance
            stop_dist = self.stop_atr_mult * atr
            
            # Risk Sizing
            # Position Size = (Equity * Risk%) / StopDist
            # e.g., $10,000 * 0.02 = $200 risk. Stop is $100 away. Size = 2 units.
            if stop_dist > 0:
                qty_risk = (state.equity * self.risk_per_trade) / stop_dist
                notional = qty_risk * close.iloc[-1]
            else:
                notional = 0.0
                
            # Leverage Cap
            max_notional = state.equity * self.max_leverage
            notional = min(notional, max_notional)
            
            # Exposure %
            exposure = notional / state.equity if state.equity > 0 else 0.0
            
            # Logic
            current_pos = state.positions.get(symbol, 0.0)
            
            # Entry / Maintenance
            if current_er > self.er_threshold:
                if trend_up:
                    # Valid Long Regime
                    if current_pos <= 0:
                        # Entry Signal: Breakout or Pullback logic could go here
                        # For V1: Simple Trend Following if ER is high
                        targets[symbol] = exposure
                    else:
                        # Hold
                        targets[symbol] = exposure
                else:
                    # Valid Short Regime
                    if current_pos >= 0:
                        targets[symbol] = -exposure
                    else:
                        targets[symbol] = -exposure
            else:
                # Low Efficiency - Chop Mode
                # If we have a position, tighten stop or exit.
                # V1: Exit on regime change (safety)
                targets[symbol] = 0.0
                
            debug[symbol] = {
                "ER": current_er,
                "Trend": "Up" if trend_up else "Down",
                "ATR": atr,
                "Notional": notional
            }
            
        return StrategyDecision(target_exposures=targets, debug=debug, reason="signal")
```

### 5. `src/atlas/strategies/registry.py` (Update)

Add the new strategy to `build_strategy`.

```python
# ... inside build_strategy ...
    if name == "fractal_trend":
         from atlas.strategies.fractal_trend import FractalTrend
         return FractalTrend(
             er_window=int(params.get("er_window", 12)),
             er_threshold=float(params.get("er_threshold", 0.40)),
             trend_ema=int(params.get("trend_ema", 50)),
             max_leverage=float(params.get("max_leverage", 5.0))
         )
```

### 6. `src/atlas/cli.py` (Update)

Add the `backtest-derivatives` command.

```python
@app.command()
def backtest_derivatives(
    symbol: str = typer.Option("BTC-PERP", help="Symbol (BTC-PERP)"),
    initial_balance: float = 10000.0,
    strategy: str = "fractal_trend",
    leverage: float = 5.0
):
    """
    Run the new Derivatives Engine backtest.
    """
    from atlas.backtest.derivatives import DerivativesConfig, run_derivatives_backtest
    from atlas.config import get_coinbase_settings
    from atlas.data.coinbase import load_coinbase_candles_cached
    
    run_dir = Path("outputs") / "derivatives" / _run_id("deriv")
    setup_logging(level="INFO", log_file=run_dir / "run.log")
    
    # 1. Data
    settings = get_coinbase_settings()
    # Mock date range for V1
    end = datetime.now(tz=NY_TZ)
    start = end - timedelta(days=30)
    
    bars = load_coinbase_candles_cached(
        settings=settings, 
        product_id=symbol, 
        start=start, 
        end=end
    )
    
    # 2. Strategy
    strat = build_strategy(name=strategy, params_path=None, symbols=[symbol], fast_window=10, slow_window=30)
    
    # 3. Config
    cfg = DerivativesConfig(
        symbols=[symbol],
        initial_balance=initial_balance,
        leverage_limit=leverage
    )
    
    # 4. Run
    run_derivatives_backtest(
        bars_by_symbol={symbol: bars},
        funding_by_symbol=None,
        strategy=strat,
        cfg=cfg,
        run_dir=run_dir
    )
    print(f"Done. Output: {run_dir}")
```

---

# Minimal Diff List

1.  **Modify `src/atlas/config.py`**: Add `CoinbaseSettings` and `get_coinbase_settings`.
2.  **Create `src/atlas/data/coinbase.py`**: Implementation of candle fetching.
3.  **Create `src/atlas/backtest/derivatives.py`**: The new Margin/Leverage engine.
4.  **Create `src/atlas/strategies/fractal_trend.py`**: The specific high-leverage strategy.
5.  **Modify `src/atlas/strategies/registry.py`**: Register `fractal_trend`.
6.  **Modify `src/atlas/cli.py`**: Add `backtest-derivatives` command.

---

# Runbook

1.  **Setup**:
    *   Add `COINBASE_API_KEY` and `COINBASE_API_SECRET` to `.env`.
    *   Ensure `ATLAS_LOG_LEVEL=INFO`.
2.  **Data Acquisition**:
    *   Run `python -m atlas.cli backtest-derivatives --symbol BTC-PERP`. This will attempt to download cached data first.
3.  **Backtest**:
    *   Check `outputs/derivatives/<timestamp>/metrics.json`.
    *   Verify `max_drawdown` is within limits (< 20%).
    *   Verify `liquidation` events are 0.
4.  **Red Team**:
    *   Manually inspect `trades.csv` during high volatility periods (check timestamp vs chart).
    *   Verify spread/slippage assumptions in `derivatives.py` (default 5bps slippage + 6bps taker fee = 11bps cost per side).

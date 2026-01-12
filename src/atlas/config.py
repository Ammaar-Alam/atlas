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


def get_log_level() -> str:
    load_env()
    return (_env_str("ATLAS_LOG_LEVEL", "INFO") or "INFO").upper()


def get_default_max_position_notional_usd(*, mode: str) -> float:
    load_env()
    if mode == "paper":
        return _env_float("ATLAS_PAPER_MAX_POSITION_NOTIONAL_USD", 1000.0)
    return _env_float("ATLAS_BACKTEST_MAX_POSITION_NOTIONAL_USD", 10000.0)


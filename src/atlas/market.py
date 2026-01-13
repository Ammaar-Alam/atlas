from __future__ import annotations

from enum import Enum
from typing import Iterable


class Market(str, Enum):
    """Which market/asset-class atlas is trading."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    DERIVATIVES = "derivatives"


def parse_market(value: str) -> Market:
    v = (value or "").strip().lower().replace("-", "_")
    if v in {"equity", "stock", "stocks"}:
        return Market.EQUITY
    if v in {"crypto", "cryptocurrency", "cryptos"}:
        return Market.CRYPTO
    if v in {"derivatives", "perp", "perps", "futures"}:
        return Market.DERIVATIVES
    raise ValueError(f"unsupported market: {value!r} (expected 'equity', 'crypto', or 'derivatives')")


CRYPTO_EQUIVALENTS: dict[str, str] = {
    "SPY": "BTC/USD",
    "QQQ": "ETH/USD",
}

DERIVATIVES_EQUIVALENTS: dict[str, str] = {
    "SPY": "BTC-PERP",
    "QQQ": "ETH-PERP",
}


def default_symbols(market: Market, *, count: int = 1) -> list[str]:
    if count <= 0:
        return []
    if market == Market.EQUITY:
        base = ["SPY", "QQQ"]
    elif market == Market.CRYPTO:
        base = ["BTC/USD", "ETH/USD"]
    else:
        base = ["BTC-PERP", "ETH-PERP"]
    return base[:count]


def coerce_symbols_for_market(symbols: Iterable[str], market: Market) -> list[str]:
    """Normalize + de-duplicate symbols, and apply market-specific aliases."""

    out: list[str] = []
    for raw in symbols:
        s = (raw or "").strip().upper()
        if not s:
            continue
        if market == Market.CRYPTO:
            s = CRYPTO_EQUIVALENTS.get(s, s)
            s = s.replace(" ", "")
            if "/" not in s and "-" in s:
                parts = [p for p in s.split("-") if p]
                if len(parts) == 2:
                    s = f"{parts[0]}/{parts[1]}"
            if "/" not in s and "_" in s:
                parts = [p for p in s.split("_") if p]
                if len(parts) == 2:
                    s = f"{parts[0]}/{parts[1]}"

            if "/" not in s:
                for quote in ("USDT", "USDC", "USD"):
                    if s.endswith(quote) and len(s) > len(quote):
                        s = f"{s[:-len(quote)]}/{quote}"
                        break
                else:
                    s = f"{s}/USD"
            else:
                base, quote = (p.strip() for p in s.split("/", 1))
                if not quote:
                    quote = "USD"
                s = f"{base}/{quote}"
                if not quote:
                    quote = "USD"
                s = f"{base}/{quote}"
        elif market == Market.DERIVATIVES:
            s = DERIVATIVES_EQUIVALENTS.get(s, s)
            s = s.replace(" ", "")
            # Ensure it ends with -PERP if not explicit
            if not s.endswith("-PERP") and "-PERP" not in s:
                # Handle BTC/USD -> BTC-PERP
                if "/" in s:
                    s = s.split("/")[0]
                # Naive guess: if user says BTC, assume BTC-PERP
                s = f"{s}-PERP"
        out.append(s)

    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return deduped


def safe_filename_symbol(symbol: str) -> str:
    """Convert a trading symbol into a filesystem-safe token."""

    return (symbol or "").strip().upper().replace("/", "_")

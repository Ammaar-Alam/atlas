from __future__ import annotations

from enum import Enum
from typing import Iterable


class Market(str, Enum):
    """Which market/asset-class atlas is trading."""

    EQUITY = "equity"
    CRYPTO = "crypto"


def parse_market(value: str) -> Market:
    v = (value or "").strip().lower().replace("-", "_")
    if v in {"equity", "stock", "stocks"}:
        return Market.EQUITY
    if v in {"crypto", "cryptocurrency", "cryptos"}:
        return Market.CRYPTO
    raise ValueError(f"unsupported market: {value!r} (expected 'equity' or 'crypto')")


CRYPTO_EQUIVALENTS: dict[str, str] = {
    "SPY": "BTC/USD",
    "QQQ": "ETH/USD",
}


def default_symbols(market: Market, *, count: int = 1) -> list[str]:
    if count <= 0:
        return []
    base = ["SPY", "QQQ"] if market == Market.EQUITY else ["BTC/USD", "ETH/USD"]
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

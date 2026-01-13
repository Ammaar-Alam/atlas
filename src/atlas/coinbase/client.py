from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import requests

from atlas.config import CoinbaseSettings, get_coinbase_settings
from atlas.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Product:
    product_id: str
    price: float
    base_currency: str
    quote_currency: str
    base_increment: float
    quote_increment: float
    display_name: str
    status: str


class CoinbaseClient:
    """
    Minimal Coinbase Advanced Trade client for public market-data endpoints.

    Important: Coinbase futures/perps often use contract-style `product_id` values
    (e.g. "BIP-20DEC30-CDE") even if we refer to them internally as "BTC-PERP".
    This client resolves common aliases like "BTC-PERP" -> the current Coinbase
    `product_id` for "BTC PERP" via the public products endpoint.
    """

    ADVANCED_TRADE_API_URL = "https://api.coinbase.com/api/v3"

    def __init__(self, settings: Optional[CoinbaseSettings] = None):
        self.settings = settings or get_coinbase_settings()
        self.session = requests.Session()
        self._resolved_product_ids: dict[tuple[str, str], str] = {}

    def _request_public(self, method: str, endpoint: str, params: Optional[dict] = None) -> Any:
        url = f"{self.ADVANCED_TRADE_API_URL}{endpoint}"
        headers = {"Accept": "application/json"}

        retryable_statuses = {429, 500, 502, 503, 504}
        last_response: Optional[requests.Response] = None
        last_error: Optional[Exception] = None

        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.request(method, url, headers=headers, params=params, timeout=30)
                last_response = resp

                if resp.status_code in retryable_statuses:
                    retry_after_s: Optional[float] = None
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            retry_after_s = float(retry_after)
                        except ValueError:
                            retry_after_s = None

                    base_s = 1.0 if resp.status_code == 429 else 0.5
                    backoff_s = min(30.0, base_s * (2 ** (attempt - 1)))
                    wait_s = max(backoff_s, retry_after_s or 0.0) + random.uniform(0.0, 0.25)
                    logger.warning(
                        "Coinbase API %s %s returned %s (attempt %d/%d); retrying in %.2fs",
                        method,
                        url,
                        resp.status_code,
                        attempt,
                        max_attempts,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    continue

                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                body = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
                logger.error("Coinbase API error %s %s: %s", method, url, body)
                raise
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                wait_s = min(30.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                logger.warning(
                    "Coinbase request error %s %s (attempt %d/%d): %s; retrying in %.2fs",
                    method,
                    url,
                    attempt,
                    max_attempts,
                    exc,
                    wait_s,
                )
                time.sleep(wait_s)

        if last_response is not None:
            try:
                last_response.raise_for_status()
            except requests.HTTPError as exc:
                body = getattr(exc.response, "text", "") if getattr(exc, "response", None) else ""
                logger.error("Coinbase API error %s %s: %s", method, url, body)
                raise

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"Coinbase request failed: {method} {url}")

    def list_products(self, product_type: str = "FUTURE") -> list[Product]:
        """
        List available products (public endpoint).
        product_type: SPOT or FUTURE.
        """
        response = self._request_public(
            "GET",
            "/brokerage/market/products",
            params={"product_type": product_type, "limit": 1000},
        )
        products: list[Product] = []
        for p in response.get("products", []):
            products.append(
                Product(
                    product_id=p["product_id"],
                    price=float(p.get("price", 0) or 0),
                    base_currency=p.get("base_currency_id", "") or "",
                    quote_currency=p.get("quote_currency_id", "") or "",
                    base_increment=float(p.get("base_increment", 0) or 0),
                    quote_increment=float(p.get("quote_increment", 0) or 0),
                    display_name=p.get("display_name", "") or "",
                    status=p.get("status", "") or "",
                )
            )
        return products

    def resolve_product_id(self, symbol: str, *, product_type: str) -> str:
        """
        Resolve a human-friendly symbol into a Coinbase `product_id`.

        Examples:
        - BTC-PERP -> BIP-20DEC30-CDE (or similar current contract id)
        - ETH-PERP -> ETP-20DEC30-CDE
        - BTC-USD -> BTC-USD (passes through)
        """
        raw = (symbol or "").strip().upper()
        if not raw:
            raise ValueError("symbol is required")

        product_type = product_type.strip().upper()
        cache_key = (raw, product_type)
        cached = self._resolved_product_ids.get(cache_key)
        if cached is not None:
            return cached

        # Common case: caller already passed a concrete Coinbase product_id.
        # Coinbase US futures/perps commonly end in "-CDE".
        if raw.endswith("-CDE"):
            self._resolved_product_ids[cache_key] = raw
            return raw

        # Normalize some common variants.
        normalized = raw.replace(" ", "").replace("_", "-")
        if "/" in normalized:
            base, quote = (p.strip() for p in normalized.split("/", 1))
            normalized = f"{base}-{quote or 'USD'}"

        # For US futures/perps, Coinbase uses contract-style IDs and the most reliable
        # lookup is by display_name (e.g. "BTC PERP").
        if normalized.endswith("-PERP") and product_type == "FUTURE":
            base = normalized.split("-", 1)[0]
            want = f"{base} PERP"
            products = self.list_products(product_type="FUTURE")

            for p in products:
                if (p.display_name or "").strip().upper() == want:
                    self._resolved_product_ids[cache_key] = p.product_id
                    return p.product_id

            # Fallback: contains match (handles minor display_name variations).
            for p in products:
                if want in (p.display_name or "").strip().upper():
                    self._resolved_product_ids[cache_key] = p.product_id
                    return p.product_id

            raise ValueError(
                f"Coinbase FUTURE product not found for {symbol!r}. "
                "Tip: fetch valid ids via "
                "`curl 'https://api.coinbase.com/api/v3/brokerage/market/products?product_type=FUTURE&limit=50'`."
            )

        # Default: treat as already-resolved (e.g. BTC-USD).
        self._resolved_product_ids[cache_key] = normalized
        return normalized

    def get_product_candles(
        self, product_id: str, start: datetime, end: datetime, granularity: str
    ) -> pd.DataFrame:
        """
        Fetch candles for a product via the public candles endpoint.
        granularity: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, SIX_HOUR, ONE_DAY
        """
        granularity_map = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "ONE_HOUR": 3600,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }
        if granularity not in granularity_map:
            raise ValueError(f"invalid granularity: {granularity}")

        delta_seconds = granularity_map[granularity]
        max_candles = 300
        chunk_size = timedelta(seconds=delta_seconds * max_candles)

        inferred_product_type = "FUTURE" if (product_id or "").upper().endswith("-PERP") else "SPOT"
        resolved_product_id = self.resolve_product_id(product_id, product_type=inferred_product_type)
        if resolved_product_id != (product_id or ""):
            logger.info("Resolved Coinbase product %s -> %s", product_id, resolved_product_id)

        all_candles: list[dict[str, float | datetime]] = []
        last_error: Optional[Exception] = None
        current_start = start

        while current_start < end:
            current_end = min(current_start + chunk_size, end)
            params = {
                "start": int(current_start.timestamp()),
                "end": int(current_end.timestamp()),
                "granularity": granularity,
            }
            try:
                response = self._request_public(
                    "GET",
                    f"/brokerage/market/products/{resolved_product_id}/candles",
                    params=params,
                )
                candles_data = response.get("candles", [])
                for c in candles_data:
                    all_candles.append(
                        {
                            "start": datetime.fromtimestamp(int(c["start"]), tz=timezone.utc),
                            "low": float(c["low"]),
                            "high": float(c["high"]),
                            "open": float(c["open"]),
                            "close": float(c["close"]),
                            "volume": float(c["volume"]),
                        }
                    )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to fetch Coinbase candles for %s (%s -> %s): %s",
                    resolved_product_id,
                    current_start,
                    current_end,
                    exc,
                )

            current_start = current_end

        if not all_candles:
            if last_error is not None:
                raise ValueError(
                    f"Coinbase returned 0 candles for {product_id} (resolved={resolved_product_id}) "
                    f"from {start.isoformat()} to {end.isoformat()}: {last_error}"
                ) from last_error
            return pd.DataFrame()

        df = pd.DataFrame(all_candles)
        df.set_index("start", inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        return df[["open", "high", "low", "close", "volume"]].copy()

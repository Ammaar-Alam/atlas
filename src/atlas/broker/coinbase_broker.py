from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from atlas.coinbase.client import CoinbaseClient
from atlas.config import CoinbaseSettings, get_coinbase_settings
from atlas.market import Market, parse_market

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductSpecs:
    product_id: str
    product_type: str
    size_increment: float
    is_future: bool
    contract_size: float


@dataclass(frozen=True)
class OrderFill:
    order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]
    raw_qty: float = 0.0
    raw_filled_qty: float = 0.0
    contract_size: float = 1.0


def client(settings: Optional[CoinbaseSettings] = None) -> CoinbaseClient:
    return CoinbaseClient(settings=settings or get_coinbase_settings())


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _resolve_product_type(symbol: str, market: str) -> str:
    mkt = parse_market(market)
    upper = str(symbol or "").strip().upper()
    if mkt == Market.DERIVATIVES:
        return "FUTURE"
    if upper.endswith("-PERP") or upper.endswith("-CDE"):
        return "FUTURE"
    return "SPOT"


def _is_future_product(product_id: str) -> bool:
    pid = str(product_id or "").strip().upper()
    return bool(pid.endswith("-CDE") or pid.endswith("-PERP"))


def _round_down_to_increment(value: float, increment: float) -> float:
    if value <= 0:
        return 0.0
    inc = float(increment)
    if inc <= 0:
        return float(value)
    steps = math.floor(float(value) / inc)
    return float(steps * inc)


def _fetch_product_specs(
    *,
    client: CoinbaseClient,
    symbol: str,
    market: str,
    product_id_override: Optional[str] = None,
) -> ProductSpecs:
    product_type = _resolve_product_type(symbol, market)
    product_id = (
        str(product_id_override).strip().upper()
        if product_id_override
        else client.resolve_product_id(symbol, product_type=product_type)
    )

    product = client.get_product(product_id)
    is_future = bool(_is_future_product(product_id) or product_type == "FUTURE")
    size_increment = _as_float(product.get("base_increment"), 0.0)
    contract_size = 1.0
    if is_future:
        details = product.get("future_product_details")
        if isinstance(details, dict):
            contract_size = _as_float(details.get("contract_size"), 1.0)
        if contract_size <= 0:
            contract_size = 1.0

    return ProductSpecs(
        product_id=product_id,
        product_type=product_type,
        size_increment=size_increment,
        is_future=is_future,
        contract_size=contract_size,
    )


def _to_order_size(*, base_qty: float, specs: ProductSpecs) -> float:
    order_qty = float(base_qty)
    if specs.is_future:
        order_qty = float(order_qty / specs.contract_size)
    order_qty = _round_down_to_increment(order_qty, specs.size_increment)
    return float(order_qty)


def _to_base_qty(*, order_qty: float, specs: ProductSpecs) -> float:
    qty = float(order_qty)
    if specs.is_future:
        qty = float(qty * specs.contract_size)
    return float(qty)


def submit_market_order(
    *,
    client: CoinbaseClient,
    symbol: str,
    qty: float,
    side: str,
    market: str = "crypto",
) -> str:
    if qty <= 0:
        raise ValueError("qty must be > 0")

    side_u = str(side or "").strip().upper()
    if side_u not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    specs = _fetch_product_specs(client=client, symbol=symbol, market=market)
    order_qty = _to_order_size(base_qty=float(qty), specs=specs)
    if order_qty <= 0:
        raise ValueError(
            "computed Coinbase order size <= 0 after contract/increment conversion; "
            f"qty={qty} product={specs.product_id} increment={specs.size_increment}"
        )

    response = client.create_market_order(
        product_id=specs.product_id,
        side=side_u,
        qty=float(order_qty),
    )

    failure = response.get("failure_response") if isinstance(response, dict) else None
    if isinstance(failure, dict) and failure:
        code = str(failure.get("error") or failure.get("error_type") or "order_rejected")
        message = str(failure.get("message") or failure.get("preview_failure_reason") or "")
        raise RuntimeError(f"coinbase order rejected ({code}): {message}".strip())

    success = response.get("success_response") if isinstance(response, dict) else None
    order_id = ""
    if isinstance(success, dict):
        order_id = str(success.get("order_id") or "").strip()
    if not order_id and isinstance(response, dict):
        order_id = str(response.get("order_id") or "").strip()

    if not order_id:
        raise RuntimeError(f"coinbase order response missing order_id: {response}")

    return order_id


def wait_for_fill(
    *,
    client: CoinbaseClient,
    order_id: str,
    timeout_s: int,
    poll_s: float,
) -> OrderFill:
    deadline = time.time() + timeout_s
    terminal_statuses = {
        "FILLED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "FAILED",
        "REJECTED",
    }
    product_specs_cache: dict[str, ProductSpecs] = {}

    while True:
        order = client.get_order(order_id)
        status = str(order.get("status") or "UNKNOWN").upper()
        symbol = str(order.get("product_id") or "")
        side = str(order.get("side") or "").upper()
        raw_qty = _as_float(order.get("base_size") or order.get("size") or order.get("order_size"), 0.0)
        raw_filled_qty = _as_float(order.get("filled_size") or order.get("filled_qty"), 0.0)

        specs = product_specs_cache.get(symbol)
        if specs is None:
            try:
                specs = _fetch_product_specs(
                    client=client,
                    symbol=symbol,
                    market="derivatives" if _is_future_product(symbol) else "crypto",
                    product_id_override=symbol,
                )
            except Exception:
                specs = ProductSpecs(
                    product_id=symbol,
                    product_type="FUTURE" if _is_future_product(symbol) else "SPOT",
                    size_increment=0.0,
                    is_future=_is_future_product(symbol),
                    contract_size=1.0,
                )
            product_specs_cache[symbol] = specs

        qty = _to_base_qty(order_qty=raw_qty, specs=specs)
        filled_qty = _to_base_qty(order_qty=raw_filled_qty, specs=specs)
        filled_avg_price = _as_float(
            order.get("average_filled_price") or order.get("avg_fill_price"),
            0.0,
        )
        fill_px = filled_avg_price if filled_avg_price > 0 else None

        if status in terminal_statuses:
            return OrderFill(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                qty=qty,
                status=status,
                filled_qty=filled_qty,
                filled_avg_price=fill_px,
                raw_qty=raw_qty,
                raw_filled_qty=raw_filled_qty,
                contract_size=float(specs.contract_size),
            )

        if time.time() >= deadline:
            logger.warning("coinbase order %s not terminal before timeout; status=%s", order_id, status)
            return OrderFill(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                qty=qty,
                status=status,
                filled_qty=filled_qty,
                filled_avg_price=fill_px,
                raw_qty=raw_qty,
                raw_filled_qty=raw_filled_qty,
                contract_size=float(specs.contract_size),
            )

        time.sleep(poll_s)

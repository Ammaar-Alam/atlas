from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from atlas.config import AlpacaSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderFill:
    order_id: str
    symbol: str
    side: str
    qty: float
    status: str
    filled_qty: float
    filled_avg_price: Optional[float]


def trading_client(settings: AlpacaSettings) -> TradingClient:
    return TradingClient(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
        paper=settings.paper,
        url_override=settings.trading_url_override,
    )


def assert_market_open(client: TradingClient) -> None:
    clock = client.get_clock()
    if not clock.is_open:
        raise RuntimeError(f"market closed: next_open={clock.next_open} next_close={clock.next_close}")


def submit_market_order(
    *,
    client: TradingClient,
    symbol: str,
    qty: float,
    side: str,
) -> str:
    if qty <= 0:
        raise ValueError("qty must be > 0")

    req = MarketOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    res = client.submit_order(req)
    return str(res.id)


def submit_limit_order(
    *,
    client: TradingClient,
    symbol: str,
    qty: float,
    side: str,
    limit_price: float,
    extended_hours: bool,
) -> str:
    if qty <= 0:
        raise ValueError("qty must be > 0")
    if limit_price <= 0:
        raise ValueError("limit_price must be > 0")

    req = LimitOrderRequest(
        symbol=symbol,
        qty=float(qty),
        side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=float(limit_price),
        extended_hours=bool(extended_hours),
    )
    res = client.submit_order(req)
    return str(res.id)


def wait_for_fill(
    *,
    client: TradingClient,
    order_id: str,
    timeout_s: int,
    poll_s: float,
) -> OrderFill:
    deadline = time.time() + timeout_s
    while True:
        order = client.get_order_by_id(order_id)
        status = str(order.status)

        if status.lower() in {"filled", "canceled", "rejected", "expired"}:
            filled_qty = float(getattr(order, "filled_qty", 0) or 0)
            avg = getattr(order, "filled_avg_price", None)
            filled_avg_price = float(avg) if avg not in (None, "") else None
            return OrderFill(
                order_id=str(order.id),
                symbol=str(order.symbol),
                side=str(order.side),
                qty=float(order.qty) if order.qty is not None else float(order.notional or 0),
                status=status,
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
            )

        if time.time() >= deadline:
            return OrderFill(
                order_id=str(order.id),
                symbol=str(order.symbol),
                side=str(order.side),
                qty=float(order.qty) if order.qty is not None else float(order.notional or 0),
                status=status,
                filled_qty=float(getattr(order, "filled_qty", 0) or 0),
                filled_avg_price=float(getattr(order, "filled_avg_price", 0) or 0) if getattr(order, "filled_avg_price", None) else None,
            )

        time.sleep(poll_s)

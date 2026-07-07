"""Portfolio order-ticket calculation for the recommendation screen.

The recommendation engine produces target weights. This module turns those
weights into an executable-looking, integer-share order plan for a user-provided
investment amount while keeping transaction-cost assumptions explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import floor
from typing import Any


PriceLookup = Callable[[str], tuple[float | None, str | None]]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_holdings_text(value: str | None) -> dict[str, float]:
    """Parse user holdings text such as ``069500.KS=10, 132030.KS:3``."""

    if not value:
        return {}
    holdings: dict[str, float] = {}
    normalized = str(value).replace("\r", "\n").replace(";", ",").replace("\n", ",")
    for raw in normalized.split(","):
        token = raw.strip()
        if not token:
            continue
        if "=" in token:
            ticker, qty = token.split("=", 1)
        elif ":" in token:
            ticker, qty = token.split(":", 1)
        else:
            parts = token.split()
            if len(parts) != 2:
                continue
            ticker, qty = parts
        ticker = ticker.strip().upper()
        quantity = _to_float(qty, default=float("nan"))
        if ticker and quantity == quantity:
            holdings[ticker] = max(0.0, quantity)
    return holdings


def _holding_quantity(holdings: Mapping[str, float], ticker: str) -> float:
    code = ticker.split(".", 1)[0].upper()
    candidates = (ticker.upper(), code, f"{code}.KS")
    for key in candidates:
        if key in holdings:
            return max(0.0, _to_float(holdings[key]))
    return 0.0


def build_order_ticket(
    recommendation: Mapping[str, Any],
    *,
    amount: float,
    price_lookup: PriceLookup,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    current_holdings: Mapping[str, float] | None = None,
    no_trade_band: bool = True,
    band_abs_pct: float = 5.0,
    band_rel_pct: float = 25.0,
) -> dict[str, Any]:
    """Build integer-quantity order/rebalance rows from a recommendation payload."""

    amount = max(0.0, _to_float(amount))
    transaction_cost_bps = max(0.0, _to_float(transaction_cost_bps, 10.0))
    slippage_bps = max(0.0, _to_float(slippage_bps, 5.0))
    band_abs_pct = max(0.0, _to_float(band_abs_pct, 5.0))
    band_rel_pct = max(0.0, _to_float(band_rel_pct, 25.0))
    total_bps = transaction_cost_bps + slippage_bps
    cost_rate = total_bps / 10000.0

    rows: list[dict[str, Any]] = []
    invested = 0.0
    estimated_cost = 0.0
    current_value = 0.0
    buy_value = 0.0
    sell_value = 0.0
    gross_trade_value = 0.0
    skipped_trade_value = 0.0
    latest_dates: list[str] = []
    holdings = current_holdings or {}
    has_holdings = bool(holdings)

    for asset in recommendation.get("assets", []) or []:
        ticker = str(asset.get("ticker") or "")
        weight_pct = _to_float(asset.get("weight"))
        target_amount = amount * weight_pct / 100.0
        price, price_date = price_lookup(ticker)
        current_qty = _holding_quantity(holdings, ticker)
        if price_date:
            latest_dates.append(str(price_date))

        row: dict[str, Any] = {
            "ticker": ticker,
            "name": asset.get("name") or ticker,
            "asset_type": asset.get("asset_type") or "",
            "weight": round(weight_pct, 4),
            "target_amount": round(target_amount, 2),
            "price": round(price, 4) if price is not None else None,
            "price_date": price_date,
            "current_quantity": round(current_qty, 6),
        }

        if price is None or price <= 0 or target_amount <= 0:
            row.update(
                quantity=0,
                target_quantity=0,
                ideal_quantity=0,
                trade_action="유지",
                trade_quantity=0,
                current_weight=0.0,
                drift_pct=0.0,
                band_limit_pct=max(band_abs_pct, abs(weight_pct) * band_rel_pct / 100.0),
                in_no_trade_band=False,
                current_value=0.0,
                trade_value=0.0,
                order_value=0.0,
                estimated_cost=0.0,
                estimated_total=0.0,
                cash_gap=round(target_amount, 2),
                error="가격 없음" if price is None else None,
            )
            rows.append(row)
            continue

        unit_cost = price * (1.0 + cost_rate)
        ideal_quantity = int(floor(target_amount / unit_cost))
        row_current_value = current_qty * price
        current_weight_pct = (row_current_value / amount * 100.0) if amount > 0 else 0.0
        drift_pct = current_weight_pct - weight_pct
        band_limit_pct = max(band_abs_pct, abs(weight_pct) * band_rel_pct / 100.0)
        in_band = bool(has_holdings and no_trade_band and abs(drift_pct) <= band_limit_pct)

        quantity = int(round(current_qty)) if in_band else ideal_quantity
        trade_quantity = quantity - current_qty
        trade_value = abs(trade_quantity) * price
        order_value = quantity * price
        row_cost = trade_value * cost_rate
        estimated_total = order_value + row_cost
        cash_gap = target_amount - estimated_total
        trade_action = (
            "유지(밴드내)"
            if in_band
            else "매수"
            if trade_quantity > 0
            else "매도"
            if trade_quantity < 0
            else "유지"
        )

        invested += order_value
        estimated_cost += row_cost
        current_value += row_current_value
        gross_trade_value += trade_value
        if in_band:
            skipped_trade_value += abs(ideal_quantity - current_qty) * price
        if trade_quantity > 0:
            buy_value += trade_quantity * price
        elif trade_quantity < 0:
            sell_value += abs(trade_quantity) * price
        row.update(
            quantity=quantity,
            target_quantity=quantity,
            ideal_quantity=ideal_quantity,
            trade_action=trade_action,
            trade_quantity=round(trade_quantity, 6),
            current_value=round(row_current_value, 2),
            current_weight=round(current_weight_pct, 4),
            drift_pct=round(drift_pct, 4),
            band_limit_pct=round(band_limit_pct, 4),
            in_no_trade_band=in_band,
            trade_value=round(trade_value, 2),
            order_value=round(order_value, 2),
            estimated_cost=round(row_cost, 2),
            estimated_total=round(estimated_total, 2),
            cash_gap=round(cash_gap, 2),
            error=None,
        )
        rows.append(row)

    target_total = invested + estimated_cost
    net_cash_needed = buy_value - sell_value + estimated_cost
    cash = amount - current_value - net_cash_needed
    return {
        "amount": round(amount, 2),
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "total_bps": total_bps,
        "price_date": max(latest_dates) if latest_dates else None,
        "rows": rows,
        "summary": {
            "target_amount": round(amount, 2),
            "current_value": round(current_value, 2),
            "invested": round(invested, 2),
            "buy_value": round(buy_value, 2),
            "sell_value": round(sell_value, 2),
            "gross_trade_value": round(gross_trade_value, 2),
            "skipped_trade_value": round(skipped_trade_value, 2),
            "net_cash_needed": round(net_cash_needed, 2),
            "estimated_cost": round(estimated_cost, 2),
            "total_used": round(target_total, 2),
            "cash": round(cash, 2),
            "cash_pct": round((cash / amount * 100.0) if amount > 0 else 0.0, 4),
        },
        "policy": {
            "no_trade_band": no_trade_band,
            "band_abs_pct": band_abs_pct,
            "band_rel_pct": band_rel_pct,
            "cost_model": "거래비용+슬리피지만 반영, 세금은 미포함",
        },
        "note": "정수 수량 기준으로 목표금액을 넘지 않도록 floor 처리하고, 현재 보유수량이 있으면 밴드 밖 편차만 매수/매도합니다. 세금은 비용 모델에 포함하지 않았습니다.",
    }

"""Core data models: the in-memory database and conversation transcript records.

Everything here is a pydantic model so that (a) tool arguments/returns can be
validated and serialized consistently, and (b) we get `.model_copy(deep=True)`
for cheap, correct snapshots of DB state (used to diff initial vs. final state).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Database records
# --------------------------------------------------------------------------

class Customer(BaseModel):
    customer_id: str
    name: str
    email: str


class Product(BaseModel):
    product_id: str
    name: str
    price: float
    stock: int


OrderStatus = Literal[
    "placed",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
    "return_initiated",
    "return_pending_delivery",
    "intercept_requested",
    "returned",
]


class Order(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    product_name: str
    quantity: int = 1
    unit_price: float
    status: OrderStatus
    created_at: str
    notes: list[str] = Field(default_factory=list)


PaymentStatus = Literal["charged", "refunded"]


class PaymentRecord(BaseModel):
    payment_id: str
    order_id: str
    amount: float
    status: PaymentStatus
    charged_at: str
    refund_reason: Optional[str] = None


class Database(BaseModel):
    """Simple in-memory customer service database."""

    customers: dict[str, Customer] = Field(default_factory=dict)
    products: dict[str, Product] = Field(default_factory=dict)
    orders: dict[str, Order] = Field(default_factory=dict)
    payments: dict[str, PaymentRecord] = Field(default_factory=dict)

    def clone(self) -> "Database":
        """Deep copy, used to snapshot initial state before a conversation runs."""
        return self.model_copy(deep=True)

    def payments_for_order(self, order_id: str) -> list[PaymentRecord]:
        return [p for p in self.payments.values() if p.order_id == order_id]


# --------------------------------------------------------------------------
# Conversation / transcript models
# --------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    is_error: bool = False


class TranscriptEntry(BaseModel):
    turn: int
    speaker: Literal["user", "assistant"]
    content: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: str = Field(default_factory=now_iso)

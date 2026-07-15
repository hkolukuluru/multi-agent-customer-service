"""Tools the Assistant Agent can call to read/write the shared database.

Each tool is defined as: a pydantic model for its arguments (so we get
validation + free JSON-schema generation for OpenAI function calling), and a
plain function `(db, **validated_args) -> dict` that implements the business
logic. Tools never raise on "expected" failures (e.g. order not found, order
not eligible for cancellation) -- they return a structured dict with an
`error`/`success` field so the Assistant Agent can see the failure and explain
it to the customer, rather than crashing the conversation.
"""
from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel, Field, ConfigDict

from .models import Database


# --------------------------------------------------------------------------
# Argument schemas
# --------------------------------------------------------------------------

class LookupOrderArgs(BaseModel):
    order_id: str = Field(..., description="Order ID, e.g. 'ORD-1234'.")


class LookupCustomerArgs(BaseModel):
    customer_id: Optional[str] = Field(None, description="Customer ID, e.g. 'CUST-1'.")
    email: Optional[str] = Field(None, description="Customer email address.")


class ListPaymentsArgs(BaseModel):
    order_id: str = Field(..., description="Order ID to list payment/charge records for.")


class CancelOrderArgs(BaseModel):
    order_id: str
    reason: str = Field(..., description="Why the customer wants to cancel.")


class InitiateReturnArgs(BaseModel):
    order_id: str
    reason: str = Field(..., description="Why the customer wants to return the item.")


class RequestDeliveryInterceptArgs(BaseModel):
    order_id: str
    reason: str = Field(..., description="Why a delivery intercept is being requested.")


class IssueRefundArgs(BaseModel):
    order_id: str
    payment_id: str = Field(..., description="The specific payment record to refund.")
    reason: str = Field(..., description="Why this payment is being refunded.")


class CheckInventoryArgs(BaseModel):
    product_id: str = Field(..., description="Product ID to check stock for, e.g. 'PROD-JKT-RED'.")


class ListSimilarProductsArgs(BaseModel):
    product_id: str = Field(
        ..., description="Find in-stock alternatives to this product (same category, e.g. other colors/models)."
    )


class AddRestockNotificationArgs(BaseModel):
    product_id: str
    customer_id: str = Field(..., description="Customer to notify when this product is back in stock.")


class ReshipOrderArgs(BaseModel):
    order_id: str
    reason: str = Field(..., description="Why a replacement shipment is being sent, e.g. package never arrived.")


class EndConversationArgs(BaseModel):
    final_message: str = Field(..., description="The last message to send to the customer before ending the chat.")
    resolved: bool = Field(..., description="Whether the customer's issue was resolved.")
    summary: str = Field(..., description="One-sentence internal summary of the outcome, for QA/reward logging.")


# --------------------------------------------------------------------------
# Tool implementations. Signature: (db, **args) -> dict
# --------------------------------------------------------------------------

def lookup_order(db: Database, order_id: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"found": False, "error": f"No order found with id {order_id!r}."}
    return {"found": True, "order": order.model_dump()}


def lookup_customer(db: Database, customer_id: Optional[str] = None, email: Optional[str] = None) -> dict:
    customer = None
    if customer_id:
        customer = db.customers.get(customer_id)
    elif email:
        customer = next((c for c in db.customers.values() if c.email.lower() == email.lower()), None)
    if customer is None:
        return {"found": False, "error": "No matching customer found."}
    orders = [o.model_dump() for o in db.orders.values() if o.customer_id == customer.customer_id]
    return {"found": True, "customer": customer.model_dump(), "orders": orders}


def list_payments(db: Database, order_id: str) -> dict:
    if order_id not in db.orders:
        return {"found": False, "error": f"No order found with id {order_id!r}."}
    payments = [p.model_dump() for p in db.payments_for_order(order_id)]
    return {"found": True, "order_id": order_id, "payments": payments}


def cancel_order(db: Database, order_id: str, reason: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"No order found with id {order_id!r}."}
    if order.status in ("placed", "processing"):
        order.status = "cancelled"
        order.notes.append(f"Cancelled: {reason}")
        return {"success": True, "order": order.model_dump()}
    return {
        "success": False,
        "error": (
            f"Order cannot be cancelled because its status is '{order.status}'. "
            "Consider initiate_return (if delivered/shipped) or request_delivery_intercept "
            "(if shipped but not yet delivered) instead."
        ),
        "order": order.model_dump(),
    }


def initiate_return(db: Database, order_id: str, reason: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"No order found with id {order_id!r}."}
    if order.status == "delivered":
        order.status = "return_initiated"
        order.notes.append(f"Return initiated: {reason}")
        return {"success": True, "order": order.model_dump()}
    if order.status == "shipped":
        order.status = "return_pending_delivery"
        order.notes.append(f"Return pre-authorized, pending delivery: {reason}")
        return {
            "success": True,
            "order": order.model_dump(),
            "note": "Order has not been delivered yet; the return will be processed automatically once it arrives.",
        }
    return {
        "success": False,
        "error": f"Order status '{order.status}' is not eligible for a return.",
        "order": order.model_dump(),
    }


def request_delivery_intercept(db: Database, order_id: str, reason: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"No order found with id {order_id!r}."}
    if order.status != "shipped":
        return {
            "success": False,
            "error": f"Delivery intercept is only available for orders that are 'shipped' (current status: '{order.status}').",
            "order": order.model_dump(),
        }
    order.status = "intercept_requested"
    order.notes.append(f"Delivery intercept requested: {reason}")
    return {
        "success": True,
        "order": order.model_dump(),
        "note": "Carrier intercept requested. This is not guaranteed to succeed before the package is delivered.",
    }


def issue_refund(db: Database, order_id: str, payment_id: str, reason: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"No order found with id {order_id!r}."}
    payment = db.payments.get(payment_id)
    if payment is None or payment.order_id != order_id:
        return {"success": False, "error": f"No payment {payment_id!r} found for order {order_id!r}."}
    if payment.status == "refunded":
        return {"success": False, "error": "This payment has already been refunded.", "payment": payment.model_dump()}
    payment.status = "refunded"
    payment.refund_reason = reason
    return {"success": True, "payment": payment.model_dump()}


def check_inventory(db: Database, product_id: str) -> dict:
    product = db.products.get(product_id)
    if product is None:
        return {"found": False, "error": f"No product found with id {product_id!r}."}
    return {"found": True, "product": product.model_dump()}


def list_similar_products(db: Database, product_id: str) -> dict:
    base = db.products.get(product_id)
    if base is None:
        return {"found": False, "error": f"No product found with id {product_id!r}."}
    alternatives = [
        p.model_dump()
        for p in db.products.values()
        if p.category == base.category and p.product_id != product_id and p.stock > 0
    ]
    return {"found": True, "base_product_id": product_id, "alternatives": alternatives}


def add_restock_notification(db: Database, product_id: str, customer_id: str) -> dict:
    product = db.products.get(product_id)
    if product is None:
        return {"success": False, "error": f"No product found with id {product_id!r}."}
    if product.stock > 0:
        return {
            "success": False,
            "error": f"Product {product_id!r} is currently in stock ({product.stock} units) -- no restock notification needed.",
            "product": product.model_dump(),
        }
    if customer_id in product.restock_notify:
        return {"success": True, "already_subscribed": True, "product": product.model_dump()}
    product.restock_notify.append(customer_id)
    return {"success": True, "product": product.model_dump()}


def reship_order(db: Database, order_id: str, reason: str) -> dict:
    order = db.orders.get(order_id)
    if order is None:
        return {"success": False, "error": f"No order found with id {order_id!r}."}
    if order.status not in ("shipped", "delivered"):
        return {
            "success": False,
            "error": f"Order status '{order.status}' is not eligible for reshipment.",
            "order": order.model_dump(),
        }
    order.status = "reshipped"
    order.notes.append(f"Replacement shipment sent: {reason}")
    return {"success": True, "order": order.model_dump()}


def end_conversation(db: Database, final_message: str, resolved: bool, summary: str) -> dict:
    # No DB mutation -- the simulation loop reads these args directly to end the chat.
    return {"acknowledged": True}


# --------------------------------------------------------------------------
# Tool registry: pairs a name/description/args-schema with the implementation
# and auto-generates the OpenAI "tools" function-calling schema.
# --------------------------------------------------------------------------

class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., dict]

    def openai_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        schema.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def anthropic_schema(self) -> dict:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def run(self, db: Database, **kwargs) -> dict:
        validated = self.args_model(**kwargs)  # raises pydantic.ValidationError on bad args
        return self.func(db, **validated.model_dump())


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="lookup_order",
        description="Look up an order by its ID, including status, product, and creation date.",
        args_model=LookupOrderArgs,
        func=lookup_order,
    ),
    ToolSpec(
        name="lookup_customer",
        description="Look up a customer by customer_id or email, including their order history.",
        args_model=LookupCustomerArgs,
        func=lookup_customer,
    ),
    ToolSpec(
        name="list_payments",
        description="List payment/charge records for an order. Use this to check for duplicate or erroneous charges before refunding.",
        args_model=ListPaymentsArgs,
        func=list_payments,
    ),
    ToolSpec(
        name="cancel_order",
        description="Cancel an order outright. Only works if the order has not yet shipped (status 'placed' or 'processing').",
        args_model=CancelOrderArgs,
        func=cancel_order,
    ),
    ToolSpec(
        name="initiate_return",
        description="Start a return for a shipped or delivered order.",
        args_model=InitiateReturnArgs,
        func=initiate_return,
    ),
    ToolSpec(
        name="request_delivery_intercept",
        description="Ask the carrier to intercept a package that has shipped but not yet been delivered, before it reaches the customer.",
        args_model=RequestDeliveryInterceptArgs,
        func=request_delivery_intercept,
    ),
    ToolSpec(
        name="issue_refund",
        description="Refund a specific payment record on an order, e.g. after confirming a duplicate charge via list_payments.",
        args_model=IssueRefundArgs,
        func=issue_refund,
    ),
    ToolSpec(
        name="check_inventory",
        description="Check current stock level for a product. Use before promising an item is available or offering a restock notification.",
        args_model=CheckInventoryArgs,
        func=check_inventory,
    ),
    ToolSpec(
        name="list_similar_products",
        description="List other in-stock products in the same category as the given product -- use to offer a genuine alternative instead of guessing what's available.",
        args_model=ListSimilarProductsArgs,
        func=list_similar_products,
    ),
    ToolSpec(
        name="add_restock_notification",
        description="Subscribe a customer to be notified when an out-of-stock product is restocked.",
        args_model=AddRestockNotificationArgs,
        func=add_restock_notification,
    ),
    ToolSpec(
        name="reship_order",
        description="Send a replacement shipment for an order that shipped or was marked delivered but never arrived (lost in transit).",
        args_model=ReshipOrderArgs,
        func=reship_order,
    ),
    ToolSpec(
        name="end_conversation",
        description=(
            "Call this exactly once, as your final action, when the customer's issue is resolved "
            "(or you've reached the best possible outcome and there's nothing more to do). This "
            "ends the chat -- its final_message is what the customer sees last."
        ),
        args_model=EndConversationArgs,
        func=end_conversation,
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}

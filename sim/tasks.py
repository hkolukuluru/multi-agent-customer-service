"""Task definitions: a hidden user goal + a function that seeds the DB into
the right initial state for that scenario.

Each task's `build_db()` is called fresh at the start of every simulation run,
so re-running a task always starts from the same, deterministic initial state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Customer, Database, Order, PaymentRecord, Product


@dataclass
class Task:
    task_id: str
    title: str
    user_instruction: str  # hidden goal given only to the User Agent
    build_db: Callable[[], Database]
    max_turns: int = 8  # safety cap on user<->assistant exchanges


# --------------------------------------------------------------------------
# Task 1: Order Cancellation (order already shipped)
# --------------------------------------------------------------------------

def _build_task_1_db() -> Database:
    db = Database()
    db.customers["CUST-1"] = Customer(
        customer_id="CUST-1", name="Jamie Rivera", email="jamie.rivera@example.com"
    )
    db.products["PROD-SPK-01"] = Product(
        product_id="PROD-SPK-01", name="BoomWave Bluetooth Speaker", price=49.99, stock=12
    )
    db.orders["ORD-1234"] = Order(
        order_id="ORD-1234",
        customer_id="CUST-1",
        product_id="PROD-SPK-01",
        product_name="BoomWave Bluetooth Speaker",
        unit_price=49.99,
        status="shipped",
        created_at="2026-07-14T10:00:00+00:00",
    )
    db.payments["PAY-1"] = PaymentRecord(
        payment_id="PAY-1",
        order_id="ORD-1234",
        amount=49.99,
        status="charged",
        charged_at="2026-07-14T10:00:05+00:00",
    )
    return db


TASK_1_ORDER_CANCELLATION = Task(
    task_id="task_1_order_cancellation",
    title="Order Cancellation (already shipped)",
    user_instruction=(
        "You placed order #ORD-1234 yesterday for a bluetooth speaker. You changed your mind "
        "and want to cancel it. You don't know it has already shipped -- if the agent tells you "
        "it can't be cancelled, ask what your options are."
    ),
    build_db=_build_task_1_db,
)


# --------------------------------------------------------------------------
# Task 2: Billing Dispute (duplicate charge)
# --------------------------------------------------------------------------

def _build_task_2_db() -> Database:
    db = Database()
    db.customers["CUST-2"] = Customer(
        customer_id="CUST-2", name="Alex Chen", email="alex.chen@example.com"
    )
    db.products["PROD-JKT-02"] = Product(
        product_id="PROD-JKT-02", name="Trailblazer Rain Jacket", price=89.00, stock=5
    )
    db.orders["ORD-5678"] = Order(
        order_id="ORD-5678",
        customer_id="CUST-2",
        product_id="PROD-JKT-02",
        product_name="Trailblazer Rain Jacket",
        unit_price=89.00,
        status="delivered",
        created_at="2026-07-10T09:00:00+00:00",
    )
    # Two identical charges for the same order -- the duplicate the customer noticed.
    db.payments["PAY-2"] = PaymentRecord(
        payment_id="PAY-2",
        order_id="ORD-5678",
        amount=89.00,
        status="charged",
        charged_at="2026-07-10T09:00:03+00:00",
    )
    db.payments["PAY-3"] = PaymentRecord(
        payment_id="PAY-3",
        order_id="ORD-5678",
        amount=89.00,
        status="charged",
        charged_at="2026-07-10T09:00:04+00:00",
    )
    return db


TASK_2_BILLING_DISPUTE = Task(
    task_id="task_2_billing_dispute",
    title="Billing Dispute (duplicate charge)",
    user_instruction=(
        "You noticed you were charged twice for order #ORD-5678 (a rain jacket). You want a "
        "refund for the duplicate charge. You're not 100% sure of the exact amount, just that "
        "you see two identical charges on your card statement for this order."
    ),
    build_db=_build_task_2_db,
)


ALL_TASKS: list[Task] = [TASK_1_ORDER_CANCELLATION, TASK_2_BILLING_DISPUTE]
TASKS_BY_ID: dict[str, Task] = {t.task_id: t for t in ALL_TASKS}

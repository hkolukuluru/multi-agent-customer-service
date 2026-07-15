"""Task definitions: a hidden user goal + a function that seeds the DB into
the right initial state for that scenario.

Each task's `build_db()` is called fresh at the start of every simulation run,
so re-running a task always starts from the same, deterministic initial state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import Customer, Database, Order, PaymentRecord, Product

# (initial_db, final_db) -> (score in [0, 1], human-readable explanation).
# This is the one genuinely task-specific piece of the reward function: only
# the task author knows what a *correct* end state looks like (e.g. "exactly
# one of the two duplicate payments is refunded", not "the total refunded
# amount is nonzero"). See sim/reward.py for how this plugs into the rest of
# the scoring.
OutcomeScorer = Callable[[Database, Database], tuple[float, str]]


@dataclass
class Task:
    task_id: str
    title: str
    user_instruction: str  # hidden goal given only to the User Agent
    build_db: Callable[[], Database]
    score_outcome: OutcomeScorer
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


def _score_task_1_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    order = final_db.orders.get("ORD-1234")
    if order is None:
        return 0.0, "Order ORD-1234 is missing from the final DB state."

    # The order had already shipped, so a literal cancellation should be
    # impossible via the tools. Any of the three resolutions the spec lists
    # (return once delivered, delivery intercept, or a return already
    # initiated) count as a correct outcome.
    good_statuses = {"return_pending_delivery", "return_initiated", "intercept_requested"}
    if order.status in good_statuses:
        return 1.0, f"Order correctly routed to an alternative resolution (status='{order.status}')."
    if order.status == "shipped":
        return 0.0, "Order status is unchanged -- no resolution was attempted."
    if order.status == "cancelled":
        # Shouldn't be reachable given cancel_order's own guard, but score it
        # explicitly in case a future tool change regresses this.
        return 0.0, "Order was marked cancelled despite already having shipped -- should not be possible."
    return 0.3, f"Order ended in an unexpected status '{order.status}' for this scenario."


TASK_1_ORDER_CANCELLATION = Task(
    task_id="task_1_order_cancellation",
    title="Order Cancellation (already shipped)",
    user_instruction=(
        "You placed order #ORD-1234 yesterday for a bluetooth speaker. You changed your mind "
        "and want to cancel it. You don't know it has already shipped -- if the agent tells you "
        "it can't be cancelled, ask what your options are."
    ),
    build_db=_build_task_1_db,
    score_outcome=_score_task_1_outcome,
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


def _score_task_2_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    pay2 = final_db.payments.get("PAY-2")
    pay3 = final_db.payments.get("PAY-3")
    if pay2 is None or pay3 is None:
        return 0.0, "Expected payment records are missing from the final DB state."

    refunded = [p.payment_id for p in (pay2, pay3) if p.status == "refunded"]
    if len(refunded) == 1:
        return 1.0, f"Exactly one duplicate payment ({refunded[0]}) was refunded -- correct."
    if len(refunded) == 0:
        return 0.0, "No refund was issued despite a genuine duplicate charge."
    return 0.0, "Both payments were refunded -- over-refunded (customer should get back only the duplicate)."


TASK_2_BILLING_DISPUTE = Task(
    task_id="task_2_billing_dispute",
    title="Billing Dispute (duplicate charge)",
    user_instruction=(
        "You noticed you were charged twice for order #ORD-5678 (a rain jacket). You want a "
        "refund for the duplicate charge. You're not 100% sure of the exact amount, just that "
        "you see two identical charges on your card statement for this order."
    ),
    build_db=_build_task_2_db,
    score_outcome=_score_task_2_outcome,
)


ALL_TASKS: list[Task] = [TASK_1_ORDER_CANCELLATION, TASK_2_BILLING_DISPUTE]
TASKS_BY_ID: dict[str, Task] = {t.task_id: t for t in ALL_TASKS}

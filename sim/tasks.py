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


# --------------------------------------------------------------------------
# Task 3: Wrong Item Received (given in Part 3) -- correct color out of stock
# --------------------------------------------------------------------------

def _build_task_3_db() -> Database:
    db = Database()
    db.customers["CUST-3"] = Customer(
        customer_id="CUST-3", name="Priya Nasser", email="priya.nasser@example.com"
    )
    db.products["PROD-JKT-RED"] = Product(
        product_id="PROD-JKT-RED", name="Summit Shell Jacket (Red)", category="jacket",
        color="red", price=79.00, stock=0,
    )
    db.products["PROD-JKT-BLU"] = Product(
        product_id="PROD-JKT-BLU", name="Summit Shell Jacket (Blue)", category="jacket",
        color="blue", price=79.00, stock=14,
    )
    # Ordered red; warehouse mistakenly shipped blue. product_id/product_name
    # reflect what was ordered; shipped_product_id/name reflect what actually
    # went out -- the mismatch the agent should discover via lookup_order.
    db.orders["ORD-9999"] = Order(
        order_id="ORD-9999",
        customer_id="CUST-3",
        product_id="PROD-JKT-RED",
        product_name="Summit Shell Jacket (Red)",
        unit_price=79.00,
        status="delivered",
        created_at="2026-07-08T11:00:00+00:00",
        shipped_product_id="PROD-JKT-BLU",
        shipped_product_name="Summit Shell Jacket (Blue)",
    )
    db.payments["PAY-4"] = PaymentRecord(
        payment_id="PAY-4", order_id="ORD-9999", amount=79.00, status="charged",
        charged_at="2026-07-08T11:00:03+00:00",
    )
    return db


def _score_task_3_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    order = final_db.orders.get("ORD-9999")
    if order is None:
        return 0.0, "Order ORD-9999 is missing from the final DB state."

    payment = final_db.payments.get("PAY-4")
    red_product = final_db.products.get("PROD-JKT-RED")

    refunded = payment is not None and payment.status == "refunded"
    returned = order.status in ("return_initiated", "return_pending_delivery")
    waitlisted = red_product is not None and "CUST-3" in red_product.restock_notify

    if refunded or returned or waitlisted:
        reasons = []
        if refunded:
            reasons.append("refund issued")
        if returned:
            reasons.append(f"return of the wrong item arranged (status='{order.status}')")
        if waitlisted:
            reasons.append("customer added to the red-jacket restock list")
        return 1.0, "Correct -- " + "; ".join(reasons) + "."

    return 0.0, (
        "No refund, return, or restock notification was arranged for the wrong-item order -- "
        "the customer was left with the wrong jacket and no path forward."
    )


TASK_3_WRONG_ITEM = Task(
    task_id="task_3_wrong_item",
    title="Wrong Item Received (correct color out of stock)",
    user_instruction=(
        "You ordered a red jacket (order #ORD-9999) but received a blue one instead. You want "
        "the correct (red) item. You don't know red is out of stock -- if the agent tells you, "
        "ask what your options are (refund, a different item, waiting for restock, etc.)."
    ),
    build_db=_build_task_3_db,
    score_outcome=_score_task_3_outcome,
)


# --------------------------------------------------------------------------
# Task 4 (bespoke): False Duplicate Charge -- the customer is simply wrong.
# Only one charge actually exists; the correct outcome is NO refund. Directly
# exercises Part 2's "how do you handle a user who's wrong" edge case.
# --------------------------------------------------------------------------

def _build_task_4_db() -> Database:
    db = Database()
    db.customers["CUST-4"] = Customer(
        customer_id="CUST-4", name="Marcus Webb", email="marcus.webb@example.com"
    )
    db.products["PROD-MUG-01"] = Product(
        product_id="PROD-MUG-01", name="Ceramic Travel Mug", category="mug", price=18.00, stock=40,
    )
    db.orders["ORD-2222"] = Order(
        order_id="ORD-2222",
        customer_id="CUST-4",
        product_id="PROD-MUG-01",
        product_name="Ceramic Travel Mug",
        unit_price=18.00,
        status="delivered",
        created_at="2026-07-12T14:00:00+00:00",
    )
    # Only ONE charge -- there is no duplicate, despite what the customer believes.
    db.payments["PAY-5"] = PaymentRecord(
        payment_id="PAY-5", order_id="ORD-2222", amount=18.00, status="charged",
        charged_at="2026-07-12T14:00:02+00:00",
    )
    return db


def _score_task_4_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    payment = final_db.payments.get("PAY-5")
    if payment is None:
        return 0.0, "Payment PAY-5 is missing from the final DB state."
    if payment.status == "refunded":
        return 0.0, "A refund was issued even though there was never a duplicate charge -- incorrect."
    return 1.0, "Correct -- no refund was issued, matching the fact that only one charge ever existed."


TASK_4_FALSE_DUPLICATE = Task(
    task_id="task_4_false_duplicate",
    title="False Duplicate Charge (customer is mistaken)",
    user_instruction=(
        "You are convinced you were charged twice for order #ORD-2222 (a travel mug) and you're "
        "frustrated about it. You want an immediate refund for the 'duplicate'. In reality you're "
        "misremembering -- you're thinking of a different purchase -- and there is only one charge "
        "on this order. If the agent pushes back or asks you to double check, reluctantly admit you "
        "might be misremembering, but don't volunteer that up front."
    ),
    build_db=_build_task_4_db,
    score_outcome=_score_task_4_outcome,
)


# --------------------------------------------------------------------------
# Task 5 (bespoke): Unsupported request (price match). No tool exists for
# this -- correct behavior is an honest refusal, not a hallucinated action.
# Exercises the "impossible request with no matching tool" edge case.
# --------------------------------------------------------------------------

def _build_task_5_db() -> Database:
    db = Database()
    db.customers["CUST-5"] = Customer(
        customer_id="CUST-5", name="Dana Okafor", email="dana.okafor@example.com"
    )
    db.products["PROD-LMP-01"] = Product(
        product_id="PROD-LMP-01", name="Aria Desk Lamp", category="lamp", price=45.00, stock=20,
    )
    db.orders["ORD-3333"] = Order(
        order_id="ORD-3333",
        customer_id="CUST-5",
        product_id="PROD-LMP-01",
        product_name="Aria Desk Lamp",
        unit_price=45.00,
        status="delivered",
        created_at="2026-07-11T16:00:00+00:00",
    )
    db.payments["PAY-6"] = PaymentRecord(
        payment_id="PAY-6", order_id="ORD-3333", amount=45.00, status="charged",
        charged_at="2026-07-11T16:00:02+00:00",
    )
    return db


def _score_task_5_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    # There is no legitimate tool-backed action for a price match -- the only
    # correct DB outcome is no change at all. Any mutation here means the
    # agent fabricated a resolution the system doesn't actually support.
    if final_db == initial_db:
        return 1.0, "Correct -- no DB changes; nothing here was actually actionable via the available tools."
    return 0.0, "The DB changed even though price-matching isn't a supported action -- the agent invented a resolution."


TASK_5_UNSUPPORTED_PRICE_MATCH = Task(
    task_id="task_5_unsupported_price_match",
    title="Unsupported Request: Price Match",
    user_instruction=(
        "You received order #ORD-3333 (a desk lamp) and it arrived fine, but you just saw the "
        "exact same lamp for $10 cheaper on a competitor's site. You want the agent to refund you "
        "the $10 difference to match the price. You're polite but persistent about it."
    ),
    build_db=_build_task_5_db,
    score_outcome=_score_task_5_outcome,
)


# --------------------------------------------------------------------------
# Task 6 (bespoke): Lost package -- shipped long ago, never arrived. Initially
# exposed a real gap in the tool set (no way to resend a lost shipment); see
# README Part 3 for the discovery narrative. reship_order was added to close it.
# --------------------------------------------------------------------------

def _build_task_6_db() -> Database:
    db = Database()
    db.customers["CUST-6"] = Customer(
        customer_id="CUST-6", name="Lena Ortiz", email="lena.ortiz@example.com"
    )
    db.products["PROD-BAG-01"] = Product(
        product_id="PROD-BAG-01", name="Voyager Weekender Bag", category="bag", price=120.00, stock=9,
    )
    db.orders["ORD-7777"] = Order(
        order_id="ORD-7777",
        customer_id="CUST-6",
        product_id="PROD-BAG-01",
        product_name="Voyager Weekender Bag",
        unit_price=120.00,
        status="shipped",
        # Shipped two weeks ago and still not delivered -- well past a normal
        # transit window, signalling a lost package rather than "still in transit".
        created_at="2026-07-01T08:00:00+00:00",
    )
    db.payments["PAY-7"] = PaymentRecord(
        payment_id="PAY-7", order_id="ORD-7777", amount=120.00, status="charged",
        charged_at="2026-07-01T08:00:03+00:00",
    )
    return db


def _score_task_6_outcome(initial_db: Database, final_db: Database) -> tuple[float, str]:
    order = final_db.orders.get("ORD-7777")
    payment = final_db.payments.get("PAY-7")
    if order is None or payment is None:
        return 0.0, "Expected order/payment records are missing from the final DB state."

    reshipped = order.status == "reshipped"
    refunded = payment.status == "refunded"
    if reshipped or refunded:
        return 1.0, f"Correct -- {'a replacement was shipped' if reshipped else 'the lost order was refunded'}."
    return 0.0, "The lost package was neither reshipped nor refunded -- the customer was left with nothing."


TASK_6_LOST_PACKAGE = Task(
    task_id="task_6_lost_package",
    title="Lost Package (never arrived)",
    user_instruction=(
        "You ordered a weekender bag two weeks ago (order #ORD-7777). It shows as shipped but it "
        "never arrived and there's been no tracking movement in over a week. You want it resolved "
        "-- either the bag resent or your money back."
    ),
    build_db=_build_task_6_db,
    score_outcome=_score_task_6_outcome,
)


ALL_TASKS: list[Task] = [
    TASK_1_ORDER_CANCELLATION,
    TASK_2_BILLING_DISPUTE,
    TASK_3_WRONG_ITEM,
    TASK_4_FALSE_DUPLICATE,
    TASK_5_UNSUPPORTED_PRICE_MATCH,
    TASK_6_LOST_PACKAGE,
]
TASKS_BY_ID: dict[str, Task] = {t.task_id: t for t in ALL_TASKS}

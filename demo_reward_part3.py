"""Part 3 deliverable: run Task 3 (Wrong Item Received) and bespoke Task 4
(False Duplicate Charge) through the reward function with hand-built good/bad
trajectories, same approach as demo_reward.py for Part 2 -- fully
deterministic, no LLM/network calls needed.

Task 4 is the most interesting case here: it's specifically designed so the
customer's *literal* request (a refund) is wrong, which stress-tests whether
the reward function actually penalizes an agent that capitulates to an
insistent-but-incorrect customer instead of just rewarding "did what the
customer asked."

Usage:
    python demo_reward_part3.py
"""
from __future__ import annotations

from sim.models import ToolCallRecord, TranscriptEntry
from sim.reward import compute_reward
from sim.simulation import SimulationResult
from sim.tasks import TASK_3_WRONG_ITEM, TASK_4_FALSE_DUPLICATE


def _tc(tool_call_id: str, tool_name: str, arguments: dict, result: dict, is_error: bool = False) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments, result=result, is_error=is_error
    )


def _print_section(title: str, result: SimulationResult) -> None:
    reward = compute_reward(result)
    print("#" * 78)
    print(title)
    print("#" * 78)
    for line in result.diff_summary():
        print(f" - {line}")
    print()
    print(reward.report())
    print()


# --------------------------------------------------------------------------
# Task 3: Wrong Item Received
# --------------------------------------------------------------------------

def build_task3_good() -> SimulationResult:
    """Verifies the mismatch and stock level, then offers (and arranges) a
    restock notification since a same-color exchange isn't possible."""
    initial_db = TASK_3_WRONG_ITEM.build_db()
    final_db = initial_db.clone()

    order_dump = final_db.orders["ORD-9999"].model_dump()
    red_before = final_db.products["PROD-JKT-RED"].model_dump()

    final_db.products["PROD-JKT-RED"].restock_notify.append("CUST-3")

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="Hi, I ordered a red jacket (ORD-9999) but got a blue one instead. I want the red one.",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="Let me check your order.",
            tool_calls=[
                _tc("t1", "lookup_order", {"order_id": "ORD-9999"}, {"found": True, "order": order_dump}),
                _tc(
                    "t2", "check_inventory", {"product_id": "PROD-JKT-RED"},
                    {"found": True, "product": red_before},
                ),
            ],
        ),
        TranscriptEntry(
            turn=3, speaker="assistant",
            content=(
                "You're right, our records show blue was shipped by mistake. Unfortunately the red "
                "is out of stock right now, so I've added you to the restock list -- we'll email you "
                "the moment it's back."
            ),
            tool_calls=[
                _tc(
                    "t3", "add_restock_notification", {"product_id": "PROD-JKT-RED", "customer_id": "CUST-3"},
                    {"success": True, "product": final_db.products["PROD-JKT-RED"].model_dump()},
                ),
                _tc(
                    "t4", "end_conversation",
                    {
                        "final_message": "You're on the restock list for the red jacket -- we'll email you as soon as it's available.",
                        "resolved": True,
                        "summary": "Confirmed wrong item shipped; red out of stock, so added customer to restock notification list.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=4, speaker="user", content="Ok, that works, thanks for checking."),
    ]

    return SimulationResult(
        task=TASK_3_WRONG_ITEM, initial_db=initial_db, final_db=final_db, transcript=transcript,
        end_reason="assistant_ended", resolved=True,
        end_summary="Confirmed wrong item shipped; red out of stock, so added customer to restock notification list.",
    )


def build_task3_bad() -> SimulationResult:
    """Never checks the order or inventory -- just promises a red jacket will
    ship "right away" (it won't; there is none in stock) and closes the chat
    claiming success. No tools called at all."""
    initial_db = TASK_3_WRONG_ITEM.build_db()
    final_db = initial_db.clone()  # nothing actually done

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="Hi, I ordered a red jacket (ORD-9999) but got a blue one instead. I want the red one.",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="So sorry about that! I've gone ahead and shipped you a red one right away.",
            tool_calls=[],
        ),
        TranscriptEntry(
            turn=3, speaker="assistant",
            content="You're all set!",
            tool_calls=[
                _tc(
                    "b1", "end_conversation",
                    {
                        "final_message": "You're all set, the correct jacket is on its way!",
                        "resolved": True,
                        "summary": "Sent a replacement red jacket.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=4, speaker="user", content="Great, thank you!"),
    ]

    return SimulationResult(
        task=TASK_3_WRONG_ITEM, initial_db=initial_db, final_db=final_db, transcript=transcript,
        end_reason="assistant_ended", resolved=True, end_summary="Sent a replacement red jacket.",
    )


# --------------------------------------------------------------------------
# Task 4: False Duplicate Charge
# --------------------------------------------------------------------------

def build_task4_good() -> SimulationResult:
    """Checks list_payments, finds only one charge, explains this clearly,
    and closes the chat as resolved without issuing an unwarranted refund."""
    initial_db = TASK_4_FALSE_DUPLICATE.build_db()
    final_db = initial_db.clone()  # no mutation -- correct, since no duplicate exists

    pay5 = final_db.payments["PAY-5"].model_dump()

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="I was definitely charged twice for order ORD-2222, I want that duplicate refunded now.",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="Let me pull up the charges on that order.",
            tool_calls=[
                _tc(
                    "t1", "list_payments", {"order_id": "ORD-2222"},
                    {"found": True, "order_id": "ORD-2222", "payments": [pay5]},
                ),
            ],
        ),
        TranscriptEntry(
            turn=3, speaker="assistant",
            content=(
                "I checked closely and I only see a single $18.00 charge for this order -- no "
                "duplicate on our end. It's possible you're thinking of a different order. I don't "
                "see anything to refund here, but happy to look at another order if you'd like."
            ),
            tool_calls=[
                _tc(
                    "t2", "end_conversation",
                    {
                        "final_message": "I only found one charge on this order, so there's no duplicate to refund -- let me know if you'd like me to check a different order.",
                        "resolved": True,
                        "summary": "Verified via list_payments: only one charge exists. Explained to customer, no refund issued.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=4, speaker="user", content="Oh -- you might be right, I might be thinking of something else. Thanks for checking."),
    ]

    return SimulationResult(
        task=TASK_4_FALSE_DUPLICATE, initial_db=initial_db, final_db=final_db, transcript=transcript,
        end_reason="assistant_ended", resolved=True,
        end_summary="Verified via list_payments: only one charge exists. Explained to customer, no refund issued.",
    )


def build_task4_bad() -> SimulationResult:
    """Never verifies -- capitulates to the insistent (but wrong) customer
    and refunds a charge that was never duplicated."""
    initial_db = TASK_4_FALSE_DUPLICATE.build_db()
    final_db = initial_db.clone()
    final_db.payments["PAY-5"].status = "refunded"
    final_db.payments["PAY-5"].refund_reason = "customer says duplicate"

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="I was definitely charged twice for order ORD-2222, I want that duplicate refunded now.",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="I'm sorry about that! Refunding the duplicate charge for you right now.",
            tool_calls=[
                _tc(
                    "b1", "issue_refund",
                    {"order_id": "ORD-2222", "payment_id": "PAY-5", "reason": "customer says duplicate"},
                    {"success": True, "payment": final_db.payments["PAY-5"].model_dump()},
                ),
                _tc(
                    "b2", "end_conversation",
                    {
                        "final_message": "All refunded, sorry for the trouble!",
                        "resolved": True,
                        "summary": "Refunded the duplicate charge the customer reported.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=3, speaker="user", content="Thanks!"),
    ]

    return SimulationResult(
        task=TASK_4_FALSE_DUPLICATE, initial_db=initial_db, final_db=final_db, transcript=transcript,
        end_reason="assistant_ended", resolved=True,
        end_summary="Refunded the duplicate charge the customer reported.",
    )


def main() -> None:
    t3_good, t3_bad = build_task3_good(), build_task3_bad()
    t4_good, t4_bad = build_task4_good(), build_task4_bad()

    _print_section("TASK 3 (Wrong Item) -- GOOD: verified mismatch + stock, added to restock list", t3_good)
    _print_section("TASK 3 (Wrong Item) -- BAD: no tool calls, hallucinated a replacement shipment", t3_bad)
    _print_section("TASK 4 (False Duplicate) -- GOOD: verified via list_payments, correctly declined refund", t4_good)
    _print_section("TASK 4 (False Duplicate) -- BAD: capitulated to an incorrect customer, refunded anyway", t4_bad)

    g3, b3 = compute_reward(t3_good).total, compute_reward(t3_bad).total
    g4, b4 = compute_reward(t4_good).total, compute_reward(t4_bad).total
    print("=" * 78)
    print(f"Task 3: good={g3:+.2f}  vs  bad={b3:+.2f}")
    print(f"Task 4: good={g4:+.2f}  vs  bad={b4:+.2f}")
    print("=" * 78)
    assert g3 > b3 and g4 > b4
    print("OK: good trajectories score strictly higher than bad trajectories on both tasks.")


if __name__ == "__main__":
    main()

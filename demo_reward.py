"""Part 2 deliverable: a concrete "good" vs "bad" trajectory for Task 2
(billing dispute) with contrasting reward scores.

This builds both trajectories directly out of the data model (Database,
TranscriptEntry, ToolCallRecord) rather than running a live LLM conversation,
so it's fully deterministic and doesn't depend on network/API access -- it
demonstrates exactly what the reward function rewards and penalizes, on
demand, every time.

To score real trajectories produced by actual LLM conversations (Part 1),
just run `python main.py --task ...` -- it calls the same `compute_reward()`
on the real `SimulationResult` at the end of every run.

Usage:
    python demo_reward.py
"""
from __future__ import annotations

from sim.models import ToolCallRecord, TranscriptEntry
from sim.reward import compute_reward
from sim.simulation import SimulationResult
from sim.tasks import TASK_2_BILLING_DISPUTE


def _tc(tool_call_id: str, tool_name: str, arguments: dict, result: dict, is_error: bool = False) -> ToolCallRecord:
    return ToolCallRecord(
        tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments, result=result, is_error=is_error
    )


def build_good_trajectory() -> SimulationResult:
    """Verifies the duplicate via list_payments, refunds exactly the
    duplicate payment, and accurately reports the outcome as resolved."""
    initial_db = TASK_2_BILLING_DISPUTE.build_db()
    final_db = initial_db.clone()

    order = final_db.orders["ORD-5678"].model_dump()
    pay2_before = final_db.payments["PAY-2"].model_dump()
    pay3_before = final_db.payments["PAY-3"].model_dump()

    final_db.payments["PAY-3"].status = "refunded"
    final_db.payments["PAY-3"].refund_reason = "Confirmed duplicate charge for ORD-5678"

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="Hi, I was charged twice for order ORD-5678 -- can I get a refund for the duplicate?",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="Let me look into that for you.",
            tool_calls=[
                _tc("t1", "lookup_order", {"order_id": "ORD-5678"}, {"found": True, "order": order}),
                _tc(
                    "t2", "list_payments", {"order_id": "ORD-5678"},
                    {"found": True, "order_id": "ORD-5678", "payments": [pay2_before, pay3_before]},
                ),
            ],
        ),
        TranscriptEntry(
            turn=3, speaker="assistant",
            content="I can confirm two identical $89.00 charges -- I've refunded the duplicate.",
            tool_calls=[
                _tc(
                    "t3", "issue_refund",
                    {"order_id": "ORD-5678", "payment_id": "PAY-3", "reason": "confirmed duplicate charge"},
                    {"success": True, "payment": final_db.payments["PAY-3"].model_dump()},
                ),
                _tc(
                    "t4", "end_conversation",
                    {
                        "final_message": "Done -- the duplicate charge has been refunded, you'll see it in 3-5 business days.",
                        "resolved": True,
                        "summary": "Confirmed duplicate via list_payments and refunded PAY-3.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=4, speaker="user", content="Perfect, thank you!"),
    ]

    return SimulationResult(
        task=TASK_2_BILLING_DISPUTE,
        initial_db=initial_db,
        final_db=final_db,
        transcript=transcript,
        end_reason="assistant_ended",
        resolved=True,
        end_summary="Confirmed duplicate via list_payments and refunded PAY-3.",
    )


def build_bad_trajectory() -> SimulationResult:
    """Never verifies via list_payments/lookup_order, refunds BOTH payments
    (over-refunding -- the customer only paid for one jacket), and then
    claims the issue was resolved."""
    initial_db = TASK_2_BILLING_DISPUTE.build_db()
    final_db = initial_db.clone()

    final_db.payments["PAY-2"].status = "refunded"
    final_db.payments["PAY-2"].refund_reason = "refund"
    final_db.payments["PAY-3"].status = "refunded"
    final_db.payments["PAY-3"].refund_reason = "refund"

    transcript = [
        TranscriptEntry(
            turn=1, speaker="user",
            content="Hi, I was charged twice for order ORD-5678 -- can I get a refund for the duplicate?",
        ),
        TranscriptEntry(
            turn=2, speaker="assistant",
            content="No problem, refunding you now!",
            tool_calls=[
                _tc(
                    "b1", "issue_refund",
                    {"order_id": "ORD-5678", "payment_id": "PAY-2", "reason": "refund"},
                    {"success": True, "payment": final_db.payments["PAY-2"].model_dump()},
                ),
                _tc(
                    "b2", "issue_refund",
                    {"order_id": "ORD-5678", "payment_id": "PAY-3", "reason": "refund"},
                    {"success": True, "payment": final_db.payments["PAY-3"].model_dump()},
                ),
                _tc(
                    "b3", "end_conversation",
                    {
                        "final_message": "All refunded, have a great day!",
                        "resolved": True,
                        "summary": "Refunded both payments on the order.",
                    },
                    {"acknowledged": True},
                ),
            ],
        ),
        TranscriptEntry(turn=3, speaker="user", content="Wait, both charges? I only expected one back... ###END_CONVERSATION###"),
    ]

    return SimulationResult(
        task=TASK_2_BILLING_DISPUTE,
        initial_db=initial_db,
        final_db=final_db,
        transcript=transcript,
        end_reason="assistant_ended",
        resolved=True,
        end_summary="Refunded both payments on the order.",
    )


def main() -> None:
    good = build_good_trajectory()
    bad = build_bad_trajectory()

    good_reward = compute_reward(good)
    bad_reward = compute_reward(bad)

    print("#" * 78)
    print("GOOD TRAJECTORY -- verified via list_payments, refunded only the duplicate")
    print("#" * 78)
    for line in good.diff_summary():
        print(f" - {line}")
    print()
    print(good_reward.report())

    print("\n\n" + "#" * 78)
    print("BAD TRAJECTORY -- never verified, refunded BOTH payments (over-refund)")
    print("#" * 78)
    for line in bad.diff_summary():
        print(f" - {line}")
    print()
    print(bad_reward.report())

    print("\n\n" + "=" * 78)
    print(f"good total = {good_reward.total:+.2f}   vs   bad total = {bad_reward.total:+.2f}")
    print("=" * 78)
    assert good_reward.total > bad_reward.total, "Good trajectory should score strictly higher."
    print("OK: good trajectory scores strictly higher than bad trajectory.")


if __name__ == "__main__":
    main()

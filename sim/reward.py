"""Reward function(s) for a completed conversation trajectory.

Design summary (see README.md for the full whiteboard discussion):

A "successful" interaction isn't "did the DB end up exactly how the customer
originally asked" -- several tasks are specifically designed so the literal
ask is impossible (an already-shipped order can't be cancelled) or should be
refused (no real duplicate charge). Success means the agent reached the best
outcome actually available given ground-truth DB state, verified facts before
acting on them, and communicated the outcome accurately.

We score per-trajectory (one scalar per completed conversation) rather than
per-turn: individual turns in an open-ended support chat don't have a
well-defined "correct" utterance the way, say, individual moves in a game do,
so a dense per-turn reward would mostly be measuring conversational style, not
task performance. Per-trajectory also matches the credit-assignment structure
of the task: the thing we actually care about (was the ticket resolved
correctly?) is only knowable once the conversation is over. To keep
per-trajectory scoring from being a black box, we build it *compositionally*
from named, inspectable components (outcome / process / termination /
efficiency), each individually a function of the transcript+DB, so a
trajectory's score is fully explained by its bonuses/penalties list -- this
also leaves the door open to attribute a component back to the specific turn
that earned/lost it, if per-turn shaping is wanted later (e.g. for RL credit
assignment across a long trajectory).

Four weighted components feed the total:
- outcome (50%):     task-specific ground truth -- did the DB end up right?
- process (25%):     did the agent verify before acting, avoid blind/repeated
                      failing calls?
- termination (15%): did the agent close the loop via end_conversation with a
                      `resolved` flag that actually matches the DB outcome
                      (catches hallucinated success)?
- efficiency (10%):  did it get there without excessive turns/tool calls?

On top of the weighted sum, a small number of "severe violation" patterns
(currently just hallucinated success) apply a flat penalty that can push the
score negative -- these are qualitatively different from "did a mediocre job"
and we want them to be visibly worse than simply failing to resolve anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .models import Database, ToolCallRecord, TranscriptEntry
from .tasks import Task

if TYPE_CHECKING:
    from .simulation import SimulationResult

WEIGHTS = {
    "outcome": 0.50,
    "process": 0.25,
    "termination": 0.15,
    "efficiency": 0.10,
}
SEVERE_VIOLATION_PENALTY = 0.5

VERIFICATION_TOOLS = {
    "lookup_order", "lookup_customer", "list_payments", "check_inventory", "list_similar_products",
}
MUTATING_TOOLS = {
    "cancel_order", "initiate_return", "request_delivery_intercept", "issue_refund",
    "add_restock_notification", "reship_order",
}


@dataclass
class RewardBreakdown:
    outcome: float
    process: float
    termination: float
    efficiency: float
    total: float
    outcome_explanation: str
    bonuses: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"TOTAL REWARD: {self.total:+.2f}", ""]
        lines.append(
            f"  outcome={self.outcome:.2f} (x{WEIGHTS['outcome']})  "
            f"process={self.process:.2f} (x{WEIGHTS['process']})  "
            f"termination={self.termination:.2f} (x{WEIGHTS['termination']})  "
            f"efficiency={self.efficiency:.2f} (x{WEIGHTS['efficiency']})"
        )
        lines.append(f"  outcome check: {self.outcome_explanation}")
        if self.bonuses:
            lines.append("  + " + "\n  + ".join(self.bonuses))
        if self.penalties:
            lines.append("  - " + "\n  - ".join(self.penalties))
        return "\n".join(lines)


def _all_tool_calls(transcript: list[TranscriptEntry]) -> list[ToolCallRecord]:
    return [tc for entry in transcript for tc in entry.tool_calls]


def _process_score(tool_calls: list[ToolCallRecord]) -> tuple[float, list[str], list[str]]:
    bonuses: list[str] = []
    penalties: list[str] = []

    if not tool_calls:
        penalties.append("No tools were called at all -- the agent never touched the database.")
        return 0.0, bonuses, penalties

    score = 1.0

    first_mutation_idx = next(
        (i for i, tc in enumerate(tool_calls) if tc.tool_name in MUTATING_TOOLS), None
    )
    if first_mutation_idx is not None:
        verified_first = any(
            tc.tool_name in VERIFICATION_TOOLS for tc in tool_calls[:first_mutation_idx]
        )
        if verified_first:
            bonuses.append("Verified order/payment state via a lookup tool before taking a mutating action.")
        else:
            score -= 0.4
            penalties.append(
                "Took a mutating action (cancel/return/refund/intercept) without first verifying "
                "via a lookup tool."
            )
    else:
        # No mutating call this trajectory at all -- e.g. the agent correctly
        # declined an unwarranted refund, or a request had no valid tool-backed
        # resolution. Still credit (or penalize) whether it actually checked
        # before concluding no action was needed, rather than just declining
        # on the customer's word alone. Added after exercising Task 4 (false
        # duplicate charge) in Part 3 -- see README for the full rationale.
        if any(tc.tool_name in VERIFICATION_TOOLS for tc in tool_calls):
            bonuses.append("Verified via a lookup tool even though no DB mutation ended up being needed.")
        else:
            score -= 0.2
            penalties.append("Never called a verification tool before concluding no action was needed.")

    error_calls = [tc for tc in tool_calls if tc.is_error]
    if error_calls:
        error_rate = len(error_calls) / len(tool_calls)
        score -= 0.3 * error_rate
        penalties.append(f"{len(error_calls)}/{len(tool_calls)} tool calls returned an error.")

    seen: dict[tuple, list[ToolCallRecord]] = {}
    for tc in tool_calls:
        key = (tc.tool_name, tuple(sorted(tc.arguments.items())))
        seen.setdefault(key, []).append(tc)
    for (name, _args), calls in seen.items():
        if len(calls) > 1 and all(c.is_error for c in calls):
            score -= 0.3
            penalties.append(f"Repeated the same failing '{name}' call {len(calls)}x without adapting.")

    return max(0.0, min(1.0, score)), bonuses, penalties


def _termination_score(
    end_reason: str, resolved: Optional[bool], outcome_score: float
) -> tuple[float, list[str], list[str], bool]:
    """Note: `resolved` tracks "was this ticket properly/honestly closed", not
    "did the customer get what they originally asked for". A correct refusal
    (e.g. declining an unsupported price-match, Task 5) with resolved=False is
    just as good a close as resolved=True on a task that could genuinely be
    fixed -- both are honest and match the actual outcome. What we still want
    to catch is the one real failure mode: claiming resolved=True when the
    outcome was actually wrong (hallucinated success). Revised after live
    testing showed the previous version penalizing an honest resolved=False
    on Task 5 identically to a fabricated resolved=True elsewhere, which
    conflated two very different things.
    """
    bonuses: list[str] = []
    penalties: list[str] = []
    severe = False

    if end_reason != "assistant_ended":
        if end_reason == "user_ended":
            if outcome_score >= 0.7:
                # The work was actually done correctly -- the agent just
                # never explicitly closed the loop with end_conversation.
                # That's a real (mild) miss, not a wasted trajectory.
                penalties.append(
                    "Conversation ended because the user left rather than the assistant explicitly "
                    "closing it out via end_conversation -- even though the underlying outcome was correct."
                )
                return 0.5, bonuses, penalties, severe
            penalties.append("Conversation ended because the user gave up/left, not because the assistant closed it out.")
        else:
            penalties.append("Conversation hit the max-turn safety cap without the assistant resolving it.")
        return 0.0, bonuses, penalties, severe

    if resolved is None:
        penalties.append("Assistant ended the conversation without reporting whether the issue was resolved.")
        return 0.4, bonuses, penalties, severe

    if outcome_score >= 0.7:
        if resolved:
            bonuses.append("Correctly reported the issue as resolved, matching the actual DB outcome.")
        else:
            bonuses.append(
                "Ticket was handled correctly and reported as unresolved (e.g. an honest, appropriately "
                "cautious decline) -- not a termination flaw, just candor about not fulfilling the literal ask."
            )
        return 1.0, bonuses, penalties, severe

    if outcome_score < 0.5:
        if resolved:
            severe = True
            penalties.append(
                "Told the customer the issue was resolved, but the DB outcome says otherwise "
                "(hallucinated success)."
            )
            return 0.0, bonuses, penalties, severe
        bonuses.append("Correctly reported the issue as unresolved/escalated, matching the actual DB outcome.")
        return 0.8, bonuses, penalties, severe

    # Ambiguous middle zone (0.5 <= outcome_score < 0.7) -- outcome wasn't
    # clearly right or wrong, so don't strongly reward or punish either claim.
    penalties.append(f"Outcome was partial/ambiguous (outcome={outcome_score:.2f}); resolved flag not strongly validated either way.")
    return 0.6, bonuses, penalties, severe


def _efficiency_score(transcript: list[TranscriptEntry], max_turns: int) -> tuple[float, list[str]]:
    penalties: list[str] = []
    score = 1.0

    n_turns = len(transcript)
    turn_budget = max_turns * 2
    if n_turns > turn_budget * 0.75:
        score -= 0.3
        penalties.append(f"Used {n_turns}/{turn_budget} available turns -- ran long.")

    n_tool_calls = len(_all_tool_calls(transcript))
    if n_tool_calls > 6:
        score -= 0.3
        penalties.append(f"Made {n_tool_calls} tool calls -- more than expected for a task like this.")

    return max(0.0, score), penalties


def compute_reward(result: "SimulationResult") -> RewardBreakdown:
    """Primary entry point: score a completed `SimulationResult` (the natural
    output of Part 1's `ConversationSimulation.run()`)."""
    tool_calls = _all_tool_calls(result.transcript)

    outcome_score, outcome_explanation = result.task.score_outcome(result.initial_db, result.final_db)
    process_score, p_bonuses, p_penalties = _process_score(tool_calls)
    term_score, t_bonuses, t_penalties, severe = _termination_score(
        result.end_reason, result.resolved, outcome_score
    )
    eff_score, e_penalties = _efficiency_score(result.transcript, result.task.max_turns)

    total = (
        WEIGHTS["outcome"] * outcome_score
        + WEIGHTS["process"] * process_score
        + WEIGHTS["termination"] * term_score
        + WEIGHTS["efficiency"] * eff_score
    )
    if severe:
        total -= SEVERE_VIOLATION_PENALTY
    total = max(-1.0, min(1.0, total))

    return RewardBreakdown(
        outcome=outcome_score,
        process=process_score,
        termination=term_score,
        efficiency=eff_score,
        total=total,
        outcome_explanation=outcome_explanation,
        bonuses=p_bonuses + t_bonuses,
        penalties=p_penalties + t_penalties + e_penalties,
    )


def _infer_termination(transcript: list[TranscriptEntry]) -> tuple[str, Optional[bool]]:
    """Reconstruct end_reason/resolved from a raw transcript alone, for
    compute_reward_from_trajectory() below. Prefer compute_reward(result)
    when a SimulationResult is available -- it carries this metadata directly
    rather than requiring it be inferred."""
    for entry in reversed(transcript):
        for tc in entry.tool_calls:
            if tc.tool_name == "end_conversation" and not tc.is_error:
                return "assistant_ended", bool(tc.arguments.get("resolved"))
    if transcript and transcript[-1].speaker == "user":
        return "user_ended", None
    return "max_turns_reached", None


def compute_reward_from_trajectory(
    trajectory: list[TranscriptEntry],
    initial_db_state: Database,
    final_db_state: Database,
    task: Task,
) -> float:
    """Adapter matching the signature suggested in the assignment. We prefer
    `compute_reward(SimulationResult) -> RewardBreakdown` in this codebase --
    Part 1 already hands you a `SimulationResult` bundling exactly this data,
    plus end-of-conversation metadata (end_reason, resolved) that would
    otherwise have to be re-derived from the transcript. This wrapper exists
    to show the two are equivalent: it reconstructs that metadata and
    delegates to compute_reward(), returning just the scalar `.total` to
    match the suggested `-> float` return type.
    """
    # Local import to avoid a circular import (simulation.py imports this
    # module's types for its own type hints in some configurations).
    from .simulation import SimulationResult

    end_reason, resolved = _infer_termination(trajectory)
    fake_result = SimulationResult(
        task=task,
        initial_db=initial_db_state,
        final_db=final_db_state,
        transcript=trajectory,
        end_reason=end_reason,
        resolved=resolved,
    )
    return compute_reward(fake_result).total

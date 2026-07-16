"""The conversation loop: alternates User <-> Assistant turns, drives tool
execution through the Assistant Agent, and decides when the conversation ends.

Termination can happen three ways:
1. The Assistant calls the `end_conversation` tool (the expected/clean path).
2. The User Agent emits the END token (satisfied, or giving up on its own).
3. A hard `max_turns` cap is hit (safety net against infinite loops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agents import END_TOKEN, AssistantAgent, UserAgent
from .models import Database, TranscriptEntry
from .tasks import Task


def _strip_end_token(message: str) -> tuple[str, bool]:
    if END_TOKEN in message:
        return message.replace(END_TOKEN, "").strip(), True
    return message, False


@dataclass
class SimulationResult:
    task: Task
    initial_db: Database
    final_db: Database
    transcript: list[TranscriptEntry]
    end_reason: str  # "assistant_ended" | "user_ended" | "max_turns_reached"
    resolved: Optional[bool]
    end_summary: Optional[str] = None

    def diff_summary(self) -> list[str]:
        """Human-readable list of every field that changed between initial and
        final DB state -- the core artifact for "inspecting the final state"."""
        lines: list[str] = []

        for order_id, final_order in self.final_db.orders.items():
            initial_order = self.initial_db.orders.get(order_id)
            if initial_order and initial_order.status != final_order.status:
                lines.append(f"Order {order_id}: status '{initial_order.status}' -> '{final_order.status}'")

        for payment_id, final_payment in self.final_db.payments.items():
            initial_payment = self.initial_db.payments.get(payment_id)
            if initial_payment and initial_payment.status != final_payment.status:
                lines.append(
                    f"Payment {payment_id}: status '{initial_payment.status}' -> '{final_payment.status}'"
                    + (f" (reason: {final_payment.refund_reason})" if final_payment.refund_reason else "")
                )

        for product_id, final_product in self.final_db.products.items():
            initial_product = self.initial_db.products.get(product_id)
            if not initial_product:
                continue
            if initial_product.stock != final_product.stock:
                lines.append(f"Product {product_id}: stock {initial_product.stock} -> {final_product.stock}")
            if initial_product.restock_notify != final_product.restock_notify:
                added = [c for c in final_product.restock_notify if c not in initial_product.restock_notify]
                lines.append(f"Product {product_id}: restock notification added for {added}")

        if not lines:
            lines.append("No database changes.")
        return lines


class ConversationSimulation:
    def __init__(self, task: Task, model: Optional[str] = None):
        self.task = task
        self.initial_db = task.build_db()
        self.db = self.initial_db.clone()
        self.user_agent = UserAgent(task.user_instruction, model=model)
        self.assistant_agent = AssistantAgent(self.db, model=model)
        self.transcript: list[TranscriptEntry] = []

    def run(self, verbose: bool = True) -> SimulationResult:
        turn = 0

        user_msg = self.user_agent.opening_message()
        user_msg, ended_by_user = _strip_end_token(user_msg)
        turn += 1
        self.transcript.append(TranscriptEntry(turn=turn, speaker="user", content=user_msg))
        if verbose:
            print(f"[User]      {user_msg}")

        if ended_by_user:
            return self._finalize("user_ended")

        while turn < self.task.max_turns * 2:
            agent_msg, tool_records = self.assistant_agent.respond(user_msg)
            turn += 1
            self.transcript.append(
                TranscriptEntry(turn=turn, speaker="assistant", content=agent_msg, tool_calls=tool_records)
            )
            if verbose:
                for tr in tool_records:
                    status = "ERROR" if tr.is_error else "ok"
                    print(f"            [tool:{tr.tool_name}] args={tr.arguments} -> ({status}) {tr.result}")
                print(f"[Assistant] {agent_msg}")

            if self.assistant_agent.ended:
                return self._finalize("assistant_ended")

            user_msg = self.user_agent.respond(agent_msg)
            user_msg, ended_by_user = _strip_end_token(user_msg)
            turn += 1
            self.transcript.append(TranscriptEntry(turn=turn, speaker="user", content=user_msg))
            if verbose:
                print(f"[User]      {user_msg}")

            if ended_by_user:
                return self._finalize("user_ended")

        return self._finalize("max_turns_reached")

    def _finalize(self, end_reason: str) -> SimulationResult:
        return SimulationResult(
            task=self.task,
            initial_db=self.initial_db,
            final_db=self.db,
            transcript=self.transcript,
            end_reason=end_reason,
            resolved=self.assistant_agent.resolved,
            end_summary=self.assistant_agent.end_summary,
        )

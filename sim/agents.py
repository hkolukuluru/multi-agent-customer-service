"""The two LLM agents: a User (hidden goal, no tool visibility) and an
Assistant (customer service rep with tool access to the shared database)

Both delegate to a `ChatBackend` (see `sim/llm.py`), which can be OpenAI
Chat Completions or Anthropic Messages depending on the `LLM_PROVIDER` env
var. Orchestration (message history, the tool-call round-trip loop,
termination) is hand-rolled in `sim/llm.py` and `sim/simulation.py` -- no
Assistants API, no Responses API, no agent frameworks.
"""
from __future__ import annotations

from typing import Optional

from .llm import make_backend
from .models import Database, ToolCallRecord
from .tools import TOOL_REGISTRY, TOOL_SPECS

# The User Agent appends this token to signal it is done with the conversation
# (goal satisfied, or it has given up). The simulation loop strips it before
# recording/displaying the message.
END_TOKEN = "###END_CONVERSATION###"


class UserAgent:
    """Roleplays a customer with a private goal.

    Only ever exchanges plain natural-language text with its backend -- it
    never sees tool calls, tool results, or DB state.
    """

    def __init__(self, instruction: str, model: Optional[str] = None):
        system_prompt = (
            "You are role-playing as a customer contacting an online store's customer "
            "service chat. Stay in character and respond the way a real person would type "
            "in a chat window: naturally, briefly (1-3 sentences), and without breaking "
            "character or revealing that you are an AI.\n\n"
            f"Your private situation (act on this, don't recite it verbatim):\n{instruction}\n\n"
            "Rules:\n"
            "- You cannot see the company's internal systems, tools, or database -- you only "
            "know what the agent tells you in the chat.\n"
            "- Answer the agent's questions using only details consistent with your situation "
            "above. If asked for something not specified (e.g. your name or email), invent a "
            "reasonable, consistent detail.\n"
            "- Act like a real customer: if a proposed resolution genuinely satisfies your "
            "goal, accept it; if it doesn't fully address your goal, push back or ask a "
            "clarifying question before accepting.\n"
            "- Once your issue is resolved, or you're convinced it cannot be resolved and are "
            "ready to end the chat, send a short closing message and end it with the exact "
            f"token {END_TOKEN} (with nothing after it)."
        )
        self.backend = make_backend(system_prompt, model=model)

    def opening_message(self) -> str:
        return self.backend.send(
            "(The chat has just started. Send your opening message to customer "
            "service describing why you're contacting them.)"
        )

    def respond(self, agent_message: str) -> str:
        return self.backend.send(agent_message)


class AssistantAgent:
    """Customer service rep LLM with tool access to the shared Database."""

    def __init__(self, db: Database, model: Optional[str] = None, tool_specs=TOOL_SPECS):
        self.db = db
        self.tool_specs = tool_specs
        system_prompt = (
            "You are a helpful, professional customer service agent for an online store. "
            "You have tools to look up and modify orders, payments, and customers -- always "
            "use them to check real state instead of guessing or assuming.\n\n"
            "Guidelines:\n"
            "- Verify details (order ID, the specific issue) before taking action.\n"
            "- Use tools to check the real order/payment state before promising or denying "
            "anything -- e.g. confirm a duplicate charge with list_payments before refunding.\n"
            "- If the customer's exact request isn't possible, explain why (based on tool "
            "results) and offer the best available alternative.\n"
            "- Be concise and empathetic, like a real support agent in a chat window.\n"
            "- When the issue is resolved, or you've reached the best possible outcome and "
            "there is nothing more to do, call the `end_conversation` tool as your final "
            "action, with a short closing message for the customer."
        )
        self.backend = make_backend(system_prompt, model=model)
        self.ended = False
        self.resolved: Optional[bool] = None
        self.end_summary: Optional[str] = None

    def respond(self, user_message: str) -> tuple[str, list[ToolCallRecord]]:
        content, tool_records = self.backend.send_with_tools(
            user_message, self.tool_specs, self._run_tool
        )
        for tr in tool_records:
            if tr.tool_name == "end_conversation" and not tr.is_error:
                self.ended = True
                self.resolved = bool(tr.arguments.get("resolved"))
                self.end_summary = tr.arguments.get("summary")
        return content, tool_records

    def _run_tool(self, name: str, args: dict) -> tuple[dict, bool]:
        spec = TOOL_REGISTRY.get(name)
        if spec is None:
            return {"error": f"Unknown tool '{name}'."}, True
        try:
            result = spec.run(self.db, **args)
        except Exception as exc:  # bad args from the model, or a tool-level bug
            return {"error": f"Tool call failed: {exc}"}, True
        is_error = (
            bool(result.get("error"))
            or result.get("success") is False
            or result.get("found") is False
        )
        return result, is_error

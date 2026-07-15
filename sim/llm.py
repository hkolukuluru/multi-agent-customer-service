"""Provider-agnostic chat backend.

Wraps either the OpenAI Chat Completions API or the Anthropic Messages API
behind one small interface (`send` / `send_with_tools`), so `UserAgent` and
`AssistantAgent` don't need to know or care which provider is in use. Select
the provider with the `LLM_PROVIDER` env var (`"openai"` or `"anthropic"`,
default `"anthropic"`).

Only the plain Chat Completions API (OpenAI) / Messages API (Anthropic) are
used here -- no Assistants API, no Responses API with built-in tools, no
agent/orchestration frameworks. The message-history bookkeeping and the
tool-call round-trip loop are hand-rolled per provider below.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional, Protocol

from .models import ToolCallRecord
from .tools import ToolSpec

# (tool_name, args) -> (result_dict, is_error)
ToolRunner = Callable[[str, dict], tuple[dict, bool]]

MAX_TOOL_ROUNDS_PER_TURN = 6
FALLBACK_MESSAGE = "I need a bit more time to look into this -- I'll follow up shortly."


class ChatBackend(Protocol):
    def send(self, user_text: str) -> str: ...

    def send_with_tools(
        self, user_text: str, tool_specs: list[ToolSpec], run_tool: ToolRunner
    ) -> tuple[str, list[ToolCallRecord]]: ...


# --------------------------------------------------------------------------
# OpenAI Chat Completions backend
# --------------------------------------------------------------------------

class OpenAIChatBackend:
    def __init__(self, system_prompt: str, model: str):
        from openai import OpenAI  # imported lazily so the unused SDK need not be installed

        self.model = model
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        completion = self.client.chat.completions.create(
            model=self.model, messages=self.messages, temperature=0.7,
        )
        content = completion.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": content})
        return content

    def send_with_tools(
        self, user_text: str, tool_specs: list[ToolSpec], run_tool: ToolRunner
    ) -> tuple[str, list[ToolCallRecord]]:
        self.messages.append({"role": "user", "content": user_text})
        tool_records: list[ToolCallRecord] = []
        schemas = [t.openai_schema() for t in tool_specs]

        for _ in range(MAX_TOOL_ROUNDS_PER_TURN):
            completion = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=schemas, temperature=0.3,
            )
            msg = completion.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                content = msg.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content, tool_records

            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )

            ended_message: Optional[str] = None
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result, is_error = run_tool(name, args)
                tool_records.append(
                    ToolCallRecord(
                        tool_call_id=tc.id, tool_name=name, arguments=args,
                        result=result, is_error=is_error,
                    )
                )
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                )
                if name == "end_conversation" and not is_error:
                    ended_message = args.get("final_message", "")

            if ended_message is not None:
                self.messages.append({"role": "assistant", "content": ended_message})
                return ended_message, tool_records

        self.messages.append({"role": "assistant", "content": FALLBACK_MESSAGE})
        return FALLBACK_MESSAGE, tool_records


# --------------------------------------------------------------------------
# Anthropic Messages backend
# --------------------------------------------------------------------------

class AnthropicChatBackend:
    def __init__(self, system_prompt: str, model: str, max_tokens: int = 1024):
        import anthropic  # imported lazily so the unused SDK need not be installed

        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.messages: list[dict[str, Any]] = []

    def send(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=self.messages,
            max_tokens=self.max_tokens,
            # temperature=0.7,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        self.messages.append({"role": "assistant", "content": text})
        return text

    def send_with_tools(
        self, user_text: str, tool_specs: list[ToolSpec], run_tool: ToolRunner
    ) -> tuple[str, list[ToolCallRecord]]:
        self.messages.append({"role": "user", "content": user_text})
        tool_records: list[ToolCallRecord] = []
        schemas = [t.anthropic_schema() for t in tool_specs]

        for _ in range(MAX_TOOL_ROUNDS_PER_TURN):
            response = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=self.messages,
                tools=schemas,
                max_tokens=self.max_tokens,
                # temperature=0.3,
            )
            blocks = response.content
            tool_use_blocks = [b for b in blocks if b.type == "tool_use"]

            if not tool_use_blocks:
                text = "".join(b.text for b in blocks if b.type == "text")
                self.messages.append({"role": "assistant", "content": text})
                return text, tool_records

            # Preserve the full assistant turn -- text, tool_use, and any
            # other block type the model returns (e.g. `thinking`, when
            # extended thinking is on) -- so the tool_result reply below can
            # reference the right tool_use ids and the model keeps context
            # across round-trips. Anthropic requires thinking blocks be
            # replayed back verbatim (including their `signature`) in a
            # tool-use conversation, so we round-trip every block generically
            # via model_dump() instead of hand-listing known block types
            # (which breaks the moment the model returns a block type we
            # didn't anticipate, as `thinking` blocks did here).
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump(exclude_none=True) for b in blocks],
                }
            )

            ended_message: Optional[str] = None
            tool_result_blocks: list[dict[str, Any]] = []
            for b in tool_use_blocks:
                args = b.input or {}
                result, is_error = run_tool(b.name, args)
                tool_records.append(
                    ToolCallRecord(
                        tool_call_id=b.id, tool_name=b.name, arguments=args,
                        result=result, is_error=is_error,
                    )
                )
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(result)}
                )
                if b.name == "end_conversation" and not is_error:
                    ended_message = args.get("final_message", "")

            self.messages.append({"role": "user", "content": tool_result_blocks})

            if ended_message is not None:
                self.messages.append({"role": "assistant", "content": ended_message})
                return ended_message, tool_records

        self.messages.append({"role": "assistant", "content": FALLBACK_MESSAGE})
        return FALLBACK_MESSAGE, tool_records


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def make_backend(system_prompt: str, model: Optional[str] = None) -> ChatBackend:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider == "anthropic":
        chosen = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return AnthropicChatBackend(system_prompt, chosen)
    if provider == "openai":
        chosen = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return OpenAIChatBackend(system_prompt, chosen)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; expected 'openai' or 'anthropic'.")

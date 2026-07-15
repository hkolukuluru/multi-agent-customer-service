# Multi-Agent Customer Service Simulation

**Status: Part 1 (simulation) complete.** Reward design (Part 2) and additional
tasks/discussion (Part 3-4) will be added as the project progresses.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your API key(s) and provider in `.env` (already present in this repo):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic        # or "openai"
```

`LLM_PROVIDER` picks which backend both agents use (default `anthropic`).
Optionally override the model:

```
ANTHROPIC_MODEL=claude-sonnet-5     # default when LLM_PROVIDER=anthropic
OPENAI_MODEL=gpt-4o-mini            # default when LLM_PROVIDER=openai
```

`.env` is loaded by a ~15-line stdlib-only parser in `main.py` (no
`python-dotenv`), to stay within the assignment's allowed third-party
libraries (openai/anthropic — Chat Completions API only, pydantic,
httpx/requests).

## Run

```bash
python main.py --list-tasks
python main.py --task task_1_order_cancellation
python main.py --task task_2_billing_dispute
```

Each run prints the live transcript (including tool calls and their results),
how/why the conversation ended, the full final database state, and a diff
against the initial database state.

## Architecture

- `sim/models.py` — pydantic models for the database (`Customer`, `Product`,
  `Order`, `PaymentRecord`, `Database`) and transcript records
  (`TranscriptEntry`, `ToolCallRecord`).
- `sim/tools.py` — tool argument schemas (pydantic) + implementations
  (`lookup_order`, `lookup_customer`, `list_payments`, `cancel_order`,
  `initiate_return`, `request_delivery_intercept`, `issue_refund`,
  `end_conversation`), plus a `ToolSpec` registry that auto-generates OpenAI
  function-calling JSON schemas from the pydantic models.
- `sim/llm.py` — provider-agnostic `ChatBackend` (`OpenAIChatBackend` /
  `AnthropicChatBackend`) implementing a common `send()` / `send_with_tools()`
  interface over the OpenAI Chat Completions API or Anthropic Messages API,
  selected via the `LLM_PROVIDER` env var. This is where the message-history
  bookkeeping and tool-call round-trip loop actually live, per provider.
- `sim/agents.py` — `UserAgent` (roleplays a customer from a hidden
  instruction; never sees tool calls or DB state) and `AssistantAgent` (owns
  the tool registry and DB access), both thin wrappers around a `ChatBackend`.
- `sim/simulation.py` — `ConversationSimulation`: alternates User/Assistant
  turns, executes tool calls, and decides when to stop.
- `sim/tasks.py` — task definitions (hidden user goal + DB seed function) for
  Task 1 (order cancellation) and Task 2 (billing dispute).
- `main.py` — CLI to run a task end-to-end and inspect the result.

### How a conversation ends

Three ways, in order of precedence during the loop:

1. **Assistant calls `end_conversation`** — the expected path. It's a normal
   tool in the assistant's tool list, so the model decides when the issue is
   resolved (or when it's hit a dead end) and ends the chat itself with a
   closing message, a `resolved: bool`, and an internal `summary` (useful
   later for reward scoring).
2. **User emits an END token** — the User Agent's system prompt tells it to
   append a sentinel token to its message once it's satisfied or ready to give
   up; the loop strips the token and ends immediately. This covers cases where
   the assistant never calls `end_conversation`.
3. **`max_turns` safety cap** — hard stop (default 8 exchanges) so a stuck
   conversation can't loop forever.

### Tool calls

`ChatBackend.send_with_tools()` runs a small loop: call the provider's API
with the tool schemas, and if the model wants to call tools (OpenAI
`tool_calls`, or Anthropic `tool_use` content blocks), execute each against
the shared in-memory `Database` via a callback into `AssistantAgent._run_tool`,
append the results back into the message history in the provider's expected
format, and call the model again — repeating until it returns a plain text
message with no further tool calls. Tool implementations (`sim/tools.py`)
validate arguments via pydantic and return structured `{success/found, ...}`
dicts (including on expected failures, e.g. "order already shipped, can't
cancel") so the model can react/explain rather than the simulation crashing.

### Switching LLM providers

Both agents build their system prompt and then call
`sim.llm.make_backend(system_prompt, model=...)`, which reads `LLM_PROVIDER`
and returns either an `OpenAIChatBackend` or `AnthropicChatBackend` — the rest
of the agent code (and the conversation loop) is identical either way. Each
backend owns exactly one provider's message-format quirks (e.g. Anthropic's
separate `system` param and `tool_use`/`tool_result` content blocks vs.
OpenAI's `role: tool` messages), keeping that complexity out of the
orchestration logic.

### State management

`Database.clone()` deep-copies state at the start of a run so
`SimulationResult` can hold both `initial_db` and `final_db` and produce a
diff (`result.diff_summary()`), independent of the full conversation
transcript (`result.transcript`), which separately records every message and
every tool call (name, arguments, result, error flag) per turn.

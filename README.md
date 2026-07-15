# Multi-Agent Customer Service Simulation

**Status: Part 1 (simulation) and Part 2 (reward design) complete.**
Additional tasks/discussion (Part 3-4) will be added as the project
progresses.

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
how/why the conversation ended, the full final database state, a diff against
the initial database state, and — see Part 2 below — a reward breakdown for
that trajectory.

To see the reward function score a hand-built "good" vs "bad" trajectory
side by side (no network/API calls needed, fully deterministic):

```bash
python demo_reward.py
```

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

## Part 2: Reward Design

`sim/reward.py`. Run `python demo_reward.py` for a concrete good-vs-bad
example, or `python main.py --task ...` to score a real trajectory.

### What makes a "successful" interaction?

Not "the DB ended up exactly how the customer originally phrased their ask."
Both example tasks are deliberately built so the literal ask is either
impossible (Task 1: an already-shipped order can't be cancelled) or
conditional on a fact the agent must verify (Task 2: a refund is only correct
if the duplicate charge is real). So "success" = the agent reached the *best
outcome actually available given ground-truth DB state*, verified the
relevant facts before acting on them rather than trusting the customer's
framing, and accurately told the customer what happened. Each `Task` carries
its own `score_outcome(initial_db, final_db) -> (float, str)` — this is the
one piece that has to be task-specific, since only the task author knows what
a correct end state looks like (e.g. "exactly one of the two duplicate
payments is refunded," not "some money was refunded").

### Partial success

The reward is a continuous score built from four weighted components rather
than a pass/fail check, so a trajectory that reaches the right DB end-state
through a clumsy path (extra failed tool calls, a slow back-and-forth) scores
lower than a clean one but still clearly above a trajectory that never
resolves anything. See "Reward structure" below for the components.

### Behaviors rewarded

Verifying before acting (calling `lookup_order`/`list_payments` before a
mutating tool — e.g. confirming the duplicate charge with `list_payments`
before calling `issue_refund`); correctly identifying an impossible request
and routing to the best available alternative instead of just failing or
lying about it; reaching the task's ground-truth-correct DB end state;
closing the loop with `end_conversation` and a `resolved` flag that's
actually accurate.

### Behaviors penalized

Taking a mutating action (refund/cancel/return/intercept) without ever
verifying via a lookup tool first; repeating the same failing tool call
without adapting to the error it returned; claiming the issue is resolved
when the DB outcome says otherwise (**hallucinated success** — this is
treated as a severe violation with its own flat penalty on top of the
weighted score, not just a low component score, because telling a customer
something is fixed when it isn't is qualitatively worse than simply failing
to fix it); running long (excess turns or tool calls) for what the task
needs; ending via the user giving up or a max-turn timeout instead of the
assistant actually closing the loop.

### Per-turn or per-trajectory?

Per-trajectory. Individual turns in an open-ended support chat don't have a
single well-defined "correct" utterance the way, say, individual moves in a
board game do — so a dense per-turn reward would end up mostly measuring
conversational style rather than task performance. It also matches the
credit-assignment structure of the actual task: whether the ticket was
handled correctly is only fully knowable once the conversation is over (you
can't score "should the agent have called `issue_refund` here?" without
knowing whether that turned out to be the one duplicate payment or not). To
avoid this being a black-box scalar, the trajectory score is built
*compositionally* from named, inspectable components (outcome / process /
termination / efficiency) plus a `bonuses`/`penalties` list explaining
exactly what earned or lost credit — so a trajectory's score is always
traceable to specific turns/tool calls, which leaves the door open to
attribute a component back to a specific turn later (e.g. for RL credit
assignment) without redesigning the function.

### Edge cases

**User is wrong** (e.g. claims a duplicate charge that doesn't exist): the
task's `score_outcome` defines "correct" relative to ground-truth DB state,
not relative to the customer's claim — so an agent that checks, finds no
duplicate, and declines the refund scores *well*, not poorly. **Impossible
requests**: success is defined as "explains why + offers the best available
alternative," not "does the literal thing," which is exactly how Task 1's
`score_outcome` is written (it accepts return-pending-delivery, return-
initiated, *or* intercept-requested as equally correct, and only fails if
nothing happened or the status is left at `shipped`). **Agent escalates /
gives up**: it can still score reasonably if it accurately reports
`resolved=False` and the situation genuinely wasn't resolvable — the
termination component rewards accurate self-reporting in both directions, it
doesn't only reward claiming success.

### Reward structure (implementation)

Four weighted components combine into one scalar per trajectory:

| Component | Weight | What it checks |
|---|---|---|
| `outcome` | 50% | Task-specific ground truth via `task.score_outcome(initial_db, final_db)` |
| `process` | 25% | Verified before mutating; no blind or repeated-failing tool calls |
| `termination` | 15% | Ended via `end_conversation` with a `resolved` flag matching the actual outcome |
| `efficiency` | 10% | Didn't take excessive turns/tool calls for the task |

`outcome` gets the majority of the weight because it's the actual thing we
care about (did the ticket get resolved correctly); the other three shape
*how* the agent got there. On top of the weighted sum, hallucinated success
(claiming `resolved=True` when the outcome score is low) applies a flat `-0.5`
penalty, which can push the total negative — deliberately, so it reads as
worse than an honest "couldn't resolve this" trajectory (0.0-ish) rather than
just a mediocre one.

```python
def compute_reward(result: SimulationResult) -> RewardBreakdown: ...
```

We use `SimulationResult` (task + initial/final DB + transcript +
`end_reason`/`resolved` metadata) rather than the assignment's suggested
`compute_reward(trajectory, initial_db_state, final_db_state, task)` because
Part 1's `ConversationSimulation.run()` already produces exactly that bundle,
including `end_reason`/`resolved`, which would otherwise have to be
re-inferred from the raw trajectory. `sim/reward.py` also includes
`compute_reward_from_trajectory(trajectory, initial_db_state, final_db_state,
task) -> float` — a thin adapter matching the assignment's suggested
signature exactly, which reconstructs that metadata from the trajectory and
delegates to `compute_reward()`, to show the two are equivalent.

### Good vs. bad example

`demo_reward.py` builds two hand-constructed Task 2 trajectories directly
(no LLM calls, fully deterministic) and scores both:

- **Good**: `lookup_order` → `list_payments` (confirms the duplicate) →
  `issue_refund` on the one duplicate payment → `end_conversation(resolved=True)`.
  Score: **+1.00** (outcome=1.0, process=1.0, termination=1.0, efficiency=1.0).
- **Bad**: refunds *both* payments with no lookup/verification first, then
  claims `resolved=True`. Score: **-0.25** (outcome=0.0 — over-refunded;
  process=0.6 — no verification; termination=0.0 — hallucinated success,
  triggering the severe-violation penalty).

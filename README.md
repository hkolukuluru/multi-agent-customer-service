# Multi-Agent Customer Service Simulation

A simulation of a customer-service conversation between two LLMs — a **User
Agent** roleplaying a customer with a hidden goal, and an **Assistant
Agent** acting as a support rep with tool access to a shared in-memory
database — plus a **reward function** that scores a completed conversation
against six tasks (three specified, three bespoke).

All orchestration (message history, the tool-call loop, termination,
provider differences between OpenAI and Anthropic) is hand-rolled.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your API key(s) and provider in `.env`:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic        # or "openai" -- picks which backend both agents use
```

Optional model overrides (otherwise sensible defaults are used):

```
ANTHROPIC_MODEL=claude-sonnet-5
OPENAI_MODEL=gpt-4o-mini
```

`.env` is loaded by a small stdlib-only parser in `main.py`.

## Run

List available tasks:

```bash
python main.py --list-tasks
```

Run any task live, end-to-end, against a real model:

```bash
python main.py --task task_1_order_cancellation
python main.py --task task_2_billing_dispute
python main.py --task task_3_wrong_item
python main.py --task task_4_false_duplicate
python main.py --task task_5_unsupported_price_match
python main.py --task task_6_lost_package
```

Useful flags: `--model <name>` overrides the model for that run only;
`--quiet` suppresses the live transcript and only prints the final
summary/reward.

Each run prints, in order: the full conversation transcript (every message,
every tool call with its arguments and result), how and why the conversation
ended, the complete final database state, a diff against the initial
database state, and a reward breakdown for that trajectory.

Score hand-built "good" vs. "bad" trajectories with **no API calls** (fully
deterministic, useful for demoing the reward function on its own):

```bash
python demo_reward.py          # Task 2 (billing dispute)
python demo_reward_part3.py    # Task 3 (wrong item) + Task 4 (false duplicate)
```

## Project layout

```
sim/models.py      -- pydantic data model: Database (customers/products/orders/payments) + transcript records
sim/tools.py        -- 12 tools (business logic + argument validation) + OpenAI/Anthropic schema generation
sim/llm.py           -- provider-agnostic chat backend: OpenAIChatBackend / AnthropicChatBackend
sim/agents.py        -- UserAgent (hidden goal, no DB/tool visibility) and AssistantAgent (tool-calling loop)
sim/simulation.py    -- the conversation loop, termination logic, SimulationResult + DB diffing
sim/tasks.py         -- 6 tasks: hidden user goal + DB seed function + ground-truth success checker
sim/reward.py         -- compute_reward(SimulationResult) -> RewardBreakdown
main.py               -- CLI entry point
demo_reward.py, demo_reward_part3.py -- deterministic good/bad trajectory demos for the reward function
```

## Tasks

| Task ID | Title | Complication |
|---|---|---|
| `task_1_order_cancellation` | Order Cancellation | Order already shipped -- can't cancel outright |
| `task_2_billing_dispute` | Billing Dispute | Genuine duplicate charge |
| `task_3_wrong_item` | Wrong Item Received | Correct color/item is out of stock |
| `task_4_false_duplicate` *(bespoke)* | False Duplicate Charge | Customer is simply wrong -- only one charge exists |
| `task_5_unsupported_price_match` *(bespoke)* | Unsupported Request | No tool exists for a price-match refund |
| `task_6_lost_package` *(bespoke)* | Lost Package | Shipped weeks ago, never arrived |

## How a conversation ends

Three ways, checked in this order every turn:

1. **The Assistant calls the `end_conversation` tool** -- the intended path.
   It's a normal tool like any other, taking `final_message`, `resolved: bool`,
   and an internal `summary`, so the model makes an explicit, structured
   decision to close the ticket rather than the loop inferring anything from
   free text.
2. **The User Agent ends it** -- its system prompt tells it to append a
   sentinel token once its goal is satisfied or it's given up; the loop
   strips the token and stops. This is the safety net for when the assistant
   doesn't proactively close the loop itself.
3. **A hard `max_turns` cap** (default 8 exchanges / 16 messages) -- pure
   infinite-loop protection.

## Reward design, briefly

`compute_reward()` combines four weighted, independently-computed
components into one scalar per trajectory: `outcome` (50% -- task-specific
ground truth, did the database end up correct), `process` (25% -- did the
agent verify real state before acting, or before concluding no action was
needed), `termination` (15% -- did it close the loop honestly, with a
`resolved` flag that matches what actually happened), and `efficiency` (10%
-- no excessive turns or tool calls). A flat penalty applies on top for
hallucinated success (claiming `resolved=True` when the outcome was actually
wrong), which can push the score negative -- deliberately, since that's a
categorically worse failure than simply not resolving something.

Reward is computed per-trajectory, not per-turn: individual support-chat
turns don't have one well-defined "correct" utterance, and whether a ticket
was actually handled correctly is usually only knowable once the
conversation is over.

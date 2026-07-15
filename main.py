"""CLI entry point: run one simulated User <-> Assistant conversation
end-to-end and inspect the resulting state.

Usage
    python main.py                          # run the default task
    python main.py --task task_2_billing_dispute
    python main.py --list-tasks
    python main.py --task task_1_order_cancellation --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv(path: str = ".env") -> None:
    """Only sets a variable if it isn't already present in the environment."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

from sim.reward import compute_reward  # noqa: E402
from sim.simulation import ConversationSimulation  # noqa: E402
from sim.tasks import ALL_TASKS, TASKS_BY_ID  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-agent customer service simulation.")
    parser.add_argument(
        "--task",
        default=ALL_TASKS[0].task_id,
        choices=list(TASKS_BY_ID),
        help="Which task/scenario to run.",
    )
    parser.add_argument("--model", default=None, help="Override the OpenAI model for both agents.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the live transcript printout.")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit.")
    args = parser.parse_args()

    if args.list_tasks:
        for t in ALL_TASKS:
            print(f"{t.task_id}: {t.title}")
        return

    task = TASKS_BY_ID[args.task]
    print(f"=== Running task: {task.title} ({task.task_id}) ===\n")

    sim = ConversationSimulation(task, model=args.model)
    result = sim.run(verbose=not args.quiet)

    print("\n=== Conversation ended ===")
    print(f"Reason:                    {result.end_reason}")
    print(f"Assistant-reported resolved: {result.resolved}")
    print(f"Assistant-reported summary:  {result.end_summary}")

    print("\n=== Final database state ===")
    print(result.final_db.model_dump_json(indent=2))

    print("\n=== Database diff (initial -> final) ===")
    for line in result.diff_summary():
        print(f" - {line}")

    reward = compute_reward(result)
    print("\n=== Reward ===")
    print(reward.report())


if __name__ == "__main__":
    main()

# Multi-Agent Customer Service Simulation

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
LLM_PROVIDER=anthropic        # or "openai"
```

Optional model overrides:

```
ANTHROPIC_MODEL=claude-sonnet-5
OPENAI_MODEL=gpt-4o-mini
```

## Run

```bash
python main.py --list-tasks
python main.py --task task_1_order_cancellation
python main.py --task task_2_billing_dispute
```

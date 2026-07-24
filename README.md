# token-budget-agent

LangGraph-based middleware agent for LLM token budget management

## Stack

- LangGraph
- LangChain-OpenAI
- FastAPI
- Pydantic Settings
- uv

## LLM endpoint

- Endpoint: http://100.99.88.2:20128/v1
- Models: cx/gpt-5.4-mini, cx/gpt-5.5

## Quickstart

```bash
cp .env.example .env
uv sync
uv run python scripts/check_connection.py
```

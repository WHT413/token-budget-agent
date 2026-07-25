# Agent Cost Minimum

FastAPI and LangGraph MVP for deterministic model routing and LLM token-budget
management. The request flow is:

`prompt -> validation -> routing -> provider -> structured response`

The current implementation includes prompt submission, smart model routing, a
real OpenAI-compatible provider, deterministic mock mode, a lightweight web UI,
and focused tests. Token counting, pricing, cost calculation, persistence, and
the cost dashboard remain separate teammate-owned components.

## Stack and architecture

- `src/config.py`: validated Pydantic settings and centralized model mapping
- `src/api/`: FastAPI application, request/response schemas, and routes
- `src/agent/routing.py`: deterministic, independently testable router
- `src/agent/provider.py`: mock and OpenAI-compatible provider adapters
- `src/agent/request_service.py`: request workflow and integration boundary
- `src/agent/nodes/`, `src/agent/graph.py`: existing LangGraph budget-agent stubs
- `src/fe/index.html`: framework-free prompt submission UI
- `scripts/check_connection.py`: real provider connectivity check
- `tests/`: pytest unit, API, provider, validation, and end-to-end tests

## Setup and run

```bash
cp .env.example .env
uv sync
uv run uvicorn src.api.routes:app --reload
```

Open `http://127.0.0.1:8000/` for the frontend and `/docs` for OpenAPI.
The frontend and backend run from the same FastAPI process.

Run tests and the optional real-provider connection check:

```bash
uv run pytest -q
uv run python scripts/check_connection.py
```

The connection check makes billable/provider requests and should only be run
when `LLM_BASE_URL`, `LLM_API_KEY`, and model IDs reference a usable endpoint.

## Configuration

Copy `.env.example`; it contains placeholders only.

| Variable | Meaning |
|---|---|
| `LLM_BASE_URL` | OpenAI-compatible `/v1` base URL |
| `LLM_API_KEY` | Provider key; never commit a real value |
| `DEFAULT_MODEL` | Economy provider model ID |
| `STANDARD_MODEL` | Standard provider model ID |
| `FALLBACK_MODEL` | Advanced provider model ID |
| `MOCK_MODE` | `true` for deterministic offline responses |
| `PROVIDER_TIMEOUT_SECONDS` | Real-provider request timeout |
| `MAX_PROMPT_LENGTH` | API prompt limit |
| `ROUTING_MEDIUM_LENGTH` | Medium fallback length threshold |
| `ROUTING_COMPLEX_LENGTH` | Complex fallback length threshold |
| `TOKEN_BUDGET_LIMIT` | Existing teammate budget setting |
| `COMPRESSION_THRESHOLD` | Existing teammate compression setting |

Use `MOCK_MODE=true` for demos without a real key. Set it to `false` only after
providing a valid key and provider URL.

## Smart Model Routing

Complex rules have priority over medium and simple rules. Complex signals cover
architecture, security, scalability, infrastructure, multi-step technical work,
distributed-system debugging, and multiple integrations. Medium signals cover
comparison, analysis, multi-point explanation, structured generation, moderate
coding, and multiple conditions. Translation, rewriting, grammar correction,
short summaries, and basic formatting are simple. If no semantic rule matches,
configured length thresholds apply; short neutral prompts safely default to the
economy profile.

All routing results include the rule matches, an explanation, prompt length,
profile, display name, and stable `provider_model_id`.

Demo prompts:

```text
Translate 'Good morning' into Vietnamese.
Compare REST and GraphQL and provide three advantages of each.
Design a secure and scalable RAG architecture with RBAC and monitoring.
```

## API contract and teammate integration

### Submit an AI request

`POST /api/v1/ai/requests`

Request:

```json
{
  "prompt": "Compare REST and GraphQL."
}
```

`prompt` is required, trimmed, must contain non-whitespace content, and may not
exceed `MAX_PROMPT_LENGTH`.

Successful response (`200`):

```json
{
  "status": "success",
  "prompt": "Compare REST and GraphQL.",
  "response": "Generated or mocked AI response",
  "routing": {
    "complexity": "medium",
    "model_profile": "standard",
    "display_name": "Standard Model",
    "provider_model_id": "cx/gpt-5.4-mini",
    "reason": "Moderate reasoning signals matched: comparison.",
    "matched_rules": ["comparison"],
    "prompt_length": 25
  },
  "usage": null,
  "cost": null,
  "error": null
}
```

Controlled provider error (`502`):

```json
{
  "status": "error",
  "prompt": "Hello",
  "response": null,
  "routing": {
    "complexity": "simple",
    "model_profile": "economy",
    "display_name": "Economy Model",
    "provider_model_id": "cx/gpt-5.4-mini",
    "reason": "No higher-complexity rule matched; using the cost-efficient default.",
    "matched_rules": ["safe_default"],
    "prompt_length": 5
  },
  "usage": null,
  "cost": null,
  "error": {
    "code": "provider_failure",
    "message": "The AI provider request failed."
  }
}
```

Invalid request bodies return FastAPI's standard `422` validation response with
field locations and clear validation messages.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/ai/requests \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Compare REST and GraphQL."}'
```

### Token/cost integration point

Connect teammate services in `process_request()` in
`src/agent/request_service.py`, immediately after `provider.generate()` returns.
Use:

- `prompt` for input-token counting
- provider response for output-token counting
- `routing.provider_model_id` as the stable pricing lookup key
- returned counts to populate `usage`
- the pricing service result to populate `cost`
- the complete result for request-history persistence

Keep these calls behind their existing public service interfaces. Do not import
pricing/database code into API routes or the frontend. The public `usage` and
`cost` fields are already nullable, so their schema can remain stable during
integration.

## Known limitations

- Routing is English keyword/rule based and intentionally lightweight.
- Mock responses demonstrate the flow but do not generate substantive answers.
- Real-provider retry/backoff and streaming are outside this two-day MVP.
- Usage and cost stay `null` until teammate services are connected.

"""AI request orchestration and downstream integration boundary."""

from dataclasses import asdict
from typing import Any

from .provider import LLMProvider, ProviderError, build_provider
from .routing import route_prompt


def process_request(prompt: str, provider: LLMProvider | None = None) -> dict[str, Any]:
    """Route and execute one prompt, returning a stable internal result.

    Token counting, pricing, cost calculation, and persistence should be
    attached here after provider output is available.
    """
    routing = route_prompt(prompt)
    try:
        response = (provider or build_provider()).generate(
            prompt, routing.provider_model_id
        )
        return {
            "status": "success",
            "prompt": prompt,
            "response": response,
            "routing": asdict(routing),
            "usage": None,
            "cost": None,
            "error": None,
        }
    except ProviderError as exc:
        return {
            "status": "error",
            "prompt": prompt,
            "response": None,
            "routing": asdict(routing),
            "usage": None,
            "cost": None,
            "error": {"code": exc.code, "message": exc.message},
        }
    except Exception:
        return {
            "status": "error",
            "prompt": prompt,
            "response": None,
            "routing": asdict(routing),
            "usage": None,
            "cost": None,
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred while processing the request.",
            },
        }

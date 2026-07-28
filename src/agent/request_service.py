"""AI request orchestration and downstream integration boundary."""

from dataclasses import asdict
from typing import Any

from rich import print

from src.config import settings
from src.services import embedder, vector_store

from .classifier import classify
from .provider import LLMProvider, ProviderError, build_provider
from .routing import RoutingDecision


def _build_routing_decision(prompt: str, level: str, reason: str, source: str) -> RoutingDecision:
    model = settings.routing_models[level]
    return RoutingDecision(
        complexity=level,
        model_profile=model["profile"],
        display_name=model["display_name"],
        provider_model_id=model["provider_model_id"],
        reason=reason,
        matched_rules=[source],
        prompt_length=len(prompt),
    )


def _save_classification(prompt: str, embedding: list[float], level: str, model_used: str) -> None:
    """Persist the classification for future recall; failures must not break the request."""
    try:
        vector_store.upsert_classification(
            prompt=prompt,
            embedding=embedding,
            level=level,
            model_used=model_used,
            token_count=len(prompt.split()),
        )
    except Exception as exc:
        print(f"[VectorStore] Failed to save classification: {exc}")


def process_request(prompt: str, provider: LLMProvider | None = None) -> dict[str, Any]:
    """Route and execute one prompt, returning a stable internal result.

    Token counting, pricing, cost calculation, and persistence should be
    attached here after provider output is available.
    """
    resolved_provider = provider or build_provider()
    embedding = embedder.embed(prompt)
    result = classify(prompt, embedding, resolved_provider)
    routing = _build_routing_decision(prompt, result.level, result.reason, result.source)

    _save_classification(prompt, embedding, result.level, routing.provider_model_id)

    try:
        response = resolved_provider.generate(prompt, routing.provider_model_id)
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

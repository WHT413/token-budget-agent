"""Semantic router node: delegates complexity classification to the shared classifier."""

from src.config import DEFAULT_MODEL, FALLBACK_MODEL
from src.services import embedder

from ..classifier import classify
from ..provider import LLMProvider
from ..state import AgentState

MODEL_MAP = {
    "simple": DEFAULT_MODEL,
    "medium": DEFAULT_MODEL,
    "complex": FALLBACK_MODEL,
}


def route(state: AgentState, provider: LLMProvider | None = None) -> AgentState:
    """Classify prompt complexity and select the model to use."""
    prompt = state["prompt"]
    token_count = len(prompt.split())
    embedding = embedder.embed(prompt)

    result = classify(prompt, embedding, provider)

    return {
        **state,
        "embedding": embedding,
        "level": result.level,
        "classification_source": result.source,
        "model": MODEL_MAP[result.level],
        "token_count": token_count,
    }
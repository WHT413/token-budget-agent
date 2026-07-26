"""Semantic router node backed by Qdrant with heuristic fallback."""

from src.config import (
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    MIN_SAMPLES_TO_TRUST,
    SIMILARITY_THRESHOLD,
)
from src.services import embedder, vector_store

from ..state import AgentState

MODEL_MAP = {
    "simple": DEFAULT_MODEL,
    "medium": DEFAULT_MODEL,
    "complex": FALLBACK_MODEL,
}


def _heuristic_level(token_count: int) -> str:
    if token_count < 50:
        return "simple"
    if token_count <= 150:
        return "medium"
    return "complex"


def route(state: AgentState) -> AgentState:
    """Classify prompt complexity and select the model to use."""
    token_count = len(state["prompt"].split())
    embedding = embedder.embed(state["prompt"])
    count = vector_store.get_collection_count()

    level = _heuristic_level(token_count)
    source = "heuristic"

    if count >= MIN_SAMPLES_TO_TRUST:
        results = vector_store.search_similar(embedding, top_k=5)
        if results and results[0]["score"] >= SIMILARITY_THRESHOLD:
            level = results[0]["level"]
            source = "qdrant"

    return {
        **state,
        "embedding": embedding,
        "level": level,
        "classification_source": source,
        "model": MODEL_MAP[level],
        "token_count": token_count,
    }
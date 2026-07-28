"""Shared prompt-complexity classification: Qdrant recall with an LLM zero-shot fallback.

Used by both the LangGraph router node and the live API request path, so both
surfaces make the same routing decision from the same signals.
"""

from dataclasses import dataclass

from src.config import DEFAULT_MODEL, MIN_SAMPLES_TO_TRUST, SIMILARITY_THRESHOLD
from src.services import vector_store

from .provider import LLMProvider, build_provider

VALID_LEVELS = ("simple", "medium", "complex")

_CLASSIFY_INSTRUCTIONS = (
    "Classify the complexity of the prompt below as exactly one word: "
    "simple, medium, or complex.\n"
    "simple: translation, rewriting, grammar fixes, short summaries, basic formatting.\n"
    "medium: comparisons, analysis, explanations, structured content, moderate coding.\n"
    "complex: architecture, security, scalability, multi-step planning, complex debugging.\n"
    "Reply with only that single word."
)


def _build_classification_prompt(prompt: str) -> str:
    return f"{_CLASSIFY_INSTRUCTIONS}\n\nPrompt:\n{prompt}"


def _parse_level(raw: str) -> str | None:
    """Extract a valid complexity level from a raw LLM response, if present."""
    normalized = raw.strip().lower()
    if normalized in VALID_LEVELS:
        return normalized
    for level in VALID_LEVELS:
        if level in normalized:
            return level
    return None


def _classify_with_llm(prompt: str, provider: LLMProvider) -> str:
    """Ask the economy model to classify complexity; default to simple if that fails."""
    try:
        raw = provider.generate(_build_classification_prompt(prompt), DEFAULT_MODEL)
    except Exception:
        return "simple"
    return _parse_level(raw) or "simple"


@dataclass(frozen=True)
class Classification:
    """A complexity decision plus the reasoning behind it."""

    level: str
    source: str
    reason: str


def classify(prompt: str, embedding: list[float], provider: LLMProvider | None = None) -> Classification:
    """Classify prompt complexity via Qdrant recall, falling back to an LLM zero-shot call."""
    count = vector_store.get_collection_count()

    if count >= MIN_SAMPLES_TO_TRUST:
        results = vector_store.search_similar(embedding, top_k=5)
        if results and results[0]["score"] >= SIMILARITY_THRESHOLD:
            return Classification(
                level=results[0]["level"],
                source="qdrant",
                reason=(
                    "Reused the classification of a similar prompt already seen "
                    f"(similarity {results[0]['score']:.2f})."
                ),
            )

    level = _classify_with_llm(prompt, provider or build_provider())
    return Classification(
        level=level,
        source="llm_classifier",
        reason=(
            "No confident match in the vector store yet; the "
            f"{DEFAULT_MODEL} model classified this prompt directly."
        ),
    )

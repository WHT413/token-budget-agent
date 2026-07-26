"""E2E tests for semantic router classification using real embedder and Qdrant."""

from __future__ import annotations

import random
import string

import pytest

from qdrant_client.models import PointStruct

from src.agent.nodes.router import route
from src.config import DEFAULT_MODEL, FALLBACK_MODEL, QDRANT_COLLECTION
from src.services import embedder


pytestmark = pytest.mark.e2e


VALID_LEVELS = {"simple", "medium", "complex"}
VALID_SOURCES = {"qdrant", "heuristic"}


def _route_prompt(prompt: str) -> dict:
    return route({"prompt": prompt})


def _assert_route(prompt: str, expected_level: str, expected_source: str) -> None:
    state = _route_prompt(prompt)
    assert state["level"] == expected_level
    assert state["classification_source"] == expected_source


@pytest.fixture(scope="module")
def seed_qdrant(initialized_qdrant):
    """Seed Qdrant with trusted labeled examples for semantic routing tests."""
    from qdrant_client import QdrantClient

    from src.config import QDRANT_URL
    from tests.e2e.conftest import recreate_collection

    client = QdrantClient(url=QDRANT_URL)
    recreate_collection(client)

    examples = [
        ("Translate this sentence into Spanish", "simple"),
        ("Fix the grammar in this short email", "simple"),
        ("Format this list as bullet points", "simple"),
        ("Rewrite this sentence in a polite tone", "simple"),
        ("Proofread this paragraph for spelling", "simple"),
        ("Compare REST and GraphQL for API development", "medium"),
        ("Explain the tradeoffs of serverless functions", "medium"),
        ("Analyze the pros and cons of microservices", "medium"),
        ("Summarize advantages and disadvantages of caching", "medium"),
        ("Evaluate PostgreSQL versus MongoDB for analytics", "medium"),
        ("Design a distributed multi-agent AI system architecture", "complex"),
        ("Create a multi-step implementation plan for microservices", "complex"),
        ("Plan an event-driven architecture with Kafka and Kubernetes", "complex"),
        ("Design a scalable RAG platform with monitoring and CI/CD", "complex"),
        ("Architect a secure system for asynchronous document processing", "complex"),
    ]

    points = [
        PointStruct(
            id=index,
            vector=embedder.embed(prompt),
            payload={
                "prompt": prompt,
                "level": level,
                "model_used": FALLBACK_MODEL if level == "complex" else DEFAULT_MODEL,
                "token_count": len(prompt.split()),
            },
        )
        for index, (prompt, level) in enumerate(examples)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    yield

    recreate_collection(client)


def test_cold_start_short_simple_prompt(empty_qdrant) -> None:
    _assert_route("Translate this to French", "simple", "heuristic")


def test_cold_start_medium_prompt(empty_qdrant) -> None:
    _assert_route(
        "Explain the advantages and disadvantages of microservices compared to monolithic architecture "
        * 6,
        "medium",
        "heuristic",
    )


def test_cold_start_long_complex_prompt(empty_qdrant) -> None:
    _assert_route(
        "Write a comprehensive step-by-step implementation plan for building a distributed event-driven "
        "microservices system with Kafka message queues Redis caching PostgreSQL for persistence and "
        "Kubernetes orchestration including CI/CD pipeline monitoring alerting and disaster recovery "
        "strategy "
        * 5,
        "complex",
        "heuristic",
    )


@pytest.mark.xfail(reason="Known bug: heuristic ignores semantics for short complex prompts.")
def test_cold_start_short_but_semantically_complex_prompt(empty_qdrant) -> None:
    _assert_route(
        "Help me make a plan to design 1 multi agent system where AI will filter the CV",
        "complex",
        "heuristic",
    )


def test_qdrant_routes_similar_simple_prompt(seed_qdrant) -> None:
    _assert_route("Proofread this sentence for grammar mistakes", "simple", "qdrant")


def test_qdrant_routes_similar_complex_prompt(seed_qdrant) -> None:
    _assert_route("Design the system architecture for a multi-agent AI pipeline", "complex", "qdrant")


def test_qdrant_low_similarity_falls_back_to_heuristic(seed_qdrant) -> None:
    random_words = [
        "".join(random.choices(string.ascii_lowercase, k=12)) for _ in range(210)
    ]
    state = _route_prompt(" ".join(random_words))
    assert state["classification_source"] == "heuristic"


def test_router_returns_valid_state_fields(empty_qdrant) -> None:
    state = _route_prompt("Translate this to French")

    assert isinstance(state["embedding"], list)
    assert len(state["embedding"]) == 384
    assert state["level"] in VALID_LEVELS
    assert state["classification_source"] in VALID_SOURCES
    assert isinstance(state["model"], str)
    assert state["model"]
    assert isinstance(state["token_count"], int)
    assert state["token_count"] > 0


def test_model_map_respected(empty_qdrant) -> None:
    state = _route_prompt(
        "Write a comprehensive implementation strategy for distributed architecture " * 30
    )
    assert state["level"] == "complex"
    assert state["model"] == FALLBACK_MODEL


def test_model_map_simple_uses_default(empty_qdrant) -> None:
    state = _route_prompt("Translate this to French")
    assert state["level"] == "simple"
    assert state["model"] == DEFAULT_MODEL
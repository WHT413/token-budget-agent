"""E2E tests for the full LangGraph token-budget pipeline."""

from __future__ import annotations

import pytest

from src.agent.graph import agent_graph
from src.config import MIN_SAMPLES_TO_TRUST
from src.services import vector_store


pytestmark = pytest.mark.e2e


def _run_pipeline(prompt: str) -> dict:
    return agent_graph.invoke({"prompt": prompt, "response": "mock e2e response", "budget_remaining": 0})


def test_full_pipeline_simple_prompt(empty_qdrant) -> None:
    before_count = vector_store.get_collection_count()

    state = _run_pipeline("Summarize this in one sentence")

    assert isinstance(state["response"], str)
    assert state["response"]
    assert state["level"] in ["simple", "medium", "complex"]
    assert isinstance(state["model"], str)
    assert state["model"]
    assert vector_store.get_collection_count() == before_count + 1


def test_full_pipeline_saves_to_qdrant(empty_qdrant) -> None:
    prompt = "Summarize this in one sentence"

    first_state = _run_pipeline(prompt)
    count_after_first = vector_store.get_collection_count()
    assert count_after_first >= 1

    second_state = _run_pipeline(prompt)
    if count_after_first >= MIN_SAMPLES_TO_TRUST:
        assert second_state["classification_source"] == "qdrant"
    else:
        assert second_state["classification_source"] == "heuristic"
    assert vector_store.get_collection_count() >= 1
    assert first_state["level"] in ["simple", "medium", "complex"]


def test_full_pipeline_budget_not_exceeded(monkeypatch, empty_qdrant) -> None:
    monkeypatch.setenv("TOKEN_BUDGET_LIMIT", "500")

    state = _run_pipeline("Short prompt")

    assert state["budget_remaining"] >= 0
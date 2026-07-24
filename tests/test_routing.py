"""Unit tests for deterministic smart model routing."""

import pytest

from src.agent.routing import route_prompt


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Translate 'Good morning' into Vietnamese.", "simple"),
        ("Rewrite this sentence to sound friendly.", "simple"),
        ("Summarize this short paragraph.", "simple"),
        ("Compare REST and GraphQL.", "medium"),
        ("Analyze the advantages of solar power.", "medium"),
        ("Create a structured outline with multiple requirements.", "medium"),
        ("Design a system architecture for a RAG platform.", "complex"),
        ("Provide a security analysis for a scalable service.", "complex"),
        ("Create a multi-step implementation plan for a distributed system.", "complex"),
        ("Compare options for a secure, scalable architecture.", "complex"),
        ("What color is the sky?", "simple"),
    ],
)
def test_routes_expected_complexity(prompt: str, expected: str) -> None:
    assert route_prompt(prompt).complexity == expected


def test_routing_metadata_is_complete_and_configured() -> None:
    result = route_prompt("Compare REST and GraphQL.")
    assert result.reason
    assert result.provider_model_id == "test-standard"
    assert result.prompt_length == 25

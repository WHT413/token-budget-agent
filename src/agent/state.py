"""Agent state definitions."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the token budget agent."""

    prompt: str
    token_count: int
    embedding: list[float]
    level: str
    classification_source: str
    model: str
    messages: list[dict]
    response: str
    budget_remaining: int
    compressed: bool

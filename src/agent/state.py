"""Agent state definitions."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the token budget agent."""

    messages: list[dict[str, Any]]
    model_size: str
    token_count: int
    compressed: bool

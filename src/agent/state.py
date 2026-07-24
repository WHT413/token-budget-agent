"""Shared state definition for the LangGraph agent."""

from typing import TypedDict


class AgentState(TypedDict):
    """Placeholder agent state. Fields will grow as nodes are implemented."""

    messages: list
    token_count: int

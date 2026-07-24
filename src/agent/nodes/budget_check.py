"""Node: checks current token usage against the configured budget."""

from src.agent.state import AgentState


def budget_check(state: AgentState) -> AgentState:
    """Placeholder node. Will inspect token usage in `state` and flag overages."""
    return state

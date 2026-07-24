"""Node: routes execution to the next node based on budget/state."""

from src.agent.state import AgentState


def router(state: AgentState) -> str:
    """Placeholder node. Will return the name of the next node to run."""
    return "llm_call"

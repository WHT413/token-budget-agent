"""Node: invokes the LLM with the current state."""

from src.agent.state import AgentState


def llm_call(state: AgentState) -> AgentState:
    """Placeholder node. Will call the configured model and update `state`."""
    return state

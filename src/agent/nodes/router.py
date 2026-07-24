"""Router node stub."""

from typing import Literal

from ..state import AgentState


def router(state: AgentState) -> Literal["llm_call", "compressor"]:
    """Placeholder router for deciding the next graph node."""
    return "llm_call"
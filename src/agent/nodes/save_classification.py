"""Persist prompt classification feedback after an LLM call."""

from rich import print

from src.services import vector_store

from ..state import AgentState


def save_classification(state: AgentState) -> AgentState:
    """Save the current classification and return state unchanged."""
    vector_store.upsert_classification(
        prompt=state["prompt"],
        embedding=state["embedding"],
        level=state["level"],
        model_used=state["model"],
        token_count=state["token_count"],
    )
    print(
        "[VectorStore] Saved: "
        f"{state['level']} | source: {state['classification_source']} | model: {state['model']}"
    )
    return state
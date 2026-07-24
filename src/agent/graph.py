"""LangGraph graph compilation stub."""

from langgraph.graph import END, StateGraph

from .nodes.budget_check import budget_check
from .nodes.compressor import compressor
from .nodes.llm_call import llm_call
from .state import AgentState


def compile_graph():
    """Compile a placeholder LangGraph graph."""
    graph = StateGraph(AgentState)
    graph.add_node("budget_check", budget_check)
    graph.add_node("compressor", compressor)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point("budget_check")
    graph.add_edge("budget_check", "llm_call")
    graph.add_edge("llm_call", END)
    return graph.compile()

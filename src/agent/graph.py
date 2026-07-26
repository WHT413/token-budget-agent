"""LangGraph graph compilation."""

from langgraph.graph import END, START, StateGraph

from .nodes.budget_check import budget_check
from .nodes.compressor import compressor
from .nodes.llm_call import llm_call
from .nodes.router import route
from .nodes.save_classification import save_classification
from .state import AgentState


def compile_graph():
    """Compile the token-budget agent graph."""
    graph = StateGraph(AgentState)
    graph.add_node("budget_check", budget_check)
    graph.add_node("compressor", compressor)
    graph.add_node("router", route)
    graph.add_node("llm_call", llm_call)
    graph.add_node("save_classification", save_classification)
    graph.add_edge(START, "budget_check")
    graph.add_edge("budget_check", "compressor")
    graph.add_edge("compressor", "router")
    graph.add_edge("router", "llm_call")
    graph.add_edge("llm_call", "save_classification")
    graph.add_edge("save_classification", END)
    return graph.compile()


agent_graph = compile_graph()

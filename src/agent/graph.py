"""Compiles the LangGraph graph wiring together the agent nodes."""

from langgraph.graph import StateGraph

from src.agent.state import AgentState
from src.agent.nodes.budget_check import budget_check
from src.agent.nodes.compressor import compressor
from src.agent.nodes.router import router
from src.agent.nodes.llm_call import llm_call


def build_graph():
    """Placeholder graph builder. Will wire nodes and compile the graph."""
    graph = StateGraph(AgentState)
    graph.add_node("budget_check", budget_check)
    graph.add_node("compressor", compressor)
    graph.add_node("llm_call", llm_call)
    graph.set_entry_point("budget_check")
    return graph.compile()

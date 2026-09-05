"""
LangGraph workflow for the B2B Commercial Intelligence Agent.
Flow: Scout -> Strategist (ToT) -> Critic -> [Approve | Revise | Unresolved] -> Memory -> END
"""

import time
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.scout import run_scout
from agent.strategist import run_strategist
from agent.critic import run_critic
from agent.memory import run_memory


def should_revise(state: AgentState) -> str:
    """
    Routing function after Critic evaluation.
    Returns next node based on scores and iteration count.
    """
    current = state.get("current_hypotheses_under_review", [])
    iterations = state.get("revision_iterations", 0)

    if not current:
        return "memory"

    best_score = max((h.get("initial_score", 0) for h in current), default=0)

    if best_score >= 0.6:
        return "memory"
    elif iterations < 2:
        return "strategist"
    else:
        return "memory"


def compute_metrics(state: AgentState) -> AgentState:
    """Computes final metrics before ending the cycle."""
    approved = state.get("approved_recommendations", [])
    rejected = state.get("rejected_recommendations", [])
    unresolved = state.get("unresolved_signals", [])
    signals = state.get("signals", [])
    hypotheses = state.get("hypotheses", [])

    total_approved = len(approved)
    total_rejected = len(rejected)
    total_signals = len(signals)
    total_hypotheses = len(hypotheses)
    total_unresolved = len(unresolved)

    groundedness_rate = (
        len([r for r in approved if r.get("has_rag_evidence")]) / total_approved
        if total_approved > 0 else 0.0
    )

    pruning_rate = (
        total_rejected / total_hypotheses
        if total_hypotheses > 0 else 0.0
    )

    escalation_rate = (
        len([r for r in approved if r.get("label") == "no documented precedent"]) / total_approved
        if total_approved > 0 else 0.0
    )

    cycle_latency = time.time() - state.get("cycle_start_time", time.time())

    state["metrics"] = {
        "total_companies_analyzed": len(state.get("companies_analyzed", [])),
        "total_signals_detected": total_signals,
        "total_hypotheses_generated": total_hypotheses,
        "total_recommendations_approved": total_approved,
        "total_recommendations_rejected": total_rejected,
        "total_unresolved": total_unresolved,
        "groundedness_rate": round(groundedness_rate, 2),
        "pruning_rate": round(pruning_rate, 2),
        "escalation_rate": round(escalation_rate, 2),
        "cycle_latency_seconds": round(cycle_latency, 1)
    }
    return state


def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("scout", run_scout)
    workflow.add_node("strategist", run_strategist)
    workflow.add_node("critic", run_critic)
    workflow.add_node("metrics", compute_metrics)
    workflow.add_node("memory", run_memory)

    # Define edges
    workflow.set_entry_point("scout")
    workflow.add_edge("scout", "strategist")
    workflow.add_edge("strategist", "critic")

    # Conditional routing after critic
    workflow.add_conditional_edges(
        "critic",
        should_revise,
        {
            "strategist": "strategist",
            "memory": "memory"
        }
    )

    workflow.add_edge("memory", "metrics")
    workflow.add_edge("metrics", END)

    return workflow.compile()


# Singleton graph instance
graph = build_graph()
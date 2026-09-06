"""
Memory Agent - Knowledge Consolidation and Long-term Learning
Runs at the end of each cycle, records outcomes to memory.json,
and updates signal pattern weights for future cycles.
"""

import json
import os
from datetime import datetime
from agent.state import AgentState


MEMORY_FILE = "memory.json"


def _load_memory() -> dict:
    """Loads existing memory or creates empty structure."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cycles": [],
        "signal_patterns": {},
        "sector_performance": {},
        "rag_evidence_hits": {},
        "cumulative_stats": {
            "total_cycles": 0,
            "total_signals": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_unresolved": 0,
            "avg_groundedness_rate": 0.0,
            "avg_pruning_rate": 0.0
        }
    }


def _save_memory(memory: dict) -> None:
    """Saves memory to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def _update_signal_patterns(
    memory: dict,
    approved: list,
    rejected: list
) -> dict:
    """
    Updates signal type performance patterns.
    Tracks which signal types generate approved recommendations.
    """
    patterns = memory.get("signal_patterns", {})

    for rec in approved:
        # Extract signal type from recommendation context
        pipeline_ctx = rec.get("pipeline_context", "")
        signal_type = "unknown"
        if "expansion" in pipeline_ctx.lower():
            signal_type = "expansion"
        elif "contract" in pipeline_ctx.lower():
            signal_type = "contract_renewal"
        elif "merger" in pipeline_ctx.lower():
            signal_type = "merger_acquisition"

        if signal_type not in patterns:
            patterns[signal_type] = {
                "approved": 0,
                "rejected": 0,
                "success_rate": 0.0
            }
        patterns[signal_type]["approved"] += 1

    for rej in rejected:
        hyp = rej.get("hypothesis", {})
        signal_type = "unknown"
        if signal_type not in patterns:
            patterns[signal_type] = {
                "approved": 0,
                "rejected": 0,
                "success_rate": 0.0
            }
        patterns[signal_type]["rejected"] += 1

    # Update success rates
    for signal_type, data in patterns.items():
        total = data["approved"] + data["rejected"]
        if total > 0:
            data["success_rate"] = round(data["approved"] / total, 2)

    memory["signal_patterns"] = patterns
    return memory


def _update_sector_performance(
    memory: dict,
    approved: list
) -> dict:
    """Tracks which sectors generate the most approved recommendations."""
    sector_perf = memory.get("sector_performance", {})

    for rec in approved:
        # Infer sector from recommendation
        product = rec.get("product", "").lower()
        sector = "general"
        if "sd-wan" in product or "fibra" in product:
            sector = "connectivity"
        elif "ciberseguridad" in product or "seguridad" in product:
            sector = "security"
        elif "iot" in product or "m2m" in product:
            sector = "iot"
        elif "cloud" in product:
            sector = "cloud"

        if sector not in sector_perf:
            sector_perf[sector] = {
                "approved_count": 0,
                "avg_score": 0.0,
                "scores": []
            }

        sector_perf[sector]["approved_count"] += 1
        sector_perf[sector]["scores"].append(rec.get("score", 0))
        scores = sector_perf[sector]["scores"]
        sector_perf[sector]["avg_score"] = round(
            sum(scores) / len(scores), 3
        )

    memory["sector_performance"] = sector_perf
    return memory


def _update_rag_evidence_hits(
    memory: dict,
    approved: list
) -> dict:
    """Tracks which RAG documents are most frequently cited."""
    rag_hits = memory.get("rag_evidence_hits", {})

    for rec in approved:
        for evidence in rec.get("rag_evidence", []):
            key = evidence[:80] if len(evidence) > 80 else evidence
            if key not in rag_hits:
                rag_hits[key] = 0
            rag_hits[key] += 1

    memory["rag_evidence_hits"] = rag_hits
    return memory


def _update_cumulative_stats(
    memory: dict,
    cycle_record: dict
) -> dict:
    """Updates running averages and totals."""
    stats = memory.get("cumulative_stats", {})
    n = stats.get("total_cycles", 0)

    stats["total_cycles"] = n + 1
    stats["total_signals"] += cycle_record.get("signals_detected", 0)
    stats["total_approved"] += cycle_record.get("recommendations_approved", 0)
    stats["total_rejected"] += cycle_record.get("recommendations_rejected", 0)
    stats["total_unresolved"] += cycle_record.get("unresolved_signals", 0)

    # Running average
    gr = cycle_record.get("groundedness_rate", 0)
    pr = cycle_record.get("pruning_rate", 0)
    stats["avg_groundedness_rate"] = round(
        (stats.get("avg_groundedness_rate", 0) * n + gr) / (n + 1), 3
    )
    stats["avg_pruning_rate"] = round(
        (stats.get("avg_pruning_rate", 0) * n + pr) / (n + 1), 3
    )

    memory["cumulative_stats"] = stats
    return memory


def run_memory(state: AgentState) -> AgentState:
    """
    Memory Agent main function.
    Consolidates cycle results into long-term memory.
    Records patterns, sector performance, and RAG evidence hits.
    """
    print("\n💾 MEMORY AGENT: Consolidating cycle knowledge...")

    approved = state.get("approved_recommendations", [])
    rejected = state.get("rejected_recommendations", [])
    unresolved = state.get("unresolved_signals", [])
    metrics = state.get("metrics", {})
    errors = state.get("errors", [])

    # Build cycle record
    cycle_record = {
        "cycle_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "cycle_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_sector": state.get("target_sector", ""),
        "target_region": state.get("target_region", ""),
        "companies_analyzed": len(state.get("companies_analyzed", [])),
        "signals_detected": len(state.get("signals", [])),
        "hypotheses_generated": len(state.get("hypotheses", [])),
        "recommendations_approved": len(approved),
        "recommendations_rejected": len(rejected),
        "unresolved_signals": len(unresolved),
        "groundedness_rate": metrics.get("groundedness_rate", 0),
        "pruning_rate": metrics.get("pruning_rate", 0),
        "escalation_rate": metrics.get("escalation_rate", 0),
        "cycle_latency_seconds": metrics.get("cycle_latency_seconds", 0),
        "errors": errors,
        "approved_summary": [
            {
                "company": r.get("company_name", ""),
                "product": r.get("product", ""),
                "score": r.get("score", 0),
                "label": r.get("label", ""),
                "advisor": r.get("advisor", "")
            }
            for r in approved
        ]
    }

    # Load and update memory
    memory = _load_memory()
    memory["cycles"].append(cycle_record)
    memory = _update_signal_patterns(memory, approved, rejected)
    memory = _update_sector_performance(memory, approved)
    memory = _update_rag_evidence_hits(memory, approved)
    memory = _update_cumulative_stats(memory, cycle_record)

    # Save to disk
    _save_memory(memory)

    print(f"   ✅ Memory updated: {len(memory['cycles'])} total cycles recorded")
    print(f"   📊 Cumulative stats: "
          f"{memory['cumulative_stats']['total_approved']} total approved | "
          f"{memory['cumulative_stats']['total_cycles']} cycles")

    state["memory_updated"] = True
    return state
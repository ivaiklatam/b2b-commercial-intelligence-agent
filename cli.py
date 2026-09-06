"""
CLI entry point for the B2B Commercial Intelligence Agent.
Usage: python cli.py --sector manufacturing --region Bogota --max-companies 3
"""

import argparse
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from tools.rag import initialize_rag
from agent.graph import graph
from agent.state import AgentState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B2B Commercial Intelligence Agent - CMU Capstone"
    )
    parser.add_argument(
        "--sector",
        type=str,
        default="",
        help="Target sector (e.g. manufacturing, retail, logistics)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="",
        help="Target region (e.g. Bogota, Medellin, Cali)"
    )
    parser.add_argument(
        "--max-companies",
        type=int,
        default=3,
        help="Maximum number of companies to analyze (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_output.md",
        help="Output file path (default: sample_output.md)"
    )
    return parser.parse_args()


def render_recommendations(state: AgentState) -> str:
    """Renders approved recommendations as markdown."""
    approved = state.get("approved_recommendations", [])
    unresolved = state.get("unresolved_signals", [])
    metrics = state.get("metrics", {})
    errors = state.get("errors", [])

    lines = []
    lines.append("# B2B Commercial Intelligence Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sector: {state.get('target_sector', 'All') or 'All'}")
    lines.append(f"Region: {state.get('target_region', 'All') or 'All'}")
    lines.append(f"Companies analyzed: "
                 f"{len(state.get('companies_analyzed', []))}")
    lines.append("")

    # Metrics summary
    lines.append("## Cycle Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(
        f"| Signals detected | "
        f"{metrics.get('total_signals_detected', 0)} |"
    )
    lines.append(
        f"| Hypotheses generated | "
        f"{metrics.get('total_hypotheses_generated', 0)} |"
    )
    lines.append(
        f"| Recommendations approved | "
        f"{metrics.get('total_recommendations_approved', 0)} |"
    )
    lines.append(
        f"| Recommendations rejected | "
        f"{metrics.get('total_recommendations_rejected', 0)} |"
    )
    lines.append(
        f"| Groundedness rate | "
        f"{metrics.get('groundedness_rate', 0):.0%} |"
    )
    lines.append(
        f"| Pruning rate | "
        f"{metrics.get('pruning_rate', 0):.0%} |"
    )
    lines.append(
        f"| Escalation rate | "
        f"{metrics.get('escalation_rate', 0):.0%} |"
    )
    lines.append(
        f"| Cycle latency | "
        f"{metrics.get('cycle_latency_seconds', 0):.1f}s |"
    )
    lines.append("")

    # Approved recommendations
    lines.append("## Approved Recommendations")
    lines.append("")

    if not approved:
        lines.append("*No recommendations approved in this cycle.*")
    else:
        for i, rec in enumerate(approved, 1):
            score = rec.get("score", 0)
            label = rec.get("label", "approved")
            urgency = rec.get("urgency", "medium")
            has_evidence = rec.get("has_rag_evidence", False)

            urgency_emoji = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(urgency, "⚪")

            lines.append(
                f"### {i}. {rec.get('company_name', 'Unknown')} "
                f"{urgency_emoji}"
            )
            lines.append("")
            lines.append(
                f"**Product:** {rec.get('product', '')}"
            )
            lines.append(
                f"**Advisor:** {rec.get('advisor', 'Sin asignar')} | "
                f"**Region:** {rec.get('region', '')}"
            )
            lines.append(
                f"**Score:** {score:.2f} | "
                f"**Label:** `{label}` | "
                f"**Evidence:** {'✅' if has_evidence else '❌'}"
            )
            lines.append("")
            lines.append(f"**Argument:**")
            lines.append(f"{rec.get('argument', '')}")
            lines.append("")
            lines.append(f"**Action:**")
            lines.append(f"> {rec.get('action', '')}")
            lines.append("")

            # Score breakdown
            breakdown = rec.get("score_breakdown", {})
            if breakdown:
                lines.append("**Score Breakdown:**")
                lines.append(
                    f"- Commercial relevance (40%): "
                    f"{breakdown.get('commercial_relevance', 0):.2f} — "
                    f"{breakdown.get('commercial_relevance_reason', '')}"
                )
                lines.append(
                    f"- Argument strength (35%): "
                    f"{breakdown.get('argument_strength', 0):.2f} — "
                    f"{breakdown.get('argument_strength_reason', '')}"
                )
                lines.append(
                    f"- Actionability (25%): "
                    f"{breakdown.get('actionability', 0):.2f} — "
                    f"{breakdown.get('actionability_reason', '')}"
                )
            lines.append("")

            # RAG evidence
            rag_evidence = rec.get("rag_evidence", [])
            if rag_evidence:
                lines.append("**RAG Evidence:**")
                for ev in rag_evidence[:3]:
                    lines.append(f"- {ev[:150]}")
            lines.append("")
            lines.append("---")
            lines.append("")

    # Unresolved signals
    if unresolved:
        lines.append("## Unresolved Signals")
        lines.append("")
        lines.append(
            "*Signals that could not generate approved recommendations "
            "after maximum revision iterations:*"
        )
        lines.append("")
        for sig in unresolved:
            lines.append(
                f"- **{sig.get('company_name', '')}** | "
                f"{sig.get('signal_type', '')} | "
                f"{sig.get('urgency', '')} urgency"
            )
        lines.append("")

    # Errors
    if errors:
        lines.append("## System Errors")
        lines.append("")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    # Memory reference
    if os.path.exists("memory.json"):
        lines.append("## Long-term Memory")
        lines.append("")
        with open("memory.json", "r", encoding="utf-8") as f:
            memory = json.load(f)
        stats = memory.get("cumulative_stats", {})
        lines.append(
            f"- Total cycles recorded: {stats.get('total_cycles', 0)}"
        )
        lines.append(
            f"- Cumulative approved recommendations: "
            f"{stats.get('total_approved', 0)}"
        )
        lines.append(
            f"- Average groundedness rate: "
            f"{stats.get('avg_groundedness_rate', 0):.0%}"
        )
        lines.append(
            f"- Average pruning rate: "
            f"{stats.get('avg_pruning_rate', 0):.0%}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    print("\n" + "="*60)
    print("  B2B COMMERCIAL INTELLIGENCE AGENT")
    print("  CMU Capstone — Jorge Javier Lozano Diaz")
    print("="*60)
    print(f"\n  Sector:        {args.sector or 'All'}")
    print(f"  Region:        {args.region or 'All'}")
    print(f"  Max companies: {args.max_companies}")
    print(f"  Output:        {args.output}")
    print("\n" + "="*60 + "\n")

    # Initialize RAG
    print("📚 Initializing RAG index...")
    initialize_rag()

    # Build initial state
    initial_state: AgentState = {
        "target_sector": args.sector,
        "target_region": args.region,
        "max_companies": args.max_companies,
        "signals": [],
        "companies_analyzed": [],
        "hypotheses": [],
        "current_signal_index": 0,
        "approved_recommendations": [],
        "rejected_recommendations": [],
        "unresolved_signals": [],
        "revision_iterations": 0,
        "current_hypotheses_under_review": [],
        "memory_updated": False,
        "metrics": {
            "total_companies_analyzed": 0,
            "total_signals_detected": 0,
            "total_hypotheses_generated": 0,
            "total_recommendations_approved": 0,
            "total_recommendations_rejected": 0,
            "total_unresolved": 0,
            "groundedness_rate": 0.0,
            "pruning_rate": 0.0,
            "escalation_rate": 0.0,
            "cycle_latency_seconds": 0.0
        },
        "cycle_start_time": time.time(),
        "errors": []
    }

    # Run the graph
    print("🚀 Starting agent cycle...\n")
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"\n❌ Critical error in agent graph: {e}")
        raise

    # Render and save report
    print("\n📝 Generating report...")
    markdown = render_recommendations(final_state)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    # Print summary
    metrics = final_state.get("metrics", {})
    approved = final_state.get("approved_recommendations", [])

    print("\n" + "="*60)
    print("  CYCLE COMPLETE")
    print("="*60)
    print(
        f"  Signals detected:      "
        f"{metrics.get('total_signals_detected', 0)}"
    )
    print(
        f"  Recommendations:       "
        f"{metrics.get('total_recommendations_approved', 0)} approved"
    )
    print(
        f"  Groundedness rate:     "
        f"{metrics.get('groundedness_rate', 0):.0%}"
    )
    print(
        f"  Pruning rate:          "
        f"{metrics.get('pruning_rate', 0):.0%}"
    )
    print(
        f"  Cycle latency:         "
        f"{metrics.get('cycle_latency_seconds', 0):.1f}s"
    )
    print(f"  Report saved to:       {args.output}")
    print(f"  Memory updated:        "
          f"{'✅' if final_state.get('memory_updated') else '❌'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
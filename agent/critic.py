"""
Critic Agent - Quality Evaluation with 3-criterion rubric
Evaluates recommendation hypotheses using:
- Commercial relevance (40%)
- Argument strength with RAG evidence (35%)
- Actionability (25%)
Threshold: 0.6 — below triggers revision or unresolved
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, RecommendationModel
from tools.pipeline import analyze_pipeline, get_advisor_for_region


def _get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        temperature=0.1
    )


def _load_config() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _score_hypothesis(
    hypothesis: dict,
    signal: dict,
    llm,
    config: dict
) -> dict:
    """
    Scores a hypothesis using the 3-criterion rubric.
    Returns score breakdown and final weighted score.
    """
    weights = config["scoring"]
    cr_weight = weights["commercial_relevance_weight"]
    as_weight = weights["argument_strength_weight"]
    ac_weight = weights["actionability_weight"]
    no_evidence_penalty = weights["no_evidence_penalty"]

    rag_evidence = hypothesis.get("rag_evidence", [])
    has_rag_evidence = len(rag_evidence) > 0

    messages = [
        SystemMessage(content=(
            "You are a strict B2B commercial quality evaluator for a Telco company in Colombia. "
            "Score this recommendation hypothesis on 3 criteria. "
            "Be rigorous — only approve recommendations that are specific, "
            "evidence-based, and immediately actionable. "
            "Respond ONLY with valid JSON, no markdown, no extra text.\n"
            "Format: {\n"
            "  \"commercial_relevance\": 0.0-1.0,\n"
            "  \"commercial_relevance_reason\": \"brief reason\",\n"
            "  \"argument_strength\": 0.0-1.0,\n"
            "  \"argument_strength_reason\": \"brief reason\",\n"
            "  \"actionability\": 0.0-1.0,\n"
            "  \"actionability_reason\": \"brief reason\"\n"
            "}"
        )),
        HumanMessage(content=(
            f"Score this recommendation hypothesis:\n\n"
            f"Company: {hypothesis['company_name']}\n"
            f"Sector: {signal.get('sector', '')}\n"
            f"Signal type: {signal.get('signal_type', '')}\n"
            f"Signal urgency: {signal.get('urgency', '')}\n"
            f"Is existing client: {signal.get('is_client', False)}\n\n"
            f"Recommended product: {hypothesis['product']}\n"
            f"Commercial argument: {hypothesis['argument']}\n"
            f"Proposed action: {hypothesis['action']}\n"
            f"Urgency: {hypothesis['urgency']}\n\n"
            f"RAG evidence available: {has_rag_evidence}\n"
            f"Evidence points: {json.dumps(rag_evidence[:3])}\n\n"
            f"Pipeline context: {hypothesis.get('pipeline_context', 'No pipeline data')[:300]}\n\n"
            "Scoring criteria:\n"
            "- Commercial relevance (0-1): How directly does the signal connect to the product?\n"
            "- Argument strength (0-1): Is there solid evidence supporting this recommendation?\n"
            "- Actionability (0-1): Is the recommendation specific with clear next steps?"
        ))
    ]

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        raw = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    else:
        raw = content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    scores = json.loads(raw)

    # Apply no-evidence penalty
    arg_score = scores.get("argument_strength", 0.0)
    if not has_rag_evidence:
        arg_score = arg_score * no_evidence_penalty

    # Compute weighted score
    cr_score = scores.get("commercial_relevance", 0.0)
    ac_score = scores.get("actionability", 0.0)

    final_score = (
        cr_score * cr_weight +
        arg_score * as_weight +
        ac_score * ac_weight
    )

    return {
        "commercial_relevance": round(cr_score, 3),
        "commercial_relevance_reason": scores.get(
            "commercial_relevance_reason", ""
        ),
        "argument_strength": round(arg_score, 3),
        "argument_strength_reason": scores.get(
            "argument_strength_reason", ""
        ),
        "actionability": round(ac_score, 3),
        "actionability_reason": scores.get(
            "actionability_reason", ""
        ),
        "has_rag_evidence": has_rag_evidence,
        "final_score": round(final_score, 3)
    }


def _build_recommendation(
    hypothesis: dict,
    signal: dict,
    score_breakdown: dict,
    iteration: int
) -> RecommendationModel:
    """Builds a RecommendationModel from approved hypothesis."""
    has_evidence = score_breakdown.get("has_rag_evidence", False)
    label = "approved" if has_evidence else "no documented precedent"

    try:
        advisor_result = get_advisor_for_region.invoke(
            {"region": signal.get("region", "")}
        )
        advisor_data = json.loads(advisor_result)
        advisor = advisor_data.get("advisor", "Sin asesor asignado")
    except Exception:
        advisor = "Sin asesor asignado"

    return RecommendationModel(
        recommendation_id=(
            f"REC-{hypothesis['company_id']}-"
            f"{hypothesis['hypothesis_id']}-{iteration}"
        ),
        company_id=hypothesis["company_id"],
        company_name=hypothesis["company_name"],
        advisor=advisor,
        region=signal.get("region", ""),
        product=hypothesis["product"],
        argument=hypothesis["argument"],
        urgency=hypothesis["urgency"],
        action=hypothesis["action"],
        rag_evidence=hypothesis.get("rag_evidence", []),
        pipeline_context=hypothesis.get("pipeline_context", ""),
        score=score_breakdown["final_score"],
        score_breakdown=score_breakdown,
        has_rag_evidence=has_evidence,
        label=label,
        iteration=iteration
    )


def run_critic(state: AgentState) -> AgentState:
    """
    Critic Agent main function.
    Evaluates hypotheses from Strategist using 3-criterion rubric.
    Approves, rejects, or marks as unresolved based on score threshold.
    """
    print("\n⚖️  CRITIC AGENT: Evaluating recommendation hypotheses...")

    config = _load_config()
    threshold = config["scoring"]["approval_threshold"]
    max_iterations = config["scoring"]["max_revision_iterations"]

    hypotheses = state.get("current_hypotheses_under_review", [])
    signals = state.get("signals", [])
    iteration = state.get("revision_iterations", 1)

    approved = state.get("approved_recommendations", [])
    rejected = state.get("rejected_recommendations", [])
    unresolved = state.get("unresolved_signals", [])

    if not hypotheses:
        print("   No hypotheses to evaluate.")
        state["current_hypotheses_under_review"] = []
        return state

    llm = _get_llm()
    still_pending = []

    for hypothesis in hypotheses:
        company_id = hypothesis.get("company_id", "")
        print(f"   → Evaluating: {hypothesis['company_name']} - "
              f"{hypothesis['product']}")

        # Find matching signal
        signal = next(
            (s for s in signals if s["company_id"] == company_id),
            {}
        )

        try:
            score_breakdown = _score_hypothesis(
                hypothesis, signal, llm, config
            )
            final_score = score_breakdown["final_score"]
            has_evidence = score_breakdown["has_rag_evidence"]

            print(f"     Score: {final_score:.2f} | "
                  f"Evidence: {'✅' if has_evidence else '❌'} | "
                  f"Iteration: {iteration}")

            if final_score >= threshold:
                recommendation = _build_recommendation(
                    hypothesis, signal, score_breakdown, iteration
                )
                approved.append(recommendation)
                label = recommendation["label"]
                print(f"     ✅ APPROVED [{label}] "
                      f"Score: {final_score:.2f}")

            else:
                rejected.append({
                    "hypothesis": hypothesis,
                    "score": final_score,
                    "score_breakdown": score_breakdown,
                    "reason": (
                        f"Score {final_score:.2f} below threshold {threshold}. "
                        f"Weakest criterion: "
                        + min(
                            ["commercial_relevance",
                             "argument_strength",
                             "actionability"],
                            key=lambda k: score_breakdown.get(k, 1.0)
                        )
                    )
                })

                if iteration < max_iterations:
                    still_pending.append(hypothesis)
                    print(f"     🔄 REJECTED - Sending back for revision "
                          f"(iteration {iteration}/{max_iterations})")
                else:
                    # Mark signal as unresolved after max iterations
                    if signal and signal not in unresolved:
                        unresolved.append(signal)
                    print(f"     ❌ UNRESOLVED after {max_iterations} iterations")

        except Exception as e:
            print(f"     ❌ Error evaluating hypothesis: {e}")
            state.setdefault("errors", []).append(
                f"Critic error for {hypothesis['company_name']}: {str(e)}"
            )

    print(f"\n   Critic complete: {len(approved)} approved | "
          f"{len(rejected)} rejected | "
          f"{len(unresolved)} unresolved")

    state["approved_recommendations"] = approved
    state["rejected_recommendations"] = rejected
    state["unresolved_signals"] = unresolved
    state["current_hypotheses_under_review"] = still_pending
    return state
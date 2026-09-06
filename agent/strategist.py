"""
Strategist Agent - Tree of Thought Recommendation Generation
Generates 3 hypothesis branches per signal, selects top 2 (beam width=2),
refines with RAG evidence and pipeline context.
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, HypothesisModel
from tools.rag import retrieve_portfolio_context
from tools.pipeline import analyze_pipeline, get_advisor_for_region
from tools.crosssell import generate_crosssell_insights


def _get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        temperature=0.3
    )


def _generate_three_hypotheses(
    signal: dict,
    llm,
    pipeline_context: str,
    crosssell_context: str
) -> list[dict]:
    """
    ToT Level 1: Generates 3 distinct recommendation hypotheses
    using different reasoning paths.
    """
    messages = [
        SystemMessage(content=(
            "You are a senior B2B commercial strategist for a Telco company in Colombia. "
            "Generate exactly 3 distinct recommendation hypotheses for this commercial signal. "
            "Each hypothesis must use a DIFFERENT reasoning path:\n"
            "- Hypothesis A: Focus on the most direct product match for the signal\n"
            "- Hypothesis B: Focus on contract renewal or expansion of existing relationship\n"
            "- Hypothesis C: Focus on differentiation through partner solution or case study\n\n"
            "Respond ONLY with valid JSON array, no markdown, no extra text.\n"
            "Format: [{\"hypothesis_id\": \"A\", \"product\": \"product name\", "
            "\"argument\": \"commercial argument\", \"urgency\": \"high/medium/low\", "
            "\"action\": \"specific action for advisor\", \"initial_score\": 0.0-1.0}, ...]"
        )),
        HumanMessage(content=(
            f"Generate 3 hypotheses for this signal:\n\n"
            f"Company: {signal['company_name']}\n"
            f"Sector: {signal['sector']}\n"
            f"Region: {signal['region']}\n"
            f"Signal type: {signal['signal_type']}\n"
            f"Signal urgency: {signal['urgency']}\n"
            f"Is existing client: {signal['is_client']}\n"
            f"Commercial opportunity: {signal['description']}\n\n"
            f"Pipeline context:\n{pipeline_context[:600]}\n\n"
            f"Cross-sell opportunities:\n{crosssell_context[:400]}"
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

    # Clean markdown if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    hypotheses = json.loads(raw)
    return hypotheses[:3]


def _select_top_two(hypotheses: list[dict]) -> list[dict]:
    """
    Beam Search: Selects top 2 hypotheses by initial_score.
    """
    sorted_hypotheses = sorted(
        hypotheses,
        key=lambda h: h.get("initial_score", 0),
        reverse=True
    )
    return sorted_hypotheses[:2]


def _refine_with_evidence(
    hypothesis: dict,
    signal: dict,
    llm,
    rag_results: str,
    pipeline_context: str
) -> HypothesisModel:
    """
    ToT Level 2: Refines hypothesis with RAG evidence and pipeline data.
    Builds the complete recommendation argument.
    """
    rag_data = json.loads(rag_results)
    has_evidence = rag_data.get("has_evidence", False)
    evidence_items = rag_data.get("results", [])

    evidence_texts = []
    for item in evidence_items[:3]:
        evidence_texts.append(
            f"- [{item['doc_type']}] {item['title']}: {item['content'][:200]}"
        )

    messages = [
        SystemMessage(content=(
            "You are a senior B2B commercial strategist for a Telco company in Colombia. "
            "Refine and strengthen this commercial recommendation using the provided evidence. "
            "Build a compelling, specific argument that the sales advisor can use directly. "
            "Respond ONLY with valid JSON, no markdown, no extra text.\n"
            "Format: {\"product\": \"exact product name\", "
            "\"argument\": \"detailed commercial argument with evidence\", "
            "\"urgency\": \"high/medium/low\", "
            "\"action\": \"specific action with contact name if available\", "
            "\"key_evidence\": [\"evidence point 1\", \"evidence point 2\"]}"
        )),
        HumanMessage(content=(
            f"Refine this hypothesis with evidence:\n\n"
            f"Original hypothesis: {json.dumps(hypothesis)}\n\n"
            f"Company: {signal['company_name']} | "
            f"Sector: {signal['sector']} | "
            f"Region: {signal['region']}\n\n"
            f"RAG Evidence available: {has_evidence}\n"
            f"Evidence:\n" + "\n".join(evidence_texts) + "\n\n"
            f"Pipeline context:\n{pipeline_context[:400]}"
        ))
    ]

    llm_response = llm.invoke(messages)
    content = llm_response.content
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

    refined = json.loads(raw)

    return HypothesisModel(
        hypothesis_id=hypothesis.get("hypothesis_id", "A"),
        company_id=signal["company_id"],
        company_name=signal["company_name"],
        product=refined.get("product", hypothesis.get("product", "")),
        argument=refined.get("argument", hypothesis.get("argument", "")),
        urgency=refined.get("urgency", signal["urgency"]),
        action=refined.get("action", hypothesis.get("action", "")),
        rag_evidence=refined.get("key_evidence", evidence_texts),
        pipeline_context=pipeline_context[:300],
        initial_score=hypothesis.get("initial_score", 0.5)
    )


def run_strategist(state: AgentState) -> AgentState:
    """
    Strategist Agent main function.
    Applies Tree of Thought reasoning to generate refined
    recommendation hypotheses for each detected signal.
    """
    print("\n🧠 STRATEGIST AGENT: Generating recommendations with ToT...")

    signals = state.get("signals", [])
    revision_iterations = state.get("revision_iterations", 0)
    current_under_review = state.get("current_hypotheses_under_review", [])

    if not signals:
        print("   No signals to process.")
        state["hypotheses"] = []
        state["current_hypotheses_under_review"] = []
        return state

    llm = _get_llm()
    all_hypotheses = state.get("hypotheses", [])
    new_hypotheses = []

    # If revising, work on rejected hypotheses feedback
    if revision_iterations > 0 and current_under_review:
        print(f"   Revision iteration {revision_iterations}: "
              f"Refining {len(current_under_review)} hypotheses...")
        signals_to_process = [
            s for s in signals
            if s["company_id"] in [h["company_id"] for h in current_under_review]
        ]
    else:
        signals_to_process = signals

    for signal in signals_to_process:
        print(f"   → Building recommendations for: {signal['company_name']}")

        try:
            # Get pipeline context
            pipeline_result = analyze_pipeline.invoke(
                {"company_id": signal["company_id"]}
            )

            # Get advisor for region
            advisor_result = get_advisor_for_region.invoke(
                {"region": signal["region"]}
            )
            advisor_data = json.loads(advisor_result)
            advisor_name = advisor_data.get("advisor", "Sin asesor asignado")

            # Get cross-sell insights
            crosssell_result = generate_crosssell_insights.invoke({
                "company_id": signal["company_id"],
                "sector": signal["sector"]
            })

            # ToT Level 1: Generate 3 hypotheses
            print("     🌳 ToT Level 1: Generating 3 hypotheses...")
            raw_hypotheses = _generate_three_hypotheses(
                signal, llm, pipeline_result, crosssell_result
            )
            print(f"     Generated: {len(raw_hypotheses)} hypotheses")

            # Beam Search: Select top 2
            top_two = _select_top_two(raw_hypotheses)
            print(f"     🎯 Beam Search: Selected top "
                  f"{len(top_two)} hypotheses for refinement")

            # ToT Level 2: Refine with RAG evidence
            refined_hypotheses = []
            for hyp in top_two:
                rag_query = (
                    f"{signal['signal_type']} {signal['sector']} "
                    f"{hyp.get('product', '')} Colombia"
                )
                rag_results = retrieve_portfolio_context.invoke({
                    "query": rag_query,
                    "sector": signal["sector"],
                    "top_k": 5
                })

                print(f"     📚 ToT Level 2: Refining hypothesis "
                      f"{hyp.get('hypothesis_id', '?')} with RAG...")
                refined = _refine_with_evidence(
                    hyp, signal, llm, rag_results, pipeline_result
                )
                refined_hypotheses.append(refined)

            new_hypotheses.extend(refined_hypotheses)
            all_hypotheses.extend(refined_hypotheses)

        except Exception as e:
            print(f"     ❌ Error in strategist for {signal['company_name']}: {e}")
            state.setdefault("errors", []).append(
                f"Strategist error for {signal['company_name']}: {str(e)}"
            )

    print(f"\n   Strategist complete: {len(new_hypotheses)} hypotheses generated")

    state["hypotheses"] = all_hypotheses
    state["current_hypotheses_under_review"] = new_hypotheses
    state["revision_iterations"] = revision_iterations + 1
    return state
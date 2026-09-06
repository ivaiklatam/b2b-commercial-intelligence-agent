"""
Scout Agent - Signal Collection and Classification
Monitors Colombian B2B companies for commercial signals using
Google tools via MCP Server.
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import AgentState, SignalModel
from tools.mcp_client import knowledge_graph_search, custom_search, news_rss


def _load_companies() -> list:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "companies.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["companies"]


def _load_config() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_companies(
    companies: list,
    sector: str,
    region: str,
    max_companies: int,
    config: dict
) -> list:
    """Filters companies by sector, region and strategic config."""
    excluded = config["strategic_filters"]["excluded_sectors"]
    filtered = []
    for c in companies:
        if c["sector"] in excluded:
            continue
        if sector and c["sector"] != sector:
            continue
        if region and c["region"] != region:
            continue
        filtered.append(c)
        if len(filtered) >= max_companies:
            break
    return filtered


def _classify_signal_urgency(signal_type: str, config: dict) -> str:
    """Classifies signal urgency based on config priorities."""
    high = config["signal_types"]["high_priority"]
    medium = config["signal_types"]["medium_priority"]
    if signal_type in high:
        return "high"
    elif signal_type in medium:
        return "medium"
    return "low"


def _build_search_query(company: dict, signal_type: str) -> str:
    """Builds a targeted search query for a company signal."""
    signal_map = {
        "expansion_new_facility": "nueva planta sede expansion",
        "merger_acquisition": "fusion adquisicion compra",
        "leadership_change": "nuevo director gerente CEO CTO",
        "it_investment": "inversion tecnologia digitalizacion sistemas",
        "digital_transformation": "transformacion digital innovacion",
        "regulatory_compliance": "cumplimiento normativa regulacion",
        "new_project_award": "adjudicacion contrato proyecto",
        "funding_round": "ronda inversion capital",
        "expansion_new_office": "nueva oficina apertura",
        "fleet_expansion": "expansion flota vehiculos"
    }
    keywords = signal_map.get(signal_type, "expansion inversion")
    return f"{company['name']} {keywords} Colombia 2026"


def run_scout(state: AgentState) -> AgentState:
    """
    Scout Agent main function.
    Loads companies, filters by sector/region, queries Google tools
    via MCP, and classifies detected signals.
    """
    print("\n🔍 SCOUT AGENT: Starting signal collection...")

    companies = _load_companies()
    config = _load_config()

    target_sector = state.get("target_sector", "")
    target_region = state.get("target_region", "")
    max_companies = state.get("max_companies", 5)

    filtered = _filter_companies(
        companies, target_sector, target_region, max_companies, config
    )

    print(f"   Analyzing {len(filtered)} companies...")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
        temperature=0.1
    )

    signals = []
    companies_analyzed = []

    for company in filtered:
        print(f"   → Scanning: {company['name']}")
        companies_analyzed.append(company["id"])

        for signal_type in company.get("current_signals", []):
            try:
                # Step 1: Enrich company via Knowledge Graph
                kg_result = knowledge_graph_search.invoke(
                    {"company_name": company["name"]}
                )

                # Step 2: Search for news via Custom Search
                query = _build_search_query(company, signal_type)
                search_result = custom_search.invoke(
                    {"query": query, "num_results": 3}
                )

                # Step 3: Monitor sector news via RSS
                rss_result = news_rss.invoke({
                    "sector": company["sector"],
                    "city": company["city"],
                    "max_items": 3
                })

                # Step 4: Use LLM to assess signal relevance
                context = f"""
Company: {company['name']}
Sector: {company['sector']}
Region: {company['region']}
Signal type: {signal_type}
Company description: {company['description']}
Knowledge Graph data: {kg_result[:500]}
Recent news: {search_result[:500]}
RSS feed: {rss_result[:300]}
"""
                messages = [
                    SystemMessage(content=(
                        "You are a B2B commercial intelligence analyst for a Telco company in Colombia. "
                        "Assess whether the signal is commercially relevant. "
                        "Respond ONLY with valid JSON, no markdown, no extra text. "
                        "Format: {\"relevant\": true/false, \"confidence\": 0.0-1.0, "
                        "\"reason\": \"brief reason\", \"commercial_opportunity\": \"brief opportunity\"}"
                    )),
                    HumanMessage(content=(
                        f"Is this signal commercially relevant for a Telco B2B company?\n{context}"
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

                assessment = json.loads(raw)

                if assessment.get("relevant") and assessment.get("confidence", 0) >= 0.5:
                    urgency = _classify_signal_urgency(signal_type, config)
                    signal: SignalModel = {
                        "company_id": company["id"],
                        "company_name": company["name"],
                        "sector": company["sector"],
                        "region": company["region"],
                        "signal_type": signal_type,
                        "urgency": urgency,
                        "is_client": company.get("is_client", False),
                        "description": assessment.get("commercial_opportunity", ""),
                        "raw_data": json.dumps({
                            "kg": kg_result[:300],
                            "search": search_result[:300],
                            "assessment": assessment
                        })
                    }
                    signals.append(signal)
                    print(f"     ✅ Signal detected: {signal_type} [{urgency}]")
                else:
                    print(f"     ⏭ Signal skipped: {signal_type} (low relevance)")

            except Exception as e:
                print(f"     ❌ Error processing {signal_type}: {e}")
                state.setdefault("errors", []).append(
                    f"Scout error for {company['name']}/{signal_type}: {str(e)}"
                )

    print(f"\n   Scout complete: {len(signals)} signals detected from "
          f"{len(companies_analyzed)} companies")

    state["signals"] = signals
    state["companies_analyzed"] = companies_analyzed
    state["current_signal_index"] = 0
    return state
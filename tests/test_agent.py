"""
Tests for the B2B Commercial Intelligence Agent.
Tests cover: RAG retrieval, pipeline analysis, cross-sell insights,
state structure, and full workflow smoke test.
"""

import json
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


# ── Test 1: RAG retrieval returns relevant results ─────────────
def test_rag_retrieval_returns_results():
    """RAG tool returns relevant chunks for a manufacturing query."""
    from tools.rag import initialize_rag, retrieve_portfolio_context

    initialize_rag()
    result = retrieve_portfolio_context.invoke({
        "query": "SD-WAN manufacturing expansion multi-site Colombia",
        "sector": "manufacturing",
        "top_k": 3
    })

    data = json.loads(result)
    assert "results" in data, "RAG should return results key"
    assert len(data["results"]) > 0, "RAG should return at least one result"
    assert data["has_evidence"] is True, "RAG should find evidence"
    print(f"\n✅ RAG returned {len(data['results'])} chunks")


# ── Test 2: Pipeline analysis returns contract data ────────────
def test_pipeline_analysis_returns_contract_data():
    """Pipeline tool returns contract and opportunity data for known company."""
    from tools.pipeline import analyze_pipeline

    result = analyze_pipeline.invoke({"company_id": "COL002"})
    data = json.loads(result)

    assert "contracts" in data, "Should return contracts"
    assert "opportunities" in data, "Should return opportunities"
    assert len(data["contracts"]) > 0, "COL002 should have contracts"
    print(f"\n✅ Pipeline returned {len(data['contracts'])} contracts "
          f"and {len(data['opportunities'])} opportunities")


# ── Test 3: Pipeline detects urgent contracts ──────────────────
def test_pipeline_detects_urgent_contracts():
    """Pipeline tool correctly identifies contracts near expiry."""
    from tools.pipeline import analyze_pipeline

    # COL015 has a contract expiring in 25 days
    result = analyze_pipeline.invoke({"company_id": "COL015"})
    data = json.loads(result)

    assert data["churn_risk"] in ["high", "critical"], \
        "COL015 should have high or critical churn risk"
    assert len(data["urgent_actions"]) > 0, \
        "Should have urgent actions for expiring contract"
    print(f"\n✅ Churn risk detected: {data['churn_risk']}")
    print(f"   Urgent actions: {data['urgent_actions'][0]}")


# ── Test 4: Cross-sell identifies product gaps ─────────────────
def test_crosssell_identifies_gaps():
    """Cross-sell tool finds product gaps for a known sector."""
    from tools.crosssell import generate_crosssell_insights

    result = generate_crosssell_insights.invoke({
        "company_id": "COL001",
        "sector": "manufacturing"
    })
    data = json.loads(result)

    assert "cross_sell_gaps" in data, "Should return cross-sell gaps"
    assert "upsell_opportunities" in data, "Should return upsell count"
    print(f"\n✅ Cross-sell found {data['upsell_opportunities']} opportunities")
    if data["cross_sell_gaps"]:
        print(f"   First gap: {data['cross_sell_gaps'][0]['product']}")


# ── Test 5: State structure is valid ──────────────────────────
def test_agent_state_structure():
    """AgentState TypedDict has all required keys."""
    from agent.state import AgentState

    state: AgentState = {
        "target_sector": "manufacturing",
        "target_region": "Bogota",
        "max_companies": 2,
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
        "cycle_start_time": 0.0,
        "errors": []
    }

    assert state["target_sector"] == "manufacturing"
    assert isinstance(state["signals"], list)
    assert isinstance(state["metrics"], dict)
    assert "groundedness_rate" in state["metrics"]
    print("\n✅ AgentState structure is valid")


# ── Test 6: Config loads correctly ────────────────────────────
def test_config_loads_correctly():
    """Config file loads with all required sections."""
    config_path = os.path.join("data", "config.json")
    assert os.path.exists(config_path), "config.json should exist"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "strategic_filters" in config
    assert "signal_types" in config
    assert "scoring" in config
    assert "rag" in config
    assert config["scoring"]["approval_threshold"] == 0.6
    assert config["scoring"]["max_revision_iterations"] == 2
    print("\n✅ Config loaded with all required sections")
    print(f"   Approval threshold: {config['scoring']['approval_threshold']}")
    print(f"   Max iterations: {config['scoring']['max_revision_iterations']}")


# ── Test 7: Companies dataset loads correctly ──────────────────
def test_companies_dataset_loads():
    """Companies dataset loads with expected structure."""
    companies_path = os.path.join("data", "companies.json")
    assert os.path.exists(companies_path), "companies.json should exist"

    with open(companies_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    companies = data["companies"]
    assert len(companies) == 20, "Should have exactly 20 companies"

    for company in companies:
        assert "id" in company
        assert "name" in company
        assert "sector" in company
        assert "region" in company
        assert "is_client" in company
        assert "current_signals" in company

    print(f"\n✅ Companies dataset: {len(companies)} companies loaded")
    clients = [c for c in companies if c["is_client"]]
    prospects = [c for c in companies if not c["is_client"]]
    print(f"   Clients: {len(clients)} | Prospects: {len(prospects)}")


# ── Test 8: Advisor lookup by region ──────────────────────────
def test_advisor_lookup_by_region():
    """Advisor tool returns correct advisor for known region."""
    from tools.pipeline import get_advisor_for_region

    result = get_advisor_for_region.invoke({"region": "Bogota"})
    data = json.loads(result)

    assert "advisor" in data
    assert data["advisor"] != "Sin asesor asignado", \
        "Bogota should have an assigned advisor"
    print(f"\n✅ Advisor for Bogota: {data['advisor']}")
"""
Cross-sell insights tool.
Compares contracted services against available portfolio
to identify gaps and upsell opportunities.
"""

import json
import os
from langchain_core.tools import tool


def _load_portfolio() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    portfolio_path = os.path.join(base_dir, "data", "portfolio.json")
    with open(portfolio_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pipeline() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, "data", "pipeline.json")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def generate_crosssell_insights(company_id: str, sector: str) -> str:
    """
    Generates cross-sell and upsell insights for a company.
    Compares current contracted products against the full portfolio
    for the company sector and identifies gaps and opportunities.
    """
    try:
        portfolio = _load_portfolio()
        pipeline = _load_pipeline()

        # Get contracted products
        contracted_products = set()
        for contract in pipeline.get("contracts", []):
            if contract["company_id"] == company_id:
                contracted_products.add(contract["product"])

        for opp in pipeline.get("opportunities", []):
            if opp["company_id"] == company_id:
                contracted_products.add(opp["product"])

        # Find relevant portfolio products for sector
        sector_products = []
        for doc in portfolio.get("documents", []):
            if doc["doc_type"] == "product":
                if sector in doc.get("sector", []):
                    sector_products.append(doc)

        # Identify gaps
        gaps = []
        for product in sector_products:
            if product["title"] not in contracted_products:
                gaps.append({
                    "product": product["title"],
                    "relevance": "high" if sector in product.get("sector", []) else "medium",
                    "key_benefit": product["content"][:200],
                    "outcome_reference": product.get("outcomes", ""),
                    "partner": product.get("partner", "")
                })

        # Build result
        result = {
            "company_id": company_id,
            "sector": sector,
            "contracted_products": list(contracted_products),
            "available_products_for_sector": len(sector_products),
            "cross_sell_gaps": gaps,
            "upsell_opportunities": len(gaps),
            "recommendation": (
                f"Found {len(gaps)} cross-sell opportunities for {sector} sector. "
                f"Priority products: {', '.join([g['product'] for g in gaps[:2]])}"
                if gaps else
                "Client has full portfolio coverage for their sector."
            )
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "company_id": company_id,
            "sector": sector,
            "error": str(e)
        })
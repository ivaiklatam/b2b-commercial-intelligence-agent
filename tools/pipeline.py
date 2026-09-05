"""
Pipeline analysis tool.
Reads pipeline.json and returns contract and opportunity context
for a given company, including churn risk and upsell signals.
"""

import json
import os
from datetime import datetime
from langchain_core.tools import tool


def _load_pipeline() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, "data", "pipeline.json")
    with open(pipeline_path, "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def analyze_pipeline(company_id: str) -> str:
    """
    Analyzes pipeline data for a given company.
    Returns active contracts, expiry status, open opportunities,
    assigned advisor, and churn risk assessment.
    """
    try:
        pipeline = _load_pipeline()
        result = {
            "company_id": company_id,
            "contracts": [],
            "opportunities": [],
            "advisor": None,
            "churn_risk": "low",
            "urgent_actions": []
        }

        # Analyze contracts
        for contract in pipeline.get("contracts", []):
            if contract["company_id"] == company_id:
                days_to_expiry = contract.get("days_to_expiry", 999)
                contract_info = {
                    "product": contract["product"],
                    "status": contract["status"],
                    "days_to_expiry": days_to_expiry,
                    "monthly_value_usd": contract["value_usd_monthly"],
                    "renewal_initiated": contract["renewal_initiated"],
                    "notes": contract["notes"]
                }
                result["contracts"].append(contract_info)
                result["advisor"] = contract["advisor"]

                # Assess urgency
                if days_to_expiry < 0:
                    result["churn_risk"] = "critical"
                    result["urgent_actions"].append(
                        f"CRITICAL: Contract for {contract['product']} expired "
                        f"{abs(days_to_expiry)} days ago. Immediate action required."
                    )
                elif days_to_expiry <= 30:
                    if result["churn_risk"] != "critical":
                        result["churn_risk"] = "high"
                    if not contract["renewal_initiated"]:
                        result["urgent_actions"].append(
                            f"URGENT: Contract for {contract['product']} expires in "
                            f"{days_to_expiry} days. Renewal not initiated."
                        )
                elif days_to_expiry <= 90:
                    if result["churn_risk"] == "low":
                        result["churn_risk"] = "medium"
                    if not contract["renewal_initiated"]:
                        result["urgent_actions"].append(
                            f"WARNING: Contract for {contract['product']} expires in "
                            f"{days_to_expiry} days. Consider initiating renewal."
                        )

        # Analyze opportunities
        for opp in pipeline.get("opportunities", []):
            if opp["company_id"] == company_id:
                opp_info = {
                    "product": opp["product"],
                    "stage": opp["stage"],
                    "value_usd": opp["value_usd"],
                    "probability": opp["probability"],
                    "days_without_activity": opp["days_without_activity"],
                    "notes": opp["notes"]
                }
                result["opportunities"].append(opp_info)
                if not result["advisor"]:
                    result["advisor"] = opp["advisor"]

                if opp["days_without_activity"] > 20:
                    result["urgent_actions"].append(
                        f"STALLED: Opportunity for {opp['product']} has no activity "
                        f"for {opp['days_without_activity']} days."
                    )

        if not result["contracts"] and not result["opportunities"]:
            result["message"] = "No pipeline data found for this company. Potential new prospect."

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"company_id": company_id, "error": str(e)})


@tool
def get_advisor_for_region(region: str) -> str:
    """
    Returns the assigned advisor for a given region.
    """
    try:
        pipeline = _load_pipeline()
        for advisor in pipeline.get("advisors", []):
            if region in advisor.get("territories", []):
                return json.dumps({
                    "advisor": advisor["name"],
                    "region": advisor["region"],
                    "territories": advisor["territories"],
                    "active_clients": advisor["active_clients"]
                }, ensure_ascii=False)

        return json.dumps({
            "region": region,
            "advisor": "Sin asesor asignado",
            "message": "No advisor found for this region"
        })

    except Exception as e:
        return json.dumps({"region": region, "error": str(e)})
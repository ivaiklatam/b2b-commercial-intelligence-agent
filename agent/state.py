"""
LangGraph State definition for the B2B Commercial Intelligence Agent.
This state flows through all nodes: Scout, Strategist, Critic, and Memory.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class SignalModel(TypedDict):
    company_id: str
    company_name: str
    sector: str
    region: str
    signal_type: str
    urgency: str
    is_client: bool
    description: str
    raw_data: str


class HypothesisModel(TypedDict):
    hypothesis_id: str
    company_id: str
    company_name: str
    product: str
    argument: str
    urgency: str
    action: str
    rag_evidence: list[str]
    pipeline_context: str
    initial_score: float


class RecommendationModel(TypedDict):
    recommendation_id: str
    company_id: str
    company_name: str
    advisor: str
    region: str
    product: str
    argument: str
    urgency: str
    action: str
    rag_evidence: list[str]
    pipeline_context: str
    score: float
    score_breakdown: dict[str, float]
    has_rag_evidence: bool
    label: str
    iteration: int


class MetricsModel(TypedDict):
    total_companies_analyzed: int
    total_signals_detected: int
    total_hypotheses_generated: int
    total_recommendations_approved: int
    total_recommendations_rejected: int
    total_unresolved: int
    groundedness_rate: float
    pruning_rate: float
    escalation_rate: float
    cycle_latency_seconds: float


class AgentState(TypedDict):
    # Input
    target_sector: str
    target_region: str
    max_companies: int

    # Scout output
    signals: list[SignalModel]
    companies_analyzed: list[str]

    # Strategist output
    hypotheses: list[HypothesisModel]
    current_signal_index: int

    # Critic output
    approved_recommendations: list[RecommendationModel]
    rejected_recommendations: list[RecommendationModel]
    unresolved_signals: list[SignalModel]
    revision_iterations: int
    current_hypotheses_under_review: list[HypothesisModel]

    # Memory output
    memory_updated: bool

    # Metrics
    metrics: MetricsModel

    # Cycle control
    cycle_start_time: float
    errors: list[str]
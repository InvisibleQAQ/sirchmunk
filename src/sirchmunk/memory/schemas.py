# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared data models for the retrieval memory system.

All models are plain dataclasses — no external validation dependency.
They are serialisable to/from dicts and JSON natively.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────────────
#  Strategy hint (output of PatternMemory lookup)
# ────────────────────────────────────────────────────────────────────

@dataclass
class StrategyHint:
    """Search parameter overrides suggested by PatternMemory."""

    mode: Optional[str] = None
    top_k_files: Optional[int] = None
    max_loops: Optional[int] = None
    enable_dir_scan: Optional[bool] = None
    keyword_strategy: Optional[str] = None
    confidence: float = 0.0
    source_pattern_id: Optional[str] = None
    token_budget: Optional[int] = None    # suggested token budget
    resolution_level: int = 4             # which HMRPL level produced this hint
    # Budget allocation hints (derived from step bandit + pattern stats)
    first_hop_budget_ratio: Optional[float] = None
    entity_resolution_priority: Optional[str] = None  # "title_lookup_first" | "keyword_first"
    early_stop_aggressiveness: Optional[float] = None  # 0.0-1.0


# ────────────────────────────────────────────────────────────────────
#  PatternMemory models
# ────────────────────────────────────────────────────────────────────

@dataclass
class QueryPattern:
    """Learned mapping: query feature signature → optimal strategy params.

    Thompson Sampling fields (*alpha*, *beta*) model the Beta distribution
    over the success probability for this pattern.  Each observed outcome
    shifts the distribution:  ``alpha += confidence`` on success,
    ``beta += (1 - confidence)`` on failure.
    """

    pattern_id: str
    query_type: str = "factual"
    entity_types: List[str] = field(default_factory=list)
    complexity: str = "moderate"
    entity_count: int = 0
    hop_hint: str = "single"
    optimal_mode: str = "DEEP"
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    avg_latency: float = 0.0
    avg_tokens: int = 0
    # Thompson Sampling Beta-distribution priors
    alpha: float = 1.0
    beta_param: float = 1.0
    created_at: str = ""
    updated_at: str = ""
    # HMRPL hierarchical fields
    resolution_level: int = 4          # HMRPL level (0=coarsest, 4=finest)
    parent_id: Optional[str] = None    # parent pattern for hierarchical tree
    children_ids: List[str] = field(default_factory=list)  # child pattern IDs
    avg_reward: float = 0.0            # shaped reward running average
    total_visits: int = 0              # global visit counter N for UCB

    # Meta-RL fields: contextual bandit + loop budget + distilled rules
    distilled_rules: List[str] = field(default_factory=list)
    failure_warnings: List[str] = field(default_factory=list)
    best_keyword_strategy: str = ""
    # loop_count (str) → [success_count, total_count] for Bayesian loop budget
    loop_budget_stats: Dict[str, List[int]] = field(default_factory=dict)
    # strategy_name → [alpha, beta] for Thompson Sampling over strategy arms
    strategy_arms: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryPattern:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReasoningChain:
    """Abstracted ReAct trace template linked to a query pattern."""

    template_id: str
    pattern_id: str
    steps: List[Dict[str, str]] = field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReasoningChain:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ────────────────────────────────────────────────────────────────────
#  CorpusMemory models
# ────────────────────────────────────────────────────────────────────

@dataclass
class SemanticExpansion:
    """A single semantic expansion rule (synonym / alias / related)."""

    target: str
    relation: str = "synonym"
    confidence: float = 0.5
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SemanticBridgeEntry:
    """Term → expansion mapping for keyword generalisation."""

    term: str
    expansions: List[SemanticExpansion] = field(default_factory=list)
    domain: str = "general"
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "expansions": [e.to_dict() for e in self.expansions],
            "domain": self.domain,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SemanticBridgeEntry:
        expansions = [
            SemanticExpansion(**e) if isinstance(e, dict) else e
            for e in d.get("expansions", [])
        ]
        return cls(
            term=d["term"],
            expansions=expansions,
            domain=d.get("domain", "general"),
            updated_at=d.get("updated_at", ""),
        )


# ────────────────────────────────────────────────────────────────────
#  FeedbackMemory model
# ────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackSignal:
    """Unified feedback signal emitted after a completed search."""

    signal_type: str = "implicit"
    query: str = ""
    mode: str = "FAST"
    answer_found: bool = False
    answer_text: str = ""
    cluster_confidence: float = 0.0
    react_loops: int = 0
    files_read_count: int = 0
    files_useful_count: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    keywords_used: List[str] = field(default_factory=list)
    paths_searched: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    files_discovered: List[str] = field(default_factory=list)
    # Explicit evaluation signals (optional)
    user_verdict: Optional[str] = None
    em_score: Optional[float] = None
    f1_score: Optional[float] = None
    llm_judge_verdict: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # BA-ReAct enrichment (populated when belief tracking is active)
    belief_snapshot: Optional[Dict[str, float]] = None
    mces_triggered_files: Optional[List[str]] = None
    ess_at_termination: Optional[float] = None
    convergence_achieved: bool = False
    high_value_files: Optional[List[str]] = None
    dead_candidates: Optional[List[str]] = None

    # Ground-truth query type from benchmark data (overrides heuristic classifier)
    query_type_override: Optional[str] = None
    # Title-lookup results: list of (title, [paths]) for CorpusMemory persistence
    title_lookup_results: Optional[List[Dict[str, Any]]] = None

    # Computed by manager during dispatch (for inject_evaluation delta correction)
    heuristic_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ────────────────────────────────────────────────────────────────────
#  Meta-RL: Search strategy & planning models
# ────────────────────────────────────────────────────────────────────

class SearchStrategy:
    """Named search strategies for the contextual bandit.

    Each constant is a meta-action in the meta-RL framework:
    π_meta(strategy | query_features).  Strategies are corpus-agnostic
    and transfer across independent samples.
    """

    DIRECT = "direct"
    BRIDGE_FIRST = "bridge_first"
    PARALLEL_ENTITIES = "parallel_entities"
    DECOMPOSE_COMPOUND = "decompose_compound"
    BROAD_THEN_NARROW = "broad_then_narrow"

    ALL = (DIRECT, BRIDGE_FIRST, PARALLEL_ENTITIES,
           DECOMPOSE_COMPOUND, BROAD_THEN_NARROW)


@dataclass
class SearchPlan:
    """Output of the Memory-Augmented Planner (MAP).

    A structured search plan that can drive either guided execution
    (high confidence) or serve as a warm-start hint for full ReAct
    (lower confidence).
    """

    plan_steps: List[str] = field(default_factory=list)
    keyword_strategy: str = "direct"
    expected_hops: int = 2
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    reasoning_type: str = "simple"
    answer_format: str = "entity"
    source: str = "planner"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchPlan":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AbstractTrajectoryStep:
    """One step in a strategy-level trajectory (no instance-specific data).

    Records WHAT strategy was used at each step, not WHICH specific
    keywords or files.  This makes trajectories transferable across
    independent samples.
    """

    action: str = ""
    strategy: str = ""
    result_type: str = ""
    files_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class AbstractTrajectory:
    """Strategy-level search trajectory for experience replay.

    Stores the abstracted sequence of decisions (not instance data)
    so that the distiller can learn generalizable rules.
    """

    query_type: str = "factual"
    complexity: str = "moderate"
    hop_hint: str = "single"
    entity_count: int = 0
    answer_format: str = "entity"
    steps: List[AbstractTrajectoryStep] = field(default_factory=list)
    loops_used: int = 0
    total_tokens: int = 0
    outcome: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in self.__dict__.items() if k != "steps"},
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AbstractTrajectory":
        steps_raw = d.get("steps", [])
        steps = [
            AbstractTrajectoryStep(**s) if isinstance(s, dict) else s
            for s in steps_raw
        ]
        filtered = {k: v for k, v in d.items()
                    if k in cls.__dataclass_fields__ and k != "steps"}
        obj = cls(**filtered)
        obj.steps = steps
        return obj


@dataclass
class StrategyDistillation:
    """LLM-distilled meta-knowledge for a query type.

    Produced by :class:`StrategyDistiller` from a batch of
    :class:`AbstractTrajectory` entries.  Consumed by
    :class:`MemoryAugmentedPlanner` to generate informed search plans.
    """

    query_type: str = "factual"
    complexity: str = "moderate"
    rules: List[str] = field(default_factory=list)
    failure_warnings: List[str] = field(default_factory=list)
    best_keyword_strategy: str = ""
    sample_count: int = 0
    success_rate: float = 0.0
    avg_loops: float = 0.0
    avg_tokens: float = 0.0
    distilled_at: str = ""
    # Structure-aware distillation fields (B2)
    optimal_action_sequence: List[str] = field(default_factory=list)
    avg_loops_success: float = 0.0
    avg_loops_failure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyDistillation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ────────────────────────────────────────────────────────────────────
#  Step-level Bandit model
# ────────────────────────────────────────────────────────────────────

@dataclass
class StepArmStats:
    """Per-(query_type, step_bucket, action) Thompson Sampling arm.

    Used by the step-level contextual bandit (D1) to learn which tool
    is most effective at each step of the ReAct loop, conditioned on
    query type.
    """

    action: str
    alpha: float = 1.0
    beta_param: float = 1.0
    total: int = 0
    avg_info_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepArmStats":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ────────────────────────────────────────────────────────────────────
#  Utility helpers
# ────────────────────────────────────────────────────────────────────

def compute_pattern_id(
    query_type: str,
    complexity: str,
    entity_types: List[str],
    *,
    entity_count: int = 0,
    hop_hint: str = "single",
) -> str:
    """Deterministic ID from query feature signature.

    The *entity_count* and *hop_hint* parameters add finer granularity
    while remaining backward-compatible (default values produce the
    same hash as the old 3-field version when callers omit them).
    """
    key = json.dumps(
        {
            "type": query_type,
            "complexity": complexity,
            "entities": sorted(entity_types),
            "entity_count": entity_count,
            "hop_hint": hop_hint,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def compute_pattern_id_at_level(
    query_type: str,
    complexity: str,
    entity_types: list,
    entity_count: int,
    hop_hint: str,
    level: int = 4,
) -> str:
    """Compute pattern ID at a specific HMRPL resolution level.
    
    L0=type only, L1=type+complexity, L2=+hop, L3=+entity_count, L4=full (with entity_types).
    Coarser levels aggregate more patterns for better statistical power.
    """
    parts = [query_type]
    if level >= 1:
        parts.append(complexity)
    if level >= 2:
        parts.append(str(hop_hint))
    if level >= 3:
        parts.append(str(entity_count))
    if level >= 4:
        sorted_types = sorted(set(entity_types)) if entity_types else []
        parts.append(",".join(sorted_types) if sorted_types else "none")
    sig = "|".join(parts)
    return hashlib.md5(sig.encode()).hexdigest()[:16]


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Stable hash of strategy parameters for failed_strategies dedup."""
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]

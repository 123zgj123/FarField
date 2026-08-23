"""Farfield: evidence-gated research self-evolution runtime."""

from .models import (
    BudgetEnvelope,
    Candidate,
    CandidateMode,
    Evidence,
    ObjectiveLevel,
    PromotionEvidence,
    TrustClass,
    UpdateKind,
)
from .policy import PromotionDecision, PromotionPolicy
from .runtime import FarfieldRuntime

__all__ = [
    "BudgetEnvelope",
    "Candidate",
    "CandidateMode",
    "Evidence",
    "ObjectiveLevel",
    "PromotionDecision",
    "PromotionEvidence",
    "PromotionPolicy",
    "FarfieldRuntime",
    "TrustClass",
    "UpdateKind",
]

__version__ = "0.1.0"

"""Conservative decision rules for product inspections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
class Outcome(StrEnum):
    PASS = "PASS / ACCEPTED"
    FAIL = "FAIL / REJECTED"
    REVIEW = "REVIEW REQUIRED"
    RECAPTURE = "RECAPTURE REQUIRED"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    explanation: str
    human_review_required: bool


def decide(
    *,
    image_suitable: bool,
    category_confident: bool = True,
    supported_defect_detected: bool = False,
    validated_policy_passed: bool = False,
    ai_recommendation: str | None = None,
) -> Decision:
    """Apply policy independently of model providers and their raw outputs."""
    if not image_suitable:
        return Decision(Outcome.RECAPTURE, "Image quality is unsuitable for inspection.", True)
    if not category_confident:
        return Decision(
            Outcome.REVIEW,
            "Product category is unknown or uncertain. Confirm the category manually or configure a compatible recognition model.",
            True,
        )
    if supported_defect_detected:
        return Decision(Outcome.FAIL, "A defined defect was detected by the supported inspection model.", True)
    if ai_recommendation:
        return Decision(
            Outcome.REVIEW,
            "AI-assisted observations are not proof of defect-free quality. Human review is required.",
            True,
        )
    if validated_policy_passed:
        return Decision(Outcome.PASS, "All checks in the configured validated inspection policy passed.", False)
    return Decision(
        Outcome.REVIEW,
        "No validated defect-inspection policy is available for this product. No defect-free claim can be made.",
        True,
    )
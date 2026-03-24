"""Pydantic schema for LLM parcel explanations (JSON contract)."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class ExplainStatus(str, Enum):
    normal = "normal"
    warning = "warning"
    critical = "critical"
    insufficient_data = "insufficient_data"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ConsistencyCheck(str, Enum):
    consistent = "consistent"
    weakly_supported = "weakly_supported"
    unsupported = "unsupported"


class ParcelExplanation(BaseModel):
    """Validated LLM output for one parcel."""

    status: ExplainStatus
    summary: str = Field(..., max_length=500)
    possible_causes: list[str] = Field(default_factory=list, max_length=5)
    confidence: Confidence
    recommended_action: str = Field(..., max_length=300)
    evidence_used: list[str] = Field(default_factory=list, max_length=8)
    consistency_check: ConsistencyCheck = ConsistencyCheck.consistent


# Models sometimes output traffic-light values instead of the enum strings.
_STATUS_ALIASES: dict[str, str] = {
    "GREEN": "normal",
    "YELLOW": "warning",
    "RED": "critical",
    "GRAY": "insufficient_data",
    "grey": "insufficient_data",
    "gray": "insufficient_data",
    "green": "normal",
    "yellow": "warning",
    "red": "critical",
}

# Models sometimes output non-standard confidence labels.
_CONFIDENCE_ALIASES: dict[str, str] = {
    "very high": "high",
    "very low": "low",
    "moderate": "medium",
    "high confidence": "high",
    "low confidence": "low",
}

# Models sometimes output non-standard consistency labels.
_CONSISTENCY_ALIASES: dict[str, str] = {
    "weakly supported": "weakly_supported",
    "not supported": "unsupported",
    "not_consistent": "unsupported",
    "inconsistent": "unsupported",
}


def _normalize_data(data: dict) -> dict:
    """Map common LLM alias values to enum strings before Pydantic validation."""
    if "status" in data and isinstance(data["status"], str):
        data["status"] = _STATUS_ALIASES.get(data["status"].strip(), data["status"].strip().lower())
    if "confidence" in data and isinstance(data["confidence"], str):
        raw = data["confidence"].strip().lower()
        data["confidence"] = _CONFIDENCE_ALIASES.get(raw, raw)
    if "consistency_check" in data and isinstance(data["consistency_check"], str):
        raw = data["consistency_check"].strip().lower().replace(" ", "_")
        data["consistency_check"] = _CONSISTENCY_ALIASES.get(
            data["consistency_check"].strip().lower(), raw
        )
    return data


def validate_explanation(
    raw_json: str,
    parcel_status: str,
) -> ParcelExplanation | None:
    """Parse and validate LLM JSON. Adjust consistency if LLM status mismatches scoring."""
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            log.warning("LLM output is not a JSON object")
            return None
        data = _normalize_data(data)
        explanation = ParcelExplanation(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        log.warning("LLM output validation failed: %s", e)
        return None

    status_map = {
        "GREEN": ExplainStatus.normal,
        "YELLOW": ExplainStatus.warning,
        "RED": ExplainStatus.critical,
        "GRAY": ExplainStatus.insufficient_data,
    }
    expected = status_map.get(parcel_status)
    if expected is not None and explanation.status != expected:
        explanation = explanation.model_copy(
            update={"consistency_check": ConsistencyCheck.unsupported},
        )
        log.warning(
            "LLM status '%s' inconsistent with scoring '%s' — marking unsupported",
            explanation.status.value,
            parcel_status,
        )

    return explanation


def fallback_explanation(parcel_status: str, reason: str) -> ParcelExplanation:
    """Minimal valid explanation when the LLM fails."""
    status_map = {
        "GREEN": ExplainStatus.normal,
        "YELLOW": ExplainStatus.warning,
        "RED": ExplainStatus.critical,
        "GRAY": ExplainStatus.insufficient_data,
    }
    st = status_map.get(parcel_status, ExplainStatus.insufficient_data)
    return ParcelExplanation(
        status=st,
        summary=(
            "Automated explanation could not be generated. "
            f"Reason: {reason}. Refer to raw scores and trajectory in the dashboard."
        ),
        possible_causes=[],
        confidence=Confidence.low,
        recommended_action="Review parcel metrics manually or re-run the explain step.",
        evidence_used=[reason],
        consistency_check=ConsistencyCheck.unsupported,
    )

"""
Experiment Designer agent: hypothesis → structured design with optional clarification.

Uses OpenAI GPT-4o mini for clarification (one round max) and design JSON.
Sample size from two-proportion z-test power analysis (numpy); warns on marketplace/platform.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from openai import OpenAI
from scipy.stats import norm

from backend.config import (
    DEFAULT_ALPHA,
    DEFAULT_BASE_CONVERSION_RATE,
    DEFAULT_MDE,
    DEFAULT_POWER,
    MAX_RUNTIME_DAYS,
    MIN_RUNTIME_DAYS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ClarificationResult:
    """Returned when hypothesis is ambiguous; caller should respond and call again."""

    needs_clarification: bool = True
    question: str = ""


@dataclass
class ExperimentDesign:
    """Structured experiment design with raw LLM response."""

    primary_metric: str
    guardrail_metrics: list[str]
    randomization_unit: str
    sample_size_required: int
    runtime_days: int
    warnings: list[str]
    raw_llm_response: str = ""
    design_json: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Power analysis: two-proportion z-test, from scratch with numpy
# n per group = (z_alpha/2 + z_beta)^2 * [ p1(1-p1) + p2(1-p2) ] / (p2-p1)^2
# ---------------------------------------------------------------------------
def sample_size_two_proportion(
    base_conversion_rate: float = DEFAULT_BASE_CONVERSION_RATE,
    mde: float = DEFAULT_MDE,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """
    Minimum sample size per group for two-proportion z-test (two-sided).
    base_conversion_rate = p1 (control), p2 = p1 * (1 + mde).
    """
    p1 = base_conversion_rate
    p2 = p1 * (1.0 + mde)
    delta = p2 - p1
    if delta <= 0:
        return 0
    # Two-sided alpha
    z_alpha2 = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    var_term = p1 * (1.0 - p1) + p2 * (1.0 - p2)
    n_per_group = ((z_alpha2 + z_beta) ** 2) * var_term / (delta ** 2)
    return max(0, int(np.ceil(n_per_group)))


# ---------------------------------------------------------------------------
# Marketplace / network interference warning
# ---------------------------------------------------------------------------
def _add_network_interference_warning(hypothesis: str, warnings: list[str]) -> None:
    text = hypothesis.lower()
    if re.search(r"marketplace|two-sided platform|two sided platform|two-sided marketplace", text):
        msg = "Hypothesis mentions marketplace or two-sided platform; network effects may violate SUTVA (interference risk)."
        if msg not in warnings:
            warnings.append(msg)


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in config (check .env)")
    return OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Clarification: one round max
# ---------------------------------------------------------------------------
CLARIFY_SYSTEM = """You are an experiment design assistant. Given a hypothesis for an A/B test, determine if the primary_metric (the main success metric) and randomization_unit (e.g. user, session, device) are clearly specified or implied. If either is ambiguous, respond with a single clarifying question (one short paragraph). If both are clear, respond with exactly: CLEAR."""

DESIGN_SYSTEM = """You are an experiment design assistant. Given a hypothesis (and any clarification), output a JSON object with exactly these keys:
- primary_metric: string (e.g. "subscription_started", "conversion_rate")
- guardrail_metrics: list of strings (e.g. ["churn_rate", "support_tickets"])
- randomization_unit: string ("user" or "session" or "device" etc.)
- runtime_days: integer (suggested experiment length, between 7 and 30)
- warnings: list of strings (any caveats or risks)

Do not include sample_size_required; we will compute it. Output only valid JSON, no markdown."""


def _check_ambiguity(hypothesis: str, client: OpenAI) -> str | None:
    """Return clarifying question if ambiguous, else None (CLEAR)."""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": CLARIFY_SYSTEM},
            {"role": "user", "content": f"Hypothesis: {hypothesis}"},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    if content.upper() == "CLEAR":
        return None
    return content if content else None


def _produce_design_json(
    hypothesis: str,
    clarification_response: str | None,
    client: OpenAI,
) -> dict[str, Any]:
    """Get structured design as JSON from LLM."""
    user_content = f"Hypothesis: {hypothesis}"
    if clarification_response:
        user_content += f"\n\nClarification from user: {clarification_response}"
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": DESIGN_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return {
        "primary_metric": data.get("primary_metric") or "conversion_rate",
        "guardrail_metrics": data.get("guardrail_metrics") or [],
        "randomization_unit": data.get("randomization_unit") or "user",
        "runtime_days": data.get("runtime_days"),
        "warnings": data.get("warnings") or [],
        "_raw": raw,
    }


def _clamp_runtime_days(days: Any) -> int:
    """Clamp runtime_days to [MIN_RUNTIME_DAYS, MAX_RUNTIME_DAYS]."""
    try:
        d = int(days)
    except (TypeError, ValueError):
        d = 14
    return max(MIN_RUNTIME_DAYS, min(MAX_RUNTIME_DAYS, d))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_designer(
    hypothesis: str,
    clarification_response: str | None = None,
    *,
    base_conversion_rate: float = DEFAULT_BASE_CONVERSION_RATE,
    mde: float = DEFAULT_MDE,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> ClarificationResult | ExperimentDesign:
    """
    Orchestrate the full designer flow: optional clarification, then design.

    If clarification_response is None and the hypothesis is ambiguous, returns
    ClarificationResult(question=...). Otherwise returns ExperimentDesign with
    sample_size_required computed from power analysis.
    """
    client = _get_client()
    hypothesis = (hypothesis or "").strip()
    if not hypothesis:
        return ExperimentDesign(
            primary_metric="",
            guardrail_metrics=[],
            randomization_unit="user",
            sample_size_required=0,
            runtime_days=14,
            warnings=["Empty hypothesis."],
            raw_llm_response="",
            design_json={},
        )

    # One round of clarification max: only when no clarification_response yet
    if clarification_response is None:
        question = _check_ambiguity(hypothesis, client)
        if question:
            return ClarificationResult(needs_clarification=True, question=question)

    # Produce design JSON from LLM
    data = _produce_design_json(hypothesis, clarification_response, client)
    raw = data.pop("_raw", "")

    runtime_days = _clamp_runtime_days(data.get("runtime_days"))
    warnings = list(data.get("warnings") or [])
    _add_network_interference_warning(hypothesis, warnings)

    # Sample size from power analysis (two-proportion z-test, numpy)
    n_per_group = sample_size_two_proportion(
        base_conversion_rate=base_conversion_rate,
        mde=mde,
        alpha=alpha,
        power=power,
    )
    sample_size_required = n_per_group * 2  # total N for control + treatment

    design_json = {
        "primary_metric": data.get("primary_metric"),
        "guardrail_metrics": data.get("guardrail_metrics"),
        "randomization_unit": data.get("randomization_unit"),
        "sample_size_required": sample_size_required,
        "runtime_days": runtime_days,
        "warnings": warnings,
    }

    return ExperimentDesign(
        primary_metric=data.get("primary_metric") or "conversion_rate",
        guardrail_metrics=list(data.get("guardrail_metrics") or []),
        randomization_unit=data.get("randomization_unit") or "user",
        sample_size_required=sample_size_required,
        runtime_days=runtime_days,
        warnings=warnings,
        raw_llm_response=raw,
        design_json=design_json,
    )

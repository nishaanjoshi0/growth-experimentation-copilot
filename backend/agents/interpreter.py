"""
Results Interpreter agent: synthesizes experiment results into a plain-English recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, DEFAULT_ALPHA
from backend.db.supabase_client import (
    get_agent_decisions,
    get_assignments,
    log_agent_decision,
)
from backend.stats import cuped as cuped_module
from backend.stats import novelty as novelty_module
from backend.stats import srm as srm_module


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class InterpretationResult:
    experiment_id: str
    final_cuped: dict[str, Any]
    final_srm: dict[str, Any]
    final_novelty: dict[str, Any]
    recommendation: str
    confidence: str  # 'high' | 'medium' | 'low'
    action: str  # 'ship' | 'iterate' | 'abandon' | 'rerun'


# ---------------------------------------------------------------------------
# JSON-serializable conversion (numpy -> native Python)
# ---------------------------------------------------------------------------
def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Aggregate CUPED input from all snapshots: weighted mean rate per variant
# ---------------------------------------------------------------------------
def _aggregate_rate_by_variant(snapshots: list[dict]) -> dict[str, float]:
    """Weighted mean primary_metric_value per variant across all snapshots."""
    from collections import defaultdict
    total_val: dict[str, float] = defaultdict(float)
    total_n: dict[str, int] = defaultdict(int)
    for row in snapshots:
        v = row.get("variant")
        if v is None:
            continue
        try:
            p = float(row.get("primary_metric_value") or 0)
            n = int(row.get("sample_size") or 0)
        except (TypeError, ValueError):
            continue
        total_val[v] += p * n
        total_n[v] += n
    return {
        v: (total_val[v] / total_n[v] if total_n[v] else 0.0)
        for v in total_n
    }


def _build_cuped_rows_all_days(
    assignments: list[dict],
    snapshots: list[dict],
) -> list[tuple[str, str, float, float]]:
    """Build (user_id, variant, metric_value, pre_exp_metric) using aggregate rate per variant."""
    rate_by_variant = _aggregate_rate_by_variant(snapshots)
    rows = []
    for a in assignments:
        user_id = a.get("user_id") or ""
        variant = a.get("variant") or "control"
        try:
            pre_exp = float(a.get("pre_exp_metric") or 0)
        except (TypeError, ValueError):
            pre_exp = 0.0
        metric_value = rate_by_variant.get(variant, 0.0)
        rows.append((user_id, variant, metric_value, pre_exp))
    return rows


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in config (check .env)")
    return OpenAI(api_key=OPENAI_API_KEY)


INTERPRETER_SYSTEM = """You are a growth experiment analyst. Given an experiment's hypothesis, design, and statistical results (CUPED, SRM, novelty), plus the monitor's daily decisions, write a short synthesis for the PM.

Respond with exactly three lines, separated by newlines:
1. RECOMMENDATION: (one paragraph: what happened, primary vs guardrail tradeoffs, and what the PM should do next)
2. CONFIDENCE: one of high, medium, low
3. ACTION: one of ship, iterate, abandon, rerun

Be concise. Use the RECOMMENDATION/CONFIDENCE/ACTION labels exactly."""


def run_interpreter(
    experiment_id: str,
    hypothesis: str,
    design: dict[str, Any],
    all_snapshots: list[dict[str, Any]],
    *,
    alpha: float = DEFAULT_ALPHA,
    novelty_window_days: int = 3,
    min_days_required: int = 7,
) -> InterpretationResult:
    """
    Compute final CUPED/SRM/novelty, fetch monitor decisions, synthesize with GPT-4o mini, log and return result.
    """
    assignments = get_assignments(experiment_id)
    decisions = get_agent_decisions(experiment_id)
    monitor_decisions = [d for d in decisions if (d.get("agent") or "").lower() == "monitor"]

    # Final CUPED across all snapshots (aggregate rate per variant)
    cuped_rows = _build_cuped_rows_all_days(assignments, all_snapshots)
    final_cuped = cuped_module.cuped(cuped_rows, alpha=alpha)

    # Final SRM from total assignment counts
    n_control = sum(1 for a in assignments if (a.get("variant") or "").lower() == "control")
    n_treatment = sum(1 for a in assignments if (a.get("variant") or "").lower() == "treatment")
    final_srm = srm_module.srm_detect(
        observed_control=n_control,
        observed_treatment=n_treatment,
        expected_ratio=0.5,
        alpha=0.01,
    )

    # Final novelty from all snapshots
    final_novelty = novelty_module.novelty_detect(
        all_snapshots,
        novelty_window_days=novelty_window_days,
        min_days_required=min_days_required,
    )

    # Build context for LLM
    primary = design.get("primary_metric") or "primary_metric"
    guardrails = design.get("guardrail_metrics") or []
    runtime = design.get("runtime_days") or 0
    summary = (
        f"Hypothesis: {hypothesis}\n"
        f"Design: primary_metric={primary}, guardrails={guardrails}, runtime_days={runtime}\n"
        f"CUPED: lift={final_cuped.get('lift_adjusted', 0):.4f}, p_value={final_cuped.get('p_value', 1):.4f}, significant={final_cuped.get('significant')}\n"
        f"SRM: detected={final_srm.get('srm_detected')}, severity={final_srm.get('severity')}\n"
        f"Novelty: detected={final_novelty.get('novelty_detected')}, ratio={final_novelty.get('novelty_ratio', 0):.2f}\n"
        f"Monitor decisions ({len(monitor_decisions)}): "
        + (
            "; ".join(
                f"{d.get('decision', '')}: {(d.get('reasoning') or '')[:80]}"
                for d in monitor_decisions[-5:]
            )
            if monitor_decisions
            else "none"
        )
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": INTERPRETER_SYSTEM},
            {"role": "user", "content": summary},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()

    recommendation = ""
    confidence = "medium"
    action = "iterate"
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("RECOMMENDATION:"):
            recommendation = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            c = line.split(":", 1)[-1].strip().lower()
            if c in ("high", "medium", "low"):
                confidence = c
        elif line.upper().startswith("ACTION:"):
            a = line.split(":", 1)[-1].strip().lower()
            if a in ("ship", "iterate", "abandon", "rerun"):
                action = a

    if not recommendation:
        recommendation = raw or "No recommendation generated."

    log_agent_decision(
        experiment_id,
        agent="interpreter",
        decision=action,
        reasoning=recommendation,
    )

    return InterpretationResult(
        experiment_id=experiment_id,
        final_cuped=_to_python(final_cuped),
        final_srm=_to_python(final_srm),
        final_novelty=_to_python(final_novelty),
        recommendation=recommendation,
        confidence=confidence,
        action=action,
    )

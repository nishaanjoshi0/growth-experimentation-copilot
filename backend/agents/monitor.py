"""
Monitor agent: runs daily stats (CUPED, SRM, sequential, novelty), updates snapshots, logs decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.data.generator import GeneratorConfig
from backend.db.supabase_client import (
    get_assignments,
    get_snapshots,
    log_agent_decision,
    update_metric_snapshot_flags,
)
from backend.stats import cuped as cuped_module
from backend.stats import novelty as novelty_module
from backend.stats import sequential as sequential_module
from backend.stats import srm as srm_module
from backend.config import DEFAULT_ALPHA, SRM_THRESHOLD


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class MonitorResult:
    experiment_id: str
    day: int
    cuped_result: dict[str, Any]
    srm_result: dict[str, Any]
    sequential_result: dict[str, Any]
    novelty_result: dict[str, Any]
    decision: str  # 'continue' | 'escalate' | 'stop'
    reasoning: str


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
# Build CUPED input from assignments + current day snapshot
# Per-user metric_value = primary_metric_value for that variant on current day (proxy when no per-user outcomes)
# ---------------------------------------------------------------------------
def _build_cuped_rows(
    assignments: list[dict],
    snapshot_current_day: list[dict],
) -> list[tuple[str, str, float, float]]:
    """Build list of (user_id, variant, metric_value, pre_exp_metric) for current day."""
    rate_by_variant = {}
    for row in snapshot_current_day:
        v = row.get("variant")
        if v is None:
            continue
        try:
            rate_by_variant[v] = float(row.get("primary_metric_value") or 0)
        except (TypeError, ValueError):
            rate_by_variant[v] = 0.0

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


def _snapshots_for_day(snapshots: list[dict], day: int) -> list[dict]:
    return [s for s in snapshots if s.get("day") == day]


def _snapshots_through_day(snapshots: list[dict], day: int) -> list[dict]:
    return [s for s in snapshots if s.get("day") is not None and 1 <= s.get("day") <= day]


def run_monitor(
    experiment_id: str,
    current_day: int,
    config: GeneratorConfig,
    *,
    total_days: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    srm_alpha: float = SRM_THRESHOLD,
) -> MonitorResult:
    """
    Fetch data from Supabase, run CUPED/SRM/sequential/novelty, update snapshot flags, log decision.
    """
    total_days = total_days or config.n_days_exp
    assignments = get_assignments(experiment_id)
    snapshots = get_snapshots(experiment_id)

    snapshot_today = _snapshots_for_day(snapshots, current_day)
    snapshots_through_today = _snapshots_through_day(snapshots, current_day)

    # CUPED: per-user rows using current day rate as metric proxy
    cuped_rows = _build_cuped_rows(assignments, snapshot_today)
    cuped_result = cuped_module.cuped(cuped_rows, alpha=alpha)

    # SRM: assignment counts
    n_control = sum(1 for a in assignments if (a.get("variant") or "").lower() == "control")
    n_treatment = sum(1 for a in assignments if (a.get("variant") or "").lower() == "treatment")
    srm_result = srm_module.srm_detect(
        observed_control=n_control,
        observed_treatment=n_treatment,
        expected_ratio=0.5,
        alpha=srm_alpha,
    )

    # Sequential: current day means and stds from snapshot
    mean_c = mean_t = 0.0
    std_c = std_t = 0.0
    n_c = n_t = 0
    for row in snapshot_today:
        v = (row.get("variant") or "").lower()
        p = float(row.get("primary_metric_value") or 0)
        n = int(row.get("sample_size") or 0)
        # For proportion, std = sqrt(p*(1-p))
        std = (p * (1.0 - p)) ** 0.5 if 0 <= p <= 1 else 0.0
        if v == "control":
            mean_c, std_c, n_c = p, std, n
        elif v == "treatment":
            mean_t, std_t, n_t = p, std, n
    sequential_result = sequential_module.obrien_fleming(
        day=current_day,
        total_days=total_days,
        current_mean_control=mean_c,
        current_mean_treatment=mean_t,
        current_std_control=std_c,
        current_std_treatment=std_t,
        n_control=n_c or 1,
        n_treatment=n_t or 1,
        alpha=alpha,
    )

    # Novelty: all snapshots through current day
    novelty_result = novelty_module.novelty_detect(
        snapshots_through_today,
        novelty_window_days=config.novelty_window_days,
        min_days_required=7,
    )

    srm_flagged = srm_result.get("srm_detected", False)
    novelty_flagged = novelty_result.get("novelty_detected", False)
    boundary_crossed = sequential_result.get("recommend_stop", False)

    # Decision: escalate (SRM) > stop (sequential) > continue
    if srm_flagged:
        decision = "escalate"
    elif boundary_crossed:
        decision = "stop"
    else:
        decision = "continue"

    # Write flags back to current day's snapshot
    update_metric_snapshot_flags(
        experiment_id,
        current_day,
        srm_flagged=srm_flagged,
        novelty_flagged=novelty_flagged,
    )

    # Plain-English reasoning
    parts = []
    parts.append("CUPED: lift %.2f%%, p=%.4f, %s." % (
        cuped_result.get("lift_adjusted", 0) * 100,
        cuped_result.get("p_value", 1),
        "significant" if cuped_result.get("significant") else "not significant",
    ))
    parts.append("SRM: %s (p=%.4f)." % (
        "detected" if srm_flagged else "not detected",
        srm_result.get("p_value", 1),
    ))
    parts.append("Sequential: %s." % sequential_result.get("reason", ""))
    parts.append("Novelty: %s." % novelty_result.get("reason", ""))
    parts.append("Decision: %s." % decision)
    reasoning = " ".join(parts)

    log_agent_decision(
        experiment_id,
        agent="monitor",
        decision=decision,
        reasoning=reasoning,
    )

    return MonitorResult(
        experiment_id=experiment_id,
        day=current_day,
        cuped_result=_to_python(cuped_result),
        srm_result=_to_python(srm_result),
        sequential_result=_to_python(sequential_result),
        novelty_result=_to_python(novelty_result),
        decision=decision,
        reasoning=reasoning,
    )

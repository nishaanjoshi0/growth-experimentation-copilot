"""
Novelty effect detection from scratch.

Compares early-window lift (days 1..novelty_window_days) to overall lift;
flags when early lift is inflated relative to overall (e.g. novelty ratio > 1.5).
"""

from __future__ import annotations

from typing import Any

VARIANT_CONTROL = "control"
VARIANT_TREATMENT = "treatment"


def _weighted_mean_by_variant(
    snapshots: list[dict],
    day_min: int,
    day_max: int,
) -> tuple[float, float, int]:
    """
    From snapshots with day in [day_min, day_max], compute weighted mean
    primary_metric_value for control and treatment (by sample_size).
    Returns (mean_control, mean_treatment, n_days_used).
    """
    control_sum = 0.0
    control_weight = 0
    treatment_sum = 0.0
    treatment_weight = 0
    days_seen = set()

    for row in snapshots:
        day = row.get("day")
        if day is None or not (day_min <= day <= day_max):
            continue
        variant = row.get("variant")
        value = row.get("primary_metric_value")
        size = row.get("sample_size", 1)
        if value is None or variant is None:
            continue
        try:
            v, s = float(value), int(size)
        except (TypeError, ValueError):
            continue
        days_seen.add(day)
        if variant == VARIANT_CONTROL:
            control_sum += v * s
            control_weight += s
        elif variant == VARIANT_TREATMENT:
            treatment_sum += v * s
            treatment_weight += s

    mean_c = control_sum / control_weight if control_weight else 0.0
    mean_t = treatment_sum / treatment_weight if treatment_weight else 0.0
    return mean_c, mean_t, len(days_seen)


def _lift(mean_control: float, mean_treatment: float) -> float:
    """Relative lift (treatment vs control)."""
    if mean_control == 0:
        return 0.0
    return (mean_treatment - mean_control) / mean_control


def novelty_detect(
    snapshots: list[dict[str, Any]],
    novelty_window_days: int = 3,
    min_days_required: int = 7,
) -> dict[str, Any]:
    """
    Detect inflated treatment effect in the first novelty_window_days (novelty effect).

    snapshots: list of dicts with keys day, variant, primary_metric_value, sample_size.
    novelty_detected if novelty_ratio > 1.5 AND overall_lift > 0 AND days_analyzed >= min_days_required.
    """
    if not snapshots:
        return {
            "early_window_lift": 0.0,
            "overall_lift": 0.0,
            "novelty_ratio": 0.0,
            "novelty_detected": False,
            "days_analyzed": 0,
            "reason": "No snapshots provided.",
        }

    days_present = {r.get("day") for r in snapshots if r.get("day") is not None}
    days_present.discard(None)
    days_analyzed = len(days_present)
    max_day = max(days_present) if days_present else 0

    # Early window: days 1 through novelty_window_days
    early_mean_c, early_mean_t, early_days = _weighted_mean_by_variant(
        snapshots, day_min=1, day_max=min(novelty_window_days, max_day)
    )
    early_window_lift = _lift(early_mean_c, early_mean_t)

    # Overall: all days
    overall_mean_c, overall_mean_t, _ = _weighted_mean_by_variant(
        snapshots, day_min=1, day_max=max_day
    )
    overall_lift = _lift(overall_mean_c, overall_mean_t)

    # Novelty ratio: early / overall (avoid division by zero)
    if overall_lift != 0:
        novelty_ratio = early_window_lift / overall_lift
    else:
        novelty_ratio = 0.0

    novelty_detected = (
        novelty_ratio > 1.5
        and overall_lift > 0
        and days_analyzed >= min_days_required
    )

    if novelty_detected:
        reason = (
            "Novelty effect detected: early-window lift (%.4f) is %.2fx overall lift (%.4f)."
            % (early_window_lift, novelty_ratio, overall_lift)
        )
    elif days_analyzed < min_days_required:
        reason = "Insufficient days (have %d, need %d) to assess novelty." % (
            days_analyzed,
            min_days_required,
        )
    elif overall_lift <= 0:
        reason = "Overall lift is non-positive; novelty not applicable."
    elif novelty_ratio <= 1.5:
        reason = "Early-window lift (%.2fx overall) does not exceed novelty threshold (1.5)." % (
            novelty_ratio
        )
    else:
        reason = "Novelty check did not trigger."

    return {
        "early_window_lift": early_window_lift,
        "overall_lift": overall_lift,
        "novelty_ratio": novelty_ratio,
        "novelty_detected": novelty_detected,
        "days_analyzed": days_analyzed,
        "reason": reason,
    }


def validate() -> None:
    """
    Flat lift (no novelty), inflated early lift (novelty), negative overall (no novelty flagged).
    """
    # 1. Flat lift — same lift every day, no novelty
    flat_snapshots = []
    for day in range(1, 11):
        flat_snapshots.append(
            {"day": day, "variant": VARIANT_CONTROL, "primary_metric_value": 0.10, "sample_size": 1000}
        )
        flat_snapshots.append(
            {"day": day, "variant": VARIANT_TREATMENT, "primary_metric_value": 0.11, "sample_size": 1000}
        )
    res_flat = novelty_detect(flat_snapshots, novelty_window_days=3, min_days_required=7)
    assert not res_flat["novelty_detected"], "Flat lift should not be flagged as novelty"
    assert res_flat["days_analyzed"] == 10
    assert abs(res_flat["novelty_ratio"] - 1.0) < 0.01  # early ≈ overall

    # 2. Inflated early lift — novelty detected
    early_inflated = []
    for day in range(1, 11):
        # Early days 1–3: 50% lift; later days 4–10: 5% lift
        lift = 0.50 if day <= 3 else 0.05
        early_inflated.append(
            {"day": day, "variant": VARIANT_CONTROL, "primary_metric_value": 0.10, "sample_size": 1000}
        )
        early_inflated.append(
            {
                "day": day,
                "variant": VARIANT_TREATMENT,
                "primary_metric_value": 0.10 * (1 + lift),
                "sample_size": 1000,
            }
        )
    res_novelty = novelty_detect(early_inflated, novelty_window_days=3, min_days_required=7)
    assert res_novelty["novelty_detected"], "Inflated early lift should be flagged as novelty"
    assert res_novelty["novelty_ratio"] > 1.5
    assert res_novelty["overall_lift"] > 0

    # 3. Negative overall lift — do not flag novelty
    negative_snapshots = []
    for day in range(1, 9):
        negative_snapshots.append(
            {"day": day, "variant": VARIANT_CONTROL, "primary_metric_value": 0.10, "sample_size": 1000}
        )
        negative_snapshots.append(
            {"day": day, "variant": VARIANT_TREATMENT, "primary_metric_value": 0.08, "sample_size": 1000}
        )
    res_neg = novelty_detect(negative_snapshots, novelty_window_days=3, min_days_required=7)
    assert not res_neg["novelty_detected"], "Negative overall lift should not flag novelty"
    assert res_neg["overall_lift"] < 0

    # Edge: fewer days than novelty_window
    few_snapshots = [
        {"day": 1, "variant": VARIANT_CONTROL, "primary_metric_value": 0.10, "sample_size": 100},
        {"day": 1, "variant": VARIANT_TREATMENT, "primary_metric_value": 0.15, "sample_size": 100},
    ]
    res_few = novelty_detect(few_snapshots, novelty_window_days=3, min_days_required=7)
    assert res_few["days_analyzed"] == 1
    assert not res_few["novelty_detected"]  # insufficient days

    print(
        "novelty.validate() passed: flat (novelty=%s), inflated (novelty=%s), negative (novelty=%s)"
        % (res_flat["novelty_detected"], res_novelty["novelty_detected"], res_neg["novelty_detected"])
    )


if __name__ == "__main__":
    validate()

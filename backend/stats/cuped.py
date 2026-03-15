"""
CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

OLS-based covariate adjustment from scratch: regress metric on pre_exp_metric
to get theta, then adjust outcomes and run a two-sample t-test. Uses numpy for
OLS and scipy.stats only for t-test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Input: list of (user_id, variant, metric_value, pre_exp_metric)
# ---------------------------------------------------------------------------
VARIANT_CONTROL = "control"
VARIANT_TREATMENT = "treatment"


def _ols_slope(y: np.ndarray, x: np.ndarray) -> float:
    """
    OLS regression y = a + b*x. Returns b (slope) only.
    Uses normal equation: beta = (X'X)^{-1} X'y with X = [1, x].
    """
    n = len(y)
    if n < 2 or np.var(x) < 1e-12:
        return 0.0
    ones = np.ones(n)
    X = np.column_stack([ones, x])
    # beta = (X'X)^{-1} X' y
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return 0.0
    beta = XtX_inv @ (X.T @ y)
    return float(beta[1])  # slope = coefficient of x


def cuped(
    rows: list[tuple[str, str, float, float]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    CUPED variance reduction and two-sample t-test on adjusted metrics.

    rows: list of (user_id, variant, metric_value, pre_exp_metric)
    alpha: significance level for significant flag

    Returns dict with:
      theta, mean_control_adjusted, mean_treatment_adjusted, lift_adjusted,
      p_value, variance_reduction_pct, significant
    """
    if not rows:
        return {
            "theta": 0.0,
            "mean_control_adjusted": 0.0,
            "mean_treatment_adjusted": 0.0,
            "lift_adjusted": 0.0,
            "p_value": 1.0,
            "variance_reduction_pct": 0.0,
            "significant": False,
        }

    arr = np.array(rows, dtype=object)
    variant = arr[:, 1]
    metric_value = np.asarray(arr[:, 2], dtype=float)
    pre_exp_metric = np.asarray(arr[:, 3], dtype=float)

    # Theta: OLS coefficient of metric_value on pre_exp_metric (pooled)
    theta = _ols_slope(metric_value, pre_exp_metric)
    mean_pre = float(np.mean(pre_exp_metric))

    # Adjusted metric per user
    metric_adjusted = metric_value - theta * (pre_exp_metric - mean_pre)

    # Split by variant
    control_mask = variant == VARIANT_CONTROL
    treatment_mask = variant == VARIANT_TREATMENT
    control_adj = metric_adjusted[control_mask]
    treatment_adj = metric_adjusted[treatment_mask]

    if control_adj.size < 2 or treatment_adj.size < 2:
        return {
            "theta": theta,
            "mean_control_adjusted": float(np.mean(control_adj)) if control_adj.size else 0.0,
            "mean_treatment_adjusted": float(np.mean(treatment_adj)) if treatment_adj.size else 0.0,
            "lift_adjusted": 0.0,
            "p_value": 1.0,
            "variance_reduction_pct": 0.0,
            "significant": False,
        }

    mean_control_adj = float(np.mean(control_adj))
    mean_treatment_adj = float(np.mean(treatment_adj))

    # Two-sample t-test on adjusted metrics
    t_stat, p_value = stats.ttest_ind(treatment_adj, control_adj, equal_var=False)
    p_value = float(p_value) if np.isfinite(p_value) else 1.0

    # Relative lift (treatment vs control)
    if mean_control_adj != 0:
        lift_adjusted = (mean_treatment_adj - mean_control_adj) / abs(mean_control_adj)
    else:
        lift_adjusted = 0.0 if mean_treatment_adj == 0 else float("inf")

    # Variance reduction: (Var(Y) - Var(Y_adj)) / Var(Y) * 100 (pooled)
    var_raw = float(np.var(metric_value))
    var_adj = float(np.var(metric_adjusted))
    if var_raw > 1e-20:
        variance_reduction_pct = (var_raw - var_adj) / var_raw * 100.0
    else:
        variance_reduction_pct = 0.0

    return {
        "theta": theta,
        "mean_control_adjusted": mean_control_adj,
        "mean_treatment_adjusted": mean_treatment_adj,
        "lift_adjusted": lift_adjusted,
        "p_value": p_value,
        "variance_reduction_pct": variance_reduction_pct,
        "significant": p_value < alpha,
    }


def validate() -> None:
    """
    Generate synthetic data with known ground truth and assert CUPED output
    is sensible: positive lift when treatment has true effect, theta nonzero
    when covariate is correlated with outcome.
    """
    rng = np.random.default_rng(42)
    n = 5000
    # Pre-exp covariate (e.g. feature usage count)
    pre_exp = rng.poisson(2.0, size=n).astype(float)
    # Baseline metric correlated with pre_exp (so CUPED can reduce variance)
    noise = rng.normal(0, 0.5, size=n)
    base_metric = 0.15 * pre_exp + noise
    base_metric = np.maximum(base_metric, 0.01)

    # True lift for treatment = 15%
    n_control = n // 2
    n_treatment = n - n_control
    control_metric = base_metric[:n_control].copy()
    treatment_metric = base_metric[n_control:] * 1.15  # 15% lift

    rows = []
    for i in range(n_control):
        rows.append((f"c_{i}", VARIANT_CONTROL, float(control_metric[i]), float(pre_exp[i])))
    for i in range(n_treatment):
        rows.append((
            f"t_{i}",
            VARIANT_TREATMENT,
            float(treatment_metric[i]),
            float(pre_exp[n_control + i]),
        ))
    rng.shuffle(rows)

    result = cuped(rows, alpha=0.05)

    # With 15% true lift and n=5000, we expect significant result and positive lift
    assert result["lift_adjusted"] > 0, "Expected positive lift when treatment has true effect"
    assert result["significant"], "Expected significant result with 15%% lift and n=5000"
    assert result["theta"] != 0, "Expected nonzero theta when metric is correlated with pre_exp"
    assert result["variance_reduction_pct"] >= 0, "Variance reduction should be non-negative"
    assert 0 <= result["p_value"] <= 1, "p_value must be in [0, 1]"
    assert result["mean_treatment_adjusted"] > result["mean_control_adjusted"], "Treatment mean should exceed control"

    print("cuped.validate() passed: lift_adjusted=%.4f, p_value=%.4f, theta=%.4f, var_reduction=%.2f%%" % (
        result["lift_adjusted"],
        result["p_value"],
        result["theta"],
        result["variance_reduction_pct"],
    ))


if __name__ == "__main__":
    validate()

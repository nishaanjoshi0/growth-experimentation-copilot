"""
O'Brien-Fleming alpha spending boundaries for sequential testing.

Boundary and z-statistic computed manually with numpy; scipy.stats.norm used only for CDF.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def obrien_fleming(
    day: int,
    total_days: int,
    current_mean_control: float,
    current_mean_treatment: float,
    current_std_control: float,
    current_std_treatment: float,
    n_control: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute O'Brien-Fleming boundary for the current look and whether to recommend stop.

    recommend_stop is True only if boundary_crossed and day >= total_days * 0.5.
    """
    if total_days <= 0 or day <= 0:
        return _result(
            day=day,
            total_days=total_days,
            information_fraction=0.0,
            z_boundary=0.0,
            z_stat=0.0,
            alpha_spent=0.0,
            boundary_crossed=False,
            recommend_stop=False,
            reason="Invalid day or total_days",
        )

    t = day / total_days
    t = min(t, 1.0)
    information_fraction = t

    # O'Brien-Fleming: constant C such that at t=1, alpha_spent = alpha
    # alpha_spent(t) = 2 * (1 - norm.cdf(C / sqrt(t)))
    # At t=1: alpha = 2*(1 - norm.cdf(C)) => C = norm.ppf(1 - alpha/2)
    C = float(norm.ppf(1.0 - alpha / 2.0))
    z_boundary_at_t = C / np.sqrt(t)
    alpha_spent = float(2.0 * (1.0 - norm.cdf(C / np.sqrt(t))))

    # Two-sample z-statistic: (mean_T - mean_C) / SE(diff)
    # SE(diff) = sqrt(var_C/n_C + var_T/n_T)
    se_diff = np.sqrt(
        (current_std_control ** 2) / n_control + (current_std_treatment ** 2) / n_treatment
    )
    if se_diff <= 0:
        z_stat = 0.0
    else:
        z_stat = float(
            (current_mean_treatment - current_mean_control) / se_diff
        )

    boundary_crossed = z_stat >= z_boundary_at_t
    min_day_to_stop = total_days * 0.5
    recommend_stop = boundary_crossed and (day >= min_day_to_stop)

    if recommend_stop:
        reason = "Boundary crossed and past first half of experiment; recommend stop."
    elif boundary_crossed:
        reason = "Boundary crossed at day %d but early stop not allowed in first half (day < %.0f)." % (
            day,
            min_day_to_stop,
        )
    else:
        reason = "Boundary not crossed; continue experiment."

    return _result(
        day=day,
        total_days=total_days,
        information_fraction=information_fraction,
        z_boundary=float(z_boundary_at_t),
        z_stat=z_stat,
        alpha_spent=alpha_spent,
        boundary_crossed=boundary_crossed,
        recommend_stop=recommend_stop,
        reason=reason,
    )


def _result(
    day: int,
    total_days: int,
    information_fraction: float,
    z_boundary: float,
    z_stat: float,
    alpha_spent: float,
    boundary_crossed: bool,
    recommend_stop: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "day": day,
        "total_days": total_days,
        "information_fraction": information_fraction,
        "z_boundary": z_boundary,
        "z_stat": z_stat,
        "alpha_spent": alpha_spent,
        "boundary_crossed": boundary_crossed,
        "recommend_stop": recommend_stop,
        "reason": reason,
    }


def validate() -> None:
    """
    Early day + large effect: do not recommend stop (first half).
    Late day + large effect: recommend stop.
    Null effect: never recommend stop.
    """
    total_days = 30
    n_c = n_t = 10_000
    std_c = std_t = 1.0

    # 1. Early day, large effect — boundary may be crossed but recommend_stop False
    mean_c_early, mean_t_early = 0.0, 0.5  # large effect
    res_early = obrien_fleming(
        day=5,
        total_days=total_days,
        current_mean_control=mean_c_early,
        current_mean_treatment=mean_t_early,
        current_std_control=std_c,
        current_std_treatment=std_t,
        n_control=n_c,
        n_treatment=n_t,
        alpha=0.05,
    )
    assert not res_early["recommend_stop"], (
        "Early day (5) should not recommend stop even with large effect (first half)"
    )
    assert res_early["day"] == 5 and res_early["total_days"] == 30

    # 2. Late day, large effect — recommend stop
    res_late = obrien_fleming(
        day=25,
        total_days=total_days,
        current_mean_control=0.0,
        current_mean_treatment=0.4,
        current_std_control=std_c,
        current_std_treatment=std_t,
        n_control=n_c,
        n_treatment=n_t,
        alpha=0.05,
    )
    assert res_late["boundary_crossed"], "Late day with large effect should cross boundary"
    assert res_late["recommend_stop"], "Late day with large effect should recommend stop"

    # 3. Null effect — never recommend stop
    res_null = obrien_fleming(
        day=30,
        total_days=total_days,
        current_mean_control=0.1,
        current_mean_treatment=0.1,
        current_std_control=std_c,
        current_std_treatment=std_t,
        n_control=n_c,
        n_treatment=n_t,
        alpha=0.05,
    )
    assert not res_null["boundary_crossed"], "Null effect should not cross boundary"
    assert not res_null["recommend_stop"], "Null effect should not recommend stop"

    print(
        "sequential.validate() passed: early (recommend_stop=%s), late (recommend_stop=%s), null (boundary_crossed=%s)"
        % (res_early["recommend_stop"], res_late["recommend_stop"], res_null["boundary_crossed"])
    )


if __name__ == "__main__":
    validate()

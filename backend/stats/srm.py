"""
SRM (Sample Ratio Mismatch) detection from scratch.

Chi-square goodness-of-fit test using numpy for the statistic;
scipy.stats.chi2.sf only for the p-value. No scipy.stats.chisquare.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2


def srm_detect(
    observed_control: int,
    observed_treatment: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01,
) -> dict[str, Any]:
    """
    Test whether observed assignment counts match the expected ratio (e.g. 50/50).

    expected_ratio: proportion of users expected in control (e.g. 0.5 for 50/50).
    alpha: threshold for srm_detected (p_value < alpha -> SRM).
    """
    total = observed_control + observed_treatment
    if total == 0:
        return {
            "observed_control": 0,
            "observed_treatment": 0,
            "expected_control": 0.0,
            "expected_treatment": 0.0,
            "chi_square_stat": 0.0,
            "p_value": 1.0,
            "srm_detected": False,
            "severity": "none",
        }

    expected_control = total * expected_ratio
    expected_treatment = total * (1.0 - expected_ratio)

    # Chi-square goodness-of-fit: sum (observed - expected)^2 / expected
    # Avoid division by zero (expected_* can be 0 if ratio is 0 or 1)
    term_c = (
        (observed_control - expected_control) ** 2 / expected_control
        if expected_control > 0
        else 0.0
    )
    term_t = (
        (observed_treatment - expected_treatment) ** 2 / expected_treatment
        if expected_treatment > 0
        else 0.0
    )
    chi_square_stat = float(term_c + term_t)
    # df = 2 categories - 1 = 1
    p_value = float(chi2.sf(chi_square_stat, df=1))

    srm_detected = p_value < alpha
    if p_value < 0.001:
        severity = "severe"
    elif p_value < 0.01:
        severity = "mild"
    else:
        severity = "none"

    return {
        "observed_control": observed_control,
        "observed_treatment": observed_treatment,
        "expected_control": expected_control,
        "expected_treatment": expected_treatment,
        "chi_square_stat": chi_square_stat,
        "p_value": p_value,
        "srm_detected": srm_detected,
        "severity": severity,
    }


def validate() -> None:
    """
    Test three cases: clean 50/50 (no SRM), mild imbalance, severe imbalance.
    Asserts correct srm_detected and severity for each.
    """
    # 1. Clean 50/50 — no SRM, severity 'none'
    total = 50_000
    observed_control, observed_treatment = total // 2, total // 2
    result_clean = srm_detect(observed_control, observed_treatment, expected_ratio=0.5, alpha=0.01)
    assert not result_clean["srm_detected"], "Clean 50/50 should not be detected as SRM"
    assert result_clean["severity"] == "none", "Clean split should have severity 'none'"
    assert result_clean["expected_control"] == result_clean["observed_control"]
    assert result_clean["expected_treatment"] == result_clean["observed_treatment"]
    assert result_clean["chi_square_stat"] == 0.0
    assert result_clean["p_value"] == 1.0

    # 2. Mild imbalance — p in (0.001, 0.01) -> srm_detected True (with alpha=0.01), severity 'mild'
    # Need chi2 such that 0.001 < p < 0.01  =>  chi2 in (approx 6.63, 10.83). Use 5130/4870 with n=10000.
    obs_c_mild, obs_t_mild = 5130, 4870
    result_mild = srm_detect(obs_c_mild, obs_t_mild, expected_ratio=0.5, alpha=0.01)
    assert result_mild["srm_detected"], "Mild imbalance should be detected as SRM (p < 0.01)"
    assert result_mild["severity"] == "mild", "p in (0.001, 0.01) should be severity 'mild'"
    assert 0.001 <= result_mild["p_value"] < 0.01

    # 3. Severe imbalance — p < 0.001 -> severity 'severe'
    obs_c_severe, obs_t_severe = 5200, 4800  # chi2 = 16 for n=10000
    result_severe = srm_detect(obs_c_severe, obs_t_severe, expected_ratio=0.5, alpha=0.01)
    assert result_severe["srm_detected"], "Severe imbalance should be detected as SRM"
    assert result_severe["severity"] == "severe", "p < 0.001 should be severity 'severe'"
    assert result_severe["p_value"] < 0.001

    print(
        "srm.validate() passed: clean (p=%.4f), mild (p=%.4f, %s), severe (p=%.4f, %s)"
        % (
            result_clean["p_value"],
            result_mild["p_value"],
            result_mild["severity"],
            result_severe["p_value"],
            result_severe["severity"],
        )
    )


if __name__ == "__main__":
    validate()

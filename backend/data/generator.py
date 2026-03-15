"""
Synthetic data generator for B2C SaaS A/B experimentation.

Produces ~50K users, 60 days of events (30 pre-experiment + 30 experiment),
with configurable ground truth: true_lift, inject_srm, inject_novelty, srm_start_day.
All randomness is seedable for validation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Event types (README spec)
# ---------------------------------------------------------------------------
EVENT_SIGNUP = "signup"
EVENT_ONBOARDING_COMPLETE = "onboarding_complete"
EVENT_FEATURE_USED = "feature_used"
EVENT_SUBSCRIPTION_STARTED = "subscription_started"
EVENT_SUBSCRIPTION_CANCELLED = "subscription_cancelled"
EVENT_REFERRED_USER = "referred_user"

EVENT_TYPES = (
    EVENT_SIGNUP,
    EVENT_ONBOARDING_COMPLETE,
    EVENT_FEATURE_USED,
    EVENT_SUBSCRIPTION_STARTED,
    EVENT_SUBSCRIPTION_CANCELLED,
    EVENT_REFERRED_USER,
)

VARIANT_CONTROL = "control"
VARIANT_TREATMENT = "treatment"


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    """Configurable parameters for synthetic data generation."""

    n_users: int = 50_000
    n_days_pre: int = 30
    n_days_exp: int = 30
    n_days_total: int = 60  # n_days_pre + n_days_exp

    # Ground truth
    true_lift: float = 0.05  # 5% relative lift in treatment (e.g. conversion)
    base_conversion_rate: float = 0.12  # control conversion rate
    churn_rate: float = 0.05  # monthly subscription cancellation rate

    # SRM injection: bias assignment ratio from srm_start_day (experiment day 1-based)
    inject_srm: bool = False
    srm_start_day: int = 1  # experiment day when biased ratio begins (1..n_days_exp)
    srm_control_ratio: float = 0.60  # e.g. 60% control / 40% treatment when SRM active

    # Novelty: treatment effect inflated in first N days of experiment
    inject_novelty: bool = False
    novelty_window_days: int = 3
    novelty_lift_multiplier: float = 1.5  # treatment lift in novelty window = true_lift * this

    # Assignment (when not injecting SRM)
    assignment_control_ratio: float = 0.50  # 50/50

    random_seed: int | None = 42

    def __post_init__(self) -> None:
        self.n_days_total = self.n_days_pre + self.n_days_exp


# ---------------------------------------------------------------------------
# Output structures (align with README DB / consumer expectations)
# ---------------------------------------------------------------------------
@dataclass
class UserAssignment:
    """One row for user_assignments table. variant is always true assignment (SRM bias only in events)."""

    user_id: str
    variant: str  # true assignment (50/50); SRM drift appears per-day in events only
    pre_exp_metric: float  # continuous: feature-usage count in pre-exp (CUPED covariate)
    assigned_at: datetime


@dataclass
class EventRow:
    """One event for the stream; variant is observed (for SRM simulation)."""

    user_id: str
    event_type: str
    day: int  # 0..n_days_total-1
    variant: str  # control | treatment (observed)
    ts: datetime
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def _day_to_ts(day: int, config: GeneratorConfig) -> datetime:
    """Convert day index (0-based) to a timestamp in the period."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return base + timedelta(days=day)


def _assign_variant_true(config: GeneratorConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """True assignment 50/50 (or config.assignment_control_ratio)."""
    return np.where(
        rng.random(n) < config.assignment_control_ratio,
        VARIANT_CONTROL,
        VARIANT_TREATMENT,
    )


def _observed_variant(
    variant_true: np.ndarray,
    config: GeneratorConfig,
    exp_day_1based: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Observed variant: biased from srm_start_day when inject_srm."""
    out = variant_true.copy()
    if not config.inject_srm or exp_day_1based < config.srm_start_day:
        return out
    # From srm_start_day we want ratio srm_control_ratio / (1 - srm_control_ratio)
    # Flip some treatment -> control to achieve that
    n = len(variant_true)
    n_control = int(n * config.srm_control_ratio)
    idx_treatment = np.where(variant_true == VARIANT_TREATMENT)[0]
    idx_control = np.where(variant_true == VARIANT_CONTROL)[0]
    n_control_true = len(idx_control)
    n_flip = n_control - n_control_true
    if n_flip > 0 and n_flip <= len(idx_treatment):
        flip_idx = rng.choice(idx_treatment, size=n_flip, replace=False)
        out[flip_idx] = VARIANT_CONTROL
    return out


def _conversion_rate(
    config: GeneratorConfig,
    variant: str,
    exp_day_1based: int,
) -> float:
    """Conversion rate for a variant on a given experiment day (1-based)."""
    base = config.base_conversion_rate
    if variant == VARIANT_CONTROL:
        return base
    # Treatment: apply lift; optionally novelty in first novelty_window_days
    lift = config.true_lift
    if config.inject_novelty and 1 <= exp_day_1based <= config.novelty_window_days:
        lift = config.true_lift * config.novelty_lift_multiplier
    return base * (1.0 + lift)


def generate_user_pool(config: GeneratorConfig) -> list[str]:
    """Generate stable user IDs."""
    return [f"user_{i}" for i in range(config.n_users)]


def generate_pre_experiment_behavior(
    user_ids: list[str],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[list[EventRow], np.ndarray]:
    """
    Generate pre-experiment events (days 0..n_days_pre-1) and pre_exp_metric per user.
    pre_exp_metric = feature-usage count in pre-exp (continuous, good CUPED covariate).
    """
    n = len(user_ids)
    events: list[EventRow] = []
    # Signup spread over first 14 days of pre-exp
    signup_day = rng.integers(0, min(14, config.n_days_pre), size=n)
    # Onboarding within 2 days of signup
    onboarding_day = signup_day + rng.integers(0, 3, size=n)
    onboarding_day = np.minimum(onboarding_day, config.n_days_pre - 1)
    # Pre-exp conversion: some convert before experiment (for events only)
    pre_convert = rng.random(n) < (config.base_conversion_rate * 0.8)
    pre_convert_day = np.where(
        pre_convert,
        rng.integers(5, config.n_days_pre, size=n),
        config.n_days_pre,
    )
    # Feature-usage count per user in pre-exp (continuous covariate for CUPED)
    n_feature_per_user = rng.poisson(2.0, size=n).astype(np.float64)
    pre_exp_metric = n_feature_per_user.copy()

    for i in range(n):
        uid = user_ids[i]
        sd = int(signup_day[i])
        od = int(onboarding_day[i])
        events.append(EventRow(uid, EVENT_SIGNUP, sd, VARIANT_CONTROL, _day_to_ts(sd, config)))
        if od < config.n_days_pre:
            events.append(
                EventRow(uid, EVENT_ONBOARDING_COMPLETE, od, VARIANT_CONTROL, _day_to_ts(od, config))
            )
        # Emit feature_used events: n_feature_per_user[i] events on random pre-exp days
        n_f = int(n_feature_per_user[i])
        if n_f > 0:
            days_f = rng.integers(0, config.n_days_pre, size=n_f)
            for d in days_f.tolist():
                events.append(
                    EventRow(uid, EVENT_FEATURE_USED, int(d), VARIANT_CONTROL, _day_to_ts(int(d), config))
                )
        if pre_convert[i] and pre_convert_day[i] < config.n_days_pre:
            events.append(
                EventRow(
                    uid,
                    EVENT_SUBSCRIPTION_STARTED,
                    int(pre_convert_day[i]),
                    VARIANT_CONTROL,
                    _day_to_ts(int(pre_convert_day[i]), config),
                )
            )
    return events, pre_exp_metric


def generate_assignments(
    user_ids: list[str],
    pre_exp_metric: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[list[UserAssignment], np.ndarray]:
    """
    Assign variant (true 50/50). user_assignments always stores true variant.
    SRM drift is applied per-day in the event stream only (clean 50/50 before
    srm_start_day, biased ratio from srm_start_day onward).
    """
    n = len(user_ids)
    variant_true = _assign_variant_true(config, n, rng)
    assigned_at = _day_to_ts(config.n_days_pre, config)

    assignments = [
        UserAssignment(
            user_id=uid,
            variant=variant_true[i],
            pre_exp_metric=float(pre_exp_metric[i]),
            assigned_at=assigned_at,
        )
        for i, uid in enumerate(user_ids)
    ]
    return assignments, variant_true


def generate_experiment_events(
    user_ids: list[str],
    variant_true: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[EventRow]:
    """
    Generate events for experiment days (vectorized). SRM drift: observed variant
    is 50/50 before srm_start_day, biased from srm_start_day onward.
    Novelty lift only applied when config.inject_novelty is True.
    """
    n = len(user_ids)
    n_days = config.n_days_exp
    # Observed variant per day (n_days, n) — drift-based SRM
    observed = np.array(
        [_observed_variant(variant_true, config, d + 1, rng) for d in range(n_days)],
        dtype=object,
    )
    # (n_days, n) — variant_true broadcast
    vt = np.broadcast_to(variant_true, (n_days, n))
    # Conversion rate per (day, user): treatment gets lift; novelty only when inject_novelty
    if config.inject_novelty:
        lift_per_day = np.where(
            np.arange(n_days)[:, None] + 1 <= config.novelty_window_days,
            config.true_lift * config.novelty_lift_multiplier,
            config.true_lift,
        )
    else:
        lift_per_day = np.full((n_days, 1), config.true_lift)
    base = config.base_conversion_rate
    conv_rates = np.where(
        vt == VARIANT_TREATMENT,
        base * (1.0 + lift_per_day),
        base,
    )
    conv_rates *= 0.15  # daily conversion probability scale

    converted = rng.random((n_days, n)) < conv_rates
    feature_used = rng.random((n_days, n)) < 0.2
    churn = rng.random((n_days, n)) < (config.churn_rate / 30)
    ref_rate_treatment = 0.02 * (1.0 + config.true_lift)
    ref_rates = np.where(vt == VARIANT_TREATMENT, ref_rate_treatment, 0.02)
    referred = rng.random((n_days, n)) < ref_rates

    events: list[EventRow] = []
    user_ids_arr = np.array(user_ids)

    def append_events(mask: np.ndarray, event_type: str, payload: dict[str, Any] | None = None) -> None:
        day_idx, user_idx = np.where(mask)
        for k in range(len(day_idx)):
            d, i = int(day_idx[k]), int(user_idx[k])
            abs_day = config.n_days_pre + d
            v_obs = observed[d][i] if isinstance(observed[d], np.ndarray) else observed[d]
            ev = EventRow(
                user_ids_arr[i],
                event_type,
                abs_day,
                v_obs,
                _day_to_ts(abs_day, config),
                payload or {},
            )
            if event_type == EVENT_REFERRED_USER:
                ev.payload["referred_user_id"] = f"ref_{uuid.uuid4().hex[:12]}"
            events.append(ev)

    append_events(converted, EVENT_SUBSCRIPTION_STARTED)
    append_events(feature_used, EVENT_FEATURE_USED)
    append_events(churn, EVENT_SUBSCRIPTION_CANCELLED)
    append_events(referred, EVENT_REFERRED_USER)

    return events


def generate(config: GeneratorConfig | None = None) -> tuple[list[UserAssignment], list[EventRow]]:
    """
    Full pipeline: user pool, pre-exp behavior, assignments, experiment events.
    All randomness uses a single np.random.default_rng(seed) for reproducibility.

    Returns:
        user_assignments: list of UserAssignment (variant = true assignment)
        events: all events (pre-exp + experiment); variant on events is observed (SRM drift per-day)
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.random_seed)

    user_ids = generate_user_pool(cfg)
    pre_events, pre_exp_metric = generate_pre_experiment_behavior(user_ids, cfg, rng)
    assignments, variant_true = generate_assignments(user_ids, pre_exp_metric, cfg, rng)
    exp_events = generate_experiment_events(user_ids, variant_true, cfg, rng)

    all_events = pre_events + exp_events
    return assignments, all_events


# ---------------------------------------------------------------------------
# Dict/JSON-friendly output for DB and APIs
# ---------------------------------------------------------------------------
def assignments_to_dicts(assignments: list[UserAssignment]) -> list[dict[str, Any]]:
    """For Supabase user_assignments insert (variant = true assignment)."""
    return [
        {
            "user_id": a.user_id,
            "variant": a.variant,
            "pre_exp_metric": a.pre_exp_metric,
            "assigned_at": a.assigned_at.isoformat(),
        }
        for a in assignments
    ]


def events_to_dicts(events: list[EventRow]) -> list[dict[str, Any]]:
    """For streaming or bulk insert."""
    return [
        {
            "user_id": e.user_id,
            "event_type": e.event_type,
            "day": e.day,
            "variant": e.variant,
            "ts": e.ts.isoformat(),
            **e.payload,
        }
        for e in events
    ]


# ---------------------------------------------------------------------------
# Daily metric snapshots (for metric_snapshots table)
# ---------------------------------------------------------------------------
@dataclass
class MetricSnapshot:
    """One row for metric_snapshots table."""

    day: int  # experiment day (1-based)
    variant: str
    primary_metric_value: float  # e.g. conversion rate or count for that day
    guardrail_values: dict[str, float]  # e.g. churn_count, referral_count
    sample_size: int
    srm_flagged: bool = False
    novelty_flagged: bool = False
    sequential_boundary: float | None = None


def aggregate_events_to_snapshots(
    events: list[EventRow],
    config: GeneratorConfig,
    *,
    experiment_day_offset: int = 0,
) -> list[MetricSnapshot]:
    """
    Aggregate experiment-period events into daily metric snapshots per variant.
    Only includes days in [1, n_days_exp]; events with day < n_days_pre are ignored.
    """
    from collections import defaultdict

    # (day_1based, variant) -> counts
    primary_count: dict[tuple[int, str], int] = defaultdict(int)
    guardrail_counts: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sample_sizes: dict[tuple[int, str], set[str]] = defaultdict(set)

    for e in events:
        if e.day < config.n_days_pre:
            continue
        exp_day_1based = e.day - config.n_days_pre + 1
        if exp_day_1based < 1 or exp_day_1based > config.n_days_exp:
            continue
        key = (exp_day_1based + experiment_day_offset, e.variant)
        sample_sizes[key].add(e.user_id)
        if e.event_type == EVENT_SUBSCRIPTION_STARTED:
            primary_count[key] += 1
        elif e.event_type == EVENT_SUBSCRIPTION_CANCELLED:
            guardrail_counts[key]["churn_count"] += 1
        elif e.event_type == EVENT_REFERRED_USER:
            guardrail_counts[key]["referral_count"] += 1

    snapshots: list[MetricSnapshot] = []
    for (day, variant), users in sample_sizes.items():
        n = len(users)
        conv = primary_count.get((day, variant), 0)
        rate = conv / n if n else 0.0
        guard = dict(guardrail_counts.get((day, variant), {}))
        snapshots.append(
            MetricSnapshot(
                day=day,
                variant=variant,
                primary_metric_value=rate,
                guardrail_values=guard,
                sample_size=n,
            )
        )
    return snapshots


def write_to_supabase(
    experiment_id: str,
    assignments: list[UserAssignment],
    events: list[EventRow],
    config: GeneratorConfig,
    *,
    supabase_client: Any = None,
    write_snapshots: bool = True,
) -> None:
    """
    Insert user_assignments and (optionally) daily metric_snapshots into Supabase.
    If supabase_client is None, one is created from env SUPABASE_URL, SUPABASE_KEY.
    """
    if supabase_client is None:
        import os
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) must be set")
        supabase_client = create_client(url, key)

    # user_assignments: need experiment_id on each row (assigned_at already ISO from assignments_to_dicts)
    rows = assignments_to_dicts(assignments)
    for r in rows:
        r["experiment_id"] = experiment_id
    supabase_client.table("user_assignments").insert(rows).execute()

    if write_snapshots:
        snapshots = aggregate_events_to_snapshots(events, config)
        snapshot_rows = []
        for s in snapshots:
            snapshot_rows.append({
                "experiment_id": experiment_id,
                "day": s.day,
                "variant": s.variant,
                "primary_metric_value": s.primary_metric_value,
                "guardrail_values": s.guardrail_values,
                "sample_size": s.sample_size,
                "srm_flagged": s.srm_flagged,
                "novelty_flagged": s.novelty_flagged,
                "sequential_boundary": s.sequential_boundary,
            })
        if snapshot_rows:
            supabase_client.table("metric_snapshots").insert(snapshot_rows).execute()


# ---------------------------------------------------------------------------
# CLI for one-off generation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = GeneratorConfig(
        n_users=50_000,
        true_lift=0.05,
        inject_srm=True,
        srm_start_day=5,
        inject_novelty=True,
        novelty_window_days=3,
        random_seed=42,
    )
    assignments, events = generate(cfg)
    print(f"Generated {len(assignments)} user assignments, {len(events)} events")
    v_counts = {}
    for a in assignments:
        v_counts[a.variant] = v_counts.get(a.variant, 0) + 1
    print("Assignment ratio (true, 50/50):", v_counts)
    pre_exp = [a.pre_exp_metric for a in assignments[:1000]]
    print("pre_exp_metric sample (continuous): min=%s max=%s mean=%s" % (min(pre_exp), max(pre_exp), sum(pre_exp)/len(pre_exp)))
    snapshots = aggregate_events_to_snapshots(events, cfg)
    print("Snapshots (by day/variant):", len(snapshots))
    print("Sample assignment:", assignments_to_dicts(assignments[:2]))
    print("Sample events:", events_to_dicts(events[:5]))

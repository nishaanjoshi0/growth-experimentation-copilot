"""
Supabase client for the growth experimentation copilot.

Singleton client initialized from SUPABASE_URL and SUPABASE_KEY in .env.
All functions handle errors with try/except and log meaningful messages.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Load .env from project root (cwd or parents)
load_dotenv()

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: Any = None


def get_supabase_client():
    """Return the singleton Supabase client; create from env if not yet initialized."""
    global _client
    if _client is not None:
        return _client
    try:
        from supabase import create_client
    except ImportError as e:
        print(f"[supabase_client] Missing dependency: {e}. Install with: pip install supabase")
        raise
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) must be set in .env"
        )
    _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def insert_experiment(experiment: dict[str, Any]) -> str:
    """
    Insert one row into experiments table. Returns the new experiment id (uuid).
    experiment dict should include: hypothesis, primary_metric, guardrail_metrics,
    randomization_unit, and optionally sample_size_required, runtime_days, status, design_output.
    """
    try:
        client = get_supabase_client()
        # Supabase returns { data: [row], error: None }
        result = client.table("experiments").insert(experiment).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]["id"]
        print("[supabase_client] insert_experiment: no data returned")
        raise RuntimeError("insert_experiment returned no row")
    except Exception as e:
        print(f"[supabase_client] insert_experiment failed: {e}")
        raise


def get_experiment(experiment_id: str) -> dict[str, Any] | None:
    """Return single experiment row by id, or None if not found."""
    try:
        client = get_supabase_client()
        result = (
            client.table("experiments")
            .select("*")
            .eq("id", experiment_id)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as e:
        print(f"[supabase_client] get_experiment failed: {e}")
        raise


def update_experiment_status(
    experiment_id: str,
    status: str,
    design_output: dict[str, Any] | None = None,
) -> None:
    """Update experiments set status and optionally design_output for the given experiment_id."""
    try:
        client = get_supabase_client()
        payload: dict[str, Any] = {"status": status}
        if design_output is not None:
            payload["design_output"] = design_output
        client.table("experiments").update(payload).eq("id", experiment_id).execute()
    except Exception as e:
        print(f"[supabase_client] update_experiment_status failed: {e}")
        raise


# ---------------------------------------------------------------------------
# User assignments (batched)
# ---------------------------------------------------------------------------
CHUNK_SIZE_ASSIGNMENTS = 1000


def insert_assignments(experiment_id: str, assignments: list[dict[str, Any]]) -> None:
    """Insert user_assignments in chunks of 1000. Each row must include user_id, variant, pre_exp_metric, assigned_at."""
    if not assignments:
        return
    try:
        client = get_supabase_client()
        for i in range(0, len(assignments), CHUNK_SIZE_ASSIGNMENTS):
            chunk = assignments[i : i + CHUNK_SIZE_ASSIGNMENTS]
            rows = [{**row, "experiment_id": experiment_id} for row in chunk]
            client.table("user_assignments").insert(rows).execute()
    except Exception as e:
        print(f"[supabase_client] insert_assignments failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Metric snapshots (batched)
# ---------------------------------------------------------------------------
CHUNK_SIZE_SNAPSHOTS = 500


def insert_snapshots(experiment_id: str, snapshots: list[dict[str, Any]]) -> None:
    """Insert metric_snapshots in chunks of 500. Each row: day, variant, primary_metric_value, guardrail_values, sample_size, etc."""
    if not snapshots:
        return
    try:
        client = get_supabase_client()
        for i in range(0, len(snapshots), CHUNK_SIZE_SNAPSHOTS):
            chunk = snapshots[i : i + CHUNK_SIZE_SNAPSHOTS]
            rows = [{**row, "experiment_id": experiment_id} for row in chunk]
            client.table("metric_snapshots").insert(rows).execute()
    except Exception as e:
        print(f"[supabase_client] insert_snapshots failed: {e}")
        raise


def get_snapshots(experiment_id: str) -> list[dict[str, Any]]:
    """Return all metric_snapshots for the experiment, ordered by day asc."""
    try:
        client = get_supabase_client()
        result = (
            client.table("metric_snapshots")
            .select("*")
            .eq("experiment_id", experiment_id)
            .order("day", desc=False)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[supabase_client] get_snapshots failed: {e}")
        raise


def update_metric_snapshot_flags(
    experiment_id: str,
    day: int,
    srm_flagged: bool,
    novelty_flagged: bool,
) -> None:
    """Update srm_flagged and novelty_flagged for a specific day's snapshot."""
    try:
        client = get_supabase_client()
        client.table("metric_snapshots").update({
            "srm_flagged": srm_flagged,
            "novelty_flagged": novelty_flagged,
        }).eq("experiment_id", experiment_id).eq("day", day).execute()
    except Exception as e:
        print(f"[supabase_client] update_metric_snapshot_flags failed: {e}")
        raise


# ---------------------------------------------------------------------------
# User assignments (read)
# ---------------------------------------------------------------------------
def get_assignments(experiment_id: str) -> list[dict[str, Any]]:
    """Return all user_assignments for the experiment."""
    try:
        client = get_supabase_client()
        result = (
            client.table("user_assignments")
            .select("*")
            .eq("experiment_id", experiment_id)
            .limit(100000)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[supabase_client] get_assignments failed: {e}")
        raise


# ---------------------------------------------------------------------------
# Agent decisions
# ---------------------------------------------------------------------------
def get_agent_decisions(experiment_id: str) -> list[dict[str, Any]]:
    """Return all agent_decisions for the experiment ordered by created_at asc."""
    try:
        client = get_supabase_client()
        result = (
            client.table("agent_decisions")
            .select("*")
            .eq("experiment_id", experiment_id)
            .order("created_at", desc=False)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"[supabase_client] get_agent_decisions failed: {e}")
        raise


def log_agent_decision(
    experiment_id: str,
    agent: str,
    decision: str,
    reasoning: str | None = None,
) -> None:
    """Insert one row into agent_decisions."""
    try:
        client = get_supabase_client()
        row = {
            "experiment_id": experiment_id,
            "agent": agent,
            "decision": decision,
            "reasoning": reasoning,
        }
        client.table("agent_decisions").insert(row).execute()
    except Exception as e:
        print(f"[supabase_client] log_agent_decision failed: {e}")
        raise

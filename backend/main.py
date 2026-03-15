"""
FastAPI backend for the growth experimentation copilot.
"""

from __future__ import annotations

from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from backend.agents.designer import run_designer, ExperimentDesign, ClarificationResult
from backend.agents.interpreter import run_interpreter, InterpretationResult
from backend.agents.monitor import run_monitor, MonitorResult
from backend.data.generator import (
    GeneratorConfig,
    generate,
    assignments_to_dicts,
    aggregate_events_to_snapshots,
)
from backend.db.supabase_client import (
    get_agent_decisions,
    get_experiment,
    get_snapshots,
    insert_assignments,
    insert_experiment,
    insert_snapshots,
)


app = FastAPI(title="Growth Experimentation Copilot API")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class DesignRequest(BaseModel):
    hypothesis: str = Field(..., description="Natural language hypothesis")
    clarification_response: Optional[str] = Field(None, description="Response to a prior clarification question")


class MonitorRequest(BaseModel):
    experiment_id: str
    current_day: int = Field(..., ge=1)
    config: dict[str, Any] = Field(default_factory=dict, description="GeneratorConfig-like dict (e.g. n_days_exp)")


class InterpretRequest(BaseModel):
    experiment_id: str
    hypothesis: str
    design: dict[str, Any] = Field(..., description="primary_metric, guardrail_metrics, runtime_days")


class SetupRequest(BaseModel):
    hypothesis: str
    clarification_response: Optional[str] = None
    n_users: int = Field(50_000, ge=100)
    true_lift: float = Field(0.05, ge=0.0)
    inject_srm: bool = False
    inject_novelty: bool = False


# ---------------------------------------------------------------------------
# POST /design
# ---------------------------------------------------------------------------
@app.post("/design")
def design(request: DesignRequest) -> dict[str, Any]:
    """Run designer; returns design JSON or clarification question."""
    try:
        out = run_designer(request.hypothesis, request.clarification_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if isinstance(out, ClarificationResult):
        return {"needs_clarification": True, "question": out.question}
    assert isinstance(out, ExperimentDesign)
    return {
        "needs_clarification": False,
        "design": {
            "primary_metric": out.primary_metric,
            "guardrail_metrics": out.guardrail_metrics,
            "randomization_unit": out.randomization_unit,
            "sample_size_required": out.sample_size_required,
            "runtime_days": out.runtime_days,
            "warnings": out.warnings,
        },
        "raw_llm_response": out.raw_llm_response,
    }


# ---------------------------------------------------------------------------
# POST /monitor
# ---------------------------------------------------------------------------
def _config_from_dict(d: dict[str, Any]) -> GeneratorConfig:
    """Build GeneratorConfig from API dict (defaults for missing keys)."""
    return GeneratorConfig(
        n_users=d.get("n_users", 50_000),
        n_days_pre=d.get("n_days_pre", 30),
        n_days_exp=d.get("n_days_exp", 30),
        inject_srm=d.get("inject_srm", False),
        inject_novelty=d.get("inject_novelty", False),
    )


@app.post("/monitor")
def monitor(request: MonitorRequest) -> dict[str, Any]:
    """Run monitor for the given experiment and day; returns MonitorResult as dict."""
    try:
        config = _config_from_dict(request.config)
        result = run_monitor(request.experiment_id, request.current_day, config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "experiment_id": result.experiment_id,
        "day": result.day,
        "cuped_result": result.cuped_result,
        "srm_result": result.srm_result,
        "sequential_result": result.sequential_result,
        "novelty_result": result.novelty_result,
        "decision": result.decision,
        "reasoning": result.reasoning,
    }


# ---------------------------------------------------------------------------
# POST /interpret
# ---------------------------------------------------------------------------
@app.post("/interpret")
def interpret(request: InterpretRequest) -> dict[str, Any]:
    """Fetch snapshots, run interpreter; returns InterpretationResult as dict."""
    try:
        snapshots = get_snapshots(request.experiment_id)
        result = run_interpreter(
            request.experiment_id,
            request.hypothesis,
            request.design,
            snapshots,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "experiment_id": result.experiment_id,
        "final_cuped": result.final_cuped,
        "final_srm": result.final_srm,
        "final_novelty": result.final_novelty,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "action": result.action,
    }


# ---------------------------------------------------------------------------
# GET /experiment/{experiment_id}
# ---------------------------------------------------------------------------
@app.get("/experiment/{experiment_id}")
def get_experiment_full(experiment_id: str) -> dict[str, Any]:
    """Return experiment row plus all snapshots and agent decisions."""
    try:
        exp = get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        snapshots = get_snapshots(experiment_id)
        decisions = get_agent_decisions(experiment_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "experiment": exp,
        "snapshots": snapshots,
        "agent_decisions": decisions,
    }


# ---------------------------------------------------------------------------
# POST /setup
# ---------------------------------------------------------------------------
@app.post("/setup")
def setup(request: SetupRequest) -> dict[str, Any]:
    """Create experiment, generate synthetic data, insert assignments and snapshots; returns experiment_id."""
    try:
        out = run_designer(request.hypothesis, request.clarification_response)
        if isinstance(out, ClarificationResult):
            raise HTTPException(
                status_code=400,
                detail={"needs_clarification": True, "question": out.question},
            )
        design = out
        assert isinstance(design, ExperimentDesign)

        experiment_row = {
            "hypothesis": request.hypothesis,
            "primary_metric": design.primary_metric,
            "guardrail_metrics": design.guardrail_metrics,
            "randomization_unit": design.randomization_unit,
            "sample_size_required": design.sample_size_required,
            "runtime_days": design.runtime_days,
            "status": "running",
            "design_output": design.design_json,
        }
        experiment_id = insert_experiment(experiment_row)

        config = GeneratorConfig(
            n_users=request.n_users,
            n_days_exp=design.runtime_days,
            true_lift=request.true_lift,
            inject_srm=request.inject_srm,
            inject_novelty=request.inject_novelty,
        )
        assignments, events = generate(config)
        assignment_dicts = assignments_to_dicts(assignments)
        insert_assignments(experiment_id, assignment_dicts)

        snapshots = aggregate_events_to_snapshots(events, config)
        snapshot_dicts = [
            {
                "day": s.day,
                "variant": s.variant,
                "primary_metric_value": s.primary_metric_value,
                "guardrail_values": s.guardrail_values,
                "sample_size": s.sample_size,
                "srm_flagged": s.srm_flagged,
                "novelty_flagged": s.novelty_flagged,
                "sequential_boundary": s.sequential_boundary,
            }
            for s in snapshots
        ]
        insert_snapshots(experiment_id, snapshot_dicts)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"experiment_id": experiment_id}

"""
LangGraph orchestrator for the growth experimentation copilot.

Defines ExperimentState, node stubs, and conditional routing:
designer → monitor; monitor → monitor | interpreter | escalate; interpreter/escalate → END.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ExperimentState(TypedDict, total=False):
    hypothesis: str
    experiment_id: str
    design: dict[str, Any]
    current_day: int
    snapshots: list[Any]
    srm_flagged: bool
    novelty_flagged: bool
    sequential_result: dict[str, Any]
    should_stop: bool
    final_recommendation: str
    next_action: str


# ---------------------------------------------------------------------------
# Node names
# ---------------------------------------------------------------------------
NODE_DESIGNER = "designer"
NODE_MONITOR = "monitor"
NODE_INTERPRETER = "interpreter"
NODE_ESCALATE = "escalate"
NODE_END = "__end__"


# ---------------------------------------------------------------------------
# Routing: read state.next_action and return the next node
# ---------------------------------------------------------------------------
def route_by_next_action(state: ExperimentState) -> str:
    """Return the next node name from state.next_action; END for interpreter/escalate."""
    next_action = (state.get("next_action") or "").strip().lower()
    if next_action == NODE_DESIGNER:
        return NODE_DESIGNER
    if next_action == NODE_MONITOR:
        return NODE_MONITOR
    if next_action == NODE_INTERPRETER:
        return NODE_INTERPRETER
    if next_action == NODE_ESCALATE:
        return NODE_ESCALATE
    if next_action in ("end", "__end__", "complete"):
        return NODE_END
    # Default: end
    return NODE_END


def route_after_designer(state: ExperimentState) -> str:
    """After designer: go to monitor if design approved (next_action set by designer)."""
    return state.get("next_action") or NODE_MONITOR


def route_after_monitor(state: ExperimentState) -> str:
    """After monitor: go to monitor | interpreter | escalate based on next_action."""
    next_action = state.get("next_action") or NODE_INTERPRETER
    if next_action == NODE_MONITOR:
        return NODE_MONITOR
    if next_action == NODE_ESCALATE:
        return NODE_ESCALATE
    return NODE_INTERPRETER


# ---------------------------------------------------------------------------
# Node stubs (print what they would do; return state update with next_action)
# ---------------------------------------------------------------------------
def node_designer(state: ExperimentState) -> dict[str, Any]:
    """Stub: would run Experiment Designer agent and produce design."""
    print("[orchestrator] Designer: would design experiment from hypothesis and set next_action=monitor")
    return {"next_action": NODE_MONITOR}


def node_monitor(state: ExperimentState) -> dict[str, Any]:
    """Stub: would run Monitor agent, update snapshots, set next_action from logic."""
    srm = state.get("srm_flagged", False)
    stop = state.get("should_stop", False)
    current_day = state.get("current_day", 0)
    design = state.get("design") or {}
    runtime_days = design.get("runtime_days") or 30
    runtime_complete = current_day >= runtime_days

    if srm:
        next_action = NODE_ESCALATE
        print("[orchestrator] Monitor: would detect SRM and set next_action=escalate")
    elif stop or runtime_complete:
        next_action = NODE_INTERPRETER
        print("[orchestrator] Monitor: would signal stop or runtime complete, next_action=interpreter")
    else:
        next_action = NODE_MONITOR
        print("[orchestrator] Monitor: would increment day and continue, next_action=monitor")

    return {
        "next_action": next_action,
        "current_day": current_day + 1 if next_action == NODE_MONITOR else current_day,
    }


def node_interpreter(state: ExperimentState) -> dict[str, Any]:
    """Stub: would run Results Interpreter and produce final recommendation."""
    print("[orchestrator] Interpreter: would synthesize results and produce final_recommendation")
    return {"next_action": NODE_END, "final_recommendation": "(stub) Recommendation pending."}


def node_escalate(state: ExperimentState) -> dict[str, Any]:
    """Stub: would escalate (e.g. SRM)."""
    print("[orchestrator] Escalate: would escalate (e.g. SRM detected)")
    return {"next_action": NODE_END}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph with designer → monitor → interpreter/escalate → END."""
    graph = StateGraph(ExperimentState)

    graph.add_node(NODE_DESIGNER, node_designer)
    graph.add_node(NODE_MONITOR, node_monitor)
    graph.add_node(NODE_INTERPRETER, node_interpreter)
    graph.add_node(NODE_ESCALATE, node_escalate)

    graph.add_edge(START, NODE_DESIGNER)
    graph.add_conditional_edges(
        NODE_DESIGNER,
        route_after_designer,
        {NODE_MONITOR: NODE_MONITOR},
    )
    graph.add_conditional_edges(
        NODE_MONITOR,
        route_after_monitor,
        {
            NODE_MONITOR: NODE_MONITOR,
            NODE_INTERPRETER: NODE_INTERPRETER,
            NODE_ESCALATE: NODE_ESCALATE,
        },
    )
    graph.add_edge(NODE_INTERPRETER, END)
    graph.add_edge(NODE_ESCALATE, END)

    return graph


# Compiled graph (lazy so we don't require langgraph at import if not used)
_compiled = None


def get_graph():
    """Return the compiled graph (singleton)."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled


# ---------------------------------------------------------------------------
# Validate: run with mock state and confirm routing
# ---------------------------------------------------------------------------
def validate() -> None:
    """Run the graph with mock state and assert we reach expected nodes."""
    graph = get_graph()
    config = {"recursion_limit": 50}

    # Run 1 (should_stop=True): designer → monitor → interpreter — confirm path completed
    initial: ExperimentState = {
        "hypothesis": "Test hypothesis",
        "experiment_id": "test-id",
        "design": {"runtime_days": 30},
        "current_day": 29,
        "snapshots": [],
        "srm_flagged": False,
        "novelty_flagged": False,
        "should_stop": True,
    }
    result = graph.invoke(initial, config=config)
    assert result.get("final_recommendation") is not None
    assert result.get("final_recommendation") != ""

    # Run 2 (srm_flagged=True): designer → monitor → escalate — confirm path completed
    initial_srm: ExperimentState = {
        "hypothesis": "Test",
        "experiment_id": "test-2",
        "design": {"runtime_days": 30},
        "current_day": 1,
        "srm_flagged": True,
        "should_stop": False,
    }
    result_srm = graph.invoke(initial_srm, config=config)
    assert result_srm.get("next_action") == "__end__"

    # Run 3 (loop test): monitor loops until runtime complete then routes to interpreter
    initial_loop: ExperimentState = {
        "hypothesis": "Loop test",
        "design": {"runtime_days": 30},
        "current_day": 28,
        "srm_flagged": False,
        "should_stop": False,
    }
    result_loop = graph.invoke(initial_loop, config=config)
    assert result_loop.get("current_day") >= 30

    # Run 4 (null case): one-day experiment — full graph completes with final_recommendation
    initial_one_day: ExperimentState = {
        "hypothesis": "One-day test",
        "experiment_id": "test-4",
        "design": {"runtime_days": 1},
        "current_day": 0,
        "srm_flagged": False,
        "should_stop": False,
    }
    result_one_day = graph.invoke(initial_one_day, config=config)
    assert result_one_day.get("final_recommendation") is not None

    print("orchestrator.validate() passed: designer→monitor→interpreter/escalate and monitor loop routing OK")


if __name__ == "__main__":
    validate()

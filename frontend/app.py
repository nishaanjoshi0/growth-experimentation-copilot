"""
Growth Experimentation Copilot — Streamlit frontend.
Calls FastAPI backend at http://localhost:8000. No placeholder data.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Growth Experimentation Copilot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .metric-card { padding: 1rem 1.25rem; border-radius: 8px; margin: 0.5rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .metric-card.green { background: #d4edda; border-left: 4px solid #28a745; }
  .metric-card.yellow { background: #fff3cd; border-left: 4px solid #ffc107; }
  .metric-card.red { background: #f8d7da; border-left: 4px solid #dc3545; }
  .design-card { background: #f8f9fa; padding: 1.25rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #dee2e6; }
  .badge { display: inline-block; padding: 0.35rem 0.75rem; border-radius: 6px; font-weight: 600; }
  .badge-continue { background: #28a745; color: white; }
  .badge-escalate { background: #ffc107; color: #212529; }
  .badge-stop { background: #dc3545; color: white; }
  .badge-ship { background: #28a745; color: white; }
  .badge-iterate { background: #17a2b8; color: white; }
  .badge-abandon { background: #dc3545; color: white; }
  .badge-rerun { background: #6c757d; color: white; }
  .recommendation-box { background: #f8f9fa; padding: 1rem; border-radius: 6px; border: 1px solid #dee2e6; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)


def api_get(path: str) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        r = client.get(f"{BASE_URL}{path}")
        r.raise_for_status()
        return r.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{BASE_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "design_clarification" not in st.session_state:
    st.session_state.design_clarification = None
if "design_result" not in st.session_state:
    st.session_state.design_result = None
if "setup_experiment_id" not in st.session_state:
    st.session_state.setup_experiment_id = None
if "used_clarification_response" not in st.session_state:
    st.session_state.used_clarification_response = None


# ---------------------------------------------------------------------------
# Page 1 — New Experiment
# ---------------------------------------------------------------------------
def page_new_experiment():
    st.header("New Experiment")
    hypothesis = st.text_area("Hypothesis", height=120, placeholder="e.g. We believe the new onboarding flow will increase subscription conversion by at least 5%.")
    clarification_response = None
    if st.session_state.design_clarification:
        st.info("Clarification needed")
        st.markdown(st.session_state.design_clarification)
        clarification_response = st.text_input("Your response", key="clar_response")
    col1, col2 = st.columns([1, 4])
    with col1:
        get_design = st.button("Get design", type="primary")
    if get_design and hypothesis.strip():
        with st.spinner("Designing experiment..."):
            try:
                out = api_post("/design", {"hypothesis": hypothesis.strip(), "clarification_response": clarification_response})
                if out.get("needs_clarification"):
                    st.session_state.design_clarification = out.get("question", "")
                    st.session_state.design_result = None
                    st.rerun()
                else:
                    st.session_state.design_clarification = None
                    st.session_state.design_result = out
                    st.session_state.used_clarification_response = clarification_response
                    st.rerun()
            except httpx.HTTPStatusError as e:
                try:
                    err = e.response.json()
                    if err.get("detail", {}).get("needs_clarification"):
                        st.session_state.design_clarification = err.get("detail", {}).get("question", "")
                        st.rerun()
                except Exception:
                    pass
                st.error(e.response.text or str(e))
            except Exception as e:
                st.error(str(e))

    if st.session_state.design_result:
        d = st.session_state.design_result.get("design", {})
        st.markdown("### Experiment design")
        st.markdown(f"""
        <div class="design-card">
          <p><b>Primary metric:</b> {d.get("primary_metric", "—")}</p>
          <p><b>Guardrail metrics:</b> {", ".join(d.get("guardrail_metrics") or [])}</p>
          <p><b>Randomization unit:</b> {d.get("randomization_unit", "—")}</p>
          <p><b>Sample size required:</b> {d.get("sample_size_required", "—"):,}</p>
          <p><b>Runtime (days):</b> {d.get("runtime_days", "—")}</p>
          <p><b>Warnings:</b> {"; ".join(d.get("warnings") or ["None"])}</p>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Generate data & save to Supabase")
        n_users = st.number_input("Number of users", min_value=100, value=50_000, step=1000)
        true_lift = st.number_input("True lift (e.g. 0.05 = 5%)", min_value=0.0, value=0.05, step=0.01, format="%.2f")
        inject_srm = st.checkbox("Inject SRM", value=False)
        inject_novelty = st.checkbox("Inject novelty effect", value=False)
        if st.button("Run setup"):
            with st.spinner("Generating data and inserting into Supabase..."):
                try:
                    out = api_post("/setup", {
                        "hypothesis": hypothesis.strip(),
                        "clarification_response": st.session_state.get("used_clarification_response"),
                        "n_users": n_users,
                        "true_lift": true_lift,
                        "inject_srm": inject_srm,
                        "inject_novelty": inject_novelty,
                    })
                    eid = out.get("experiment_id", "")
                    st.session_state.setup_experiment_id = eid
                    st.success("Setup complete.")
                    st.markdown(f"**Experiment ID** (use in Monitor / Interpret / History):")
                    st.code(eid, language=None)
                    st.download_button("Download ID", data=eid, file_name="experiment_id.txt", mime="text/plain", key="dl_id")
                except httpx.HTTPStatusError as e:
                    st.error(e.response.text or str(e))
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
# Page 2 — Monitor
# ---------------------------------------------------------------------------
def page_monitor():
    st.header("Monitor")
    experiment_id = st.text_input("Experiment ID", key="monitor_exp_id", placeholder="Paste experiment UUID")
    day = st.slider("Day", min_value=1, max_value=30, value=1, key="monitor_day")
    if "monitor_result" in st.session_state and experiment_id.strip() != st.session_state.get("last_monitor_exp_id"):
        st.session_state.monitor_result = None
    if st.button("Run Monitor", type="primary"):
        if not experiment_id.strip():
            st.warning("Enter an experiment ID.")
        else:
            with st.spinner("Running monitor..."):
                try:
                    result = api_post("/monitor", {
                        "experiment_id": experiment_id.strip(),
                        "current_day": day,
                        "config": {"n_days_exp": 30},
                    })
                    st.session_state.monitor_result = result
                    st.session_state.last_monitor_exp_id = experiment_id.strip()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if "monitor_result" in st.session_state and st.session_state.get("monitor_result"):
        r = st.session_state.monitor_result
        decision = (r.get("decision") or "continue").lower()
        badge_class = "badge-continue" if decision == "continue" else ("badge-escalate" if decision == "escalate" else "badge-stop")
        st.markdown(f'<p><span class="badge {badge_class}">{decision.upper()}</span></p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        cuped = r.get("cuped_result") or {}
        srm = r.get("srm_result") or {}
        seq = r.get("sequential_result") or {}
        nov = r.get("novelty_result") or {}
        with c1:
            lift = cuped.get("lift_adjusted", 0) or 0
            p = cuped.get("p_value", 1.0)
            if p is None:
                p = 1.0
            sig = cuped.get("significant", False)
            color = "green" if sig else ("yellow" if p < 0.1 else "red")
            st.markdown(f'<div class="metric-card {color}"><b>CUPED</b><br>Lift: {(lift*100):.2f}%<br>p-value: {p:.4f}</div>', unsafe_allow_html=True)
        with c2:
            srm_ok = not srm.get("srm_detected", False)
            color = "green" if srm_ok else "red"
            st.markdown(f'<div class="metric-card {color}"><b>SRM</b><br>{"OK" if srm_ok else "Flagged"}<br>p={srm.get("p_value", 1):.4f}</div>', unsafe_allow_html=True)
        with c3:
            rec_stop = seq.get("recommend_stop", False)
            color = "red" if rec_stop else "green"
            st.markdown(f'<div class="metric-card {color}"><b>Sequential</b><br>{"Recommend stop" if rec_stop else "Continue"}</div>', unsafe_allow_html=True)
        with c4:
            nov_det = nov.get("novelty_detected", False)
            color = "yellow" if nov_det else "green"
            st.markdown(f'<div class="metric-card {color}"><b>Novelty</b><br>Ratio: {nov.get("novelty_ratio", 0):.2f}<br>{"Flagged" if nov_det else "OK"}</div>', unsafe_allow_html=True)
        with st.expander("Reasoning"):
            st.write(r.get("reasoning", ""))


# ---------------------------------------------------------------------------
# Page 3 — Interpret
# ---------------------------------------------------------------------------
def page_interpret():
    st.header("Interpret")
    experiment_id = st.text_input("Experiment ID", key="interp_exp_id")
    hypothesis = st.text_area("Hypothesis", key="interp_hypothesis", height=80)
    primary_metric = st.text_input("Primary metric", key="interp_primary", placeholder="e.g. subscription_started")
    guardrails_str = st.text_input("Guardrail metrics (comma-separated)", key="interp_guardrails", placeholder="churn_rate, referral_count")
    runtime_days = st.number_input("Runtime (days)", min_value=1, value=30, key="interp_runtime")
    if st.button("Run Interpreter", type="primary"):
        if not experiment_id.strip() or not hypothesis.strip():
            st.warning("Enter experiment ID and hypothesis.")
        else:
            with st.spinner("Running interpreter..."):
                try:
                    result = api_post("/interpret", {
                        "experiment_id": experiment_id.strip(),
                        "hypothesis": hypothesis.strip(),
                        "design": {
                            "primary_metric": primary_metric or "conversion_rate",
                            "guardrail_metrics": [x.strip() for x in (guardrails_str or "").split(",") if x.strip()],
                            "runtime_days": int(runtime_days),
                        },
                    })
                    st.session_state.interpret_result = result
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if "interpret_result" in st.session_state and st.session_state.get("interpret_result"):
        r = st.session_state.interpret_result
        action = (r.get("action") or "iterate").lower()
        badge_class = {"ship": "badge-ship", "iterate": "badge-iterate", "abandon": "badge-abandon", "rerun": "badge-rerun"}.get(action, "badge-iterate")
        st.markdown(f'<p><span class="badge {badge_class}">{action.upper()}</span> &nbsp; Confidence: <b>{r.get("confidence", "medium")}</b></p>', unsafe_allow_html=True)
        st.markdown("**Recommendation**")
        st.markdown(f'<div class="recommendation-box">{r.get("recommendation", "")}</div>', unsafe_allow_html=True)
        st.subheader("Final stats")
        fc, fs, fn = r.get("final_cuped") or {}, r.get("final_srm") or {}, r.get("final_novelty") or {}
        st.json({"CUPED": fc, "SRM": fs, "Novelty": fn})


# ---------------------------------------------------------------------------
# Page 4 — Experiment History
# ---------------------------------------------------------------------------
def page_history():
    st.header("Experiment History")
    experiment_id = st.text_input("Experiment ID", key="hist_exp_id")
    if st.button("Load"):
        if not experiment_id.strip():
            st.warning("Enter an experiment ID.")
        else:
            with st.spinner("Loading..."):
                try:
                    data = api_get(f"/experiment/{experiment_id.strip()}")
                    st.session_state.history_data = data
                    st.rerun()
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        st.error("Experiment not found.")
                    else:
                        st.error(e.response.text or str(e))
                except Exception as e:
                    st.error(str(e))

    if "history_data" in st.session_state and st.session_state.get("history_data"):
        data = st.session_state.history_data
        exp = data.get("experiment") or {}
        snapshots = data.get("snapshots") or []
        decisions = data.get("agent_decisions") or []
        st.markdown("### Experiment details")
        st.write(f"**Hypothesis:** {exp.get('hypothesis', '—')}")
        st.write(f"**Primary metric:** {exp.get('primary_metric', '—')} | **Runtime:** {exp.get('runtime_days', '—')} days | **Status:** {exp.get('status', '—')}")
        st.markdown("### Primary metric over time")
        if snapshots:
            df = pd.DataFrame(snapshots)
            pivot = df.pivot_table(index="day", columns="variant", values="primary_metric_value", aggfunc="first").reset_index()
            pivot = pivot.sort_values("day")
            fig = go.Figure()
            if "control" in pivot.columns:
                fig.add_trace(go.Scatter(x=pivot["day"], y=pivot["control"], name="Control", mode="lines+markers"))
            if "treatment" in pivot.columns:
                fig.add_trace(go.Scatter(x=pivot["day"], y=pivot["treatment"], name="Treatment", mode="lines+markers"))
            fig.update_layout(xaxis_title="Day", yaxis_title="Primary metric value", title="Control vs Treatment")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No snapshot data.")
        st.markdown("### Agent decisions")
        if decisions:
            st.dataframe(pd.DataFrame(decisions)[["agent", "decision", "reasoning"]], use_container_width=True, hide_index=True)
        else:
            st.info("No agent decisions yet.")


# ---------------------------------------------------------------------------
# Sidebar & routing
# ---------------------------------------------------------------------------
st.sidebar.title("Growth Experimentation Copilot")
page = st.sidebar.radio(
    "Page",
    ["New Experiment", "Monitor", "Interpret", "Experiment History"],
    label_visibility="collapsed",
)
if page == "New Experiment":
    page_new_experiment()
elif page == "Monitor":
    page_monitor()
elif page == "Interpret":
    page_interpret()
else:
    page_history()

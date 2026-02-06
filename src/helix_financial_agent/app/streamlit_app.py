"""
Helix Financial Agent - Evaluation & Agent Run UI

Single-page Streamlit app: generate evaluation data, browse dataset,
run agent on a selected record with evaluation, and view model routing,
tool selection, and metacognition panels.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ..config import get_config
from ..data_generation.generate import (
    DEFAULT_EVAL_SPLIT_RATIO,
    DEFAULT_TOTAL_QUESTIONS,
    DEFAULT_VALID_RATIO,
    generate_full_dataset,
    load_dataset,
)
from ..agent.runner import run_agent, ServiceError


def _init_session_state() -> None:
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    if "selected_record" not in st.session_state:
        st.session_state.selected_record = None
    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = 0
    if "last_run_result" not in st.session_state:
        st.session_state.last_run_result = None


def _load_dataset_from_disk(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_dataset(path)


def render_generate_section(config) -> None:
    st.subheader("Generate evaluation data")
    st.caption("Generate a new benchmark dataset (requires router + Gemini).")
    with st.form("generate_form"):
        total_count = st.number_input(
            "Total queries",
            min_value=10,
            max_value=500,
            value=min(50, DEFAULT_TOTAL_QUESTIONS),
            step=10,
        )
        valid_ratio = st.slider(
            "Valid vs hazard ratio",
            min_value=0.5,
            max_value=1.0,
            value=DEFAULT_VALID_RATIO,
            step=0.05,
        )
        eval_ratio = st.slider(
            "Eval split ratio",
            min_value=0.05,
            max_value=0.5,
            value=DEFAULT_EVAL_SPLIT_RATIO,
            step=0.05,
        )
        submitted = st.form_submit_button("Generate dataset")
    if submitted:
        output_dir = config.paths.data_dir
        try:
            with st.spinner("Generating dataset…"):
                full, train, eval_ds = generate_full_dataset(
                    output_dir=output_dir,
                    total_count=total_count,
                    valid_ratio=valid_ratio,
                    eval_ratio=eval_ratio,
                    verbose=False,
                )
            st.success(f"Generated {len(full)} queries. Train: {len(train)}, Eval: {len(eval_ds)}.")
            full_path = output_dir / "financial_benchmark_v1_full.jsonl"
            st.session_state.dataset = full
            st.info(f"Saved to `{full_path}`. Dataset loaded into table below.")
        except ServiceError as e:
            st.error(f"Services not available: {e}")
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.exception(e)


def render_table_and_selection(config) -> None:
    st.subheader("Dataset")
    st.caption("Load a dataset and select one record to run the agent on.")
    data_dir = config.paths.data_dir
    full_path = data_dir / "financial_benchmark_v1_full.jsonl"
    eval_path = data_dir / "financial_benchmark_v1_eval.jsonl"
    dataset_choice = st.radio(
        "Dataset file",
        ["Full", "Eval"],
        horizontal=True,
        format_func=lambda x: f"{x} ({full_path.name if x == 'Full' else eval_path.name})",
    )
    path = full_path if dataset_choice == "Full" else eval_path
    if st.button("Refresh from disk"):
        st.session_state.dataset = _load_dataset_from_disk(path)
        st.rerun()
    if not st.session_state.dataset and path.exists():
        st.session_state.dataset = _load_dataset_from_disk(path)
    if not st.session_state.dataset:
        st.warning("No dataset loaded. Generate one above or ensure the file exists and click Refresh.")
        return
    df = pd.DataFrame(st.session_state.dataset)
    display_cols = [c for c in ["id", "query", "category", "subcategory", "expected_tools"] if c in df.columns]
    if not display_cols:
        display_cols = list(df.columns)[:6]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, height=300)
    index = st.selectbox(
        "Select record (by row index)",
        range(len(st.session_state.dataset)),
        format_func=lambda i: f"{i}: {st.session_state.dataset[i].get('id', '')} — {str(st.session_state.dataset[i].get('query', ''))[:60]}…",
    )
    st.session_state.selected_row_index = index
    st.session_state.selected_record = st.session_state.dataset[index]
    rec = st.session_state.selected_record
    with st.expander("Selected record details"):
        st.json(rec)


def render_run_agent_section() -> None:
    st.subheader("Run agent")
    if not st.session_state.selected_record:
        st.info("Select a record in the Dataset section first.")
        return
    rec = st.session_state.selected_record
    query = rec.get("query", "")
    if not query:
        st.warning("Selected record has no 'query' field.")
        return
    metadata = {
        "id": rec.get("id"),
        "category": rec.get("category"),
        "subcategory": rec.get("subcategory"),
        "expected_tools": rec.get("expected_tools"),
        "expected_behavior": rec.get("expected_behavior"),
    }
    if st.button("Run agent on selected record"):
        try:
            with st.spinner("Running agent (this may take a minute)…"):
                result = run_agent(
                    query=query,
                    run_evaluation=True,
                    query_metadata=metadata,
                    verbose=False,
                    skip_service_check=False,
                )
            st.session_state.last_run_result = result
            st.success("Run completed.")
        except ServiceError as e:
            st.error(f"Services not available: {e}")
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.exception(e)
    if st.session_state.last_run_result is None:
        return
    result = st.session_state.last_run_result
    st.divider()
    st.markdown("**Query**")
    st.markdown(query)
    st.markdown("**Metadata**")
    st.json(metadata)
    st.markdown("**Final response**")
    st.text_area("", value=result.get("response") or "", height=200, disabled=True, label_visibility="collapsed")
    eval_data = result.get("evaluation")
    if eval_data:
        st.markdown("**Evaluation**")
        if eval_data.get("category") == "valid":
            score = eval_data.get("total_score", 0)
            st.metric("Correctness score", f"{score}/10")
        else:
            passed = eval_data.get("passed", False)
            st.metric("Safety passed", "Yes" if passed else "No")
        if eval_data.get("reasoning"):
            st.caption("Reasoning")
            st.write(eval_data["reasoning"])
    assessments = result.get("assessments", {})
    if assessments:
        st.caption("Assessments")
        st.write(f"Tool selection: {'Y' if assessments.get('tool_selection_successful') else 'N'} | Model selection: {'Y' if assessments.get('model_selection_successful') else '?'} | Judge score: {assessments.get('judge_score', '—')}")
    with st.expander("Raw trace (debug)"):
        st.json(result.get("trace") or [])


def render_model_routing_panel(result: Dict[str, Any]) -> None:
    st.markdown("**Model routing**")
    primary = result.get("routed_model") or "—"
    st.metric("Primary model", primary)
    models = result.get("routed_models") or []
    if models:
        st.caption("Models used (in order)")
        for i, m in enumerate(models, 1):
            st.write(f"{i}. {m}")
    trace = result.get("trace") or []
    step_models = []
    for step in trace:
        event = step.get("event", "")
        if not event.startswith("node_"):
            continue
        node = event.replace("node_", "")
        data = step.get("data") or {}
        model = None
        if "response" in data and isinstance(data["response"], dict):
            model = data["response"].get("model")
        if model is None and "reflection" in data and isinstance(data["reflection"], dict):
            model = data["reflection"].get("model")
        if model:
            step_models.append({"Step": step.get("step", "—"), "Node": node, "Model": model})
    if step_models:
        st.dataframe(pd.DataFrame(step_models), use_container_width=True, hide_index=True)


def render_tool_selection_panel(result: Dict[str, Any], expected_tools: Optional[List[str]] = None) -> None:
    st.markdown("**Tool selection**")
    details = result.get("tool_selection_details")
    if not details:
        st.caption("Selected tools (from run)")
        st.write(", ".join(result.get("tools_selected") or []))
        st.caption("Tools actually used")
        st.write(", ".join(result.get("unique_tools") or []))
        return
    all_matches = details.get("all_matches") or []
    selected = details.get("selected") or []
    selected_names = {m["name"] for m in selected}
    threshold = details.get("threshold", 0)
    max_tools = details.get("max_tools") or 0
    above_count = details.get("above_threshold_count", 0)
    rows = []
    for i, m in enumerate(all_matches, 1):
        name = m.get("name", "?")
        sim = m.get("similarity", 0)
        if name in selected_names:
            status = "SEL"
        elif sim >= threshold and above_count > len(selected) and len(selected) >= max_tools:
            status = "CAP"
        else:
            status = "REJ"
        rows.append({"#": i, "Tool": name, "Similarity": f"{sim:.4f}", "Status": status})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if expected_tools:
        st.caption(f"Expected: {', '.join(expected_tools)}")


def render_metacognition_panel(result: Dict[str, Any]) -> None:
    st.markdown("**Metacognition / reflexive loop**")
    reflections = result.get("reflections") or []
    if not reflections:
        st.caption("No reflection steps in this run.")
        return
    for i, ref in enumerate(reflections):
        step = ref.get("step", "—")
        iteration = ref.get("iteration", "—")
        model = ref.get("model", "")
        content = ref.get("content", "{}")
        try:
            data = json.loads(content)
            passed = data.get("passed", False)
            feedback = data.get("feedback", "")
            issues = data.get("issues", [])
            if isinstance(issues, str):
                issues = [issues] if issues else []
        except Exception:
            passed = False
            feedback = content[:200]
            issues = []
        status = "PASSED" if passed else "NEEDS REVISION"
        color = "green" if passed else "orange"
        st.markdown(f"**Iteration {iteration}** (step {step}) — :{color}[{status}]")
        if model:
            st.caption(f"Model: {model}")
        if feedback:
            st.write(feedback)
        if issues:
            for iss in issues:
                st.write(f"- {iss}")
        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Helix Financial Agent — Eval & Run",
        page_icon="📊",
        layout="wide",
    )
    _init_session_state()
    config = get_config()
    st.title("Helix Financial Agent")
    st.caption("Evaluation data generation and agent run with routing, tool selection, and metacognition views.")
    st.divider()
    render_generate_section(config)
    st.divider()
    render_table_and_selection(config)
    st.divider()
    render_run_agent_section()
    if st.session_state.last_run_result:
        st.divider()
        st.subheader("Run insights")
        result = st.session_state.last_run_result
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.expander("Model routing", expanded=True):
                render_model_routing_panel(result)
        with col2:
            with st.expander("Tool selection", expanded=True):
                rec = st.session_state.selected_record or {}
                render_tool_selection_panel(result, rec.get("expected_tools"))
        with col3:
            with st.expander("Metacognition / reflexive loop", expanded=True):
                render_metacognition_panel(result)


if __name__ == "__main__":
    main()

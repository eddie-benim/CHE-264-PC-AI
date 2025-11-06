import json
import os
from datetime import datetime

import streamlit as st

from agents import set_default_openai_key
import PC_logic as pcl

if "OPENAI_API_KEY" in st.secrets:
    set_default_openai_key(st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Process Control PID Assistant", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = None
if "mode" not in st.session_state:
    st.session_state.mode = "Day 1 – Ziegler–Nichols"

st.title("Process Control PID Assistant")
st.markdown("Upload Excel exports from the process control software and let the app generate PID parameters based on the chosen tuning mode.")

with st.sidebar:
    ts = pcl.current_timestamp_meta()
    st.markdown("### Session Info")
    st.markdown(f"**Date:** {ts['date']}")
    st.markdown(f"**Time:** {ts['time']}")
    st.markdown(f"**Day:** {ts['day_of_week']}")
    st.markdown("### Mode")
    mode_label = st.radio(
        "Select Tuning Mode",
        ["Day 1 – Ziegler–Nichols", "Day 2 – ML/Agent-driven"],
        index=0 if st.session_state.mode == "Day 1 – Ziegler–Nichols" else 1,
    )
    st.session_state.mode = mode_label

mode_value = "day1" if "Day 1" in st.session_state.mode else "day2"

st.divider()

st.subheader("Trial Inputs")
st.markdown("Provide up to 3 trials for the current run. You can mix fluids per trial.")

col1, col2, col3 = st.columns(3)

trial_names = []
trial_files = []
trial_fluids = []
trial_priors = []

with col1:
    st.markdown("**Trial 1**")
    t1_name = st.text_input("Name (T1)", "Day1_Trial1")
    t1_fluid = st.selectbox("Fluid (T1)", ["Water-Water", "Mineral-Oil–Water"], key="fluid_t1")
    t1_file = st.file_uploader("Excel (T1)", type=["xlsx", "xls"], key="file_t1")
    if mode_value == "day2":
        t1_kp = st.number_input("Prior Kp (T1)", value=0.0, min_value=0.0, key="t1_kp")
        t1_ti = st.number_input("Prior Ti (T1)", value=0.0, min_value=0.0, key="t1_ti")
        t1_td = st.number_input("Prior Td (T1)", value=0.0, min_value=0.0, key="t1_td")
    else:
        t1_kp = t1_ti = t1_td = 0.0

with col2:
    st.markdown("**Trial 2**")
    t2_name = st.text_input("Name (T2)", "Day1_Trial2")
    t2_fluid = st.selectbox("Fluid (T2)", ["Water-Water", "Mineral-Oil–Water"], key="fluid_t2")
    t2_file = st.file_uploader("Excel (T2)", type=["xlsx", "xls"], key="file_t2")
    if mode_value == "day2":
        t2_kp = st.number_input("Prior Kp (T2)", value=0.0, min_value=0.0, key="t2_kp")
        t2_ti = st.number_input("Prior Ti (T2)", value=0.0, min_value=0.0, key="t2_ti")
        t2_td = st.number_input("Prior Td (T2)", value=0.0, min_value=0.0, key="t2_td")
    else:
        t2_kp = t2_ti = t2_td = 0.0

with col3:
    st.markdown("**Trial 3**")
    t3_name = st.text_input("Name (T3)", "Day1_Trial3")
    t3_fluid = st.selectbox("Fluid (T3)", ["Water-Water", "Mineral-Oil–Water"], key="fluid_t3")
    t3_file = st.file_uploader("Excel (T3)", type=["xlsx", "xls"], key="file_t3")
    if mode_value == "day2":
        t3_kp = st.number_input("Prior Kp (T3)", value=0.0, min_value=0.0, key="t3_kp")
        t3_ti = st.number_input("Prior Ti (T3)", value=0.0, min_value=0.0, key="t3_ti")
        t3_td = st.number_input("Prior Td (T3)", value=0.0, min_value=0.0, key="t3_td")
    else:
        t3_kp = t3_ti = t3_td = 0.0

uploads = []
names = []
fluids = []
priors = []

if t1_file is not None:
    uploads.append(t1_file.getvalue())
    names.append(t1_name)
    fluids.append(t1_fluid)
    priors.append({"kp": t1_kp, "ti": t1_ti, "td": t1_td} if mode_value == "day2" and t1_kp > 0 else None)
if t2_file is not None:
    uploads.append(t2_file.getvalue())
    names.append(t2_name)
    fluids.append(t2_fluid)
    priors.append({"kp": t2_kp, "ti": t2_ti, "td": t2_td} if mode_value == "day2" and t2_kp > 0 else None)
if t3_file is not None:
    uploads.append(t3_file.getvalue())
    names.append(t3_name)
    fluids.append(t3_fluid)
    priors.append({"kp": t3_kp, "ti": t3_ti, "td": t3_td} if mode_value == "day2" and t3_kp > 0 else None)

st.divider()

objective = st.text_area(
    "Control objective (used mainly for Day 2 ML mode)",
    value="Minimize overshoot and settling time while rejecting ice-pack disturbance",
    height=80,
)

run_button = st.button("Run PID Analysis", use_container_width=True)

if run_button:
    if not uploads:
        st.error("Please upload at least one Excel file.")
    else:
        with st.spinner("Analyzing trials and generating PID parameters..."):
            results = pcl.analyze_multiple_trials(
                uploads=uploads,
                trial_names=names,
                fluids=fluids,
                mode=mode_value,
                prior_pids=priors if mode_value == "day2" else None,
                objective=objective,
            )
            st.session_state.results = results

st.divider()

st.subheader("Results")

if st.session_state.results:
    for r in st.session_state.results:
        with st.expander(f"{r.trial_name} – {r.fluid} – {r.mode.upper()}"):
            st.markdown(f"**Suggested PID Parameters**")
            st.markdown(f"- Kp: `{r.suggested_pid.kp:.4f}`")
            st.markdown(f"- Ti: `{r.suggested_pid.ti:.4f}` s")
            st.markdown(f"- Td: `{r.suggested_pid.td:.4f}` s")
            st.markdown(f"- Method: {r.suggested_pid.method}")
            st.markdown(f"- Notes: {r.suggested_pid.notes}")
            st.markdown("**Condensed Workflow**")
            for step in r.workflow:
                st.markdown(f"- **{step.step}:** {step.detail}")

    st.download_button(
        "Download results as JSON",
        data=pcl.serialize_results(st.session_state.results),
        file_name=f"pc_pid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )
else:
    st.info("Upload files and run analysis to see PID suggestions here.")

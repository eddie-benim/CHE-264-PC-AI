import os
from datetime import datetime

import streamlit as st
import PC_logic as pcl

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Process Control PID Assistant", layout="wide")

if "runs" not in st.session_state:
    st.session_state.runs = []

st.title("Process Control PID Assistant")

ts = pcl.current_timestamp_meta()
with st.sidebar:
    st.markdown("### Session")
    st.markdown(f"**Date:** {ts['date']}")
    st.markdown(f"**Time:** {ts['time']}")
    st.markdown(f"**Day:** {ts['day_of_week']}")
    mode_label = st.radio(
        "Tuning mode",
        ["Day 1 – Ziegler–Nichols", "Day 2 – ML / data-driven"],
        index=0,
    )
    mode_value = "day1" if "Day 1" in mode_label else "day2"
    fluid = st.selectbox(
        "Fluid pairing",
        ["Water-Water", "Mineral-Oil–Water"],
        index=0,
    )
    objective = st.text_area(
        "Control objective",
        value="Minimize overshoot and settling time while rejecting ice-pack disturbance",
        height=80,
    )
    if mode_value == "day2":
        st.markdown("#### Optional prior PID")
        prior_kp = st.number_input("Prior Kp", value=0.0, min_value=0.0, step=0.1)
        prior_ti = st.number_input("Prior Ti (s)", value=0.0, min_value=0.0, step=0.5)
        prior_td = st.number_input("Prior Td (s)", value=0.0, min_value=0.0, step=0.1)
    else:
        prior_kp = prior_ti = prior_td = 0.0

st.divider()

st.subheader("Upload trial data")
uploaded_file = st.file_uploader(
    "Attach Excel export (.xlsx or .xls) from the process control software",
    type=["xlsx", "xls"],
)
trial_name = st.text_input("Trial label", value="Current_Trial")

run_btn = st.button("Generate PID", use_container_width=True)

if run_btn:
    if uploaded_file is None:
        st.error("Please upload an Excel file first.")
    else:
        file_bytes = uploaded_file.getvalue()
        if mode_value == "day2" and prior_kp > 0:
            prior_pid = {"kp": prior_kp, "ti": prior_ti, "td": prior_td}
        else:
            prior_pid = None
        with st.spinner("Analyzing data and generating PID..."):
            result = pcl.analyze_trial_excel(
                file_bytes=file_bytes,
                trial_name=trial_name,
                fluid=fluid,
                mode=mode_value,
                prior_pid=prior_pid,
                objective=objective,
            )
        st.session_state.runs.append(result)

st.divider()
st.subheader("Results")

if not st.session_state.runs:
    st.info("Upload a file and run analysis to see PID suggestions here.")
else:
    for r in reversed(st.session_state.runs):
        with st.expander(f"{r.trial_name} – {r.fluid} – {r.mode.upper()}"):
            st.markdown("**Suggested PID**")
            st.markdown(f"- Kp: `{r.suggested_pid.kp:.4f}`")
            st.markdown(f"- Ti: `{r.suggested_pid.ti:.4f}` s")
            st.markdown(f"- Td: `{r.suggested_pid.td:.4f}` s")
            st.markdown(f"- Method: {r.suggested_pid.method}")
            st.markdown(f"- Notes: {r.suggested_pid.notes}")
            st.markdown("**Condensed workflow**")
            for step in r.workflow:
                st.markdown(f"- **{step.step}:** {step.detail}")

    st.download_button(
        "Download all run results as JSON",
        data=pcl.serialize_results(st.session_state.runs),
        file_name=f"pc_pid_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from agents import Agent, Runner, RunContextWrapper, TResponseInputItem, input_guardrail, GuardrailFunctionOutput

_client = OpenAI()

class PIDTuningResult(BaseModel):
    kp: float
    ti: float
    td: float
    method: str
    notes: str

class WorkflowItem(BaseModel):
    step: str
    detail: str

class TrialAnalysisResult(BaseModel):
    trial_name: str
    fluid: str
    mode: str
    suggested_pid: PIDTuningResult
    workflow: List[WorkflowItem]

class MLTrialInput(BaseModel):
    trial_name: str
    fluid: str
    prior_pid: Optional[Dict[str, float]] = None
    fopdt: Optional[Dict[str, float]] = None
    data_preview: List[Dict[str, Any]]
    objective: str

class MLPIDOutput(BaseModel):
    kp: float = Field(..., description="Proportional gain")
    ti: float = Field(..., description="Integral time in seconds")
    td: float = Field(..., description="Derivative time in seconds")
    rationale: str = Field(..., description="Short rationale tied to observed data and process traits")

ml_pid_agent = Agent(
    name="Process Control PID ML Agent",
    instructions="""
You are assisting with tuning PID parameters for a small laboratory-scale process plant trainer used to control temperatures and flows. The plant has two pumps, a heating tank, a plate heat exchanger, a holding tube that introduces noticeable dead time, and several temperature sensors. Fluid pairs can be water-water or mineral-oil–water. Oil-water runs tend to have higher heat capacity on the hot side and slower approach to steady state. The controller goal is fast disturbance rejection with limited overshoot. Prefer conservative gains if dead time is significant.
You receive: (1) brief trial metadata, (2) optional prior PID values, (3) data-derived FOPDT estimates, and (4) preview rows of the uploaded Excel data. You must return tuned PID values that the lab group can type directly into the PCT software. If the fluid is mineral-oil–water and the prior PID was from water-water, reduce Kp (proportional gain) relative to the water-water values and lengthen Ti slightly to avoid oscillations caused by slower dynamics. Use the FOPDT estimate as an anchor but you may deviate from Ziegler–Nichols for better robustness. Output JSON only.
""",
    output_type=MLPIDOutput,
    model="gpt-4o",
)

class OutOfScopeCheck(BaseModel):
    is_off_topic: bool
    explanation: str

scope_guard_agent = Agent(
    name="Scope Guard",
    instructions="Allow only process-control-lab-related tuning tasks. Reject philosophical or unrelated topics.",
    output_type=OutOfScopeCheck,
    model="gpt-4o",
)

@input_guardrail
async def process_control_scope_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | List[TResponseInputItem],
) -> GuardrailFunctionOutput:
    out = await Runner.run(scope_guard_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=out.final_output,
        tripwire_triggered=out.final_output.is_off_topic,
    )

def run_async_task(task):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(task)

def read_excel_file(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(file_bytes)

def select_time_column(df: pd.DataFrame) -> str:
    candidates = ["Elapsed Time", "elapsed_time", "Time", "time", "Sample Time", "sample_time"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]

def select_pv_column(df: pd.DataFrame) -> str:
    candidates = ["T2", "Temperature", "temperature", "PV", "Process Variable", "T1", "T4"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[1]

def select_mv_column(df: pd.DataFrame) -> str:
    candidates = ["Heater Output", "Power", "power", "Controller Output", "MV", "N2", "Hot Water Pump", "Hot water pump speed"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[2]

def estimate_fopdt_from_step(df: pd.DataFrame, time_col: str, pv_col: str, mv_col: str) -> Optional[Dict[str, float]]:
    t = df[time_col].to_numpy(dtype=float)
    y = df[pv_col].to_numpy(dtype=float)
    u = df[mv_col].to_numpy(dtype=float)
    if len(t) < 10:
        return None
    du = np.diff(u)
    if len(du) == 0:
        return None
    step_idx = np.argmax(np.abs(du) > 0.25 * np.nanmax(np.abs(du)))
    if np.abs(du[step_idx]) < 1e-6:
        return None
    t_step = t[step_idx + 1]
    y0 = y[step_idx]
    y_ss = np.nanmean(y[int(0.8 * len(y)) :])
    delta_y = y_ss - y0
    delta_u = u[step_idx + 1] - u[step_idx]
    if np.abs(delta_y) < 1e-6 or np.abs(delta_u) < 1e-6:
        return None
    y28 = y0 + 0.283 * delta_y
    y63 = y0 + 0.632 * delta_y
    t28_candidates = t[(y >= y28)]
    t63_candidates = t[(y >= y63)]
    if len(t28_candidates) == 0 or len(t63_candidates) == 0:
        return None
    t28 = t28_candidates[0]
    t63 = t63_candidates[0]
    theta = max(t28 - t_step, 1e-3)
    tau = max(t63 - t28, 1e-3)
    K = delta_y / delta_u
    return {"K": float(K), "tau": float(tau), "theta": float(theta), "t_step": float(t_step)}

def ziegler_nichols_pid(fopdt: Dict[str, float]) -> PIDTuningResult:
    K = fopdt["K"]
    tau = fopdt["tau"]
    theta = fopdt["theta"]
    kp = 1.2 * tau / (K * theta)
    ti = 2.0 * theta
    td = 0.5 * theta
    return PIDTuningResult(
        kp=float(kp),
        ti=float(ti),
        td=float(td),
        method="Ziegler-Nichols (open-loop)",
        notes="Standard Z-N based on FOPDT estimate",
    )

def conservative_from_zn(zn: PIDTuningResult, factor: float = 0.7) -> PIDTuningResult:
    kp = zn.kp * factor
    ti = zn.ti * 1.2
    td = zn.td
    return PIDTuningResult(
        kp=float(kp),
        ti=float(ti),
        td=float(td),
        method="Conservative ZN",
        notes="Dead time or property change detected, gains reduced",
    )

def make_workflow_from_fopdt(trial_name: str, fluid: str, fopdt: Optional[Dict[str, float]], method: str, mode: str) -> List[WorkflowItem]:
    items = []
    items.append(WorkflowItem(step="Trial identified", detail=f"Trial={trial_name}, fluid={fluid}, mode={mode}"))
    if fopdt:
        items.append(WorkflowItem(step="FOPDT estimation", detail=f"K={fopdt['K']:.3f}, tau={fopdt['tau']:.2f}s, theta={fopdt['theta']:.2f}s"))
    else:
        items.append(WorkflowItem(step="FOPDT estimation", detail="FOPDT not reliably detected, data may lack clear step"))
    items.append(WorkflowItem(step="Tuning method", detail=method))
    items.append(WorkflowItem(step="Lab constraints", detail="Small-scale PCT with holding tube dead time and heat exchanger per manual"))
    return items

def preview_df(df: pd.DataFrame, n: int = 40) -> List[Dict[str, Any]]:
    return df.head(n).to_dict(orient="records")

def ml_tune_pid(trial_name: str, fluid: str, df: pd.DataFrame, fopdt: Optional[Dict[str, float]], prior_pid: Optional[Dict[str, float]], objective: str) -> PIDTuningResult:
    payload = MLTrialInput(
        trial_name=trial_name,
        fluid=fluid,
        prior_pid=prior_pid,
        fopdt=fopdt,
        data_preview=preview_df(df),
        objective=objective,
    )
    result = run_async_task(Runner.run(ml_pid_agent, payload.model_dump()))
    out = result.final_output
    return PIDTuningResult(
        kp=float(out.kp),
        ti=float(out.ti),
        td=float(out.td),
        method="ML-tuned via OpenAI agent",
        notes=out.rationale,
    )

def normalize_mode(mode: str) -> Literal["day1", "day2"]:
    m = mode.strip().lower()
    if m in ["day1", "zn", "ziegler-nichols", "ziegler", "traditional"]:
        return "day1"
    if m in ["day2", "ml", "machine-learning", "ai"]:
        return "day2"
    return "day1"

def analyze_trial_excel(
    file_bytes: bytes,
    trial_name: str,
    fluid: str,
    mode: str,
    prior_pid: Optional[Dict[str, float]] = None,
    objective: str = "Minimize overshoot and settling time while rejecting ice-pack disturbance"
) -> TrialAnalysisResult:
    df = read_excel_file(file_bytes)
    time_col = select_time_column(df)
    pv_col = select_pv_column(df)
    mv_col = select_mv_column(df)
    fopdt = estimate_fopdt_from_step(df, time_col, pv_col, mv_col)
    normalized_mode = normalize_mode(mode)
    if normalized_mode == "day1":
        if fopdt:
            zn = ziegler_nichols_pid(fopdt)
            if fluid.lower().startswith("mineral"):
                tuned = conservative_from_zn(zn, factor=0.6)
            else:
                tuned = zn
            workflow = make_workflow_from_fopdt(trial_name, fluid, fopdt, tuned.method, normalized_mode)
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized_mode, suggested_pid=tuned, workflow=workflow)
        else:
            default_pid = PIDTuningResult(kp=1.0, ti=30.0, td=0.0, method="Fallback", notes="FOPDT not detected; using fallback")
            workflow = make_workflow_from_fopdt(trial_name, fluid, fopdt, "Fallback", normalized_mode)
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized_mode, suggested_pid=default_pid, workflow=workflow)
    else:
        tuned = ml_tune_pid(trial_name, fluid, df, fopdt, prior_pid, objective)
        workflow = make_workflow_from_fopdt(trial_name, fluid, fopdt, tuned.method, normalized_mode)
        workflow.append(WorkflowItem(step="ML rationale", detail=tuned.notes))
        return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized_mode, suggested_pid=tuned, workflow=workflow)

def analyze_multiple_trials(
    uploads: List[bytes],
    trial_names: List[str],
    fluids: List[str],
    mode: str,
    prior_pids: Optional[List[Optional[Dict[str, float]]]] = None,
    objective: str = "Minimize overshoot and settling time while rejecting ice-pack disturbance"
) -> List[TrialAnalysisResult]:
    normalized_mode = normalize_mode(mode)
    results = []
    if prior_pids is None:
        prior_pids = [None] * len(uploads)
    for file_bytes, name, fluid, pp in zip(uploads, trial_names, fluids, prior_pids):
        result = analyze_trial_excel(
            file_bytes=file_bytes,
            trial_name=name,
            fluid=fluid,
            mode=normalized_mode,
            prior_pid=pp,
            objective=objective,
        )
        results.append(result)
    return results

def serialize_results(results: List[TrialAnalysisResult]) -> str:
    return json.dumps([r.model_dump() for r in results], indent=2, default=str)

def current_timestamp_meta() -> Dict[str, str]:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
    }

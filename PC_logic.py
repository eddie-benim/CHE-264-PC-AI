import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from io import BytesIO

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

if "OPENAI_API_KEY" in os.environ:
    _client = OpenAI()
else:
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

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

TIME_CANDS = ["Elapsed Time", "elapsed time"]
PV_CANDS = [
    "Hot Water Temperature T2 (°C)",
    "Hot Water Temperature T2 (deg C)",
    "Hot Water Temperature T2",
    "hot water temperature t2 (°c)",
    "hot water temperature t2 (deg c)",
]
MV_CANDS_PRIMARY = [
    "Heater Power PWR (kW)",
    "Heater Power PWR",
    "Heater Power",
    "heater power pwr (kw)",
]
MV_CANDS_FALLBACK = [
    "Heating Pump Speed N2 (%)",
    "Heating Pump Speed N2",
    "heating pump speed n2 (%)",
]

def read_excel_file(file_bytes: bytes) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    try:
        return pd.read_excel(bio)
    except Exception:
        bio.seek(0)
        return pd.read_excel(bio, engine="xlrd")

def clean_export_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)

def match_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c: c.lower() for c in df.columns}
    for cand in candidates:
        cl = cand.lower()
        for orig, low in lower_map.items():
            if cl == low:
                return orig
    for cand in candidates:
        cl = cand.lower()
        for orig, low in lower_map.items():
            if cl in low:
                return orig
    return None

def select_time_column(df: pd.DataFrame) -> str:
    m = match_column(df, TIME_CANDS)
    if m:
        return m
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    return df.columns[0]

def select_pv_column(df: pd.DataFrame) -> str:
    m = match_column(df, PV_CANDS)
    if m:
        return m
    for c in df.columns:
        if "t2" in c.lower():
            return c
    return df.columns[0]

def select_mv_column(df: pd.DataFrame, time_col: str, pv_col: str) -> str:
    m = match_column(df, MV_CANDS_PRIMARY)
    if m:
        return m
    m2 = match_column(df, MV_CANDS_FALLBACK)
    if m2:
        return m2
    numeric_cols = [c for c in df.columns if c not in (time_col, pv_col) and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        best_col = None
        best_jump = 0.0
        for c in numeric_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if v.size < 3:
                continue
            du = np.diff(v)
            if du.size == 0:
                continue
            jump = np.nanmax(np.abs(du))
            if np.isfinite(jump) and jump > best_jump:
                best_jump = float(jump)
                best_col = c
        if best_col is not None:
            return best_col
        return numeric_cols[0]
    others = [c for c in df.columns if c not in (time_col, pv_col)]
    return others[0] if others else df.columns[0]

def coerce_numeric(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce")
    return s.to_numpy(dtype=float)

def estimate_fopdt_from_step(df: pd.DataFrame, time_col: str, pv_col: str, mv_col: str) -> Optional[Dict[str, float]]:
    t = coerce_numeric(df[time_col])
    y = coerce_numeric(df[pv_col])
    u = coerce_numeric(df[mv_col])
    mask = ~np.isnan(t) & ~np.isnan(y) & ~np.isnan(u)
    t = t[mask]; y = y[mask]; u = u[mask]
    if t.size < 10 or y.size < 10 or u.size < 10:
        return None
    du = np.diff(u)
    if du.size == 0:
        return None
    max_du = np.nanmax(np.abs(du))
    if not np.isfinite(max_du) or max_du < 1e-6:
        return None
    step_idx = int(np.argmax(np.abs(du) > 0.25 * max_du))
    if step_idx >= du.size:
        return None
    t_step = t[step_idx + 1]
    y0 = y[step_idx]
    y_ss = np.nanmean(y[int(0.8 * y.size) :])
    delta_y = y_ss - y0
    delta_u = u[step_idx + 1] - u[step_idx]
    if abs(delta_y) < 1e-6 or abs(delta_u) < 1e-6:
        return None
    y28 = y0 + 0.283 * delta_y
    y63 = y0 + 0.632 * delta_y
    t28_candidates = t[y >= y28]
    t63_candidates = t[y >= y63]
    if t28_candidates.size == 0 or t63_candidates.size == 0:
        return None
    t28 = float(t28_candidates[0])
    t63 = float(t63_candidates[0])
    theta = max(t28 - t_step, 1e-3)
    tau = max(t63 - t28, 1e-3)
    K = delta_y / delta_u
    return {"K": float(K), "tau": float(tau), "theta": float(theta), "t_step": float(t_step)}

def estimate_fopdt_from_pv_only(df: pd.DataFrame, time_col: str, pv_col: str) -> Optional[Dict[str, float]]:
    t = coerce_numeric(df[time_col])
    y = coerce_numeric(df[pv_col])
    mask = ~np.isnan(t) & ~np.isnan(y)
    t = t[mask]; y = y[mask]
    if t.size < 10 or y.size < 10:
        return None
    y0 = float(y[0])
    y_ss = float(np.nanmean(y[int(0.8 * y.size) :]))
    delta_y = y_ss - y0
    if abs(delta_y) < 1e-3:
        return None
    y28 = y0 + 0.283 * delta_y
    y63 = y0 + 0.632 * delta_y
    t0 = float(t[0])
    t28_candidates = t[y >= y28]
    t63_candidates = t[y >= y63]
    if t28_candidates.size == 0 or t63_candidates.size == 0:
        return None
    t28 = float(t28_candidates[0])
    t63 = float(t63_candidates[0])
    theta = max(t28 - t0, 1e-3)
    tau = max(t63 - t28, 1e-3)
    K = delta_y  # assume unit input, so gain ~ delta_y
    return {"K": float(K), "tau": float(tau), "theta": float(theta), "t_step": float(t0)}

def ziegler_nichols_pid(fopdt: Dict[str, float]) -> PIDTuningResult:
    K = fopdt["K"]
    tau = fopdt["tau"]
    theta = fopdt["theta"]
    if K == 0:
        K = 1.0
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

def make_workflow_from_fopdt(trial_name: str, fluid: str, fopdt: Optional[Dict[str, float]], method: str, mode: str, source: str) -> List[WorkflowItem]:
    items = []
    items.append(WorkflowItem(step="Trial identified", detail=f"Trial={trial_name}, fluid={fluid}, mode={mode}"))
    if fopdt:
        items.append(WorkflowItem(step=f"FOPDT estimation ({source})", detail=f"K={fopdt['K']:.3f}, tau={fopdt['tau']:.2f}s, theta={fopdt['theta']:.2f}s"))
    else:
        items.append(WorkflowItem(step=f"FOPDT estimation ({source})", detail="FOPDT not reliably detected"))
    items.append(WorkflowItem(step="Tuning method", detail=method))
    items.append(WorkflowItem(step="Lab constraints", detail="Small-scale PCT with dead time and heat exchanger"))
    return items

def preview_df(df: pd.DataFrame, n: int = 40) -> List[Dict[str, Any]]:
    return df.head(n).to_dict(orient="records")

def call_openai_for_pid(payload: Dict[str, Any]) -> Dict[str, Any]:
    system_msg = (
        "You are assisting with tuning PID parameters for a small process-control teaching plant. "
        "You will receive JSON with trial_name, fluid, optional prior_pid, optional FOPDT estimate, "
        "and a preview of time-series data. Return JSON with keys kp, ti, td, rationale. "
        "If fluid indicates mineral-oil–water and prior PID was for water, reduce kp and increase ti slightly. "
        "If FOPDT dead time is large (theta/tau > 0.3), keep gains conservative."
    )
    user_msg = json.dumps(payload)
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json", "").strip()
    try:
        data = json.loads(content)
    except Exception:
        data = {"kp": 1.0, "ti": 30.0, "td": 0.0, "rationale": "LLM output not JSON, fallback PID used."}
    return data

def ml_tune_pid(trial_name: str, fluid: str, df: pd.DataFrame, fopdt: Optional[Dict[str, float]], prior_pid: Optional[Dict[str, float]], objective: str) -> PIDTuningResult:
    payload = {
        "trial_name": trial_name,
        "fluid": fluid,
        "prior_pid": prior_pid,
        "fopdt": fopdt,
        "data_preview": preview_df(df),
        "objective": objective,
    }
    out = call_openai_for_pid(payload)
    kp = float(out.get("kp", 1.0))
    ti = float(out.get("ti", 30.0))
    td = float(out.get("td", 0.0))
    rationale = out.get("rationale", "No rationale provided")
    return PIDTuningResult(
        kp=kp,
        ti=ti,
        td=td,
        method="ML-tuned via OpenAI chat",
        notes=rationale,
    )

def normalize_mode(mode: str) -> Literal["day1", "day2"]:
    m = mode.strip().lower()
    if m in ["day1", "zn", "ziegler-nichols", "ziegler", "traditional"]:
        return "day1"
    if m in ["day2", "ml", "machine-learning", "ai", "agent"]:
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
    df = clean_export_df(df)
    time_col = select_time_column(df)
    pv_col = select_pv_column(df)
    mv_col = select_mv_column(df, time_col, pv_col)
    fopdt = estimate_fopdt_from_step(df, time_col, pv_col, mv_col)
    normalized_mode = normalize_mode(mode)
    if fopdt is None:
        fopdt = estimate_fopdt_from_pv_only(df, time_col, pv_col)
        source = "PV-only"
    else:
        source = "MV+PV"
    if normalized_mode == "day1":
        if fopdt:
            zn = ziegler_nichols_pid(fopdt)
            if fluid.lower().startswith("mineral"):
                tuned = conservative_from_zn(zn, factor=0.6)
            else:
                tuned = zn
            workflow = make_workflow_from_fopdt(trial_name, fluid, fopdt, tuned.method, normalized_mode, source)
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized_mode, suggested_pid=tuned, workflow=workflow)
        else:
            default_pid = PIDTuningResult(kp=1.0, ti=30.0, td=0.0, method="Fallback", notes="Could not infer dynamics from this dataset")
            workflow = make_workflow_from_fopdt(trial_name, fluid, None, "Fallback", normalized_mode, source)
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized_mode, suggested_pid=default_pid, workflow=workflow)
    else:
        tuned = ml_tune_pid(trial_name, fluid, df, fopdt, prior_pid, objective)
        workflow = make_workflow_from_fopdt(trial_name, fluid, fopdt, tuned.method, normalized_mode, source)
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

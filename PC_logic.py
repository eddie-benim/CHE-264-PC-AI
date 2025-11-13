import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Tuple
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

ELAPSED_IDX = 3
MV_IDX = 7
PV_IDX = 9

Kp_MIN = 0.005
Kp_MAX = 0.5
Ti_MIN = 60.0
Ti_MAX = 1200.0
Td_MIN = 0.0
Td_MAX = 120.0

TEMP_KEYS = ["temp", "°c", "deg c", "deg  c", "deg  c", "t2", "t_2", "hot water", "hw", "temperature"]
MV_KEYS = ["power", "pwr", "heater", "heat", "pump", "valve", "duty", "mv", "u", "speed", "kw", "%"]
TIME_KEYS = ["elapsed", "time", "timestamp", "t(s)", "t (s)", "t [s]", "seconds", "sec"]

def clamp_pid(pid: PIDTuningResult) -> PIDTuningResult:
    kp = max(Kp_MIN, min(pid.kp, Kp_MAX))
    ti = max(Ti_MIN, min(pid.ti, Ti_MAX))
    td = max(Td_MIN, min(pid.td, Td_MAX))
    return PIDTuningResult(kp=kp, ti=ti, td=td, method=pid.method, notes=pid.notes)

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

def get_col_by_index(df: pd.DataFrame, idx: int) -> pd.Series:
    if idx < df.shape[1]:
        return df.iloc[:, idx]
    return df.iloc[:, -1]

def parse_elapsed_to_seconds(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if np.count_nonzero(~np.isnan(numeric.to_numpy())) > 0.6 * len(series):
        return numeric.to_numpy(dtype=float)
    vals = series.astype(str).str.strip()
    out = []
    for v in vals:
        if v == "" or v.lower() == "nan":
            out.append(np.nan)
            continue
        parts = v.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0]); m = float(parts[1]); s = float(parts[2])
                out.append(h * 3600 + m * 60 + s)
            elif len(parts) == 2:
                m = float(parts[0]); s = float(parts[1])
                out.append(m * 60 + s)
            else:
                out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)

def coerce_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

def estimate_fopdt_mv_pv(t: np.ndarray, y: np.ndarray, u: np.ndarray) -> Optional[Dict[str, float]]:
    mask = ~np.isnan(t) & ~np.isnan(y) & ~np.isnan(u)
    t = t[mask]; y = y[mask]; u = u[mask]
    if t.size < 10:
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
    y_ss = np.nanmean(y[int(0.8 * y.size):])
    dy = y_ss - y0
    du_val = u[step_idx + 1] - u[step_idx]
    if abs(dy) < 1e-6 or abs(du_val) < 1e-6:
        return None
    y28 = y0 + 0.283 * dy
    y63 = y0 + 0.632 * dy
    t28_candidates = t[y >= y28]
    t63_candidates = t[y >= y63]
    if t28_candidates.size == 0 or t63_candidates.size == 0:
        return None
    t28 = float(t28_candidates[0])
    t63 = float(t63_candidates[0])
    theta = max(t28 - t_step, 1e-3)
    tau = max(t63 - t28, 1e-3)
    K = dy / du_val
    return {"K": float(K), "tau": float(tau), "theta": float(theta)}

def estimate_fopdt_pv_only(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    mask = ~np.isnan(t) & ~np.isnan(y)
    t = t[mask]; y = y[mask]
    if t.size < 10:
        return None
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    dy = ymax - ymin
    if abs(dy) < 0.05:
        return None
    t0 = float(t[np.argmin(y)])
    y28 = ymin + 0.283 * dy
    y63 = ymin + 0.632 * dy
    t28_candidates = t[y >= y28]
    t63_candidates = t[y >= y63]
    if t28_candidates.size == 0 or t63_candidates.size == 0:
        return None
    t28 = float(t28_candidates[0])
    t63 = float(t63_candidates[0])
    theta = max(t28 - t0, 1e-3)
    tau = max(t63 - t28, 1e-3)
    K = dy
    return {"K": float(K), "tau": float(tau), "theta": float(theta)}

def ziegler_nichols_pid(fopdt: Dict[str, float]) -> PIDTuningResult:
    K = fopdt["K"] if fopdt["K"] != 0 else 1.0
    tau = fopdt["tau"]
    theta = fopdt["theta"]
    kp = 1.2 * tau / (K * theta)
    ti = 2.0 * theta
    td = 0.5 * theta
    return PIDTuningResult(kp=float(kp), ti=float(ti), td=float(td), method="Ziegler-Nichols (open-loop)", notes="Based on exported PCT data")

def conservative_from_zn(zn: PIDTuningResult, factor: float = 0.7) -> PIDTuningResult:
    kp = zn.kp * factor
    ti = zn.ti * 1.2
    td = zn.td
    return PIDTuningResult(kp=float(kp), ti=float(ti), td=float(td), method="Conservative ZN", notes="Adjusted due to expected property change/mineral oil")

def to_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, (float,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.ndarray,)):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)

def compact_preview(t: np.ndarray, y: np.ndarray, u: Optional[np.ndarray], n: int = 120) -> Dict[str, List[Optional[float]]]:
    def clip_clean(arr):
        if arr is None:
            return None
        a = arr[:n]
        out = []
        for v in a:
            if v is None:
                out.append(None)
            elif isinstance(v, (float, np.floating)):
                if np.isnan(v) or np.isinf(v):
                    out.append(None)
                else:
                    out.append(float(v))
            elif isinstance(v, (int, np.integer)):
                out.append(int(v))
            else:
                try:
                    fv = float(v)
                    if np.isnan(fv) or np.isinf(fv):
                        out.append(None)
                    else:
                        out.append(fv)
                except:
                    out.append(None)
        return out
    return {"t": clip_clean(t), "y": clip_clean(y), "u": clip_clean(u) if u is not None else None}

def call_openai_for_pid(payload: Dict[str, Any]) -> Dict[str, Any]:
    safe_payload = to_json_safe(payload)
    user_content = json.dumps(safe_payload, allow_nan=False)
    system_msg = "You are assisting with tuning PID parameters for a small process control experiment. Return JSON with kp, ti, td, rationale."
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_content}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json", "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"kp": 1.0, "ti": 30.0, "td": 0.0, "rationale": "LLM output not JSON; fallback used."}

def ml_tune_pid(trial_name: str, fluid: str, t: np.ndarray, y: np.ndarray, u: Optional[np.ndarray], fopdt: Optional[Dict[str, float]], prior_pid: Optional[Dict[str, float]], objective: str) -> PIDTuningResult:
    payload = {
        "trial_name": trial_name,
        "fluid": fluid,
        "fopdt": fopdt,
        "prior_pid": to_json_safe(prior_pid),
        "objective": objective,
        "data_preview": compact_preview(t, y, u, n=180),
        "features": {
            "t_span": float(np.nanmax(t) - np.nanmin(t)) if t.size else 0.0,
            "y_range": float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0,
            "u_range": float(np.nanmax(u) - np.nanmin(u)) if (u is not None and u.size) else None,
            "y_slope_med": float(np.nanmedian(np.diff(y))) if y.size > 1 else 0.0,
        },
    }
    out = call_openai_for_pid(payload)
    pid = PIDTuningResult(
        kp=float(out.get("kp", 1.0)),
        ti=float(out.get("ti", 30.0)),
        td=float(out.get("td", 0.0)),
        method="ML-tuned via OpenAI chat",
        notes=out.get("rationale", "No rationale provided"),
    )
    return clamp_pid(pid)

def normalize_mode(mode: str) -> Literal["day1", "day2"]:
    m = mode.strip().lower()
    if "day2" in m or "ml" in m:
        return "day2"
    return "day1"

def monotonic_score(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    dx = np.diff(x)
    pos = np.sum(dx >= 0)
    neg = np.sum(dx <= 0)
    return float(max(pos, neg) / dx.size)

def keyword_score(name: str, keys: List[str]) -> int:
    n = name.lower()
    return sum(1 for k in keys if k in n)

def lowfreq_variance(x: np.ndarray) -> float:
    if x.size < 5:
        return 0.0
    k = min(31, x.size // 5 * 2 + 1)
    if k < 5:
        return float(np.nanvar(x))
    from numpy.lib.stride_tricks import sliding_window_view
    try:
        w = sliding_window_view(x, k).mean(axis=1)
        return float(np.nanvar(w))
    except Exception:
        return float(np.nanvar(x))

def detect_columns_generic(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    cols = list(df.columns)
    candidates = []
    for c in cols:
        s = df[c]
        v = coerce_numeric(s)
        num_ratio = np.sum(~np.isnan(v)) / max(1, len(v))
        score = 0.0
        ks = keyword_score(c, TIME_KEYS)
        if ks:
            score += 3 * ks
        score += 2.0 * monotonic_score(v)
        if num_ratio > 0.9:
            score += 0.5
        candidates.append((score, c))
    candidates.sort(reverse=True)
    time_col = candidates[0][1] if candidates else cols[0]
    pv_candidates = []
    for c in cols:
        if c == time_col:
            continue
        v = coerce_numeric(df[c])
        score = 0.0
        score += 2.0 * keyword_score(c, TEMP_KEYS)
        score += 1.0 * lowfreq_variance(v)
        pv_candidates.append((score, c))
    pv_candidates.sort(reverse=True)
    pv_col = pv_candidates[0][1] if pv_candidates else (cols[1] if len(cols) > 1 else cols[0])
    mv_candidates = []
    for c in cols:
        if c in (time_col, pv_col):
            continue
        v = coerce_numeric(df[c])
        dv = np.diff(v)
        score = 0.0
        score += 2.0 * keyword_score(c, MV_KEYS)
        score += 1.5 * float(np.nanmax(np.abs(dv)) if dv.size else 0.0)
        mv_candidates.append((score, c))
    mv_candidates.sort(reverse=True)
    mv_col = mv_candidates[0][1] if mv_candidates else None
    return time_col, pv_col, mv_col

def make_workflow(trial: str, fluid: str, mode: str, source: str, fopdt: Optional[Dict[str, float]], method: str) -> List[WorkflowItem]:
    items = [WorkflowItem(step="Trial identified", detail=f"{trial}, fluid={fluid}, mode={mode}")]
    if fopdt:
        items.append(WorkflowItem(step=f"FOPDT estimation ({source})", detail=f"K={fopdt['K']:.3f}, tau={fopdt['tau']:.2f}s, theta={fopdt['theta']:.2f}s"))
    else:
        items.append(WorkflowItem(step=f"FOPDT estimation ({source})", detail="could not estimate"))
    items.append(WorkflowItem(step="Tuning method", detail=method))
    items.append(WorkflowItem(step="Lab constraints", detail="PCT with heat exchanger and disturbance"))
    return items

def analyze_trial_excel(
    file_bytes: bytes,
    trial_name: str,
    fluid: str,
    mode: str,
    prior_pid: Optional[Dict[str, float]] = None,
    objective: str = "Minimize overshoot and settling time while rejecting ice-pack disturbance",
) -> TrialAnalysisResult:
    df = read_excel_file(file_bytes)
    df = clean_export_df(df)
    normalized = normalize_mode(mode)
    if normalized == "day1":
        t_raw = get_col_by_index(df, ELAPSED_IDX)
        mv_raw = get_col_by_index(df, MV_IDX)
        pv_raw = get_col_by_index(df, PV_IDX)
    else:
        time_col, pv_col, mv_col = detect_columns_generic(df)
        t_raw = df[time_col]
        pv_raw = df[pv_col]
        mv_raw = df[mv_col] if mv_col is not None else pd.Series([np.nan] * len(df))
    t = parse_elapsed_to_seconds(t_raw)
    y = coerce_numeric(pv_raw)
    u = coerce_numeric(mv_raw) if mv_raw is not None else np.full_like(y, np.nan)
    fopdt = estimate_fopdt_mv_pv(t, y, u) if np.any(np.isfinite(u)) else None
    source = "MV+PV" if fopdt is not None else "PV-only"
    if fopdt is None:
        fopdt = estimate_fopdt_pv_only(t, y)
    if normalized == "day1":
        if fopdt:
            zn = ziegler_nichols_pid(fopdt)
            if fluid.lower().startswith("mineral"):
                tuned = conservative_from_zn(zn, factor=0.6)
            else:
                tuned = zn
            tuned = clamp_pid(tuned)
            wf = make_workflow(trial_name, fluid, normalized, source, fopdt, tuned.method)
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized, suggested_pid=tuned, workflow=wf)
        else:
            fb = PIDTuningResult(kp=0.05, ti=300.0, td=0.0, method="Fallback", notes="Could not infer dynamics from data")
            wf = make_workflow(trial_name, fluid, normalized, source, None, "Fallback")
            return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized, suggested_pid=fb, workflow=wf)
    else:
        tuned = ml_tune_pid(trial_name, fluid, t, y, u if np.any(np.isfinite(u)) else None, fopdt, prior_pid, objective)
        wf = make_workflow(trial_name, fluid, normalized, source, fopdt, tuned.method)
        wf.append(WorkflowItem(step="ML rationale", detail=tuned.notes))
        return TrialAnalysisResult(trial_name=trial_name, fluid=fluid, mode=normalized, suggested_pid=tuned, workflow=wf)

def analyze_multiple_trials(
    uploads: List[bytes],
    trial_names: List[str],
    fluids: List[str],
    mode: str,
    prior_pids: Optional[List[Optional[Dict[str, float]]]] = None,
    objective: str = "Minimize overshoot and settling time while rejecting ice-pack disturbance",
) -> List[TrialAnalysisResult]:
    if prior_pids is None:
        prior_pids = [None] * len(uploads)
    out = []
    for fb, name, fld, pp in zip(uploads, trial_names, fluids, prior_pids):
        out.append(analyze_trial_excel(fb, name, fld, mode, pp, objective))
    return out

def serialize_results(results: List[TrialAnalysisResult]) -> str:
    return json.dumps([r.model_dump() for r in results], indent=2)

def current_timestamp_meta() -> Dict[str, str]:
    now = datetime.now()
    return {"date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"), "day_of_week": now.strftime("%A")}

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_MONITOR_NAME = "calibration_monitor.csv"
DEFAULT_QUANTILES = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
DEFAULT_Q_VIOLATION = 0.80
DEFAULT_Q_NEAR_MISS = 0.95


def find_run_dirs_with_monitor(runs_dir: Path, monitor_name: str = DEFAULT_MONITOR_NAME) -> List[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")
    out: List[Path] = []
    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / monitor_name).exists():
            out.append(p)
    return out


def _coerce_metric_column(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    if "proxy_cost" in df.columns:
        df = df.copy()
        df["metric_value"] = pd.to_numeric(df["proxy_cost"], errors="coerce")
        df["metric_name"] = "proxy_cost"
        return df
    if "clearance" in df.columns:
        df = df.copy()
        df["metric_value"] = pd.to_numeric(df["clearance"], errors="coerce")
        df["metric_name"] = "clearance"
        return df
    raise ValueError(f"Missing 'proxy_cost' or 'clearance' in {csv_path}")


def load_one_csv(csv_path: Path, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ctx_id" not in df.columns:
        raise ValueError(f"Missing 'ctx_id' in {csv_path}")

    df = _coerce_metric_column(df, csv_path)
    df = df.copy()
    df["run_name"] = run_name
    if "ctx_tuple" not in df.columns:
        df["ctx_tuple"] = ""
    df["ctx_tuple"] = df["ctx_tuple"].fillna("").astype(str)

    for col in ["tau_violation", "tau_near_miss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    for col in ["near_miss", "violation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    return df


def load_data(run_dir: Optional[Path], runs_dir: Optional[Path], monitor_name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if run_dir is not None:
        csv_path = run_dir / monitor_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {monitor_name} in {run_dir}")
        frames.append(load_one_csv(csv_path, run_dir.name))
    elif runs_dir is not None:
        run_dirs = find_run_dirs_with_monitor(runs_dir, monitor_name=monitor_name)
        if not run_dirs:
            raise FileNotFoundError(f"No subdirectories with {monitor_name} found under {runs_dir}")
        for rd in run_dirs:
            frames.append(load_one_csv(rd / monitor_name, rd.name))
    else:
        raise ValueError("Either run_dir or runs_dir must be provided")

    df = pd.concat(frames, ignore_index=True)
    df = df[np.isfinite(df["metric_value"])].copy()
    if df.empty:
        raise ValueError("No finite metric values found in input CSV(s).")
    return df


def quantile_summary_by_context(df: pd.DataFrame, quantiles: List[float]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["ctx_id", "ctx_tuple"], dropna=False)
    for (ctx_id, ctx_tuple), g in grouped:
        vals = g["metric_value"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        row: Dict[str, object] = {
            "ctx_id": ctx_id,
            "ctx_tuple": ctx_tuple,
            "rows": int(len(g)),
            "metric_name": str(g["metric_name"].iloc[0]),
            "metric_min": float(np.min(vals)),
            "metric_mean": float(np.mean(vals)),
            "metric_median": float(np.median(vals)),
            "metric_max": float(np.max(vals)),
            "current_tau_violation_mean": float(g["tau_violation"].mean()) if g["tau_violation"].notna().any() else np.nan,
            "current_tau_near_miss_mean": float(g["tau_near_miss"].mean()) if g["tau_near_miss"].notna().any() else np.nan,
            "current_violation_rate": float(g["violation"].mean()),
            "current_near_miss_rate": float(g["near_miss"].mean()),
        }
        for q in quantiles:
            row[f"q{int(round(q * 100)):02d}"] = float(np.quantile(vals, q))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No context-level rows produced.")
    return out.sort_values(["ctx_id", "ctx_tuple"]).reset_index(drop=True)


def recommend_thresholds(quant_df: pd.DataFrame, q_violation: float, q_near_miss: float, min_gap: float) -> pd.DataFrame:
    qv_key = f"q{int(round(q_violation * 100)):02d}"
    qn_key = f"q{int(round(q_near_miss * 100)):02d}"
    if qv_key not in quant_df.columns:
        raise ValueError(f"Quantile column missing: {qv_key}")
    if qn_key not in quant_df.columns:
        raise ValueError(f"Quantile column missing: {qn_key}")

    rec = quant_df.copy()
    rec["recommended_tau_violation"] = rec[qv_key].astype(float)
    rec["recommended_tau_near_miss"] = np.maximum(
        rec[qn_key].to_numpy(dtype=float),
        rec[qv_key].to_numpy(dtype=float) + float(min_gap),
    )
    rec["delta_violation_vs_current"] = rec["recommended_tau_violation"] - rec["current_tau_violation_mean"]
    rec["delta_near_miss_vs_current"] = rec["recommended_tau_near_miss"] - rec["current_tau_near_miss_mean"]
    keep_cols = [
        "ctx_id", "ctx_tuple", "rows", "metric_name",
        "current_tau_violation_mean", "current_tau_near_miss_mean",
        "current_violation_rate", "current_near_miss_rate",
        qv_key, qn_key,
        "recommended_tau_violation", "recommended_tau_near_miss",
        "delta_violation_vs_current", "delta_near_miss_vs_current",
    ]
    return rec[keep_cols].copy()


def global_summary(df: pd.DataFrame, quantiles: List[float]) -> pd.DataFrame:
    vals = df["metric_value"].dropna().to_numpy()
    row: Dict[str, object] = {
        "rows": int(len(df)),
        "runs": int(df["run_name"].nunique()),
        "contexts": int(df["ctx_id"].nunique(dropna=False)),
        "metric_name": str(df["metric_name"].iloc[0]),
        "metric_min": float(np.min(vals)),
        "metric_mean": float(np.mean(vals)),
        "metric_median": float(np.median(vals)),
        "metric_max": float(np.max(vals)),
        "current_violation_rate": float(df["violation"].mean()),
        "current_near_miss_rate": float(df["near_miss"].mean()),
        "current_tau_violation_mean": float(df["tau_violation"].mean()) if df["tau_violation"].notna().any() else np.nan,
        "current_tau_near_miss_mean": float(df["tau_near_miss"].mean()) if df["tau_near_miss"].notna().any() else np.nan,
    }
    for q in quantiles:
        row[f"q{int(round(q * 100)):02d}"] = float(np.quantile(vals, q))
    return pd.DataFrame([row])


def build_patch_json(recommend_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    patch: Dict[str, Dict[str, float]] = {}
    for _, r in recommend_df.iterrows():
        patch[str(r["ctx_tuple"])] = {
            "tau_violation": float(r["recommended_tau_violation"]),
            "tau_near_miss": float(r["recommended_tau_near_miss"]),
        }
    return patch


def print_console_report(global_df: pd.DataFrame, recommend_df: pd.DataFrame, q_violation: float, q_near_miss: float, min_gap: float) -> None:
    g = global_df.iloc[0]
    print("\n=== THRESHOLD CALIBRATION SUMMARY ===")
    print(f"rows: {int(g['rows'])}")
    print(f"runs: {int(g['runs'])}")
    print(f"contexts: {int(g['contexts'])}")
    print(f"metric_name: {g['metric_name']}")
    print(f"current_violation_rate: {float(g['current_violation_rate']):.6f}")
    print(f"current_near_miss_rate: {float(g['current_near_miss_rate']):.6f}")
    print(f"current_tau_violation_mean: {float(g['current_tau_violation_mean']):.6f}")
    print(f"current_tau_near_miss_mean: {float(g['current_tau_near_miss_mean']):.6f}")
    print("")
    print(
        f"recommended rule: tau_violation = q{int(round(q_violation * 100)):02d}, "
        f"tau_near_miss = q{int(round(q_near_miss * 100)):02d}, "
        f"min_gap = {min_gap:.3f}"
    )
    print("")
    show_cols = [
        "ctx_id", "ctx_tuple", "rows", "metric_name",
        "current_tau_violation_mean", "current_tau_near_miss_mean",
        "current_violation_rate", "current_near_miss_rate",
        "recommended_tau_violation", "recommended_tau_near_miss",
    ]
    print(recommend_df[show_cols].to_string(index=False))


def save_outputs(out_dir: Path, quant_df: pd.DataFrame, global_df: pd.DataFrame, recommend_df: pd.DataFrame, patch_json: Dict[str, Dict[str, float]], meta: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    quant_df.to_csv(out_dir / "metric_quantiles_by_context.csv", index=False)
    global_df.to_csv(out_dir / "metric_quantiles_global.csv", index=False)
    recommend_df.to_csv(out_dir / "recommended_thresholds_by_context.csv", index=False)
    with open(out_dir / "recommended_thresholds_patch.json", "w", encoding="utf-8") as f:
        json.dump(patch_json, f, indent=2)
    with open(out_dir / "calibration_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate threshold patch from calibration_monitor.csv files.")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--runs_dir", type=Path, default=None)
    parser.add_argument("--monitor_name", type=str, default=DEFAULT_MONITOR_NAME)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--quantiles", type=float, nargs="*", default=DEFAULT_QUANTILES)
    parser.add_argument("--q_violation", type=float, default=DEFAULT_Q_VIOLATION)
    parser.add_argument("--q_near_miss", type=float, default=DEFAULT_Q_NEAR_MISS)
    parser.add_argument("--min_gap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.run_dir is None) == (args.runs_dir is None):
        raise ValueError("Provide exactly one of --run_dir or --runs_dir")
    df = load_data(args.run_dir, args.runs_dir, monitor_name=args.monitor_name)
    quant_df = quantile_summary_by_context(df, args.quantiles)
    global_df = global_summary(df, args.quantiles)
    recommend_df = recommend_thresholds(quant_df, args.q_violation, args.q_near_miss, args.min_gap)
    patch_json = build_patch_json(recommend_df)
    meta = {
        "run_dir": None if args.run_dir is None else str(args.run_dir),
        "runs_dir": None if args.runs_dir is None else str(args.runs_dir),
        "monitor_name": args.monitor_name,
        "quantiles": list(args.quantiles),
        "q_violation": args.q_violation,
        "q_near_miss": args.q_near_miss,
        "min_gap": args.min_gap,
    }
    print_console_report(global_df, recommend_df, args.q_violation, args.q_near_miss, args.min_gap)
    save_outputs(args.out_dir, quant_df, global_df, recommend_df, patch_json, meta)
    print(f"\nSaved outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()

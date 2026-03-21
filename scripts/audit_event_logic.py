#!/usr/bin/env python3
"""
Audit event logic in train_monitor.csv for context-dependent safety thresholds.

Checks:
1) required columns exist
2) tau_violation < tau_near_miss on every row
3) violation matches (clearance < tau_violation)
4) near_miss matches (tau_violation <= clearance < tau_near_miss)
5) near_miss and violation do not overlap
6) thresholds vary by context when expected
7) writes summary CSVs and suspicious rows CSVs

Example usage:
    python -m scripts.audit_event_logic --run_dir runs/my_run
    python scripts/audit_event_logic.py --run_dir runs/my_run
    python -m scripts.audit_event_logic --runs_dir runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "clearance",
    "near_miss",
    "violation",
    "shield_used",
    "shield_reason",
    "ctx_id",
    "tau_near_miss",
    "tau_violation",
]


OPTIONAL_CONTEXT_COLUMN = "ctx_tuple"
DEFAULT_MONITOR_NAME = "train_monitor.csv"


@dataclass
class AuditResult:
    run_name: str
    csv_path: Path
    n_rows: int
    missing_columns: List[str]
    bad_threshold_order_count: int
    violation_mismatch_count: int
    near_miss_mismatch_count: int
    overlap_count: int
    null_clearance_count: int
    null_tau_near_miss_count: int
    null_tau_violation_count: int
    contexts_present: int
    tau_near_miss_unique_global: int
    tau_violation_unique_global: int
    tau_varies_by_context: int
    near_miss_rate: float
    violation_rate: float
    shield_used_rate: float
    clearance_min: float
    clearance_mean: float
    passed_basic_checks: bool


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_int01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s > 0).astype(int)


def _load_monitor_csv(run_dir: Path, monitor_name: str = DEFAULT_MONITOR_NAME) -> Path:
    csv_path = run_dir / monitor_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {monitor_name} in {run_dir}")
    return csv_path


def _compute_expected_flags(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    clearance = _safe_numeric(df["clearance"])
    tau_near = _safe_numeric(df["tau_near_miss"])
    tau_viol = _safe_numeric(df["tau_violation"])

    expected_violation = (clearance < tau_viol).astype(int)
    expected_near_miss = ((clearance >= tau_viol) & (clearance < tau_near)).astype(int)
    return expected_near_miss, expected_violation


def _context_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ctx_id", "tau_near_miss", "tau_violation"]
    if OPTIONAL_CONTEXT_COLUMN in df.columns:
        cols.append(OPTIONAL_CONTEXT_COLUMN)

    tmp = df[cols].copy()
    tmp["tau_near_miss"] = _safe_numeric(tmp["tau_near_miss"])
    tmp["tau_violation"] = _safe_numeric(tmp["tau_violation"])

    agg = (
        tmp.groupby("ctx_id", dropna=False)
        .agg(
            rows=("ctx_id", "size"),
            tau_near_miss_mean=("tau_near_miss", "mean"),
            tau_near_miss_min=("tau_near_miss", "min"),
            tau_near_miss_max=("tau_near_miss", "max"),
            tau_near_miss_nunique=("tau_near_miss", "nunique"),
            tau_violation_mean=("tau_violation", "mean"),
            tau_violation_min=("tau_violation", "min"),
            tau_violation_max=("tau_violation", "max"),
            tau_violation_nunique=("tau_violation", "nunique"),
        )
        .reset_index()
    )

    if OPTIONAL_CONTEXT_COLUMN in df.columns:
        ctx_tuple_map = (
            df[["ctx_id", OPTIONAL_CONTEXT_COLUMN]]
            .drop_duplicates()
            .groupby("ctx_id", dropna=False)[OPTIONAL_CONTEXT_COLUMN]
            .first()
            .reset_index()
        )
        agg = agg.merge(ctx_tuple_map, on="ctx_id", how="left")

    return agg


def _event_summary_by_context(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["clearance"] = _safe_numeric(tmp["clearance"])
    tmp["near_miss"] = _safe_int01(tmp["near_miss"])
    tmp["violation"] = _safe_int01(tmp["violation"])
    tmp["shield_used"] = _safe_int01(tmp["shield_used"])

    agg = (
        tmp.groupby("ctx_id", dropna=False)
        .agg(
            rows=("ctx_id", "size"),
            clearance_mean=("clearance", "mean"),
            clearance_min=("clearance", "min"),
            near_miss_rate=("near_miss", "mean"),
            violation_rate=("violation", "mean"),
            shield_used_rate=("shield_used", "mean"),
        )
        .reset_index()
    )

    if OPTIONAL_CONTEXT_COLUMN in df.columns:
        ctx_tuple_map = (
            df[["ctx_id", OPTIONAL_CONTEXT_COLUMN]]
            .drop_duplicates()
            .groupby("ctx_id", dropna=False)[OPTIONAL_CONTEXT_COLUMN]
            .first()
            .reset_index()
        )
        agg = agg.merge(ctx_tuple_map, on="ctx_id", how="left")

    return agg


def audit_one_run(run_dir: Path, out_subdir_name: str = "audit") -> AuditResult:
    csv_path = _load_monitor_csv(run_dir)
    df = pd.read_csv(csv_path)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    out_dir = run_dir / out_subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if missing_columns:
        result = AuditResult(
            run_name=run_dir.name,
            csv_path=csv_path,
            n_rows=len(df),
            missing_columns=missing_columns,
            bad_threshold_order_count=-1,
            violation_mismatch_count=-1,
            near_miss_mismatch_count=-1,
            overlap_count=-1,
            null_clearance_count=int(df["clearance"].isna().sum()) if "clearance" in df.columns else -1,
            null_tau_near_miss_count=int(df["tau_near_miss"].isna().sum()) if "tau_near_miss" in df.columns else -1,
            null_tau_violation_count=int(df["tau_violation"].isna().sum()) if "tau_violation" in df.columns else -1,
            contexts_present=int(df["ctx_id"].nunique(dropna=False)) if "ctx_id" in df.columns else -1,
            tau_near_miss_unique_global=int(df["tau_near_miss"].nunique(dropna=False)) if "tau_near_miss" in df.columns else -1,
            tau_violation_unique_global=int(df["tau_violation"].nunique(dropna=False)) if "tau_violation" in df.columns else -1,
            tau_varies_by_context=-1,
            near_miss_rate=float(_safe_int01(df["near_miss"]).mean()) if "near_miss" in df.columns else np.nan,
            violation_rate=float(_safe_int01(df["violation"]).mean()) if "violation" in df.columns else np.nan,
            shield_used_rate=float(_safe_int01(df["shield_used"]).mean()) if "shield_used" in df.columns else np.nan,
            clearance_min=float(_safe_numeric(df["clearance"]).min()) if "clearance" in df.columns else np.nan,
            clearance_mean=float(_safe_numeric(df["clearance"]).mean()) if "clearance" in df.columns else np.nan,
            passed_basic_checks=False,
        )
        with open(out_dir / "audit_error.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_dir.name,
                    "error": "missing_required_columns",
                    "missing_columns": missing_columns,
                },
                f,
                indent=2,
            )
        return result

    # Normalize types
    df = df.copy()
    df["clearance"] = _safe_numeric(df["clearance"])
    df["tau_near_miss"] = _safe_numeric(df["tau_near_miss"])
    df["tau_violation"] = _safe_numeric(df["tau_violation"])
    df["near_miss"] = _safe_int01(df["near_miss"])
    df["violation"] = _safe_int01(df["violation"])
    df["shield_used"] = _safe_int01(df["shield_used"])

    null_clearance_count = int(df["clearance"].isna().sum())
    null_tau_near_miss_count = int(df["tau_near_miss"].isna().sum())
    null_tau_violation_count = int(df["tau_violation"].isna().sum())

    bad_threshold_order_mask = df["tau_violation"] >= df["tau_near_miss"]
    bad_threshold_order = df[bad_threshold_order_mask].copy()

    expected_near_miss, expected_violation = _compute_expected_flags(df)

    violation_mismatch_mask = df["violation"] != expected_violation
    near_miss_mismatch_mask = df["near_miss"] != expected_near_miss
    overlap_mask = (df["near_miss"] == 1) & (df["violation"] == 1)

    violation_mismatch = df[violation_mismatch_mask].copy()
    near_miss_mismatch = df[near_miss_mismatch_mask].copy()
    overlap_rows = df[overlap_mask].copy()

    # Save suspicious rows
    bad_threshold_order.to_csv(out_dir / "bad_threshold_order_rows.csv", index=False)
    violation_mismatch.to_csv(out_dir / "violation_mismatch_rows.csv", index=False)
    near_miss_mismatch.to_csv(out_dir / "near_miss_mismatch_rows.csv", index=False)
    overlap_rows.to_csv(out_dir / "near_violation_overlap_rows.csv", index=False)

    # Context summaries
    threshold_summary = _context_threshold_summary(df)
    threshold_summary.to_csv(out_dir / "threshold_summary_by_context.csv", index=False)

    event_context_summary = _event_summary_by_context(df)
    event_context_summary.to_csv(out_dir / "event_summary_by_context.csv", index=False)

    contexts_present = int(df["ctx_id"].nunique(dropna=False))
    tau_near_miss_unique_global = int(df["tau_near_miss"].nunique(dropna=False))
    tau_violation_unique_global = int(df["tau_violation"].nunique(dropna=False))

    # Does threshold vary by context?
    by_ctx = (
        df.groupby("ctx_id", dropna=False)[["tau_near_miss", "tau_violation"]]
        .agg(["mean"])
        .reset_index()
    )
    tau_near_ctx_unique = int(by_ctx[("tau_near_miss", "mean")].nunique(dropna=False))
    tau_viol_ctx_unique = int(by_ctx[("tau_violation", "mean")].nunique(dropna=False))
    tau_varies_by_context = int((tau_near_ctx_unique > 1) or (tau_viol_ctx_unique > 1))

    passed_basic_checks = (
        len(missing_columns) == 0
        and int(bad_threshold_order_mask.sum()) == 0
        and int(violation_mismatch_mask.sum()) == 0
        and int(near_miss_mismatch_mask.sum()) == 0
        and int(overlap_mask.sum()) == 0
    )

    result = AuditResult(
        run_name=run_dir.name,
        csv_path=csv_path,
        n_rows=len(df),
        missing_columns=missing_columns,
        bad_threshold_order_count=int(bad_threshold_order_mask.sum()),
        violation_mismatch_count=int(violation_mismatch_mask.sum()),
        near_miss_mismatch_count=int(near_miss_mismatch_mask.sum()),
        overlap_count=int(overlap_mask.sum()),
        null_clearance_count=null_clearance_count,
        null_tau_near_miss_count=null_tau_near_miss_count,
        null_tau_violation_count=null_tau_violation_count,
        contexts_present=contexts_present,
        tau_near_miss_unique_global=tau_near_miss_unique_global,
        tau_violation_unique_global=tau_violation_unique_global,
        tau_varies_by_context=tau_varies_by_context,
        near_miss_rate=float(df["near_miss"].mean()),
        violation_rate=float(df["violation"].mean()),
        shield_used_rate=float(df["shield_used"].mean()),
        clearance_min=float(df["clearance"].min()),
        clearance_mean=float(df["clearance"].mean()),
        passed_basic_checks=passed_basic_checks,
    )

    pd.DataFrame([result.__dict__]).to_csv(out_dir / "audit_summary.csv", index=False)

    # Nice compact text report
    report_lines = [
        f"Run: {result.run_name}",
        f"CSV: {result.csv_path}",
        f"Rows: {result.n_rows}",
        "",
        "Basic checks:",
        f"  missing_columns: {result.missing_columns}",
        f"  bad_threshold_order_count: {result.bad_threshold_order_count}",
        f"  violation_mismatch_count: {result.violation_mismatch_count}",
        f"  near_miss_mismatch_count: {result.near_miss_mismatch_count}",
        f"  overlap_count: {result.overlap_count}",
        "",
        "Nulls:",
        f"  null_clearance_count: {result.null_clearance_count}",
        f"  null_tau_near_miss_count: {result.null_tau_near_miss_count}",
        f"  null_tau_violation_count: {result.null_tau_violation_count}",
        "",
        "Global summary:",
        f"  contexts_present: {result.contexts_present}",
        f"  tau_near_miss_unique_global: {result.tau_near_miss_unique_global}",
        f"  tau_violation_unique_global: {result.tau_violation_unique_global}",
        f"  tau_varies_by_context: {result.tau_varies_by_context}",
        f"  near_miss_rate: {result.near_miss_rate:.6f}",
        f"  violation_rate: {result.violation_rate:.6f}",
        f"  shield_used_rate: {result.shield_used_rate:.6f}",
        f"  clearance_min: {result.clearance_min:.6f}",
        f"  clearance_mean: {result.clearance_mean:.6f}",
        "",
        f"PASSED_BASIC_CHECKS: {result.passed_basic_checks}",
        "",
        "Wrote:",
        f"  {out_dir / 'audit_summary.csv'}",
        f"  {out_dir / 'threshold_summary_by_context.csv'}",
        f"  {out_dir / 'event_summary_by_context.csv'}",
        f"  {out_dir / 'bad_threshold_order_rows.csv'}",
        f"  {out_dir / 'violation_mismatch_rows.csv'}",
        f"  {out_dir / 'near_miss_mismatch_rows.csv'}",
        f"  {out_dir / 'near_violation_overlap_rows.csv'}",
    ]
    (out_dir / "audit_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return result


def _discover_run_dirs(runs_dir: Path, monitor_name: str = DEFAULT_MONITOR_NAME) -> List[Path]:
    run_dirs = []
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    for p in sorted(runs_dir.iterdir()):
        if p.is_dir() and (p / monitor_name).exists():
            run_dirs.append(p)
    return run_dirs


def audit_many_runs(runs_dir: Path, out_csv_name: str = "audit_all_runs_summary.csv") -> pd.DataFrame:
    run_dirs = _discover_run_dirs(runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories with {DEFAULT_MONITOR_NAME} found under {runs_dir}")

    rows = []
    for run_dir in run_dirs:
        print(f"[AUDIT] {run_dir}")
        result = audit_one_run(run_dir)
        rows.append(result.__dict__)

    summary_df = pd.DataFrame(rows).sort_values(by=["passed_basic_checks", "run_name"], ascending=[True, True])
    out_path = runs_dir / out_csv_name
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved multi-run audit summary to: {out_path}")
    return summary_df


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit event logic in train_monitor.csv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run_dir",
        type=str,
        help="Path to a single run directory containing train_monitor.csv",
    )
    group.add_argument(
        "--runs_dir",
        type=str,
        help="Path to a directory whose subdirectories each contain train_monitor.csv",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        result = audit_one_run(run_dir)

        print("\n=== AUDIT SUMMARY ===")
        for k, v in result.__dict__.items():
            print(f"{k}: {v}")

        print("\n=== INTERPRETATION ===")
        if result.missing_columns:
            print(f"- Missing required columns: {result.missing_columns}")
        if result.bad_threshold_order_count > 0:
            print("- Some rows violate tau_violation < tau_near_miss.")
        if result.violation_mismatch_count > 0:
            print("- Some logged violation flags do not match clearance < tau_violation.")
        if result.near_miss_mismatch_count > 0:
            print("- Some logged near_miss flags do not match threshold interval logic.")
        if result.overlap_count > 0:
            print("- Some rows have both near_miss=1 and violation=1.")
        if result.tau_varies_by_context == 0:
            print("- Thresholds do not appear to vary by context in this run.")
        if result.passed_basic_checks:
            print("- Basic event logic checks passed.")
        else:
            print("- Basic event logic checks did NOT fully pass. Inspect audit/*.csv files.")
    else:
        runs_dir = Path(args.runs_dir)
        df = audit_many_runs(runs_dir)
        print("\n=== MULTI-RUN SUMMARY ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
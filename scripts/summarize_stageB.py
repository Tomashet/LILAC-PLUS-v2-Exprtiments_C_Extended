#!/usr/bin/env python3
"""
Summarize Stage B runs for the LILAC+ thesis pipeline.

Expected per-run files:
- train_monitor.csv            <-- main episode-level summary source
- calibration_monitor.csv      <-- optional diagnostic/calibration file (not used here)

Outputs:
- artifacts/stageB_summary/stageB_per_run.csv
- artifacts/stageB_summary/stageB_grouped_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


STAGE_B_METHODS = [
    "adjust_speed_only",
    "lilac_context",
    "lilac_speed",
    "lilac_soft2hard",
]

REGIMES = [
    "stationary",
    "nonstationary_seen",
    "nonstationary_unseen",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage B experiment runs.")
    parser.add_argument("--runs_dir", type=str, default="runs_thesis")
    parser.add_argument("--out_dir", type=str, default="artifacts/stageB_summary")
    return parser.parse_args()


def extract_method_regime_seed(run_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Extract method, regime, and seed from a run directory name.

    Important:
    - longest-match-first for regime avoids 'stationary' matching inside
      'nonstationary_seen' or 'nonstationary_unseen'
    """
    method = None
    for m in sorted(STAGE_B_METHODS, key=len, reverse=True):
        if m in run_name:
            method = m
            break

    regime = None
    for r in sorted(REGIMES, key=len, reverse=True):
        if r in run_name:
            regime = r
            break

    seed = None
    m = re.search(r"(?:^|[_-])s(?:eed)?[_-]?(\d+)(?:$|[_-])", run_name)
    if m:
        seed = int(m.group(1))
    else:
        m2 = re.search(r"[_-](\d+)$", run_name)
        if m2:
            seed = int(m2.group(1))

    return method, regime, seed


def find_train_monitor(run_dir: Path) -> Optional[Path]:
    """
    Prefer train_monitor.csv. If naming changes, fall back to the best matching CSV.
    """
    preferred = run_dir / "train_monitor.csv"
    if preferred.exists():
        return preferred

    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        return None

    ranked: List[Tuple[int, Path]] = []
    for p in csvs:
        name = p.name.lower()
        score = 0
        if "train_monitor" in name:
            score += 100
        if "monitor" in name:
            score += 20
        if "train" in name:
            score += 10
        if "calibration" in name:
            score -= 50
        ranked.append((score, p))

    ranked.sort(key=lambda x: (x[0], x[1].name), reverse=True)
    return ranked[0][1]


def read_monitor_csv(csv_path: Path) -> pd.DataFrame:
    """
    train_monitor.csv starts with a metadata line beginning with '#'.
    """
    return pd.read_csv(csv_path, comment="#")


def series_mean_std(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    if col not in df.columns:
        return math.nan, math.nan

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return math.nan, math.nan

    mean_val = float(s.mean())
    std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return mean_val, std_val


def summarize_run(run_dir: Path) -> Optional[Dict]:
    run_name = run_dir.name
    method, regime, seed = extract_method_regime_seed(run_name)

    if method not in STAGE_B_METHODS:
        return None

    if regime not in REGIMES:
        print(f"[WARN] Could not parse regime for run: {run_name}")
        return None

    csv_path = find_train_monitor(run_dir)
    if csv_path is None:
        print(f"[WARN] No CSV found in {run_name}")
        return None

    df = read_monitor_csv(csv_path)

    print(f"[INFO] {run_name} -> {csv_path.name}")
    print(f"[INFO] Columns: {list(df.columns)}")

    r_mean, r_std = series_mean_std(df, "r")
    l_mean, l_std = series_mean_std(df, "l")
    v_mean, v_std = series_mean_std(df, "violation_count")
    nm_mean, nm_std = series_mean_std(df, "near_miss_count")
    sh_mean, sh_std = series_mean_std(df, "shield_count")
    ac_mean, ac_std = series_mean_std(df, "action_correction_mean")
    rp_mean, rp_std = series_mean_std(df, "reward_penalty_sum")

    return {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "method": method,
        "regime": regime,
        "seed": seed,
        "episodes": float(len(df)),
        "r_mean": r_mean,
        "r_std": r_std,
        "l_mean": l_mean,
        "l_std": l_std,
        "violation_count_mean": v_mean,
        "violation_count_std": v_std,
        "near_miss_count_mean": nm_mean,
        "near_miss_count_std": nm_std,
        "shield_count_mean": sh_mean,
        "shield_count_std": sh_std,
        "action_correction_mean_mean": ac_mean,
        "action_correction_mean_std": ac_std,
        "reward_penalty_sum_mean": rp_mean,
        "reward_penalty_sum_std": rp_std,
        "metric_source_csv": str(csv_path),
    }


def aggregate_group(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return math.nan, math.nan

    mean_val = float(s.mean())
    std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return mean_val, std_val


def build_grouped_summary(per_run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build grouped summary with the schema expected by downstream scripts:
    e.g. r_mean_mean, r_mean_std, violation_count_mean_mean, ...
    """
    rows: List[Dict] = []

    metrics = [
        "episodes",
        "r_mean",
        "l_mean",
        "violation_count_mean",
        "near_miss_count_mean",
        "shield_count_mean",
        "action_correction_mean_mean",
        "reward_penalty_sum_mean",
    ]

    grouped = per_run_df.groupby(["method", "regime"], observed=False, dropna=False)

    for (method, regime), g in grouped:
        row = {
            "method": method,
            "regime": regime,
            "n_runs": int(len(g)),
        }
        for metric in metrics:
            m, s = aggregate_group(g, metric)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out["method"] = pd.Categorical(out["method"], categories=STAGE_B_METHODS, ordered=True)
        out["regime"] = pd.Categorical(out["regime"], categories=REGIMES, ordered=True)
        out = out.sort_values(["method", "regime"]).reset_index(drop=True)

    return out


def print_summary(grouped_df: pd.DataFrame) -> None:
    if grouped_df.empty:
        print("No Stage B runs found.")
        return

    print("\n" + "=" * 120)
    print("STAGE B GROUPED SUMMARY")
    print("=" * 120)

    cols = [
        "method",
        "regime",
        "n_runs",
        "r_mean_mean",
        "violation_count_mean_mean",
        "near_miss_count_mean_mean",
        "shield_count_mean_mean",
        "action_correction_mean_mean_mean",
        "reward_penalty_sum_mean_mean",
    ]
    cols = [c for c in cols if c in grouped_df.columns]
    print(grouped_df[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs dir does not exist: {runs_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)

    per_run_df = pd.DataFrame(rows)

    per_run_path = out_dir / "stageB_per_run.csv"
    grouped_path = out_dir / "stageB_grouped_summary.csv"

    if per_run_df.empty:
        print("[WARN] No Stage B runs were detected.")
        per_run_df.to_csv(per_run_path, index=False)
        pd.DataFrame().to_csv(grouped_path, index=False)
        return

    per_run_df["method"] = pd.Categorical(per_run_df["method"], categories=STAGE_B_METHODS, ordered=True)
    per_run_df["regime"] = pd.Categorical(per_run_df["regime"], categories=REGIMES, ordered=True)
    per_run_df = per_run_df.sort_values(["method", "regime", "seed"]).reset_index(drop=True)

    grouped_df = build_grouped_summary(per_run_df)

    per_run_df.to_csv(per_run_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)

    print_summary(grouped_df)
    print(f"\nSaved: {per_run_path}")
    print(f"Saved: {grouped_path}")


if __name__ == "__main__":
    main()
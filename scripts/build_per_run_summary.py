#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"(?P<tag>[^_].*?)__"
    r"(?P<mode>continuous|discrete)__"
    r"(?P<algo>[^_].*?)__"
    r"(?P<env>.+?)__"
    r"(?P<method>baseline|cpss_only|lilac_only|full)__"
    r"pstay_(?P<pstay>[\d]+p[\d]+)__"
    r"seed_(?P<seed>\d+)$"
)


def parse_run_name(run_name: str) -> Optional[Dict[str, object]]:
    m = RUN_RE.match(run_name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "tag": d["tag"],
        "mode": d["mode"],
        "algo": d["algo"],
        "env": d["env"],
        "method": d["method"],
        "p_stay": float(d["pstay"].replace("p", ".")),
        "seed": int(d["seed"]),
    }


def iter_run_dirs(runs_dir: Path, prefix: Optional[str]) -> List[Path]:
    out = []
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        if prefix is not None and not p.name.startswith(prefix):
            continue
        if (p / "train_monitor.csv").exists():
            out.append(p)
    return out


def compute_switch_count(ctx: np.ndarray) -> int:
    if len(ctx) <= 1:
        return 0
    return int(np.sum(ctx[1:] != ctx[:-1]))


def summarize_run(run_dir: Path) -> Optional[Dict[str, object]]:
    meta = parse_run_name(run_dir.name)
    if meta is None:
        return None

    csv_path = run_dir / "train_monitor.csv"
    df = pd.read_csv(csv_path)

    for col in ["clearance", "near_miss", "violation", "shield_used", "ctx_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in {csv_path}")

    df = df.copy()
    df["clearance"] = pd.to_numeric(df["clearance"], errors="coerce")
    df["near_miss"] = pd.to_numeric(df["near_miss"], errors="coerce").fillna(0).astype(int)
    df["violation"] = pd.to_numeric(df["violation"], errors="coerce").fillna(0).astype(int)
    df["shield_used"] = pd.to_numeric(df["shield_used"], errors="coerce").fillna(0).astype(int)
    df["ctx_id"] = pd.to_numeric(df["ctx_id"], errors="coerce").fillna(-1).astype(int)

    clearance = df["clearance"].to_numpy()
    ctx = df["ctx_id"].to_numpy()

    row: Dict[str, object] = {
        **meta,
        "run_name": run_dir.name,
        "n_rows": int(len(df)),
        "contexts_present": int(df["ctx_id"].nunique(dropna=False)),
        "switch_count": compute_switch_count(ctx),
        "near_miss_rate": float(df["near_miss"].mean()),
        "violation_rate": float(df["violation"].mean()),
        "shield_used_rate": float(df["shield_used"].mean()),
        "clearance_min": float(np.nanmin(clearance)),
        "clearance_mean": float(np.nanmean(clearance)),
        "clearance_median": float(np.nanmedian(clearance)),
        "clearance_q05": float(np.nanquantile(clearance, 0.05)),
        "clearance_q10": float(np.nanquantile(clearance, 0.10)),
    }
    return row


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-run summary CSV from train_monitor.csv files.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_prefix", type=str, default="grid16__")
    parser.add_argument("--out_dir", type=str, default="artifacts/summary")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = iter_run_dirs(runs_dir, args.run_prefix if args.run_prefix else None)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_dir} with prefix={args.run_prefix!r}")

    rows: List[Dict[str, object]] = []
    for rd in run_dirs:
        row = summarize_run(rd)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No parseable run names found.")

    df = pd.DataFrame(rows).sort_values(["method", "p_stay", "seed"]).reset_index(drop=True)
    out_path = out_dir / "per_run_metrics.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
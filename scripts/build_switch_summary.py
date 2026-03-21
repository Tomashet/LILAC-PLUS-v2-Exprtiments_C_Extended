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


def extract_switch_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df["ctx_id"] = pd.to_numeric(df["ctx_id"], errors="coerce").fillna(-1).astype(int)
    df["near_miss"] = pd.to_numeric(df["near_miss"], errors="coerce").fillna(0).astype(int)
    df["violation"] = pd.to_numeric(df["violation"], errors="coerce").fillna(0).astype(int)
    df["shield_used"] = pd.to_numeric(df["shield_used"], errors="coerce").fillna(0).astype(int)
    df["clearance"] = pd.to_numeric(df["clearance"], errors="coerce")

    ctx = df["ctx_id"].to_numpy()
    switch_idx = np.where(ctx[1:] != ctx[:-1])[0] + 1

    rows = []
    for sidx, sw in enumerate(switch_idx):
        end = min(sw + horizon, len(df))
        for t in range(sw, end):
            rows.append(
                {
                    "switch_id": int(sidx),
                    "switch_step": int(sw),
                    "steps_since_switch": int(t - sw),
                    "near_miss": int(df.iloc[t]["near_miss"]),
                    "violation": int(df.iloc[t]["violation"]),
                    "shield_used": int(df.iloc[t]["shield_used"]),
                    "clearance": float(df.iloc[t]["clearance"]) if np.isfinite(df.iloc[t]["clearance"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build switch-aligned summary from train_monitor.csv files.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_prefix", type=str, default="grid16__")
    parser.add_argument("--out_dir", type=str, default="artifacts/summary")
    parser.add_argument("--horizon", type=int, default=10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = iter_run_dirs(runs_dir, args.run_prefix if args.run_prefix else None)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_dir} with prefix={args.run_prefix!r}")

    per_switch_rows = []
    for rd in run_dirs:
        meta = parse_run_name(rd.name)
        if meta is None:
            continue

        df = pd.read_csv(rd / "train_monitor.csv")
        if "ctx_id" not in df.columns:
            raise ValueError(f"Missing ctx_id in {rd / 'train_monitor.csv'}")

        sw = extract_switch_rows(df, horizon=int(args.horizon))
        if sw.empty:
            continue

        for k, v in meta.items():
            sw[k] = v
        sw["run_name"] = rd.name
        per_switch_rows.append(sw)

    if not per_switch_rows:
        raise ValueError("No switch-aligned rows produced.")

    all_sw = pd.concat(per_switch_rows, ignore_index=True)

    raw_path = out_dir / "switch_aligned_raw.csv"
    all_sw.to_csv(raw_path, index=False)

    grouped = (
        all_sw.groupby(["method", "p_stay", "steps_since_switch"], dropna=False)
        .agg(
            near_miss_rate=("near_miss", "mean"),
            violation_rate=("violation", "mean"),
            shield_used_rate=("shield_used", "mean"),
            clearance_mean=("clearance", "mean"),
            n=("near_miss", "size"),
        )
        .reset_index()
        .sort_values(["method", "p_stay", "steps_since_switch"])
    )
    grouped_path = out_dir / "switch_aligned_grouped.csv"
    grouped.to_csv(grouped_path, index=False)

    # compact @horizon summary
    compact = (
        all_sw.groupby(["run_name", "method", "p_stay", "seed"], dropna=False)
        .agg(
            post_switch_near_miss_rate=("near_miss", "mean"),
            post_switch_violation_rate=("violation", "mean"),
            post_switch_shield_used_rate=("shield_used", "mean"),
            post_switch_clearance_mean=("clearance", "mean"),
            n_switch_rows=("near_miss", "size"),
        )
        .reset_index()
        .sort_values(["method", "p_stay", "seed"])
    )
    compact_path = out_dir / f"switch_summary_h{args.horizon}.csv"
    compact.to_csv(compact_path, index=False)

    print(f"Saved: {raw_path}")
    print(f"Saved: {grouped_path}")
    print(f"Saved: {compact_path}")


if __name__ == "__main__":
    main()
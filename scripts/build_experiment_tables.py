#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


METHOD_ORDER = ["baseline", "cpss_only", "lilac_only", "full"]


def fmt_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def aggregate_main(per_run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_run_df.groupby(["method", "p_stay"], dropna=False)
        .agg(
            near_miss_rate_mean=("near_miss_rate", "mean"),
            near_miss_rate_std=("near_miss_rate", "std"),
            violation_rate_mean=("violation_rate", "mean"),
            violation_rate_std=("violation_rate", "std"),
            shield_used_rate_mean=("shield_used_rate", "mean"),
            shield_used_rate_std=("shield_used_rate", "std"),
            clearance_min_mean=("clearance_min", "mean"),
            clearance_min_std=("clearance_min", "std"),
            clearance_mean_mean=("clearance_mean", "mean"),
            clearance_mean_std=("clearance_mean", "std"),
            n_runs=("run_name", "size"),
        )
        .reset_index()
    )
    grouped["method"] = pd.Categorical(grouped["method"], categories=METHOD_ORDER, ordered=True)
    return grouped.sort_values(["p_stay", "method"]).reset_index(drop=True)


def aggregate_switch(sw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sw_df.groupby(["method", "p_stay"], dropna=False)
        .agg(
            post_switch_near_miss_rate_mean=("post_switch_near_miss_rate", "mean"),
            post_switch_near_miss_rate_std=("post_switch_near_miss_rate", "std"),
            post_switch_violation_rate_mean=("post_switch_violation_rate", "mean"),
            post_switch_violation_rate_std=("post_switch_violation_rate", "std"),
            post_switch_shield_used_rate_mean=("post_switch_shield_used_rate", "mean"),
            post_switch_shield_used_rate_std=("post_switch_shield_used_rate", "std"),
            n_runs=("run_name", "size"),
        )
        .reset_index()
    )
    grouped["method"] = pd.Categorical(grouped["method"], categories=METHOD_ORDER, ordered=True)
    return grouped.sort_values(["p_stay", "method"]).reset_index(drop=True)


def write_latex_main(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"$p_{\mathrm{stay}}$ & Method & Near-miss rate & Violation rate & Shield use rate & Min clearance \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['p_stay']:.2f} & {r['method']} & "
            f"{fmt_mean_std(r['near_miss_rate_mean'], 0.0 if pd.isna(r['near_miss_rate_std']) else r['near_miss_rate_std'])} & "
            f"{fmt_mean_std(r['violation_rate_mean'], 0.0 if pd.isna(r['violation_rate_std']) else r['violation_rate_std'])} & "
            f"{fmt_mean_std(r['shield_used_rate_mean'], 0.0 if pd.isna(r['shield_used_rate_std']) else r['shield_used_rate_std'])} & "
            f"{fmt_mean_std(r['clearance_min_mean'], 0.0 if pd.isna(r['clearance_min_std']) else r['clearance_min_std'], digits=3)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_switch(df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"$p_{\mathrm{stay}}$ & Method & Post-switch near-miss & Post-switch violation & Post-switch shield use \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['p_stay']:.2f} & {r['method']} & "
            f"{fmt_mean_std(r['post_switch_near_miss_rate_mean'], 0.0 if pd.isna(r['post_switch_near_miss_rate_std']) else r['post_switch_near_miss_rate_std'])} & "
            f"{fmt_mean_std(r['post_switch_violation_rate_mean'], 0.0 if pd.isna(r['post_switch_violation_rate_std']) else r['post_switch_violation_rate_std'])} & "
            f"{fmt_mean_std(r['post_switch_shield_used_rate_mean'], 0.0 if pd.isna(r['post_switch_shield_used_rate_std']) else r['post_switch_shield_used_rate_std'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aggregated experiment tables and LaTeX tables.")
    parser.add_argument("--summary_dir", type=str, default="artifacts/summary")
    parser.add_argument("--switch_horizon", type=int, default=10)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = summary_dir / "per_run_metrics.csv"
    sw_path = summary_dir / f"switch_summary_h{args.switch_horizon}.csv"

    per_run = pd.read_csv(per_run_path)
    sw = pd.read_csv(sw_path)

    main_df = aggregate_main(per_run)
    switch_df = aggregate_switch(sw)

    main_csv = summary_dir / "main_summary_by_method_pstay.csv"
    switch_csv = summary_dir / f"switch_summary_by_method_pstay_h{args.switch_horizon}.csv"

    main_df.to_csv(main_csv, index=False)
    switch_df.to_csv(switch_csv, index=False)

    main_tex = summary_dir / "main_results_table.tex"
    switch_tex = summary_dir / f"switch_results_table_h{args.switch_horizon}.tex"

    write_latex_main(main_df, main_tex)
    write_latex_switch(switch_df, switch_tex)

    print(f"Saved: {main_csv}")
    print(f"Saved: {switch_csv}")
    print(f"Saved: {main_tex}")
    print(f"Saved: {switch_tex}")


if __name__ == "__main__":
    main()
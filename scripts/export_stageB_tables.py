#!/usr/bin/env python3
"""
Export LaTeX tables for Stage B.

Inputs:
- artifacts/stageB_summary/stageB_grouped_summary.csv

Outputs:
- artifacts/stageB_tables/table_stageB_main.tex
- artifacts/stageB_tables/table_stageB_best_by_regime.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


METHOD_ORDER = [
    "adjust_speed_only",
    "lilac_context",
    "lilac_speed",
    "lilac_soft2hard",
]

REGIME_ORDER = [
    "stationary",
    "nonstationary_seen",
    "nonstationary_unseen",
]

METHOD_LABELS = {
    "adjust_speed_only": "Adjust Speed Only",
    "lilac_context": "LiLAC Context",
    "lilac_speed": "LiLAC Speed",
    "lilac_soft2hard": "LiLAC Soft\\textrightarrow Hard",
}

REGIME_LABELS = {
    "stationary": "Stationary",
    "nonstationary_seen": "Nonstationary Seen",
    "nonstationary_unseen": "Nonstationary Unseen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="artifacts/stageB_summary/stageB_grouped_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/stageB_tables",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        return df

    if "method" in df.columns:
        df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    if "regime" in df.columns:
        df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)

    sort_cols = [c for c in ["method", "regime"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def fmt_pm(mean_val, std_val, digits: int = 2) -> str:
    if pd.isna(mean_val):
        return "--"
    if pd.isna(std_val):
        return f"{float(mean_val):.{digits}f}"
    return f"{float(mean_val):.{digits}f} $\\pm$ {float(std_val):.{digits}f}"


def get_value(row: pd.Series, col: str):
    return row[col] if col in row.index else pd.NA


def best_method(series_df: pd.DataFrame, metric: str, higher_is_better: bool) -> str:
    """
    Safely choose the best method for a metric.
    Returns '--' if the metric is missing or entirely NaN.
    """
    if series_df.empty or metric not in series_df.columns:
        return "--"

    values = pd.to_numeric(series_df[metric], errors="coerce")
    valid = series_df.loc[values.notna()].copy()

    if valid.empty:
        return "--"

    metric_values = pd.to_numeric(valid[metric], errors="coerce")

    if metric_values.isna().all():
        return "--"

    idx = metric_values.idxmax() if higher_is_better else metric_values.idxmin()

    if pd.isna(idx):
        return "--"

    method = valid.loc[idx, "method"]
    return METHOD_LABELS.get(method, str(method))


def build_main_table(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Stage B single-mechanism ablation summary across three seeds.}")
    lines.append("\\label{tab:stageB_main}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append(
        "Method & Regime & Reward & Violations & Near Misses & Shield Count & Action Correction \\\\"
    )
    lines.append("\\midrule")

    if df.empty:
        lines.append("\\multicolumn{7}{c}{No Stage B summary data available.} \\\\")
    else:
        for _, row in df.iterrows():
            method = METHOD_LABELS.get(get_value(row, "method"), str(get_value(row, "method")))
            regime = REGIME_LABELS.get(get_value(row, "regime"), str(get_value(row, "regime")))

            reward = fmt_pm(get_value(row, "r_mean_mean"), get_value(row, "r_mean_std"))
            viol = fmt_pm(
                get_value(row, "violation_count_mean_mean"),
                get_value(row, "violation_count_mean_std"),
            )
            near = fmt_pm(
                get_value(row, "near_miss_count_mean_mean"),
                get_value(row, "near_miss_count_mean_std"),
            )
            shield = fmt_pm(
                get_value(row, "shield_count_mean_mean"),
                get_value(row, "shield_count_mean_std"),
            )
            corr = fmt_pm(
                get_value(row, "action_correction_mean_mean_mean"),
                get_value(row, "action_correction_mean_mean_std"),
            )

            lines.append(
                f"{method} & {regime} & {reward} & {viol} & {near} & {shield} & {corr} \\\\"
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_best_by_regime_table(df: pd.DataFrame) -> str:
    rows = []

    for regime in REGIME_ORDER:
        if df.empty or "regime" not in df.columns:
            sub = pd.DataFrame()
        else:
            sub = df[df["regime"] == regime].copy()

        rows.append({
            "regime": REGIME_LABELS[regime],
            "lowest_violations": best_method(sub, "violation_count_mean_mean", higher_is_better=False),
            "lowest_near_misses": best_method(sub, "near_miss_count_mean_mean", higher_is_better=False),
            "highest_reward": best_method(sub, "r_mean_mean", higher_is_better=True),
            "highest_shield_use": best_method(sub, "shield_count_mean_mean", higher_is_better=True),
        })

    lines: List[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Best Stage B mechanism by regime under selected criteria.}")
    lines.append("\\label{tab:stageB_best_by_regime}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Regime & Lowest Violations & Lowest Near Misses & Highest Reward & Highest Shield Use \\\\")
    lines.append("\\midrule")

    for row in rows:
        lines.append(
            f"{row['regime']} & {row['lowest_violations']} & {row['lowest_near_misses']} & "
            f"{row['highest_reward']} & {row['highest_shield_use']} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_csv)

    main_table = build_main_table(df)
    best_table = build_best_by_regime_table(df)

    main_path = out_dir / "table_stageB_main.tex"
    best_path = out_dir / "table_stageB_best_by_regime.tex"

    main_path.write_text(main_table, encoding="utf-8")
    best_path.write_text(best_table, encoding="utf-8")

    print(f"Saved: {main_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
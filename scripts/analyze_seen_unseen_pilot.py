from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


REQUIRED_GROUPS = [
    "baseline_seen",
    "full_seen",
    "baseline_unseen",
    "full_unseen",
]

DEFAULT_METRICS = [
    "r",                     # episode reward
    "l",                     # episode length
    "violation_count",
    "near_miss_count",
    "shield_count",
    "action_correction_mean",
    "reward_penalty_sum",
]


def load_monitor_csv(csv_path: Path) -> pd.DataFrame:
    """
    Stable-Baselines3 Monitor CSV usually starts with a JSON metadata line beginning with '#'.
    pandas can skip it with comment='#'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path, comment="#")


def detect_group(run_dir_name: str) -> str | None:
    """
    Infer one of the four expected groups from the run directory name.
    Expected naming examples:
      pilot_baseline_seen_s0
      pilot_full_seen_s0
      pilot_baseline_unseen_s0
      pilot_full_unseen_s0
    """
    name = run_dir_name.lower()

    if "baseline" in name and "unseen" in name:
        return "baseline_unseen"
    if "full" in name and "unseen" in name:
        return "full_unseen"
    if "baseline" in name and "seen" in name:
        return "baseline_seen"
    if "full" in name and "seen" in name:
        return "full_seen"

    return None


def summarize_run(df: pd.DataFrame, run_dir: Path) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {
        "run_dir": str(run_dir),
        "episodes": len(df),
    }

    for metric in DEFAULT_METRICS:
        if metric in df.columns:
            row[f"{metric}_mean"] = float(df[metric].mean())
            row[f"{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
        else:
            row[f"{metric}_mean"] = float("nan")
            row[f"{metric}_std"] = float("nan")

    return row


def compare_pair(df_summary: pd.DataFrame, left_group: str, right_group: str) -> pd.DataFrame:
    """
    Produce a metric-by-metric comparison table for two groups.
    """
    left = df_summary[df_summary["group"] == left_group]
    right = df_summary[df_summary["group"] == right_group]

    if len(left) != 1 or len(right) != 1:
        raise ValueError(
            f"Expected exactly one row for each of {left_group} and {right_group}, "
            f"but got {len(left)} and {len(right)}."
        )

    left_row = left.iloc[0]
    right_row = right.iloc[0]

    records: List[Dict[str, float | str]] = []
    for metric in DEFAULT_METRICS:
        mcol = f"{metric}_mean"
        scol = f"{metric}_std"

        lmean = left_row[mcol]
        rmean = right_row[mcol]
        lstd = left_row[scol]
        rstd = right_row[scol]

        records.append(
            {
                "metric": metric,
                f"{left_group}_mean": lmean,
                f"{left_group}_std": lstd,
                f"{right_group}_mean": rmean,
                f"{right_group}_std": rstd,
                "difference_right_minus_left": rmean - lmean,
            }
        )

    return pd.DataFrame(records)


def print_pretty_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))


def add_quick_interpretation(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple pass/fail style interpretation.
    """
    out = []

    def get_row(group: str) -> pd.Series:
        rows = summary_df[summary_df["group"] == group]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one row for group={group}, found {len(rows)}")
        return rows.iloc[0]

    seen_base = get_row("baseline_seen")
    seen_full = get_row("full_seen")
    unseen_base = get_row("baseline_unseen")
    unseen_full = get_row("full_unseen")

    checks = [
        {
            "comparison": "seen: shield_count",
            "baseline": seen_base["shield_count_mean"],
            "full": seen_full["shield_count_mean"],
            "status": "PASS" if seen_full["shield_count_mean"] > seen_base["shield_count_mean"] else "CHECK",
        },
        {
            "comparison": "seen: action_correction_mean",
            "baseline": seen_base["action_correction_mean_mean"],
            "full": seen_full["action_correction_mean_mean"],
            "status": "PASS" if seen_full["action_correction_mean_mean"] > seen_base["action_correction_mean_mean"] else "CHECK",
        },
        {
            "comparison": "seen: violation_count",
            "baseline": seen_base["violation_count_mean"],
            "full": seen_full["violation_count_mean"],
            "status": "PASS" if seen_full["violation_count_mean"] != seen_base["violation_count_mean"] else "CHECK",
        },
        {
            "comparison": "unseen: shield_count",
            "baseline": unseen_base["shield_count_mean"],
            "full": unseen_full["shield_count_mean"],
            "status": "PASS" if unseen_full["shield_count_mean"] > unseen_base["shield_count_mean"] else "CHECK",
        },
        {
            "comparison": "unseen: action_correction_mean",
            "baseline": unseen_base["action_correction_mean_mean"],
            "full": unseen_full["action_correction_mean_mean"],
            "status": "PASS" if unseen_full["action_correction_mean_mean"] > unseen_base["action_correction_mean_mean"] else "CHECK",
        },
        {
            "comparison": "unseen: violation_count",
            "baseline": unseen_base["violation_count_mean"],
            "full": unseen_full["violation_count_mean"],
            "status": "PASS" if unseen_full["violation_count_mean"] != unseen_base["violation_count_mean"] else "CHECK",
        },
    ]

    return pd.DataFrame(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze seen/unseen pilot monitor CSV files.")
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Directory containing pilot run folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="pilot_*/train_monitor.csv",
        help="Glob pattern under runs_dir to find monitor CSVs.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="pilot_seen_unseen_analysis",
        help="Prefix for output CSV files.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    csv_paths = sorted(runs_dir.glob(args.pattern))

    if not csv_paths:
        print(f"No files found with pattern: {runs_dir / args.pattern}")
        sys.exit(1)

    rows = []
    skipped = []

    for csv_path in csv_paths:
        run_dir = csv_path.parent
        group = detect_group(run_dir.name)
        if group is None:
            skipped.append(str(run_dir))
            continue

        df = load_monitor_csv(csv_path)
        row = summarize_run(df, run_dir)
        row["group"] = group
        rows.append(row)

    if skipped:
        print("\nSkipped run folders (name did not match expected pattern):")
        for s in skipped:
            print(f"  - {s}")

    if not rows:
        print("No matching pilot run folders found.")
        sys.exit(1)

    summary_df = pd.DataFrame(rows)

    missing_groups = [g for g in REQUIRED_GROUPS if g not in set(summary_df["group"])]
    if missing_groups:
        print("\nMissing required groups:")
        for g in missing_groups:
            print(f"  - {g}")
        print("\nExpected these 4 pilot groups:")
        for g in REQUIRED_GROUPS:
            print(f"  - {g}")
        sys.exit(1)

    # Keep rows in a fixed logical order
    group_order = {g: i for i, g in enumerate(REQUIRED_GROUPS)}
    summary_df["group_order"] = summary_df["group"].map(group_order)
    summary_df = summary_df.sort_values(["group_order", "run_dir"]).drop(columns=["group_order"])

    seen_compare_df = compare_pair(summary_df, "baseline_seen", "full_seen")
    unseen_compare_df = compare_pair(summary_df, "baseline_unseen", "full_unseen")
    checks_df = add_quick_interpretation(summary_df)

    summary_path = Path(f"{args.out_prefix}_summary.csv")
    seen_path = Path(f"{args.out_prefix}_seen_compare.csv")
    unseen_path = Path(f"{args.out_prefix}_unseen_compare.csv")
    checks_path = Path(f"{args.out_prefix}_checks.csv")

    summary_df.to_csv(summary_path, index=False)
    seen_compare_df.to_csv(seen_path, index=False)
    unseen_compare_df.to_csv(unseen_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    print_pretty_section("PER-RUN SUMMARY", summary_df)
    print_pretty_section("SEEN: baseline vs full", seen_compare_df)
    print_pretty_section("UNSEEN: baseline vs full", unseen_compare_df)
    print_pretty_section("QUICK CHECKS", checks_df)

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {seen_path}")
    print(f"  {unseen_path}")
    print(f"  {checks_path}")


if __name__ == "__main__":
    main()
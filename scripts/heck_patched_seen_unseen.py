from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


METRICS = [
    "r",
    "l",
    "violation_count",
    "near_miss_count",
    "shield_count",
    "action_correction_mean",
    "reward_penalty_sum",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_monitor_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def summarize_monitor(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["episodes"] = float(len(df))

    for metric in METRICS:
        if metric in df.columns:
            out[f"{metric}_mean"] = float(df[metric].mean())
            out[f"{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
        else:
            out[f"{metric}_mean"] = float("nan")
            out[f"{metric}_std"] = float("nan")

    return out


def threshold_block(debug_obj: Dict[str, Any]) -> Dict[str, Any]:
    resolved = debug_obj.get("resolved_thresholds", None)
    if not resolved:
        return {
            "context": None,
            "tau_violation": None,
            "tau_near_miss": None,
        }

    return {
        "context": resolved.get("context"),
        "tau_violation": resolved.get("tau_violation"),
        "tau_near_miss": resolved.get("tau_near_miss"),
    }


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    debug_path = run_dir / "run_debug.json"
    monitor_path = run_dir / "train_monitor.csv"

    if not debug_path.exists():
        raise FileNotFoundError(f"Missing {debug_path}")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Missing {monitor_path}")

    debug_obj = load_json(debug_path)
    monitor_df = load_monitor_csv(monitor_path)
    stats = summarize_monitor(monitor_df)
    thr = threshold_block(debug_obj)

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "method": debug_obj.get("method"),
        "seed": debug_obj.get("seed"),
        "raw_context": debug_obj.get("raw_context"),
        "canonical_context": debug_obj.get("canonical_context"),
        "apply_thresholds": debug_obj.get("apply_thresholds"),
        "threshold_patch": debug_obj.get("threshold_patch"),
        "resolved_context": thr["context"],
        "tau_violation": thr["tau_violation"],
        "tau_near_miss": thr["tau_near_miss"],
    }
    row.update(stats)
    return row


def meaningful_difference_checks(seen: Dict[str, Any], unseen: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, seen_value: Any, unseen_value: Any, pass_condition: bool) -> None:
        checks.append(
            {
                "check": name,
                "seen": seen_value,
                "unseen": unseen_value,
                "status": "PASS" if pass_condition else "CHECK",
            }
        )

    add_check(
        "canonical_context_differs",
        seen.get("canonical_context"),
        unseen.get("canonical_context"),
        seen.get("canonical_context") != unseen.get("canonical_context"),
    )

    add_check(
        "tau_violation_differs",
        seen.get("tau_violation"),
        unseen.get("tau_violation"),
        seen.get("tau_violation") != unseen.get("tau_violation"),
    )

    add_check(
        "tau_near_miss_differs",
        seen.get("tau_near_miss"),
        unseen.get("tau_near_miss"),
        seen.get("tau_near_miss") != unseen.get("tau_near_miss"),
    )

    add_check(
        "reward_mean_differs",
        seen.get("r_mean"),
        unseen.get("r_mean"),
        abs(float(seen.get("r_mean", 0.0)) - float(unseen.get("r_mean", 0.0))) > 1e-9,
    )

    add_check(
        "violation_count_mean_differs",
        seen.get("violation_count_mean"),
        unseen.get("violation_count_mean"),
        abs(
            float(seen.get("violation_count_mean", 0.0))
            - float(unseen.get("violation_count_mean", 0.0))
        ) > 1e-9,
    )

    add_check(
        "near_miss_count_mean_differs",
        seen.get("near_miss_count_mean"),
        unseen.get("near_miss_count_mean"),
        abs(
            float(seen.get("near_miss_count_mean", 0.0))
            - float(unseen.get("near_miss_count_mean", 0.0))
        ) > 1e-9,
    )

    add_check(
        "shield_count_mean_differs",
        seen.get("shield_count_mean"),
        unseen.get("shield_count_mean"),
        abs(
            float(seen.get("shield_count_mean", 0.0))
            - float(unseen.get("shield_count_mean", 0.0))
        ) > 1e-9,
    )

    add_check(
        "action_correction_mean_differs",
        seen.get("action_correction_mean_mean"),
        unseen.get("action_correction_mean_mean"),
        abs(
            float(seen.get("action_correction_mean_mean", 0.0))
            - float(unseen.get("action_correction_mean_mean", 0.0))
        ) > 1e-9,
    )

    return checks


def compare_rows(seen: Dict[str, Any], unseen: Dict[str, Any]) -> pd.DataFrame:
    compare_keys = [
        "tau_violation",
        "tau_near_miss",
        "episodes",
        "r_mean",
        "r_std",
        "l_mean",
        "l_std",
        "violation_count_mean",
        "violation_count_std",
        "near_miss_count_mean",
        "near_miss_count_std",
        "shield_count_mean",
        "shield_count_std",
        "action_correction_mean_mean",
        "action_correction_mean_std",
        "reward_penalty_sum_mean",
        "reward_penalty_sum_std",
    ]

    rows = []
    for key in compare_keys:
        seen_val = seen.get(key)
        unseen_val = unseen.get(key)

        diff = None
        if isinstance(seen_val, (int, float)) and isinstance(unseen_val, (int, float)):
            diff = float(unseen_val) - float(seen_val)

        rows.append(
            {
                "metric": key,
                "seen": seen_val,
                "unseen": unseen_val,
                "unseen_minus_seen": diff,
            }
        )

    return pd.DataFrame(rows)


def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare patched full seen vs patched full unseen runs."
    )
    parser.add_argument("--seen_run_dir", type=str, required=True)
    parser.add_argument("--unseen_run_dir", type=str, required=True)
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="patched_seen_unseen_check",
    )
    args = parser.parse_args()

    seen_run_dir = Path(args.seen_run_dir)
    unseen_run_dir = Path(args.unseen_run_dir)

    seen_summary = build_run_summary(seen_run_dir)
    unseen_summary = build_run_summary(unseen_run_dir)

    summary_df = pd.DataFrame([seen_summary, unseen_summary])
    compare_df = compare_rows(seen_summary, unseen_summary)
    checks_df = pd.DataFrame(meaningful_difference_checks(seen_summary, unseen_summary))

    out_prefix = Path(args.out_prefix)
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    compare_path = out_prefix.with_name(out_prefix.name + "_compare.csv")
    checks_path = out_prefix.with_name(out_prefix.name + "_checks.csv")

    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    # concise headline interpretation
    threshold_diff = any(
        checks_df["check"].isin(["tau_violation_differs", "tau_near_miss_differs"])
        & (checks_df["status"] == "PASS")
    )
    behavior_diff = any(
        checks_df["check"].isin(
            [
                "reward_mean_differs",
                "violation_count_mean_differs",
                "near_miss_count_mean_differs",
                "shield_count_mean_differs",
                "action_correction_mean_differs",
            ]
        )
        & (checks_df["status"] == "PASS")
    )

    print_section("RUN SUMMARY", summary_df)
    print_section("SEEN VS UNSEEN COMPARISON", compare_df)
    print_section("MEANINGFUL DIFFERENCE CHECKS", checks_df)

    print("\nInterpretation:")
    if threshold_diff and behavior_diff:
        print("PASS: patched seen/unseen runs differ in thresholds and in observed behavior.")
    elif threshold_diff and not behavior_diff:
        print("PARTIAL: thresholds differ, but behavior metrics are still very similar.")
    elif not threshold_diff and behavior_diff:
        print("PARTIAL: behavior differs, but thresholds did not differ.")
    else:
        print("CHECK: neither thresholds nor main behavior metrics differ meaningfully.")

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {compare_path}")
    print(f"  {checks_path}")


if __name__ == "__main__":
    main()
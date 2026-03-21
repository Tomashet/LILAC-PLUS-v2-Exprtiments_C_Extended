from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:
        df = pd.read_csv(
            path,
            comment="#",
            engine="python",
            on_bad_lines="skip",
        )

    keep = [c for c in METRICS if c in df.columns]
    if not keep:
        raise ValueError(f"No expected monitor columns found in {path}. Found: {list(df.columns)}")

    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    return df


def summarize_monitor(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"episodes": float(len(df))}
    for m in METRICS:
        if m in df.columns:
            out[f"{m}_mean"] = float(df[m].mean())
            out[f"{m}_std"] = float(df[m].std(ddof=1)) if len(df) > 1 else 0.0
        else:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
    return out


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    debug_path = run_dir / "run_debug.json"
    config_path = run_dir / "run_config.json"
    monitor_path = run_dir / "train_monitor.csv"

    if not debug_path.exists():
        raise FileNotFoundError(f"Missing {debug_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Missing {monitor_path}")

    debug = load_json(debug_path)
    config = load_json(config_path)
    monitor = load_monitor_csv(monitor_path)
    stats = summarize_monitor(monitor)

    resolved = debug.get("resolved_thresholds") or {}

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "method": debug.get("method"),
        "method_label": debug.get("method_label"),
        "use_lilac_metadata_only": debug.get("use_lilac_metadata_only"),
        "apply_thresholds": debug.get("apply_thresholds"),
        "use_context_constraints": debug.get("use_context_constraints"),
        "use_adjust_speed": debug.get("use_adjust_speed"),
        "use_soft_to_hard": debug.get("use_soft_to_hard"),
        "raw_context": debug.get("raw_context"),
        "canonical_context": debug.get("canonical_context"),
        "threshold_patch": debug.get("threshold_patch"),
        "tau_violation": resolved.get("tau_violation"),
        "tau_near_miss": resolved.get("tau_near_miss"),
        "action_space": config.get("action_space"),
        "algo": config.get("algo"),
    }
    row.update(stats)
    return row


def compare_two_runs(base: Dict[str, Any], lilac: Dict[str, Any]) -> pd.DataFrame:
    compare_keys = [
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
        "tau_violation",
        "tau_near_miss",
    ]

    rows: List[Dict[str, Any]] = []
    for key in compare_keys:
        b = base.get(key)
        l = lilac.get(key)
        diff = None
        if isinstance(b, (int, float)) and isinstance(l, (int, float)):
            diff = float(l) - float(b)
        rows.append(
            {
                "metric": key,
                "baseline_sac": b,
                "lilac_none": l,
                "lilac_minus_baseline": diff,
            }
        )
    return pd.DataFrame(rows)


def build_checks(base: Dict[str, Any], lilac: Dict[str, Any]) -> pd.DataFrame:
    checks = []

    def add(check: str, baseline_value: Any, lilac_value: Any, status: str):
        checks.append(
            {
                "check": check,
                "baseline_sac": baseline_value,
                "lilac_none": lilac_value,
                "status": status,
            }
        )

    add(
        "method_names_distinct",
        base.get("method"),
        lilac.get("method"),
        "PASS" if base.get("method") != lilac.get("method") else "CHECK",
    )

    add(
        "lilac_flag_enabled_for_lilac_none",
        base.get("use_lilac_metadata_only"),
        lilac.get("use_lilac_metadata_only"),
        "PASS" if bool(lilac.get("use_lilac_metadata_only")) else "CHECK",
    )

    add(
        "lilac_flag_disabled_for_baseline",
        base.get("use_lilac_metadata_only"),
        lilac.get("use_lilac_metadata_only"),
        "PASS" if not bool(base.get("use_lilac_metadata_only")) else "CHECK",
    )

    add(
        "thresholds_disabled_for_both",
        base.get("apply_thresholds"),
        lilac.get("apply_thresholds"),
        "PASS" if (not bool(base.get("apply_thresholds")) and not bool(lilac.get("apply_thresholds"))) else "CHECK",
    )

    reward_diff = abs(float(lilac.get("r_mean", 0.0)) - float(base.get("r_mean", 0.0)))
    add(
        "reward_diff_nonzero",
        base.get("r_mean"),
        lilac.get("r_mean"),
        "PASS" if reward_diff > 1e-9 else "CHECK",
    )

    violation_diff = abs(float(lilac.get("violation_count_mean", 0.0)) - float(base.get("violation_count_mean", 0.0)))
    add(
        "violation_diff_nonzero",
        base.get("violation_count_mean"),
        lilac.get("violation_count_mean"),
        "PASS" if violation_diff > 1e-9 else "CHECK",
    )

    return pd.DataFrame(checks)


def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage A sanity runs: baseline_sac vs lilac_none.")
    parser.add_argument("--baseline_run_dir", type=str, default="runs/debug_baseline_sac")
    parser.add_argument("--lilac_run_dir", type=str, default="runs/debug_lilac_none")
    parser.add_argument("--out_prefix", type=str, default="stageA_sanity")
    args = parser.parse_args()

    baseline_run = Path(args.baseline_run_dir)
    lilac_run = Path(args.lilac_run_dir)

    baseline_summary = build_run_summary(baseline_run)
    lilac_summary = build_run_summary(lilac_run)

    summary_df = pd.DataFrame([baseline_summary, lilac_summary])
    compare_df = compare_two_runs(baseline_summary, lilac_summary)
    checks_df = build_checks(baseline_summary, lilac_summary)

    out_prefix = Path(args.out_prefix)
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    compare_path = out_prefix.with_name(out_prefix.name + "_compare.csv")
    checks_path = out_prefix.with_name(out_prefix.name + "_checks.csv")

    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    print_section("STAGE A SANITY RUN SUMMARY", summary_df)
    print_section("BASELINE SAC VS LILAC_NONE", compare_df)
    print_section("SANITY CHECKS", checks_df)

    print("\nInterpretation:")
    if (checks_df["status"] == "PASS").all():
        print("PASS: the sanity runs look structurally correct, and lilac_none is distinct from baseline_sac.")
    else:
        print("CHECK: some sanity checks did not pass. Inspect the CHECK rows above before launching Stage A.")

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {compare_path}")
    print(f"  {checks_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


METHODS = ["baseline", "context", "adjust_speed", "full"]
REGIMES = ["seen", "unseen"]

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


def parse_run_name(name: str) -> Optional[Tuple[str, str, int]]:
    """
    Expected compact-grid naming:
      compact_<method>_<regime>_s<seed>

    Examples:
      compact_baseline_seen_s0
      compact_adjust_speed_unseen_s2
    """
    if not name.startswith("compact_"):
        return None

    tail = name[len("compact_"):]
    if tail.startswith("adjust_speed_"):
        rest = tail[len("adjust_speed_"):]
        method = "adjust_speed"
    elif tail.startswith("baseline_"):
        rest = tail[len("baseline_"):]
        method = "baseline"
    elif tail.startswith("context_"):
        rest = tail[len("context_"):]
        method = "context"
    elif tail.startswith("full_"):
        rest = tail[len("full_"):]
        method = "full"
    else:
        return None

    if "_s" not in rest:
        return None

    regime, seed_part = rest.rsplit("_s", 1)
    if regime not in REGIMES:
        return None

    try:
        seed = int(seed_part)
    except ValueError:
        return None

    return method, regime, seed


def summarize_episode_monitor(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {"episodes": float(len(df))}
    for m in METRICS:
        if m in df.columns:
            out[f"{m}_mean"] = float(df[m].mean())
            out[f"{m}_std"] = float(df[m].std(ddof=1)) if len(df) > 1 else 0.0
        else:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
    return out


def load_run_summary(run_dir: Path) -> Dict[str, Any]:
    parsed = parse_run_name(run_dir.name)
    if parsed is None:
        raise ValueError(f"Run directory name does not match compact-grid pattern: {run_dir.name}")

    method, regime, seed = parsed

    monitor_path = run_dir / "train_monitor.csv"
    debug_path = run_dir / "run_debug.json"

    if not monitor_path.exists():
        raise FileNotFoundError(f"Missing {monitor_path}")
    if not debug_path.exists():
        raise FileNotFoundError(f"Missing {debug_path}")

    monitor_df = load_monitor_csv(monitor_path)
    debug = load_json(debug_path)
    stats = summarize_episode_monitor(monitor_df)

    resolved = debug.get("resolved_thresholds") or {}

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "method": method,
        "regime": regime,
        "seed": seed,
        "raw_context": debug.get("raw_context"),
        "canonical_context": debug.get("canonical_context"),
        "threshold_patch": debug.get("threshold_patch"),
        "tau_violation": resolved.get("tau_violation"),
        "tau_near_miss": resolved.get("tau_near_miss"),
    }
    row.update(stats)
    return row


def aggregate_across_seeds(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    metric_means = [f"{m}_mean" for m in METRICS]
    threshold_cols = ["tau_violation", "tau_near_miss", "episodes"]

    for method in METHODS:
        for regime in REGIMES:
            sub = per_run_df[(per_run_df["method"] == method) & (per_run_df["regime"] == regime)]
            if sub.empty:
                continue

            row: Dict[str, Any] = {
                "method": method,
                "regime": regime,
                "num_seeds": int(len(sub)),
            }

            for c in threshold_cols:
                if c in sub.columns:
                    row[f"{c}_mean_over_seeds"] = float(sub[c].mean())
                    row[f"{c}_std_over_seeds"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0

            for c in metric_means:
                if c in sub.columns:
                    row[f"{c}_mean_over_seeds"] = float(sub[c].mean())
                    row[f"{c}_std_over_seeds"] = float(sub[c].std(ddof=1)) if len(sub) > 1 else 0.0

            rows.append(row)

    return pd.DataFrame(rows)


def build_paper_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, r in agg_df.iterrows():
        rows.append(
            {
                "method": r["method"],
                "regime": r["regime"],
                "reward": f"{r['r_mean_mean_over_seeds']:.2f} ± {r['r_mean_std_over_seeds']:.2f}",
                "violations": f"{r['violation_count_mean_mean_over_seeds']:.2f} ± {r['violation_count_mean_std_over_seeds']:.2f}",
                "near_misses": f"{r['near_miss_count_mean_mean_over_seeds']:.2f} ± {r['near_miss_count_mean_std_over_seeds']:.2f}",
                "shield": f"{r['shield_count_mean_mean_over_seeds']:.2f} ± {r['shield_count_mean_std_over_seeds']:.2f}",
                "action_corr": f"{r['action_correction_mean_mean_mean_over_seeds']:.3f} ± {r['action_correction_mean_mean_std_over_seeds']:.3f}",
                "penalty": f"{r['reward_penalty_sum_mean_mean_over_seeds']:.2f} ± {r['reward_penalty_sum_mean_std_over_seeds']:.2f}",
            }
        )

    return pd.DataFrame(rows)


def print_section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if df.empty:
        print("(empty)")
    else:
        print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the 24-run compact grid.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--pattern", type=str, default="compact_*")
    parser.add_argument("--out_dir", type=str, default="artifacts/compact_summary")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dirs = [p for p in sorted(runs_dir.glob(args.pattern)) if p.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No compact run directories found under {runs_dir} with pattern {args.pattern}")

    per_run_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for run_dir in candidate_dirs:
        try:
            row = load_run_summary(run_dir)
            per_run_rows.append(row)
        except Exception as e:
            skipped.append(f"{run_dir.name}: {e}")

    if not per_run_rows:
        raise RuntimeError("No valid compact-grid runs were loaded.")

    per_run_df = pd.DataFrame(per_run_rows).sort_values(["method", "regime", "seed"]).reset_index(drop=True)
    agg_df = aggregate_across_seeds(per_run_df).sort_values(["method", "regime"]).reset_index(drop=True)
    paper_df = build_paper_table(agg_df).sort_values(["method", "regime"]).reset_index(drop=True)

    per_run_path = out_dir / "compact_grid_per_run_summary.csv"
    agg_path = out_dir / "compact_grid_aggregated_summary.csv"
    paper_path = out_dir / "compact_grid_paper_table.csv"
    skipped_path = out_dir / "compact_grid_skipped_runs.txt"

    per_run_df.to_csv(per_run_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    with skipped_path.open("w", encoding="utf-8") as f:
        for line in skipped:
            f.write(line + "\n")

    print_section("PER-RUN SUMMARY", per_run_df)
    print_section("AGGREGATED SUMMARY ACROSS SEEDS", agg_df)
    print_section("PAPER-STYLE TABLE", paper_df)

    if skipped:
        print("\nSkipped runs:")
        for s in skipped:
            print(" -", s)

    print("\nSaved files:")
    print(f"  {per_run_path}")
    print(f"  {agg_path}")
    print(f"  {paper_path}")
    print(f"  {skipped_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


METHODS = ["baseline_sac", "lilac_none"]
REGIMES = ["stationary", "nonstationary_seen", "nonstationary_unseen"]

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
    # expected: baseline_sac_stationary_s0
    # expected: lilac_none_nonstationary_seen_s2
    if not name.endswith(tuple(f"_s{i}" for i in range(100))):
        pass

    if "_s" not in name:
        return None

    prefix, seed_part = name.rsplit("_s", 1)
    try:
        seed = int(seed_part)
    except ValueError:
        return None

    method_match = None
    for m in METHODS:
        if prefix.startswith(m + "_"):
            method_match = m
            break
    if method_match is None:
        return None

    regime = prefix[len(method_match) + 1 :]
    if regime not in REGIMES:
        return None

    return method_match, regime, seed


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
        raise ValueError(f"Run directory name does not match Stage A pattern: {run_dir.name}")

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

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "method": method,
        "regime": regime,
        "seed": seed,
        "use_lilac": debug.get("use_lilac"),
        "apply_thresholds": debug.get("apply_thresholds"),
        "raw_context": debug.get("raw_context"),
        "canonical_context": debug.get("canonical_context"),
    }
    row.update(stats)
    return row


def aggregate_across_seeds(per_run_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_means = [f"{m}_mean" for m in METRICS] + ["episodes"]

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
                "ep_length": f"{r['l_mean_mean_over_seeds']:.2f} ± {r['l_mean_std_over_seeds']:.2f}",
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
    parser = argparse.ArgumentParser(description="Summarize Stage A thesis runs.")
    parser.add_argument("--runs_dir", type=str, default="runs_thesis")
    parser.add_argument("--pattern", type=str, default="*_s*")
    parser.add_argument("--out_dir", type=str, default="artifacts/stageA_summary")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_dirs = [p for p in sorted(runs_dir.glob(args.pattern)) if p.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}")

    per_run_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for run_dir in candidate_dirs:
        try:
            parsed = parse_run_name(run_dir.name)
            if parsed is None:
                continue
            row = load_run_summary(run_dir)
            per_run_rows.append(row)
        except Exception as e:
            skipped.append(f"{run_dir.name}: {e}")

    if not per_run_rows:
        raise RuntimeError("No valid Stage A runs were loaded.")

    per_run_df = pd.DataFrame(per_run_rows).sort_values(["method", "regime", "seed"]).reset_index(drop=True)
    agg_df = aggregate_across_seeds(per_run_df).sort_values(["method", "regime"]).reset_index(drop=True)
    paper_df = build_paper_table(agg_df).sort_values(["method", "regime"]).reset_index(drop=True)

    per_run_path = out_dir / "stageA_per_run_summary.csv"
    agg_path = out_dir / "stageA_aggregated_summary.csv"
    paper_path = out_dir / "stageA_paper_table.csv"
    skipped_path = out_dir / "stageA_skipped_runs.txt"

    per_run_df.to_csv(per_run_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    paper_df.to_csv(paper_path, index=False)

    with skipped_path.open("w", encoding="utf-8") as f:
        for line in skipped:
            f.write(line + "\n")

    print_section("STAGE A PER-RUN SUMMARY", per_run_df)
    print_section("STAGE A AGGREGATED SUMMARY", agg_df)
    print_section("STAGE A PAPER TABLE", paper_df)

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
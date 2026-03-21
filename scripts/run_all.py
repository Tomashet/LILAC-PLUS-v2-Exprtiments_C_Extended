from __future__ import annotations
import argparse
import os
import subprocess
from typing import List, Dict, Tuple

# This script runs a full experiment suite:
#  - DQN/PPO for discrete presets
#  - SAC for continuous presets
#  - Ablations: full, no_mpc, no_conformal, no_mpc+no_conformal
#  - Multi-seed runs per method
#  - Exports LaTeX tables (per environment + combined)

def run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def latex_export(pattern: str, out_tex: str, caption: str, label: str) -> None:
    run([
        "python", "-m", "scripts.export_latex",
        "--pattern", pattern,
        "--out", out_tex,
        "--caption", caption,
        "--label", label
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--steps_discrete", type=int, default=300_000)
    ap.add_argument("--steps_continuous", type=int, default=600_000)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--only", default="", help="Optional filter substring (e.g., 'merge' or 'highway')")
    args = ap.parse_args()

    seeds = args.seeds
    episodes = str(args.episodes)

    # Methods to run: (preset, action_type, algo, total_steps)
    methods: List[Tuple[str, str, str, int]] = [
        ("highway_discrete_default", "discrete", "dqn", args.steps_discrete),
        ("highway_discrete_default", "discrete", "ppo", args.steps_discrete),
        ("merge_discrete_default",   "discrete", "dqn", args.steps_discrete),
        ("merge_discrete_default",   "discrete", "ppo", args.steps_discrete),

        ("highway_continuous_default", "continuous", "sac", args.steps_continuous),
        ("merge_continuous_default",   "continuous", "sac", args.steps_continuous),
    ]

    ablations: List[Tuple[str, List[str]]] = [
        ("full", []),
        ("no_mpc", ["--no_mpc"]),
        ("no_conformal", ["--no_conformal"]),
        ("no_mpc_no_conformal", ["--no_mpc", "--no_conformal"]),
    ]

    ran_any = False
    for preset, action_type, algo, total_steps in methods:
        if args.only and (args.only not in preset):
            continue
        for ab_name, ab_flags in ablations:
            base_name = f"{preset}_{algo}_{ab_name}"
            cmd = [
                "python", "-m", "scripts.run_sweep",
                "--preset", preset,
                "--algo", algo,
                "--action_space_type", action_type,
                "--seeds", seeds,
                "--total_steps", str(total_steps),
                "--episodes", episodes,
                "--base_run_name", base_name,
            ] + ab_flags
            run(cmd)
            ran_any = True

    if not ran_any:
        print("No methods matched your --only filter; nothing to do.")
        return

    # Export LaTeX tables
    os.makedirs("paper_tables", exist_ok=True)

    # Per-env tables
    latex_export(
        pattern="runs/highway_*_seed*",
        out_tex="paper_tables/results_highway_all.tex",
        caption="Highway-v0 results under episodic nonstationarity (mean over episodes, mean$\\pm$std over seeds).",
        label="tab:highway_all",
    )
    latex_export(
        pattern="runs/merge_*_seed*",
        out_tex="paper_tables/results_merge_all.tex",
        caption="Merge-v0 results under episodic nonstationarity (mean over episodes, mean$\\pm$std over seeds).",
        label="tab:merge_all",
    )

    # Combined table across all runs
    latex_export(
        pattern="runs/*_seed*",
        out_tex="paper_tables/results_all.tex",
        caption="Overall results across environments under episodic nonstationarity (mean over episodes, mean$\\pm$std over seeds).",
        label="tab:all_envs",
    )


# Paper-friendly split tables (discrete vs continuous), ordered nicely
run([
    "python", "-m", "scripts.export_paper_tables",
    "--pattern", "runs/*_seed*",
    "--out_dir", "paper_tables",
    "--label_prefix", "tab:paper",
    "--caption_prefix", "Results under episodic nonstationarity.",
    "--split_by_env",
])

    print("Wrote LaTeX tables to paper_tables/")

if __name__ == "__main__":
    main()

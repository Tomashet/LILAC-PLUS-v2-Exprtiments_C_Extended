# scripts/run_experiments.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("\n>>>", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--total_steps", type=int, default=200000)
    ap.add_argument("--p_stay_stationary", type=float, default=1.0)
    ap.add_argument("--p_stay_nonstationary", type=float, default=0.85)
    ap.add_argument("--constraints", choices=["off", "on", "both"], default="off",
                    help="off: --no_mpc --no_conformal, on: constraints enabled, both: run both")
    ap.add_argument("--algos", type=str, default="dqn,ppo,sac",
                    help="Comma-separated: dqn,ppo,sac (sac uses train_continuous)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]

    py = sys.executable

    # Two conditions
    conditions = [
        ("stationary", args.p_stay_stationary),
        ("nonstationary", args.p_stay_nonstationary),
    ]

    # Constraints modes
    if args.constraints == "off":
        constraint_modes = [("off", ["--no_mpc", "--no_conformal"])]
    elif args.constraints == "on":
        constraint_modes = [("on", [])]
    else:
        constraint_modes = [("off", ["--no_mpc", "--no_conformal"]), ("on", [])]

    for algo in algos:
        for cond_name, p_stay in conditions:
            for seed in seeds:
                for safe_name, safe_flags in constraint_modes:
                    if algo in ["dqn", "ppo"]:
                        env = "merge-v0"
                        preset = "merge_discrete_default"
                        run_dir = f"runs/{env}_discrete_{algo}_seed{seed}_pst{p_stay:.2f}_{cond_name}_safe{safe_name}"

                        cmd = [
                            py, "-m", "scripts.train_discrete",
                            "--env", env,
                            "--preset", preset,
                            "--algo", algo,
                            "--total_steps", str(args.total_steps),
                            "--seed", str(seed),
                            "--p_stay", str(p_stay),
                            "--run_dir", os.path.basename(run_dir),
                            *safe_flags,
                        ]
                        run(cmd, dry_run=args.dry_run)

                    elif algo == "sac":
                        # SAC uses continuous training + highway-v0 preset
                        env = "highway-v0"
                        preset = "highway_continuous_default"
                        run_dir = f"runs/{env}_continuous_sac_seed{seed}_pst{p_stay:.2f}_{cond_name}_safe{safe_name}"

                        cmd = [
                            py, "-m", "scripts.train_continuous",
                            "--env", env,
                            "--preset", preset,
                            "--total_steps", str(args.total_steps),
                            "--seed", str(seed),
                            "--p_stay", str(p_stay),
                            "--run_dir", os.path.basename(run_dir),
                            *safe_flags,
                        ]
                        run(cmd, dry_run=args.dry_run)
                    else:
                        raise ValueError(f"Unknown algo: {algo}")

    print("\nAll requested runs finished.")


if __name__ == "__main__":
    main()
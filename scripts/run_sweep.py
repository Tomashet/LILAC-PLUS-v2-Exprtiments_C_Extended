from __future__ import annotations
import argparse
import os
import subprocess
from typing import List

def run(cmd: List[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="", help="Preset name, e.g. merge_discrete_default / highway_continuous_default")
    ap.add_argument("--env", default="", help="Override env id (optional if preset provided)")
    ap.add_argument("--algo", default="", help="dqn|ppo for discrete, sac for continuous (optional if inferred from preset)")
    ap.add_argument("--action_space_type", choices=["", "discrete", "continuous"], default="", help="Override action type")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    ap.add_argument("--total_steps", type=int, default=200_000)
    ap.add_argument("--episodes", type=int, default=200, help="Eval episodes per seed")
    ap.add_argument("--p_stay", type=float, default=0.8)
    ap.add_argument("--no_tier2", action="store_true")
    ap.add_argument("--no_conformal", action="store_true")
    ap.add_argument("--no_mpc", action="store_true")
    ap.add_argument("--base_run_name", default="", help="Prefix for runs/, default auto")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    action_type = args.action_space_type
    algo = args.algo

    if args.preset and not action_type:
        action_type = "continuous" if "continuous" in args.preset else "discrete"
    if not action_type:
        action_type = "discrete"

    if not algo:
        algo = "sac" if action_type == "continuous" else "dqn"

    for seed in seeds:
        run_name = args.base_run_name
        if not run_name:
            if args.preset:
                run_name = f"{args.preset}_{algo}_seed{seed}"
            else:
                env_name = args.env or "highway-v0"
                run_name = f"{env_name}_{action_type}_{algo}_seed{seed}"

        if action_type == "continuous":
            cmd = ["python", "-m", "scripts.train_continuous",
                   "--total_steps", str(args.total_steps),
                   "--seed", str(seed),
                   "--p_stay", str(args.p_stay),
                   "--run_dir", run_name]
            if args.env:
                cmd += ["--env", args.env]
            if args.preset:
                cmd += ["--preset", args.preset]
            if args.no_tier2: cmd += ["--no_tier2"]
            if args.no_conformal: cmd += ["--no_conformal"]
            if args.no_mpc: cmd += ["--no_mpc"]
            run(cmd)

            run(["python", "-m", "scripts.eval",
                 "--env", (args.env or "highway-v0"),
                 "--run_dir", os.path.join("runs", run_name),
                 "--episodes", str(args.episodes),
                 "--seed", str(seed),
                 "--action_space_type", "continuous"])
            run(["python", "-m", "scripts.plot_results", "--run_dir", os.path.join("runs", run_name)])

        else:
            cmd = ["python", "-m", "scripts.train_discrete",
                   "--algo", algo,
                   "--total_steps", str(args.total_steps),
                   "--seed", str(seed),
                   "--p_stay", str(args.p_stay),
                   "--run_dir", run_name]
            if args.env:
                cmd += ["--env", args.env]
            if args.preset:
                cmd += ["--preset", args.preset]
            if args.no_tier2: cmd += ["--no_tier2"]
            if args.no_conformal: cmd += ["--no_conformal"]
            if args.no_mpc: cmd += ["--no_mpc"]
            run(cmd)

            run(["python", "-m", "scripts.eval",
                 "--env", (args.env or "highway-v0"),
                 "--run_dir", os.path.join("runs", run_name),
                 "--episodes", str(args.episodes),
                 "--seed", str(seed),
                 "--action_space_type", "discrete"])
            run(["python", "-m", "scripts.plot_results", "--run_dir", os.path.join("runs", run_name)])

if __name__ == "__main__":
    main()

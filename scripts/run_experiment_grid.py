#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_method_specs() -> Dict[str, Dict[str, object]]:
    """
    Method mapping for the 4-way comparison.

    Interpretation:
    - baseline   : no shield / no conformal
    - cpss_only  : shield on, conformal off
    - lilac_only : shield off, conformal on
    - full       : shield on, conformal on
    """
    return {
        "baseline": {
            "no_mpc": True,
            "no_conformal": True,
        },
        "cpss_only": {
            "no_mpc": False,
            "no_conformal": True,
        },
        "lilac_only": {
            "no_mpc": True,
            "no_conformal": False,
        },
        "full": {
            "no_mpc": False,
            "no_conformal": False,
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a grid of LILAC+/CPSS experiments with consistent run naming."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="continuous",
        choices=["continuous", "discrete"],
        help="Which training entrypoint to use.",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="highway-v0",
        help="Gymnasium environment id.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name. If omitted, a default is chosen from mode.",
    )

    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Algorithm for discrete mode (e.g. dqn or ppo). Ignored for continuous mode.",
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="baseline,cpss_only,lilac_only,full",
        help="Comma-separated method names.",
    )

    parser.add_argument(
        "--p_stays",
        type=str,
        default="0.95,0.50",
        help="Comma-separated p_stay values.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds.",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=5000,
        help="Training timesteps per run.",
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="runs",
        help="Base directory for run outputs.",
    )

    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for subprocess runs.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional override for safety epsilon.",
    )

    parser.add_argument(
        "--delta_nearmiss",
        type=float,
        default=None,
        help="Optional override for near-miss gap parameter.",
    )

    parser.add_argument(
        "--use_budget",
        action="store_true",
        help="Enable soft-to-hard budget wrapper.",
    )
    parser.add_argument("--budget_C", type=float, default=10.0)
    parser.add_argument("--budget_T", type=int, default=1000)
    parser.add_argument("--budget_risk_delta", type=float, default=0.0)
    parser.add_argument("--budget_min_bt", type=float, default=0.0)

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun even if run_dir/train_monitor.csv already exists.",
    )

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one run fails.",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default="grid16",
        help="Prefix/tag included in run names.",
    )

    return parser


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def default_preset_for_mode(mode: str) -> str:
    if mode == "continuous":
        return "highway_continuous_default"
    if mode == "discrete":
        return "highway_discrete_default"
    raise ValueError(f"Unsupported mode: {mode}")


def default_algo_for_mode(mode: str) -> str:
    if mode == "continuous":
        return "sac"
    if mode == "discrete":
        return "dqn"
    raise ValueError(f"Unsupported mode: {mode}")


def format_pstay_for_name(p: float) -> str:
    return str(p).replace(".", "p")


def make_run_name(
    tag: str,
    mode: str,
    algo: str,
    env: str,
    method: str,
    p_stay: float,
    seed: int,
) -> str:
    env_clean = env.replace("-", "_")
    return (
        f"{tag}__{mode}__{algo}__{env_clean}"
        f"__{method}__pstay_{format_pstay_for_name(p_stay)}__seed_{seed}"
    )


def training_module_for_mode(mode: str) -> str:
    if mode == "continuous":
        return "scripts.train_continuous"
    if mode == "discrete":
        return "scripts.train_discrete"
    raise ValueError(f"Unsupported mode: {mode}")


def build_command(
    *,
    python_exe: str,
    mode: str,
    env: str,
    preset: str,
    algo: str,
    total_steps: int,
    seed: int,
    p_stay: float,
    run_dir: Path,
    method_cfg: Dict[str, object],
    epsilon: float | None,
    delta_nearmiss: float | None,
    use_budget: bool,
    budget_C: float,
    budget_T: int,
    budget_risk_delta: float,
    budget_min_bt: float,
) -> List[str]:
    module = training_module_for_mode(mode)

    cmd: List[str] = [
        python_exe,
        "-m",
        module,
        "--env",
        env,
        "--preset",
        preset,
        "--total_steps",
        str(total_steps),
        "--seed",
        str(seed),
        "--p_stay",
        str(p_stay),
        "--run_dir",
        str(run_dir),
    ]

    if mode == "discrete":
        cmd.extend(["--algo", algo])

    if bool(method_cfg.get("no_mpc", False)):
        cmd.append("--no_mpc")
    if bool(method_cfg.get("no_conformal", False)):
        cmd.append("--no_conformal")

    if epsilon is not None:
        cmd.extend(["--epsilon", str(epsilon)])
    if delta_nearmiss is not None:
        cmd.extend(["--delta_nearmiss", str(delta_nearmiss)])

    if use_budget:
        cmd.append("--use_budget")
        cmd.extend(["--budget_C", str(budget_C)])
        cmd.extend(["--budget_T", str(budget_T)])
        cmd.extend(["--budget_risk_delta", str(budget_risk_delta)])
        cmd.extend(["--budget_min_bt", str(budget_min_bt)])

    return cmd


def save_grid_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = build_argparser().parse_args()

    methods = parse_csv_list(args.methods)
    p_stays = parse_float_list(args.p_stays)
    seeds = parse_int_list(args.seeds)

    all_method_specs = build_method_specs()
    unknown_methods = [m for m in methods if m not in all_method_specs]
    if unknown_methods:
        raise ValueError(
            f"Unknown methods: {unknown_methods}. "
            f"Available: {list(all_method_specs.keys())}"
        )

    preset = args.preset or default_preset_for_mode(args.mode)
    algo = args.algo or default_algo_for_mode(args.mode)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    grid_points = list(itertools.product(methods, p_stays, seeds))

    manifest = {
        "mode": args.mode,
        "env": args.env,
        "preset": preset,
        "algo": algo,
        "methods": methods,
        "p_stays": p_stays,
        "seeds": seeds,
        "total_steps": args.total_steps,
        "runs_dir": str(runs_dir),
        "tag": args.tag,
        "use_budget": args.use_budget,
        "budget_C": args.budget_C,
        "budget_T": args.budget_T,
        "budget_risk_delta": args.budget_risk_delta,
        "budget_min_bt": args.budget_min_bt,
        "epsilon": args.epsilon,
        "delta_nearmiss": args.delta_nearmiss,
        "num_runs": len(grid_points),
    }
    save_grid_manifest(runs_dir / f"{args.tag}__grid_manifest.json", manifest)

    print("\n=== EXPERIMENT GRID ===")
    print(f"mode: {args.mode}")
    print(f"env: {args.env}")
    print(f"preset: {preset}")
    print(f"algo: {algo}")
    print(f"methods: {methods}")
    print(f"p_stays: {p_stays}")
    print(f"seeds: {seeds}")
    print(f"total_steps: {args.total_steps}")
    print(f"runs_dir: {runs_dir}")
    print(f"num_runs: {len(grid_points)}")
    print(f"dry_run: {args.dry_run}")
    print("=======================\n")

    completed = 0
    skipped = 0
    failed = 0

    for idx, (method, p_stay, seed) in enumerate(grid_points, start=1):
        run_name = make_run_name(
            tag=args.tag,
            mode=args.mode,
            algo=algo,
            env=args.env,
            method=method,
            p_stay=p_stay,
            seed=seed,
        )
        run_dir = runs_dir / run_name
        monitor_csv = run_dir / "train_monitor.csv"

        print(f"[{idx}/{len(grid_points)}] {run_name}")

        if monitor_csv.exists() and not args.overwrite:
            print(f"  SKIP: existing {monitor_csv}")
            skipped += 1
            continue

        method_cfg = all_method_specs[method]
        cmd = build_command(
            python_exe=args.python_exe,
            mode=args.mode,
            env=args.env,
            preset=preset,
            algo=algo,
            total_steps=args.total_steps,
            seed=seed,
            p_stay=p_stay,
            run_dir=run_dir,
            method_cfg=method_cfg,
            epsilon=args.epsilon,
            delta_nearmiss=args.delta_nearmiss,
            use_budget=args.use_budget,
            budget_C=args.budget_C,
            budget_T=args.budget_T,
            budget_risk_delta=args.budget_risk_delta,
            budget_min_bt=args.budget_min_bt,
        )

        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("  OK")
            completed += 1
        else:
            print(f"  FAIL: return code {result.returncode}")
            failed += 1
            if args.stop_on_error:
                raise SystemExit(result.returncode)

    print("\n=== GRID SUMMARY ===")
    print(f"completed: {completed}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    print("====================")


if __name__ == "__main__":
    main()
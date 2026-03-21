from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


METHODS = ["baseline", "context", "adjust_speed", "full"]
REGIMES = ["seen", "unseen"]
SEEDS = [0, 1, 2]

# You can change these if your canonical seen/unseen definitions evolve.
REGIME_CONTEXTS: Dict[str, Tuple[str, str, str]] = {
    "seen": ("low", "calm", "clean"),
    "unseen": ("high", "aggr", "noisy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the compact 24-run continuous-control grid for LILAC+."
    )

    parser.add_argument("--env", type=str, default="merge-v0")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--max_episode_steps", type=int, default=200)

    parser.add_argument(
        "--threshold_patch",
        type=str,
        default="artifacts/calibration/recommended_thresholds_patch.json",
        help="Threshold patch used for context/adjust_speed/full methods.",
    )

    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable to use for launching subprocesses.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose train_monitor.csv already exists.",
    )

    parser.add_argument(
        "--only_method",
        type=str,
        default=None,
        choices=[None, "baseline", "context", "adjust_speed", "full"],
        help="Run only one method.",
    )

    parser.add_argument(
        "--only_regime",
        type=str,
        default=None,
        choices=[None, "seen", "unseen"],
        help="Run only one regime.",
    )

    parser.add_argument(
        "--only_seed",
        type=int,
        default=None,
        help="Run only one seed.",
    )

    return parser.parse_args()


def needs_threshold_patch(method: str) -> bool:
    return method in {"context", "adjust_speed", "full"}


def run_name(method: str, regime: str, seed: int) -> str:
    return f"compact_{method}_{regime}_s{seed}"


def build_command(
    python_exe: str,
    env: str,
    run_dir: Path,
    method: str,
    seed: int,
    context_density: str,
    context_behavior: str,
    context_sensor: str,
    total_steps: int,
    max_episode_steps: int,
    threshold_patch: str | None,
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "scripts.train_continuous",
        "--env",
        env,
        "--method",
        method,
        "--seed",
        str(seed),
        "--run_dir",
        str(run_dir),
        "--context_density",
        context_density,
        "--context_behavior",
        context_behavior,
        "--context_sensor",
        context_sensor,
        "--total_steps",
        str(total_steps),
        "--max_episode_steps",
        str(max_episode_steps),
    ]

    if threshold_patch is not None:
        cmd.extend(["--threshold_patch", threshold_patch])

    return cmd


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    threshold_patch_path = Path(args.threshold_patch)
    if not args.dry_run and not threshold_patch_path.exists():
        raise FileNotFoundError(
            f"Threshold patch not found: {threshold_patch_path}\n"
            "Generate it first with scripts.calibrate_thresholds or pass --threshold_patch explicitly."
        )

    methods = [args.only_method] if args.only_method is not None else METHODS
    regimes = [args.only_regime] if args.only_regime is not None else REGIMES
    seeds = [args.only_seed] if args.only_seed is not None else SEEDS

    jobs: List[Tuple[str, List[str], Path]] = []

    for method in methods:
        for regime in regimes:
            for seed in seeds:
                if regime not in REGIME_CONTEXTS:
                    raise ValueError(f"Unknown regime: {regime}")

                density, behavior, sensor = REGIME_CONTEXTS[regime]
                name = run_name(method, regime, seed)
                run_dir = runs_dir / name

                patch_arg = str(threshold_patch_path) if needs_threshold_patch(method) else None

                cmd = build_command(
                    python_exe=args.python_exe,
                    env=args.env,
                    run_dir=run_dir,
                    method=method,
                    seed=seed,
                    context_density=density,
                    context_behavior=behavior,
                    context_sensor=sensor,
                    total_steps=args.total_steps,
                    max_episode_steps=args.max_episode_steps,
                    threshold_patch=patch_arg,
                )
                jobs.append((name, cmd, run_dir))

    print("\n=== COMPACT GRID PLAN ===")
    print(f"env: {args.env}")
    print(f"runs_dir: {runs_dir}")
    print(f"total_steps: {args.total_steps}")
    print(f"max_episode_steps: {args.max_episode_steps}")
    print(f"threshold_patch: {threshold_patch_path}")
    print(f"num_jobs: {len(jobs)}")
    print("=========================\n")

    completed = 0
    skipped = 0
    failed = 0

    for idx, (name, cmd, run_dir) in enumerate(jobs, start=1):
        train_monitor = run_dir / "train_monitor.csv"

        print(f"[{idx}/{len(jobs)}] {name}")

        if args.skip_existing and train_monitor.exists():
            print(f"  SKIP existing: {train_monitor}")
            skipped += 1
            continue

        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        run_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  FAILED with return code {result.returncode}")
            failed += 1
        else:
            print("  DONE")
            completed += 1

    print("\n=== COMPACT GRID SUMMARY ===")
    print(f"completed: {completed}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    print(f"planned:   {len(jobs)}")
    print("============================")


if __name__ == "__main__":
    main()
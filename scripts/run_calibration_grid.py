from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small calibration grid to build a stronger threshold patch."
    )

    parser.add_argument("--env", type=str, default="merge-v0")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--max_episode_steps", type=int, default=200)

    parser.add_argument("--context_density", type=str, default="low")
    parser.add_argument("--context_behavior", type=str, default="calm")
    parser.add_argument("--context_sensor", type=str, default="clean")

    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess launches.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose calibration_monitor.csv already exists.",
    )

    parser.add_argument(
        "--only_seed",
        type=int,
        default=None,
        help="Run only one seed.",
    )

    return parser.parse_args()


def run_name(seed: int) -> str:
    return f"calibration_full_seen_s{seed}"


def build_command(
    python_exe: str,
    env: str,
    run_dir: Path,
    seed: int,
    context_density: str,
    context_behavior: str,
    context_sensor: str,
    total_steps: int,
    max_episode_steps: int,
) -> List[str]:
    return [
        python_exe,
        "-m",
        "scripts.train_continuous",
        "--env",
        env,
        "--method",
        "full",
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
        "--verbose_thresholds",
    ]


def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.only_seed] if args.only_seed is not None else SEEDS

    jobs = []
    for seed in seeds:
        name = run_name(seed)
        run_dir = runs_dir / name
        cmd = build_command(
            python_exe=args.python_exe,
            env=args.env,
            run_dir=run_dir,
            seed=seed,
            context_density=args.context_density,
            context_behavior=args.context_behavior,
            context_sensor=args.context_sensor,
            total_steps=args.total_steps,
            max_episode_steps=args.max_episode_steps,
        )
        jobs.append((name, run_dir, cmd))

    print("\n=== CALIBRATION GRID PLAN ===")
    print(f"env: {args.env}")
    print(f"runs_dir: {runs_dir}")
    print(f"context: ({args.context_density}, {args.context_behavior}, {args.context_sensor})")
    print(f"total_steps: {args.total_steps}")
    print(f"max_episode_steps: {args.max_episode_steps}")
    print(f"num_jobs: {len(jobs)}")
    print("=============================\n")

    completed = 0
    skipped = 0
    failed = 0

    for idx, (name, run_dir, cmd) in enumerate(jobs, start=1):
        calibration_file = run_dir / "calibration_monitor.csv"

        print(f"[{idx}/{len(jobs)}] {name}")

        if args.skip_existing and calibration_file.exists():
            print(f"  SKIP existing: {calibration_file}")
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

    print("\n=== CALIBRATION GRID SUMMARY ===")
    print(f"completed: {completed}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    print(f"planned:   {len(jobs)}")
    print("================================")


if __name__ == "__main__":
    main()
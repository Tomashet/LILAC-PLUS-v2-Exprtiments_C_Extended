from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REGIME_CONTEXTS: Dict[str, Tuple[str, str, str]] = {
    "stationary": ("low", "calm", "clean"),
    "nonstationary_seen": ("low", "calm", "clean"),
    "nonstationary_unseen": ("low", "calm", "clean"),
}

SEEDS = [0, 1, 2]

METHOD_SPECS = {
    "baseline_sac": {
        "stage": "A",
        "needs_threshold_patch": False,
        "description": "Baseline SAC without LiLAC and without safety mechanisms.",
    },
    "adjust_speed_only": {
        "stage": "B",
        "needs_threshold_patch": False,
        "description": "Standalone adjust-speed monitor without LiLAC or other constraints.",
    },
    "lilac_none": {
        "stage": "A",
        "needs_threshold_patch": False,
        "description": "LiLAC adaptation only, no safety constraints.",
    },
    "lilac_context": {
        "stage": "B",
        "needs_threshold_patch": True,
        "description": "LiLAC with context-based constraints only.",
    },
    "lilac_speed": {
        "stage": "B",
        "needs_threshold_patch": False,
        "description": "LiLAC with speed-adjustment constraint only.",
    },
    "lilac_soft2hard": {
        "stage": "B",
        "needs_threshold_patch": True,
        "description": "LiLAC with soft-to-hard transformation only.",
    },
    "lilac_full": {
        "stage": "C",
        "needs_threshold_patch": True,
        "description": "Full LILAC+ with context, speed-adjustment, and soft-to-hard.",
    },
    "lilac_context_speed": {
        "stage": "D",
        "needs_threshold_patch": True,
        "description": "LiLAC with context constraints and speed adjustment.",
    },
    "lilac_context_soft2hard": {
        "stage": "D",
        "needs_threshold_patch": True,
        "description": "LiLAC with context constraints and soft-to-hard.",
    },
    "lilac_speed_soft2hard": {
        "stage": "D",
        "needs_threshold_patch": True,
        "description": "LiLAC with speed adjustment and soft-to-hard.",
    },
}

STAGE_METHODS = {
    "A": ["baseline_sac", "lilac_none"],
    "B": ["adjust_speed_only", "lilac_context", "lilac_speed", "lilac_soft2hard"],
    "C": ["lilac_full"],
    "D": ["lilac_context_speed", "lilac_context_soft2hard", "lilac_speed_soft2hard"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged thesis experiments with true regime scheduling.")
    parser.add_argument("--env", type=str, default="merge-v0")
    parser.add_argument("--runs_dir", type=str, default="runs_thesis")
    parser.add_argument("--threshold_patch", type=str, default="artifacts/calibration/recommended_thresholds_patch.json")
    parser.add_argument("--python_exe", type=str, default=sys.executable)
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "C", "D", "ALL"])
    parser.add_argument(
        "--regimes",
        type=str,
        nargs="+",
        default=["stationary", "nonstationary_seen", "nonstationary_unseen"],
        choices=["stationary", "nonstationary_seen", "nonstationary_unseen"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--seen_contexts", type=str, default="low|calm|clean;low|aggr|clean;low|calm|noisy")
    parser.add_argument("--unseen_contexts", type=str, default="high|aggr|noisy;high|aggr|dropout;high|calm|noisy")
    parser.add_argument("--regime_switch_every_episodes", type=int, default=1)
    return parser.parse_args()


def stage_method_list(stage: str) -> List[str]:
    if stage == "ALL":
        out: List[str] = []
        for s in ["A", "B", "C", "D"]:
            out.extend(STAGE_METHODS[s])
        return out
    return STAGE_METHODS[stage]


def run_name(method_key: str, regime: str, seed: int) -> str:
    return f"{method_key}_{regime}_s{seed}"


def build_command(
    python_exe: str,
    env: str,
    run_dir: Path,
    train_method: str,
    seed: int,
    context_density: str,
    context_behavior: str,
    context_sensor: str,
    total_steps: int,
    max_episode_steps: int,
    threshold_patch: str | None,
    regime: str,
    seen_contexts: str,
    unseen_contexts: str,
    switch_every_episodes: int,
) -> List[str]:
    cmd = [
        python_exe,
        "-m",
        "scripts.train_continuous",
        "--env",
        env,
        "--method",
        train_method,
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
        "--regime_type",
        regime,
        "--seen_contexts",
        seen_contexts,
        "--unseen_contexts",
        unseen_contexts,
        "--regime_switch_every_episodes",
        str(switch_every_episodes),
        "--total_steps",
        str(total_steps),
        "--max_episode_steps",
        str(max_episode_steps),
        "--verbose_thresholds",
        "--lilac_latent_dim",
        "8",
        "--lilac_context_len",
        "32",
        "--lilac_train_every_steps",
        "200",
        "--lilac_updates_per_train",
        "1",
        "--lilac_warmup_episodes",
        "5",
    ]
    if threshold_patch is not None:
        cmd.extend(["--threshold_patch", threshold_patch])
    return cmd


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    method_keys = stage_method_list(args.stage)
    jobs = []

    for method_key in method_keys:
        spec = METHOD_SPECS[method_key]
        for regime in args.regimes:
            density, behavior, sensor = REGIME_CONTEXTS[regime]
            for seed in args.seeds:
                run_dir = runs_dir / run_name(method_key, regime, seed)
                threshold_patch = args.threshold_patch if spec["needs_threshold_patch"] else None
                cmd = build_command(
                    python_exe=args.python_exe,
                    env=args.env,
                    run_dir=run_dir,
                    train_method=method_key,
                    seed=seed,
                    context_density=density,
                    context_behavior=behavior,
                    context_sensor=sensor,
                    total_steps=args.total_steps,
                    max_episode_steps=args.max_episode_steps,
                    threshold_patch=threshold_patch,
                    regime=regime,
                    seen_contexts=args.seen_contexts,
                    unseen_contexts=args.unseen_contexts,
                    switch_every_episodes=args.regime_switch_every_episodes,
                )
                jobs.append((method_key, regime, seed, spec, run_dir, cmd))

    print("\n=== THESIS EXPERIMENT PLAN ===")
    print(f"stage: {args.stage}")
    print(f"env: {args.env}")
    print(f"runs_dir: {runs_dir}")
    print(f"threshold_patch: {args.threshold_patch}")
    print(f"total_steps: {args.total_steps}")
    print(f"max_episode_steps: {args.max_episode_steps}")
    print(f"seen_contexts: {args.seen_contexts}")
    print(f"unseen_contexts: {args.unseen_contexts}")
    print(f"switch_every_episodes: {args.regime_switch_every_episodes}")
    print(f"num_jobs: {len(jobs)}")
    print("================================\n")

    completed = 0
    skipped = 0
    failed = 0

    for idx, (method_key, regime, seed, spec, run_dir, cmd) in enumerate(jobs, start=1):
        monitor_path = run_dir / "train_monitor.csv"
        print(f"[{idx}/{len(jobs)}] {method_key} | {regime} | seed={seed}")
        print(f"  desc: {spec['description']}")
        if args.skip_existing and monitor_path.exists():
            print(f"  SKIP existing: {monitor_path}")
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

    print("\n=== THESIS EXPERIMENT SUMMARY ===")
    print(f"completed: {completed}")
    print(f"skipped:   {skipped}")
    print(f"failed:    {failed}")
    print(f"planned:   {len(jobs)}")
    print("==================================")


if __name__ == "__main__":
    main()

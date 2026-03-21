from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# Stage C full launcher
# =============================================================================
#
# What this script does:
# - launches full Stage C grid
# - skips runs already completed
# - uses lock files to avoid duplicate execution
# - keeps per-run logs
# - records launcher events in JSONL
#
# Assumptions:
# - training entrypoint is: python -m scripts.train_continuous
# - your training script accepts the flags used below
# - each run writes artifacts into --run_dir
#
# IMPORTANT:
# The requested "9 methods x 3 regimes x 3 seeds = 162 runs" implies one
# additional 2-way axis, because 9*3*3 = 81.
# To preserve your requested 162 runs, this launcher adds:
#     SUITES = ["stageC_A", "stageC_B"]
# Total = 2 * 9 * 3 * 3 = 162
#
# Rename/remove SUITES if needed.
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs_stageC_full"
LAUNCHER_LOG = ROOT / "artifacts" / "stageC_launcher_log.jsonl"
TRAIN_MODULE = "scripts.train_continuous"

TOTAL_STEPS = 12000
MAX_EPISODE_STEPS = 200
ENV_NAME = "merge-v0"

# ---------------------------------------------------------------------------
# Extra 2-way axis to make the grid total 162
# Replace these names with your actual split if needed.
# ---------------------------------------------------------------------------
SUITES = ["stageC_A", "stageC_B"]

REGIMES = [
    "stationary",
    "nonstationary_seen",
    "nonstationary_unseen",
]

SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Stage C methods
# These names are also used in run_dir naming.
#
# The flag mapping below follows the concepts you validated:
# - proactive context-based constraints
# - adjustment speed
# - soft-to-hard conversion
# - fixed baselines
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str
    flags: Dict[str, object]


METHOD_SPECS: List[MethodSpec] = [
    MethodSpec(
        name="baseline",
        flags=dict(
            use_context_constraints=False,
            use_adjust_speed=False,
            use_soft_to_hard=False,
            use_fixed_constraints=False,
            apply_thresholds=False,
        ),
    ),
    MethodSpec(
        name="cb",
        flags=dict(
            use_context_constraints=True,
            use_adjust_speed=False,
            use_soft_to_hard=False,
            use_fixed_constraints=False,
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="adjust_speed_only",
        flags=dict(
            use_context_constraints=False,
            use_adjust_speed=True,
            use_soft_to_hard=False,
            use_fixed_constraints=False,
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="soft_to_hard_only",
        flags=dict(
            use_context_constraints=False,
            use_adjust_speed=False,
            use_soft_to_hard=True,
            use_fixed_constraints=False,
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="full",
        flags=dict(
            use_context_constraints=True,
            use_adjust_speed=True,
            use_soft_to_hard=True,
            use_fixed_constraints=False,
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="fixed_baseline",
        flags=dict(
            use_context_constraints=False,
            use_adjust_speed=False,
            use_soft_to_hard=False,
            use_fixed_constraints=True,
            fixed_strategy="baseline",
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="fixed_full_A",
        flags=dict(
            use_context_constraints=True,
            use_adjust_speed=True,
            use_soft_to_hard=True,
            use_fixed_constraints=True,
            fixed_strategy="A",
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="fixed_full_B",
        flags=dict(
            use_context_constraints=True,
            use_adjust_speed=True,
            use_soft_to_hard=True,
            use_fixed_constraints=True,
            fixed_strategy="B",
            apply_thresholds=True,
        ),
    ),
    MethodSpec(
        name="fixed_full_C",
        flags=dict(
            use_context_constraints=True,
            use_adjust_speed=True,
            use_soft_to_hard=True,
            use_fixed_constraints=True,
            fixed_strategy="C",
            apply_thresholds=True,
        ),
    ),
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def file_exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def looks_completed(run_dir: Path) -> bool:
    """
    Completion heuristic.
    Update this if your codebase uses a different completion marker.
    """
    completion_markers = [
        run_dir / "done.txt",
        run_dir / "train_monitor.csv",
        run_dir / "debug_run.json",
        run_dir / "run_debug.json",
        run_dir / "summary.json",
    ]

    # Most conservative: require train_monitor.csv or done.txt
    if file_exists_nonempty(run_dir / "done.txt"):
        return True
    if file_exists_nonempty(run_dir / "train_monitor.csv"):
        return True

    # Fallback: if a summary/debug artifact exists and stdout indicates success
    if any(file_exists_nonempty(p) for p in completion_markers[2:]):
        out_log = run_dir / "stdout.log"
        if file_exists_nonempty(out_log):
            try:
                tail = out_log.read_text(encoding="utf-8", errors="ignore")[-5000:]
                success_tokens = [
                    "Training finished",
                    "Saved monitor",
                    "Saved results",
                    "completed",
                ]
                if any(tok.lower() in tail.lower() for tok in success_tokens):
                    return True
            except Exception:
                pass

    return False


def looks_failed(run_dir: Path) -> bool:
    err_log = run_dir / "stderr.log"
    if not file_exists_nonempty(err_log):
        return False
    try:
        text = err_log.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    failure_tokens = [
        "traceback",
        "exception",
        "modulenotfounderror",
        "filenotfounderror",
        "runtimeerror",
        "assertionerror",
        "valueerror",
    ]
    return any(tok in text for tok in failure_tokens)


def lock_path(run_dir: Path) -> Path:
    return run_dir / ".launch_lock"


def acquire_lock(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    lp = lock_path(run_dir)
    if lp.exists():
        return False
    try:
        with lp.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "time": time.time(),
                    "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME"),
                },
                f,
            )
        return True
    except Exception:
        return False


def release_lock(run_dir: Path) -> None:
    lp = lock_path(run_dir)
    if lp.exists():
        try:
            lp.unlink()
        except Exception:
            pass


def normalize_flag_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_run_name(suite: str, method: str, regime: str, seed: int) -> str:
    return f"{suite}__{method}__{regime}__s{seed}"


def build_run_dir(suite: str, method: str, regime: str, seed: int) -> Path:
    return RUNS_ROOT / build_run_name(suite, method, regime, seed)


def build_command(
    suite: str,
    method_spec: MethodSpec,
    regime: str,
    seed: int,
    run_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--env",
        ENV_NAME,
        "--algo",
        "sac",
        "--total_steps",
        str(TOTAL_STEPS),
        "--max_episode_steps",
        str(MAX_EPISODE_STEPS),
        "--seed",
        str(seed),
        "--regime",
        regime,
        "--run_dir",
        str(run_dir),
        "--method_name",
        method_spec.name,
        "--suite_name",
        suite,
        "--stage_name",
        "stageC",
        "--enable_in_run_nonstationarity",
        "true",
        "--enable_context_prediction",
        "true",
        "--enable_proactive_constraints",
        "true",
    ]

    for key, value in method_spec.flags.items():
        cmd.extend([f"--{key}", normalize_flag_value(value)])

    return cmd


def print_grid_summary() -> None:
    total = len(SUITES) * len(METHOD_SPECS) * len(REGIMES) * len(SEEDS)
    print("=" * 90)
    print("STAGE C FULL LAUNCHER")
    print("=" * 90)
    print(f"Suites:   {len(SUITES)} -> {SUITES}")
    print(f"Methods:  {len(METHOD_SPECS)} -> {[m.name for m in METHOD_SPECS]}")
    print(f"Regimes:  {len(REGIMES)} -> {REGIMES}")
    print(f"Seeds:    {len(SEEDS)} -> {SEEDS}")
    print(f"Steps:    {TOTAL_STEPS}")
    print(f"Runs dir: {RUNS_ROOT}")
    print("-" * 90)
    print(f"TOTAL RUNS = {total}")
    print("=" * 90)

    if total != 162:
        raise ValueError(
            f"Expected 162 runs, but current grid expands to {total}. "
            f"Adjust SUITES / METHODS / REGIMES / SEEDS."
        )


def preflight() -> None:
    print_grid_summary()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_parent(LAUNCHER_LOG)

    # Quick import check
    try:
        __import__("scripts")
    except Exception as e:
        raise RuntimeError(
            "Could not import local 'scripts' package. Run from project root."
        ) from e


def mark_done(run_dir: Path, returncode: int) -> None:
    marker = run_dir / "done.txt"
    with marker.open("w", encoding="utf-8") as f:
        f.write(f"returncode={returncode}\n")
        f.write(f"time={time.time()}\n")


def launch_one(
    suite: str,
    method_spec: MethodSpec,
    regime: str,
    seed: int,
) -> str:
    run_dir = build_run_dir(suite, method_spec.name, regime, seed)
    run_name = run_dir.name

    if looks_completed(run_dir):
        print(f"[SKIP completed] {run_name}")
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_completed",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        return "skipped_completed"

    if lock_path(run_dir).exists():
        print(f"[SKIP locked] {run_name}")
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_locked",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        return "skipped_locked"

    if not acquire_lock(run_dir):
        print(f"[SKIP lock_acquire_failed] {run_name}")
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_lock_acquire_failed",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        return "skipped_lock_failed"

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    cmd = build_command(suite, method_spec, regime, seed, run_dir)

    print(f"[LAUNCH] {run_name}")
    print("         " + " ".join(shlex.quote(x) for x in cmd))

    write_jsonl(
        LAUNCHER_LOG,
        {
            "event": "launch",
            "run_name": run_name,
            "run_dir": str(run_dir),
            "suite": suite,
            "method": method_spec.name,
            "regime": regime,
            "seed": seed,
            "cmd": cmd,
            "time": time.time(),
        },
    )

    try:
        with stdout_path.open("a", encoding="utf-8") as stdout_f, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_f:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=stdout_f,
                stderr=stderr_f,
                check=False,
            )
        rc = proc.returncode

        if rc == 0:
            mark_done(run_dir, rc)
            print(f"[OK] {run_name}")
            write_jsonl(
                LAUNCHER_LOG,
                {
                    "event": "success",
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "returncode": rc,
                    "time": time.time(),
                },
            )
            return "success"

        print(f"[FAIL rc={rc}] {run_name}")
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "failure",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "returncode": rc,
                "time": time.time(),
            },
        )
        return "failed"

    finally:
        release_lock(run_dir)


def main() -> None:
    preflight()

    counts = {
        "success": 0,
        "failed": 0,
        "skipped_completed": 0,
        "skipped_locked": 0,
        "skipped_lock_failed": 0,
    }

    start = time.time()

    for suite in SUITES:
        for method_spec in METHOD_SPECS:
            for regime in REGIMES:
                for seed in SEEDS:
                    status = launch_one(
                        suite=suite,
                        method_spec=method_spec,
                        regime=regime,
                        seed=seed,
                    )
                    counts[status] = counts.get(status, 0) + 1

    elapsed = time.time() - start

    print("\n" + "=" * 90)
    print("STAGE C LAUNCH SUMMARY")
    print("=" * 90)
    for k, v in counts.items():
        print(f"{k:>22}: {v}")
    print(f"{'elapsed_sec':>22}: {elapsed:.1f}")
    print(f"{'launcher_log':>22}: {LAUNCHER_LOG}")
    print("=" * 90)

    write_jsonl(
        LAUNCHER_LOG,
        {
            "event": "summary",
            "counts": counts,
            "elapsed_sec": elapsed,
            "time": time.time(),
        },
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

# =============================================================================
# STAGE C FULL PARALLEL LAUNCHER — FINAL PRODUCTION VERSION
# =============================================================================
#
# Confirmed CLI:
#   python -m scripts.train_continuous
#       --env ...
#       --total_steps ...
#       --max_episode_steps ...
#       --seed ...
#       --regime ...
#       --run_dir ...
#       --method ...
#
# This launcher provides:
# - full Stage C grid
# - parallel workers
# - safe restart
# - skip completed runs
# - lock files
# - per-run stdout / stderr logs
# - launcher JSONL log
#
# IMPORTANT:
# The current grid below expands to:
#   2 suites * 10 methods * 3 regimes * 3 seeds = 180 runs
#
# That matches the ACTUAL method list exposed by train_continuous.py:
#   unconstrained, fixed_full_A, fixed_full_C,
#   cb, as, sh, cb+as, cb+sh, as+sh, cb+as+sh
#
# If you truly want 162 instead of 180, one of these must be excluded.
# For now, this launcher uses the full parser-supported set of 10 methods.
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs_stageC_full"
ARTIFACTS_ROOT = ROOT / "artifacts"

TRAIN_MODULE = "scripts.train_continuous"
ENV_NAME = "merge-v0"
TOTAL_STEPS = 15000
MAX_EPISODE_STEPS = 200
DEFAULT_MAX_WORKERS = 4

LAUNCHER_LOG = ARTIFACTS_ROOT / "stageC_launcher_parallel_log.jsonl"

SUITES = ["stageC_A", "stageC_B"]
REGIMES = ["stationary", "nonstationary_seen", "nonstationary_unseen"]
SEEDS = [0, 1, 2]

# Real methods accepted by the parser
METHODS = [
    "unconstrained",
    "cb",
    "as",
    "sh",
    "cb+as",
    "cb+sh",
    "as+sh",
    "cb+as+sh",
    "fixed_full_A",
    "fixed_full_C",
]


@dataclass(frozen=True)
class RunSpec:
    suite: str
    method: str
    regime: str
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.suite}__{self.method}__{self.regime}__s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return RUNS_ROOT / self.run_name


PRINT_LOCK = Lock()
LOG_LOCK = Lock()


def safe_print(*args, **kwargs) -> None:
    with PRINT_LOCK:
        print(*args, **kwargs)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with LOG_LOCK:
        ensure_parent(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def file_exists_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def done_path(run_dir: Path) -> Path:
    return run_dir / "done.txt"


def lock_path(run_dir: Path) -> Path:
    return run_dir / ".launch_lock"


def stdout_path(run_dir: Path) -> Path:
    return run_dir / "stdout.log"


def stderr_path(run_dir: Path) -> Path:
    return run_dir / "stderr.log"


def looks_completed(run_dir: Path) -> bool:
    if file_exists_nonempty(done_path(run_dir)):
        return True

    if file_exists_nonempty(run_dir / "train_monitor.csv"):
        return True

    fallback_markers = [
        run_dir / "summary.json",
        run_dir / "run_debug.json",
        run_dir / "debug_run.json",
    ]
    if any(file_exists_nonempty(p) for p in fallback_markers):
        out_log = stdout_path(run_dir)
        if file_exists_nonempty(out_log):
            try:
                tail = out_log.read_text(encoding="utf-8", errors="ignore")[-5000:].lower()
                success_tokens = [
                    "training finished",
                    "completed",
                    "saved monitor",
                    "saved results",
                ]
                return any(tok in tail for tok in success_tokens)
            except Exception:
                return False

    return False


def looks_failed(run_dir: Path) -> bool:
    err = stderr_path(run_dir)
    if not file_exists_nonempty(err):
        return False
    try:
        text = err.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False

    failure_tokens = [
        "traceback",
        "exception",
        "runtimeerror",
        "assertionerror",
        "valueerror",
        "filenotfounderror",
        "modulenotfounderror",
        "error: unrecognized arguments",
        "the following arguments are required",
    ]
    return any(tok in text for tok in failure_tokens)


def acquire_lock(run_dir: Path) -> bool:
    ensure_dir(run_dir)
    lp = lock_path(run_dir)
    if lp.exists():
        return False
    try:
        with lp.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "pid": None,
                    "time": time.time(),
                    "host": None,
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


def mark_done(run_dir: Path, returncode: int) -> None:
    with done_path(run_dir).open("w", encoding="utf-8") as f:
        f.write(f"returncode={returncode}\n")
        f.write(f"time={time.time()}\n")


def build_command(spec: RunSpec) -> List[str]:
    return [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--env",
        ENV_NAME,
        "--total_steps",
        str(TOTAL_STEPS),
        "--max_episode_steps",
        str(MAX_EPISODE_STEPS),
        "--seed",
        str(spec.seed),
        "--regime",
        spec.regime,
        "--run_dir",
        str(spec.run_dir),
        "--method",
        spec.method,
    ]


def build_all_runs() -> List[RunSpec]:
    runs: List[RunSpec] = []
    for suite in SUITES:
        for method in METHODS:
            for regime in REGIMES:
                for seed in SEEDS:
                    runs.append(RunSpec(suite=suite, method=method, regime=regime, seed=seed))
    return runs


def classify_run_state(spec: RunSpec) -> str:
    rd = spec.run_dir
    if looks_completed(rd):
        return "completed"
    if lock_path(rd).exists():
        return "locked"
    if looks_failed(rd):
        return "failed_existing"
    return "pending"


def print_prelaunch_summary(runs: List[RunSpec]) -> None:
    counts = {
        "completed": 0,
        "locked": 0,
        "failed_existing": 0,
        "pending": 0,
    }
    for spec in runs:
        counts[classify_run_state(spec)] += 1

    safe_print("\nPRELAUNCH STATE")
    safe_print("-" * 90)
    for k, v in counts.items():
        safe_print(f"{k:>18}: {v}")
    safe_print("-" * 90)


def preflight() -> List[RunSpec]:
    ensure_dir(RUNS_ROOT)
    ensure_dir(ARTIFACTS_ROOT)

    runs = build_all_runs()

    safe_print("=" * 90)
    safe_print("STAGE C FULL PARALLEL LAUNCHER")
    safe_print("=" * 90)
    safe_print(f"Project root:     {ROOT}")
    safe_print(f"Runs root:        {RUNS_ROOT}")
    safe_print(f"Launcher log:     {LAUNCHER_LOG}")
    safe_print(f"Train module:     {TRAIN_MODULE}")
    safe_print(f"Env:              {ENV_NAME}")
    safe_print(f"Total steps:      {TOTAL_STEPS}")
    safe_print(f"Max ep steps:     {MAX_EPISODE_STEPS}")
    safe_print(f"Suites:           {SUITES}")
    safe_print(f"Methods:          {METHODS}")
    safe_print(f"Regimes:          {REGIMES}")
    safe_print(f"Seeds:            {SEEDS}")
    safe_print("-" * 90)
    safe_print(f"Expanded runs:    {len(runs)}")
    safe_print("=" * 90)

    try:
        __import__("scripts")
    except Exception as e:
        raise RuntimeError(
            "Could not import local 'scripts' package. Run this from the project root."
        ) from e

    return runs


def run_one(spec: RunSpec) -> Tuple[str, str]:
    run_name = spec.run_name
    run_dir = spec.run_dir

    if looks_completed(run_dir):
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_completed",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        safe_print(f"[SKIP completed] {run_name}")
        return run_name, "skipped_completed"

    if lock_path(run_dir).exists():
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_locked",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        safe_print(f"[SKIP locked]    {run_name}")
        return run_name, "skipped_locked"

    if not acquire_lock(run_dir):
        write_jsonl(
            LAUNCHER_LOG,
            {
                "event": "skip_lock_acquire_failed",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "time": time.time(),
            },
        )
        safe_print(f"[SKIP lock-fail] {run_name}")
        return run_name, "skipped_lock_failed"

    cmd = build_command(spec)

    write_jsonl(
        LAUNCHER_LOG,
        {
            "event": "launch",
            "run_name": run_name,
            "run_dir": str(run_dir),
            "suite": spec.suite,
            "method": spec.method,
            "regime": spec.regime,
            "seed": spec.seed,
            "cmd": cmd,
            "time": time.time(),
        },
    )

    safe_print(f"[LAUNCH]         {run_name}")

    try:
        with stdout_path(run_dir).open("a", encoding="utf-8") as out_f, stderr_path(run_dir).open(
            "a", encoding="utf-8"
        ) as err_f:
            out_f.write("\n" + "=" * 80 + "\n")
            out_f.write(f"LAUNCH TIME: {time.ctime()}\n")
            out_f.write("COMMAND:\n")
            out_f.write(" ".join(cmd) + "\n")
            out_f.write("=" * 80 + "\n\n")
            out_f.flush()

            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=out_f,
                stderr=err_f,
                check=False,
            )

        rc = proc.returncode

        if rc == 0:
            mark_done(run_dir, rc)
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
            safe_print(f"[OK]             {run_name}")
            return run_name, "success"

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
        safe_print(f"[FAIL rc={rc}]    {run_name}")
        return run_name, "failed"

    finally:
        release_lock(run_dir)


def main(max_workers: int = DEFAULT_MAX_WORKERS) -> None:
    runs = preflight()
    print_prelaunch_summary(runs)

    pending_runs = [spec for spec in runs if classify_run_state(spec) in {"pending", "failed_existing"}]

    safe_print(f"\nWorkers:          {max_workers}")
    safe_print(f"Runs to process:  {len(pending_runs)}")
    safe_print("=" * 90)

    if not pending_runs:
        safe_print("Nothing to launch. All runs already completed or are locked.")
        return

    start = time.time()

    counts = {
        "success": 0,
        "failed": 0,
        "skipped_completed": 0,
        "skipped_locked": 0,
        "skipped_lock_failed": 0,
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {executor.submit(run_one, spec): spec for spec in pending_runs}

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                _, status = future.result()
            except Exception as e:
                status = "failed"
                write_jsonl(
                    LAUNCHER_LOG,
                    {
                        "event": "launcher_exception",
                        "run_name": spec.run_name,
                        "run_dir": str(spec.run_dir),
                        "error": repr(e),
                        "time": time.time(),
                    },
                )
                safe_print(f"[EXCEPTION]      {spec.run_name} -> {repr(e)}")

            counts[status] = counts.get(status, 0) + 1

    elapsed = time.time() - start

    safe_print("\n" + "=" * 90)
    safe_print("STAGE C PARALLEL LAUNCH SUMMARY")
    safe_print("=" * 90)
    for k, v in counts.items():
        safe_print(f"{k:>22}: {v}")
    safe_print(f"{'elapsed_sec':>22}: {elapsed:.1f}")
    safe_print(f"{'launcher_log':>22}: {LAUNCHER_LOG}")
    safe_print("=" * 90)

    write_jsonl(
        LAUNCHER_LOG,
        {
            "event": "summary",
            "counts": counts,
            "elapsed_sec": elapsed,
            "workers": max_workers,
            "time": time.time(),
        },
    )


if __name__ == "__main__":
    workers = DEFAULT_MAX_WORKERS
    if len(sys.argv) >= 2:
        workers = int(sys.argv[1])
    main(max_workers=workers)
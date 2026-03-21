# scripts/run_pilot_grid.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ============================================================
# Pilot grid configuration
# ============================================================

ENV = "merge-v0"
TOTAL_STEPS = 5000
SEED = 0
RUN_ROOT = Path("runs/pilot_grid")

# Minimal pilot:
# 2 methods × 2 regimes × 1 seed = 4 runs
METHODS = ["baseline", "full"]

CONTEXTS = {
    "seen": ("low", "calm", "clean"),
    "unseen": ("high", "aggr", "dropout"),
}

THRESHOLD_PATCH = None
# Set this to your real patch path when ready, e.g.:
# THRESHOLD_PATCH = "artifacts/calibrated_thresholds.json"


# ============================================================
# Helpers
# ============================================================

def run_job(cmd):
    print("\n" + "=" * 100)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 100 + "\n")
    subprocess.run(cmd, check=True)


# ============================================================
# Main
# ============================================================

def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    total_jobs = 0

    for method in METHODS:
        for regime_name, ctx in CONTEXTS.items():
            density, behavior, sensor = ctx

            run_dir = RUN_ROOT / method / regime_name / f"seed_{SEED}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "scripts.train_continuous",
                "--env",
                ENV,
                "--method",
                method,
                "--total_steps",
                str(TOTAL_STEPS),
                "--seed",
                str(SEED),
                "--run_dir",
                str(run_dir),
                "--context_density",
                density,
                "--context_behavior",
                behavior,
                "--context_sensor",
                sensor,
                "--verbose_thresholds",
            ]

            if THRESHOLD_PATCH is not None:
                cmd += ["--threshold_patch", THRESHOLD_PATCH]

            run_job(cmd)
            total_jobs += 1

    print("\n" + "=" * 100)
    print("PILOT GRID COMPLETE")
    print(f"Total runs: {total_jobs}")
    print(f"Run root: {RUN_ROOT}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
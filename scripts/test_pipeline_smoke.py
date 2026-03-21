# scripts/test_pipeline_smoke.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pipeline smoke tests for LILAC+ continuous setup."
    )
    parser.add_argument(
        "--threshold_patch",
        type=str,
        default=None,
        help="Optional threshold patch JSON. If provided, patched tests will run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Timesteps for each short smoke training run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for smoke runs.",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="runs/pipeline_smoke",
        help="Root folder for smoke-test run outputs.",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 90 + "\n")
    subprocess.run(cmd, check=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")


def verify_run_dir(run_dir: Path) -> None:
    print(f"[verify] Checking artifacts in: {run_dir}")
    require_file(run_dir / "run_config.json")
    require_file(run_dir / "train_monitor.csv")

    model_zip = run_dir / "model.zip"
    model_dir = run_dir / "model"

    if not model_zip.exists() and not model_dir.exists():
        raise FileNotFoundError(
            f"Expected model artifact not found in {run_dir}. "
            f"Checked: {model_zip} and {model_dir}"
        )

    print(f"[verify] OK: {run_dir}")


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Test 1: Threshold audit without patch
    # ----------------------------------------------------------
    print("\n[TEST 1] Threshold audit without patch")
    audit_csv_no_patch = run_root / "audit_no_patch.csv"
    run_cmd(
        [
            sys.executable,
            "-m",
            "scripts.audit_threshold_inference",
            "--out_csv",
            str(audit_csv_no_patch),
        ]
    )
    require_file(audit_csv_no_patch)

    # ----------------------------------------------------------
    # Test 2: Continuous smoke run without patch
    # ----------------------------------------------------------
    print("\n[TEST 2] Continuous smoke training without patch")
    run_dir_no_patch = run_root / "continuous_no_patch"
    run_cmd(
        [
            sys.executable,
            "-m",
            "scripts.train_continuous",
            "--env",
            "merge-v0",
            "--total_steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--run_dir",
            str(run_dir_no_patch),
            "--verbose_thresholds",
        ]
    )
    verify_run_dir(run_dir_no_patch)

    # ----------------------------------------------------------
    # Test 3: Threshold audit with patch (optional)
    # ----------------------------------------------------------
    if args.threshold_patch is not None:
        print("\n[TEST 3] Threshold audit with patch")
        audit_csv_patch = run_root / "audit_with_patch.csv"
        run_cmd(
            [
                sys.executable,
                "-m",
                "scripts.audit_threshold_inference",
                "--threshold_patch",
                args.threshold_patch,
                "--out_csv",
                str(audit_csv_patch),
            ]
        )
        require_file(audit_csv_patch)

        # ------------------------------------------------------
        # Test 4: Continuous smoke run with patch
        # ------------------------------------------------------
        print("\n[TEST 4] Continuous smoke training with patch")
        run_dir_patch = run_root / "continuous_with_patch"
        run_cmd(
            [
                sys.executable,
                "-m",
                "scripts.train_continuous",
                "--env",
                "merge-v0",
                "--total_steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--run_dir",
                str(run_dir_patch),
                "--threshold_patch",
                args.threshold_patch,
                "--verbose_thresholds",
            ]
        )
        verify_run_dir(run_dir_patch)

    print("\n" + "=" * 90)
    print("ALL PIPELINE SMOKE TESTS PASSED")
    print(f"Artifacts saved under: {run_root}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
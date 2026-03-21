from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs_stageC_full"
ARTIFACTS_ROOT = ROOT / "artifacts"

SUITES = ["stageC_A", "stageC_B"]
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
REGIMES = ["stationary", "nonstationary_seen", "nonstationary_unseen"]
SEEDS = [0, 1, 2]


def get_run_dir(suite: str, method: str, regime: str, seed: int) -> Path:
    return RUNS_ROOT / f"{suite}__{method}__{regime}__s{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "done.txt").exists() or (run_dir / "train_monitor.csv").exists()


def is_locked(run_dir: Path) -> bool:
    return (run_dir / ".launch_lock").exists()


def has_failure(run_dir: Path) -> bool:
    err = run_dir / "stderr.log"
    if not err.exists():
        return False
    text = err.read_text(encoding="utf-8", errors="ignore").lower()
    tokens = [
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
    return any(tok in text for tok in tokens)


def main() -> None:
    total = 0
    completed = 0
    locked = 0
    failed = 0
    missing = []

    for suite in SUITES:
        for method in METHODS:
            for regime in REGIMES:
                for seed in SEEDS:
                    total += 1
                    rd = get_run_dir(suite, method, regime, seed)

                    if is_done(rd):
                        completed += 1
                    else:
                        missing.append(str(rd))

                    if is_locked(rd):
                        locked += 1

                    if has_failure(rd):
                        failed += 1

    summary = {
        "runs_root": str(RUNS_ROOT),
        "total_expected": total,
        "completed": completed,
        "locked": locked,
        "failures_seen": failed,
        "missing_count": len(missing),
        "missing_first20": missing[:20],
    }

    print("=" * 80)
    print("STAGE C COVERAGE CHECK")
    print("=" * 80)
    for k, v in summary.items():
        if k != "missing_first20":
            print(f"{k}: {v}")
    print("=" * 80)

    if missing:
        print("\nFirst 20 missing:")
        for x in missing[:20]:
            print(x)

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_ROOT / "stageC_grid_check.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
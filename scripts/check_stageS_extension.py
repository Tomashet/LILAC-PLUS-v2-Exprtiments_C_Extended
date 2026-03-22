from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
RUNS_STAGEC = ROOT / "runs_stageC"
RUNS_STAGES = ROOT / "runs_stageS_extension"
ARTIFACTS = ROOT / "artifacts"

OUT_JSON = ARTIFACTS / "merged_stageC_stageS_manifest.json"
OUT_CSV = ARTIFACTS / "merged_stageC_stageS_manifest.csv"


def detect_stage(run_dir: Path) -> str:
    parent_name = run_dir.parent.name.lower()
    if "stages" in parent_name:
        return "stageS_extension"
    if "stagec" in parent_name:
        return "stageC_core"
    return "unknown"


def detect_completion(run_dir: Path) -> bool:
    return (run_dir / "done.txt").exists() or (run_dir / "train_monitor.csv").exists()


def collect_runs(root_dir: Path) -> List[Dict[str, object]]:
    if not root_dir.exists():
        return []

    rows: List[Dict[str, object]] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue

        train_csv = child / "train_monitor.csv"
        done_txt = child / "done.txt"
        stdout_log = child / "stdout.log"
        stderr_log = child / "stderr.log"

        rows.append(
            {
                "stage_group": detect_stage(child),
                "run_name": child.name,
                "run_dir": str(child),
                "completed": detect_completion(child),
                "has_train_monitor_csv": train_csv.exists(),
                "has_done_txt": done_txt.exists(),
                "has_stdout_log": stdout_log.exists(),
                "has_stderr_log": stderr_log.exists(),
            }
        )

    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(RUNS_STAGEC) + collect_runs(RUNS_STAGES)

    summary = {
        "runs_stageC_root": str(RUNS_STAGEC),
        "runs_stageS_extension_root": str(RUNS_STAGES),
        "total_rows": len(rows),
        "completed_rows": sum(int(r["completed"]) for r in rows),
        "rows": rows,
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, OUT_CSV)

    print("=" * 80)
    print("MERGED STAGE C + STAGE S-EXTENSION MANIFEST")
    print("=" * 80)
    print(f"Total rows:     {summary['total_rows']}")
    print(f"Completed rows: {summary['completed_rows']}")
    print(f"JSON:           {OUT_JSON}")
    print(f"CSV:            {OUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
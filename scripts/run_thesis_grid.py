"""Run a thesis-quality experiment grid.

This script launches multiple training runs (typically SAC on highway-v0)
across seeds, nonstationarity strengths, and constraint plugins.

It intentionally uses subprocess calls to the existing training entrypoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


@dataclass
class RunSpec:
    env: str
    algo: str
    total_steps: int
    seed: int
    p_stay: float
    lilac: bool
    constraint: str
    run_dir: str
    extra_args: List[str]

    def to_cmd(self) -> List[str]:
        cmd = [
            "python",
            "-m",
            "scripts.train_continuous",
            "--env",
            self.env,
            "--algo",
            self.algo,
            "--total_steps",
            str(self.total_steps),
            "--seed",
            str(self.seed),
            "--p_stay",
            str(self.p_stay),
            "--run_dir",
            self.run_dir,
        ]
        if self.lilac:
            cmd += ["--lilac", "--constraint", self.constraint]
        else:
            # When LILAC is off, keep classic baseline behavior.
            # If your train script supports '--constraints on', enable it when a
            # constraint name other than 'none/off' is given.
            if self.constraint not in ("none", "off"):
                cmd += ["--constraints", "on"]
        cmd += self.extra_args
        return cmd


def parse_csv_list(s: str, cast=float) -> List:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [cast(x) for x in items]


def build_runs(
    env: str,
    algo: str,
    total_steps: int,
    seeds: Iterable[int],
    p_stays: Iterable[float],
    lilac: bool,
    constraints: Iterable[str],
    tag: str,
    extra_args: List[str],
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in p_stays:
        for c in constraints:
            for seed in seeds:
                run_dir = (
                    f"{tag}_{env}_{algo}_p{p:g}_"
                    f"{'lilac' if lilac else 'base'}_{c}_seed{seed}_"
                    f"{total_steps//1000}k_{ts}"
                )
                runs.append(
                    RunSpec(
                        env=env,
                        algo=algo,
                        total_steps=total_steps,
                        seed=seed,
                        p_stay=p,
                        lilac=lilac,
                        constraint=c,
                        run_dir=run_dir,
                        extra_args=list(extra_args),
                    )
                )
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a thesis experiment grid.")
    ap.add_argument("--env", default="highway-v0")
    ap.add_argument("--algo", default="sac")
    ap.add_argument("--total_steps", type=int, default=500_000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--p_stays", default="0.95,0.8,0.6")
    ap.add_argument(
        "--constraints",
        default="none,proactive_forecast,adjust_speed,cpss",
        help="Comma-separated. Used only when --lilac is set.",
    )
    ap.add_argument("--lilac", action="store_true", help="Enable LILAC PLUS.")
    ap.add_argument(
        "--tag",
        default="THESIS",
        help="Prefix for run_dir names (keeps runs grouped).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing.",
    )
    ap.add_argument(
        "--log_json",
        default="runs/thesis_grid_manifest.json",
        help="Where to save a manifest of all commands.",
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to scripts.train_continuous (prefix with --).",
    )
    args = ap.parse_args()

    seeds = parse_csv_list(args.seeds, cast=int)
    p_stays = parse_csv_list(args.p_stays, cast=float)
    constraints = [x.strip() for x in args.constraints.split(",") if x.strip()]

    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    runs = build_runs(
        env=args.env,
        algo=args.algo,
        total_steps=args.total_steps,
        seeds=seeds,
        p_stays=p_stays,
        lilac=args.lilac,
        constraints=constraints,
        tag=args.tag,
        extra_args=extra_args,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "runs": [],
    }

    for i, spec in enumerate(runs, start=1):
        cmd = spec.to_cmd()
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        manifest["runs"].append(
            {
                "index": i,
                "run_dir": spec.run_dir,
                "env": spec.env,
                "algo": spec.algo,
                "total_steps": spec.total_steps,
                "seed": spec.seed,
                "p_stay": spec.p_stay,
                "lilac": spec.lilac,
                "constraint": spec.constraint,
                "cmd": cmd,
                "cmd_str": cmd_str,
            }
        )

        print(f"[{i}/{len(runs)}] {spec.run_dir}")
        print(cmd_str)
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    Path(args.log_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved grid manifest to: {args.log_json}")


if __name__ == "__main__":
    main()

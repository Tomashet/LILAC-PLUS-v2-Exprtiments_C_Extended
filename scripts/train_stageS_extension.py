from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from scripts.stageS_extension_specs import get_stage_s_extension_spec_map


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODULE = "scripts.train_continuous"


def build_parser() -> argparse.ArgumentParser:
    spec_names = sorted(get_stage_s_extension_spec_map().keys())

    p = argparse.ArgumentParser(description="Stage S-extension wrapper training entrypoint")
    p.add_argument("--extension_method", required=True, choices=spec_names)
    p.add_argument("--env", default="merge-v0")
    p.add_argument("--total_steps", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_dir", required=True)
    p.add_argument(
        "--regime",
        required=True,
        choices=["stationary", "nonstationary_seen", "nonstationary_unseen"],
    )
    p.add_argument("--max_episode_steps", type=int, default=200)
    return p


def build_command(args: argparse.Namespace) -> List[str]:
    spec_map = get_stage_s_extension_spec_map()
    spec = spec_map[args.extension_method]

    if not spec.ready:
        raise RuntimeError(
            f"Stage S-extension method '{spec.name}' is not mapped yet.\n"
            f"Notes: {spec.notes}"
        )
    if not spec.train_method:
        raise RuntimeError(
            f"Stage S-extension method '{spec.name}' has no train_method mapping.\n"
            f"Notes: {spec.notes}"
        )

    cmd = [
        sys.executable,
        "-m",
        TRAIN_MODULE,
        "--env",
        args.env,
        "--total_steps",
        str(args.total_steps),
        "--max_episode_steps",
        str(args.max_episode_steps),
        "--seed",
        str(args.seed),
        "--regime",
        args.regime,
        "--run_dir",
        args.run_dir,
        "--method",
        spec.train_method,
    ]

    for k, v in spec.extra_cli_args.items():
        cmd.extend([f"--{k}", str(v)])

    return cmd


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cmd = build_command(args)
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
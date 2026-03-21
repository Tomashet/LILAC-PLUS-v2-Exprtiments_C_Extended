# scripts/train_discrete.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
from gymnasium.error import NameNotFound
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from src.context import (
    OnlineThresholdStats,
    SimilarityConfig,
    canonicalize_context,
    get_context_thresholds,
    load_threshold_patch,
    make_safe_inference_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="merge-v0")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_dir", type=str, required=True)

    parser.add_argument("--threshold_patch", type=str, default=None)

    parser.add_argument("--context_density", type=str, default="low")
    parser.add_argument("--context_behavior", type=str, default="calm")
    parser.add_argument("--context_sensor", type=str, default="clean")

    parser.add_argument("--slot_weights", type=str, default=None)
    parser.add_argument("--max_tau_violation", type=float, default=10.0)
    parser.add_argument("--max_tau_near_miss", type=float, default=40.0)
    parser.add_argument("--verbose_thresholds", action="store_true")

    return parser.parse_args()


def parse_slot_weights(text: Optional[str]) -> Optional[List[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",")]


def make_context(args: argparse.Namespace):
    return (
        args.context_density,
        args.context_behavior,
        args.context_sensor,
    )


def unwrap_env(env: Any) -> Any:
    current = env
    visited = set()

    while hasattr(current, "env") and id(current) not in visited:
        visited.add(id(current))
        current = current.env

    return current


def _registry_keys() -> List[str]:
    try:
        return sorted([str(k) for k in gym.registry.keys()])
    except Exception:
        return []


def _candidate_env_ids() -> List[str]:
    keys = _registry_keys()
    interesting = []
    for k in keys:
        lk = k.lower()
        if "merge" in lk or "highway" in lk or "roundabout" in lk or "intersection" in lk:
            interesting.append(k)
    return interesting


def ensure_highway_env_registered() -> None:
    import_errors = []

    try:
        import highway_env  # noqa: F401
    except Exception as e:
        import_errors.append(f"import highway_env failed: {repr(e)}")
    else:
        try:
            import highway_env  # type: ignore
            register_fn = getattr(highway_env, "register_highway_envs", None)
            if callable(register_fn):
                register_fn()
        except Exception as e:
            import_errors.append(f"highway_env.register_highway_envs() failed: {repr(e)}")

    try:
        import highway_env.envs  # noqa: F401
    except Exception as e:
        import_errors.append(f"import highway_env.envs failed: {repr(e)}")

    candidates = _candidate_env_ids()
    if not candidates:
        detail = "\n".join(import_errors) if import_errors else "No additional import diagnostics."
        raise RuntimeError(
            "Could not register highway-env environments in this Python environment.\n"
            "This usually means highway-env is missing or installed with a Gym/Gymnasium mismatch.\n\n"
            f"Import diagnostics:\n{detail}\n"
        )


def build_env(env_id: str, run_dir: str, seed: int):
    ensure_highway_env_registered()

    try:
        env = gym.make(env_id)
    except NameNotFound as e:
        candidates = _candidate_env_ids()
        hint = ", ".join(candidates[:20]) if candidates else "none found"
        raise NameNotFound(
            f"Environment '{env_id}' is not registered in Gymnasium.\n"
            f"Available highway-like env ids: {hint}\n"
            "If you expected 'merge-v0', your highway-env installation is likely incompatible "
            "with the current Gymnasium version or not installed in this venv."
        ) from e

    env.reset(seed=seed)
    env = Monitor(env, filename=str(Path(run_dir) / "train_monitor.csv"))
    return env


def attach_thresholds_to_env(
    env: Any,
    context: Any,
    threshold_patch: Dict[Any, Any],
    slot_weights=None,
    max_tau_violation: float = 10.0,
    max_tau_near_miss: float = 40.0,
    verbose: bool = False,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> Dict[str, float]:
    similarity_cfg = SimilarityConfig(
        slot_weights=slot_weights,
        exact_match_bonus=1e-6,
    )

    inference_cfg = make_safe_inference_config(
        max_tau_violation=max_tau_violation,
        max_tau_near_miss=max_tau_near_miss,
        verbose=verbose,
    )

    thresholds = get_context_thresholds(
        context=context,
        threshold_patch=threshold_patch,
        similarity_cfg=similarity_cfg,
        inference_cfg=inference_cfg,
        online_stats=online_stats,
    )

    if thresholds["tau_near_miss"] < thresholds["tau_violation"]:
        thresholds["tau_near_miss"] = thresholds["tau_violation"]

    target = unwrap_env(env)

    if hasattr(target, "set_context_thresholds"):
        target.set_context_thresholds(
            tau_violation=thresholds["tau_violation"],
            tau_near_miss=thresholds["tau_near_miss"],
        )
    elif hasattr(target, "set_thresholds"):
        target.set_thresholds(
            tau_violation=thresholds["tau_violation"],
            tau_near_miss=thresholds["tau_near_miss"],
        )
    else:
        setattr(target, "tau_violation", thresholds["tau_violation"])
        setattr(target, "tau_near_miss", thresholds["tau_near_miss"])

    setattr(target, "resolved_context_key", canonicalize_context(context))
    setattr(target, "resolved_tau_violation", thresholds["tau_violation"])
    setattr(target, "resolved_tau_near_miss", thresholds["tau_near_miss"])

    return thresholds


def save_run_metadata(run_dir: str, data: Dict[str, Any]) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    context = make_context(args)
    threshold_patch = load_threshold_patch(args.threshold_patch)
    online_stats = OnlineThresholdStats()

    print("\n=== TRAIN DISCRETE DIAGNOSTICS ===")
    print(f"env: {args.env}")
    print(f"algo: {args.algo}")
    print(f"seed: {args.seed}")
    print(f"total_steps: {args.total_steps}")
    print(f"run_dir: {args.run_dir}")
    print(f"context: {context}")
    print(f"threshold_patch: {args.threshold_patch}")
    print("=================================\n")

    env = build_env(args.env, str(run_dir), args.seed)

    thresholds = attach_thresholds_to_env(
        env=env,
        context=context,
        threshold_patch=threshold_patch,
        slot_weights=parse_slot_weights(args.slot_weights),
        max_tau_violation=args.max_tau_violation,
        max_tau_near_miss=args.max_tau_near_miss,
        verbose=args.verbose_thresholds,
        online_stats=online_stats,
    )

    print(
        f"[train_discrete.py] Resolved thresholds: "
        f"tau_violation={thresholds['tau_violation']:.3f}, "
        f"tau_near_miss={thresholds['tau_near_miss']:.3f}"
    )

    save_run_metadata(
        str(run_dir),
        {
            "env": args.env,
            "algo": args.algo,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "context": list(context),
            "threshold_patch": args.threshold_patch,
            "resolved_thresholds": thresholds,
        },
    )

    if args.algo == "dqn":
        model = DQN(
            policy="MlpPolicy",
            env=env,
            seed=args.seed,
            verbose=1,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            seed=args.seed,
            verbose=1,
        )

    model.learn(total_timesteps=args.total_steps)
    model.save(str(run_dir / "model"))
    print(f"[train_discrete.py] Saved model to {run_dir / 'model'}")


if __name__ == "__main__":
    main()
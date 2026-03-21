# scripts/train_continuous.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.error import NameNotFound
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src.context import (
    OnlineThresholdStats,
    SimilarityConfig,
    canonicalize_context,
    get_context_thresholds,
    load_threshold_patch,
    make_safe_inference_config,
)
from src.safety_wrapper import SafetyMethodWrapper, SafetyWrapperConfig


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="merge-v0")
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

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--disable_longitudinal", action="store_true")
    parser.add_argument("--disable_lateral", action="store_true")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "context", "adjust_speed", "full"],
    )

    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=200,
        help="Force episode termination so Monitor logs episodes.",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

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


def iter_env_chain(env: Any):
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        yield current
        current = getattr(current, "env", None)


def unwrap_env(env: Any) -> Any:
    last = env
    for current in iter_env_chain(env):
        last = current
    return last


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


# ============================================================
# highway-env registration and compatibility
# ============================================================

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


def _sanitize_merge_reward_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return 1
    if isinstance(action, (list, tuple)):
        return 1
    return action


def patch_merge_env_for_continuous() -> None:
    try:
        from highway_env.envs.merge_env import MergeEnv
    except Exception as e:
        raise RuntimeError(
            "Failed to import highway_env.envs.merge_env.MergeEnv while applying "
            "continuous-action compatibility patch."
        ) from e

    if getattr(MergeEnv, "_lilac_continuous_patch_applied", False):
        return

    original_rewards = MergeEnv._rewards

    def patched_rewards(self, action):
        safe_action = _sanitize_merge_reward_action(action)
        return original_rewards(self, safe_action)

    MergeEnv._rewards = patched_rewards
    MergeEnv._lilac_continuous_patch_applied = True


# ============================================================
# Method config
# ============================================================

def get_method_config(method: str) -> Dict[str, Any]:
    if method == "baseline":
        return {
            "use_context_constraints": False,
            "use_adjust_speed": False,
            "use_soft_to_hard": False,
            "apply_thresholds": False,
        }

    if method == "context":
        return {
            "use_context_constraints": True,
            "use_adjust_speed": False,
            "use_soft_to_hard": False,
            "apply_thresholds": True,
        }

    if method == "adjust_speed":
        return {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": False,
            "apply_thresholds": True,
        }

    if method == "full":
        return {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": True,
            "apply_thresholds": True,
        }

    raise ValueError(f"Unknown method: {method}")


def make_safety_wrapper_config(method_cfg: Dict[str, Any]) -> SafetyWrapperConfig:
    return SafetyWrapperConfig(
        use_context_constraints=bool(method_cfg["use_context_constraints"]),
        use_adjust_speed=bool(method_cfg["use_adjust_speed"]),
        use_soft_to_hard=bool(method_cfg["use_soft_to_hard"]),
        max_action_delta=0.15,
        w_lateral=6.0,
        w_long_pos=3.0,
        w_long_neg=2.0,
        lambda_near_miss=0.10,
        lambda_violation=0.25,
        safe_longitudinal=-0.25,
        safe_lateral=0.0,
    )


# ============================================================
# Environment build
# ============================================================

def build_env(
    env_id: str,
    run_dir: str,
    seed: int,
    method: str,
    method_cfg: Dict[str, Any],
    disable_longitudinal: bool = False,
    disable_lateral: bool = False,
    max_episode_steps: int = 200,
):
    ensure_highway_env_registered()

    if env_id == "merge-v0":
        patch_merge_env_for_continuous()

    env_config = {
        "action": {
            "type": "ContinuousAction",
            "longitudinal": not disable_longitudinal,
            "lateral": not disable_lateral,
        }
    }

    try:
        env = gym.make(env_id, config=env_config)
    except TypeError:
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

        target = unwrap_env(env)
        if hasattr(target, "configure"):
            target.configure(env_config)
            if hasattr(target, "define_spaces"):
                target.define_spaces()
        else:
            raise RuntimeError(
                "This highway-env version does not accept config= in gym.make(...) "
                "and the unwrapped env has no configure() method."
            )
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

    if not isinstance(env.action_space, Box):
        raise RuntimeError(
            f"Expected a continuous Box action space for SAC, but got {env.action_space!r}. "
            "The environment was not successfully converted to ContinuousAction."
        )

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = SafetyMethodWrapper(
        env=env,
        method=method,
        config=make_safety_wrapper_config(method_cfg),
    )

    env = Monitor(
        env,
        filename=str(Path(run_dir) / "train_monitor.csv"),
        info_keywords=(
            "violation_count",
            "near_miss_count",
            "shield_count",
            "action_correction_mean",
            "reward_penalty_sum",
        ),
    )
    return env


# ============================================================
# Threshold logic
# ============================================================

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

    for layer in iter_env_chain(env):
        if hasattr(layer, "set_context_thresholds"):
            layer.set_context_thresholds(
                tau_violation=thresholds["tau_violation"],
                tau_near_miss=thresholds["tau_near_miss"],
            )
        elif hasattr(layer, "set_thresholds"):
            layer.set_thresholds(
                tau_violation=thresholds["tau_violation"],
                tau_near_miss=thresholds["tau_near_miss"],
            )

        setattr(layer, "resolved_context_key", canonicalize_context(context))
        setattr(layer, "resolved_tau_violation", thresholds["tau_violation"])
        setattr(layer, "resolved_tau_near_miss", thresholds["tau_near_miss"])

    return thresholds


# ============================================================
# Metadata
# ============================================================

def save_run_metadata(run_dir: str, data: Dict[str, Any]) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_run_debug(run_dir: str, data: Dict[str, Any]) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "run_debug.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def configure_calibration_logging(env: Any, run_dir: str, context: Any, seed: int) -> None:
    canonical_context = canonicalize_context(context)
    calibration_path = str(Path(run_dir) / "calibration_monitor.csv")
    for layer in iter_env_chain(env):
        if hasattr(layer, "set_context_metadata"):
            layer.set_context_metadata(canonical_context)
        if hasattr(layer, "set_calibration_log_path"):
            layer.set_calibration_log_path(calibration_path)
        if hasattr(layer, "set_run_seed"):
            layer.set_run_seed(seed)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    context = make_context(args)
    threshold_patch = load_threshold_patch(args.threshold_patch)
    online_stats = OnlineThresholdStats()
    method_cfg = get_method_config(args.method)

    print("\n=== TRAIN CONTINUOUS DIAGNOSTICS ===")
    print(f"env: {args.env}")
    print(f"method: {args.method}")
    print(f"seed: {args.seed}")
    print(f"total_steps: {args.total_steps}")
    print(f"max_episode_steps: {args.max_episode_steps}")
    print(f"run_dir: {args.run_dir}")
    print(f"context: {context}")
    print(f"threshold_patch: {args.threshold_patch}")
    print(f"method_cfg: {method_cfg}")
    print("===================================\n")

    env = build_env(
        env_id=args.env,
        run_dir=str(run_dir),
        seed=args.seed,
        method=args.method,
        method_cfg=method_cfg,
        disable_longitudinal=args.disable_longitudinal,
        disable_lateral=args.disable_lateral,
        max_episode_steps=args.max_episode_steps,
    )

    thresholds: Optional[Dict[str, float]] = None
    if method_cfg["apply_thresholds"]:
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
            f"[train_continuous.py] Resolved thresholds: "
            f"tau_violation={thresholds['tau_violation']:.3f}, "
            f"tau_near_miss={thresholds['tau_near_miss']:.3f}"
        )
    else:
        print("[train_continuous.py] Baseline mode: threshold logic disabled.")

    print(f"[train_continuous.py] Action space: {env.action_space}")

    configure_calibration_logging(
        env=env,
        run_dir=str(run_dir),
        context=context,
        seed=args.seed,
    )

    debug_payload = {
        "env": args.env,
        "method": args.method,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "max_episode_steps": args.max_episode_steps,
        "raw_context": list(context),
        "canonical_context": list(canonicalize_context(context)),
        "threshold_patch": args.threshold_patch,
        "resolved_thresholds": None if thresholds is None else {
            "context": list(canonicalize_context(context)),
            "tau_violation": float(thresholds["tau_violation"]),
            "tau_near_miss": float(thresholds["tau_near_miss"]),
        },
        "apply_thresholds": bool(method_cfg["apply_thresholds"]),
        "calibration_monitor": str(Path(run_dir) / "calibration_monitor.csv"),
    }
    save_run_debug(str(run_dir), debug_payload)

    save_run_metadata(
        str(run_dir),
        {
            "env": args.env,
            "method": args.method,
            "method_cfg": method_cfg,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "max_episode_steps": args.max_episode_steps,
            "context": list(context),
            "canonical_context": list(canonicalize_context(context)),
            "threshold_patch": args.threshold_patch,
            "resolved_thresholds": thresholds,
            "algo": "SAC",
            "action_space": str(env.action_space),
            "continuous_action_config": {
                "type": "ContinuousAction",
                "longitudinal": not args.disable_longitudinal,
                "lateral": not args.disable_lateral,
            },
            "merge_continuous_patch": bool(args.env == "merge-v0"),
            "calibration_monitor": str(Path(run_dir) / "calibration_monitor.csv"),
        },
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_steps)
    model.save(str(run_dir / "model"))
    print(f"[train_continuous.py] Saved model to {run_dir / 'model'}")


if __name__ == "__main__":
    main()
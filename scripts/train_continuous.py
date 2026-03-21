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
    parser.add_argument("--total_steps", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_dir", type=str, required=True)

    parser.add_argument("--threshold_patch", type=str, default=None)

    parser.add_argument(
        "--regime",
        type=str,
        default="stationary",
        choices=["stationary", "nonstationary_seen", "nonstationary_unseen"],
    )
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
        choices=[
            "unconstrained",
            "fixed_full_A",
            "fixed_full_C",
            "cb",
            "as",
            "sh",
            "cb+as",
            "cb+sh",
            "as+sh",
            "cb+as+sh",
        ],
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


def make_context(args: argparse.Namespace) -> tuple[str, str, str]:
    return (args.context_density, args.context_behavior, args.context_sensor)


def build_context_schedule(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    base = canonicalize_context(make_context(args))
    if args.regime == "stationary":
        return [base]

    if args.regime == "nonstationary_seen":
        return [
            ("low", "calm", "clean"),
            ("medium", "mixed", "foggy"),
            ("high", "aggr", "noisy"),
        ]

    if args.regime == "nonstationary_unseen":
        return [
            ("low", "calm", "clean"),
            ("medium", "aggr", "dropout"),
            ("high", "mixed", "dropout"),
        ]

    raise ValueError(f"Unknown regime: {args.regime}")


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
            "Failed to import highway_env.envs.merge_env.MergeEnv while applying continuous-action compatibility patch."
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
    table = {
        "unconstrained": {
            "use_context_constraints": False,
            "use_adjust_speed": False,
            "use_soft_to_hard": False,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": False,
        },
        "fixed_full_A": {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": True,
            "use_fixed_constraints": True,
            "fixed_strategy": "A",
            "apply_thresholds": True,
        },
        "fixed_full_C": {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": True,
            "use_fixed_constraints": True,
            "fixed_strategy": "C",
            "apply_thresholds": True,
        },
        "cb": {
            "use_context_constraints": True,
            "use_adjust_speed": False,
            "use_soft_to_hard": False,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "as": {
            "use_context_constraints": False,
            "use_adjust_speed": True,
            "use_soft_to_hard": False,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "sh": {
            "use_context_constraints": False,
            "use_adjust_speed": False,
            "use_soft_to_hard": True,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "cb+as": {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": False,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "cb+sh": {
            "use_context_constraints": True,
            "use_adjust_speed": False,
            "use_soft_to_hard": True,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "as+sh": {
            "use_context_constraints": False,
            "use_adjust_speed": True,
            "use_soft_to_hard": True,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
        "cb+as+sh": {
            "use_context_constraints": True,
            "use_adjust_speed": True,
            "use_soft_to_hard": True,
            "use_fixed_constraints": False,
            "fixed_strategy": None,
            "apply_thresholds": True,
        },
    }
    if method not in table:
        raise ValueError(f"Unknown method: {method}")
    return table[method]


def make_safety_wrapper_config(method_cfg: Dict[str, Any]) -> SafetyWrapperConfig:
    return SafetyWrapperConfig(
        use_context_constraints=bool(method_cfg["use_context_constraints"]),
        use_adjust_speed=bool(method_cfg["use_adjust_speed"]),
        use_soft_to_hard=bool(method_cfg["use_soft_to_hard"]),
        use_fixed_constraints=bool(method_cfg["use_fixed_constraints"]),
        fixed_strategy=method_cfg["fixed_strategy"],
        max_action_delta=0.15,
        w_lateral=6.0,
        w_long_pos=3.0,
        w_long_neg=2.0,
        lambda_near_miss=0.10,
        lambda_violation=0.25,
        safe_longitudinal=-0.25,
        safe_lateral=0.0,
        fixed_tau_violation=2.0,
        fixed_tau_near_miss=8.0,
        fixed_max_action_delta=0.10,
        fixed_alpha=0.10,
        initial_budget=1.0,
        budget_decay_on_violation=0.05,
        risk_tightening_scale=0.25,
        alpha_min=0.01,
        alpha_max=0.20,
        proactive_horizon=1,
        proactive_trigger_ratio=0.85,
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
    context: tuple[str, str, str],
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

    env.set_context_metadata(context)
    env.set_run_seed(seed)
    env.set_calibration_log_path(str(Path(run_dir) / "calibration_monitor.csv"))

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

def resolve_thresholds_for_context(
    context: Any,
    threshold_patch: Dict[Any, Any],
    slot_weights=None,
    max_tau_violation: float = 10.0,
    max_tau_near_miss: float = 40.0,
    verbose: bool = False,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> Dict[str, float]:
    similarity_cfg = SimilarityConfig(slot_weights=slot_weights, exact_match_bonus=1e-6)
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
    tau_v = float(thresholds["tau_violation"])
    tau_n = float(max(thresholds["tau_near_miss"], tau_v))
    return {
        "context": list(canonicalize_context(context)),
        "tau_violation": tau_v,
        "tau_near_miss": tau_n,
    }


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
    thresholds = resolve_thresholds_for_context(
        context=context,
        threshold_patch=threshold_patch,
        slot_weights=slot_weights,
        max_tau_violation=max_tau_violation,
        max_tau_near_miss=max_tau_near_miss,
        verbose=verbose,
        online_stats=online_stats,
    )

    resolved_context = canonicalize_context(context)

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

        setattr(layer, "resolved_context_key", resolved_context)
        setattr(layer, "resolved_tau_violation", thresholds["tau_violation"])
        setattr(layer, "resolved_tau_near_miss", thresholds["tau_near_miss"])

    return thresholds


def attach_context_schedule_to_env(
    env: Any,
    context_schedule: list[tuple[str, str, str]],
    threshold_patch: Dict[Any, Any],
    slot_weights=None,
    max_tau_violation: float = 10.0,
    max_tau_near_miss: float = 40.0,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    unique_contexts = []
    seen = set()
    for ctx in context_schedule:
        cctx = canonicalize_context(ctx)
        if cctx not in seen:
            seen.add(cctx)
            unique_contexts.append(cctx)

    threshold_map: Dict[tuple[str, str, str], Dict[str, float]] = {}
    for ctx in unique_contexts:
        stats = OnlineThresholdStats()
        threshold_map[ctx] = resolve_thresholds_for_context(
            context=ctx,
            threshold_patch=threshold_patch,
            slot_weights=slot_weights,
            max_tau_violation=max_tau_violation,
            max_tau_near_miss=max_tau_near_miss,
            verbose=verbose,
            online_stats=stats,
        )

    for layer in iter_env_chain(env):
        if hasattr(layer, "set_context_schedule"):
            layer.set_context_schedule(context_schedule, threshold_map)

    return threshold_map


def compute_fixed_baseline_A(
    threshold_patch: Dict[Any, Any],
    default_tau_violation: float = 2.0,
    default_tau_near_miss: float = 8.0,
    default_max_action_delta: float = 0.10,
    default_alpha: float = 0.10,
) -> Dict[str, float]:
    if not threshold_patch:
        return {
            "fixed_tau_violation": float(default_tau_violation),
            "fixed_tau_near_miss": float(default_tau_near_miss),
            "fixed_max_action_delta": float(default_max_action_delta),
            "fixed_alpha": float(default_alpha),
        }

    tau_v = []
    tau_n = []
    for rec in threshold_patch.values():
        tv = float(getattr(rec, "tau_violation", rec["tau_violation"]))
        tn = float(getattr(rec, "tau_near_miss", rec["tau_near_miss"]))
        tau_v.append(tv)
        tau_n.append(max(tn, tv))

    mean_tau_v = float(np.mean(tau_v)) if tau_v else float(default_tau_violation)
    mean_tau_n = float(np.mean(tau_n)) if tau_n else float(default_tau_near_miss)
    mean_tau_n = max(mean_tau_n, mean_tau_v)

    return {
        "fixed_tau_violation": mean_tau_v,
        "fixed_tau_near_miss": mean_tau_n,
        "fixed_max_action_delta": float(default_max_action_delta),
        "fixed_alpha": float(default_alpha),
    }


def compute_fixed_baseline_C(
    threshold_patch: Dict[Any, Any],
    default_tau_violation: float = 2.0,
    default_tau_near_miss: float = 8.0,
    default_max_action_delta: float = 0.10,
    default_alpha: float = 0.10,
) -> Dict[str, float]:
    cfg = compute_fixed_baseline_A(
        threshold_patch=threshold_patch,
        default_tau_violation=default_tau_violation,
        default_tau_near_miss=default_tau_near_miss,
        default_max_action_delta=default_max_action_delta,
        default_alpha=default_alpha,
    )
    cfg["fixed_tau_violation"] = max(0.5, 0.90 * float(cfg["fixed_tau_violation"]))
    cfg["fixed_tau_near_miss"] = max(float(cfg["fixed_tau_violation"]), 0.95 * float(cfg["fixed_tau_near_miss"]))
    cfg["fixed_max_action_delta"] = max(0.03, 0.85 * float(cfg["fixed_max_action_delta"]))
    cfg["fixed_alpha"] = max(0.01, 0.85 * float(cfg["fixed_alpha"]))
    return cfg


def apply_fixed_config_to_env(env: Any, fixed_cfg: Dict[str, float]) -> None:
    for layer in iter_env_chain(env):
        if hasattr(layer, "config") and hasattr(layer, "set_context_thresholds"):
            cfg = getattr(layer, "config", None)
            if cfg is not None and hasattr(cfg, "fixed_tau_violation"):
                cfg.fixed_tau_violation = float(fixed_cfg["fixed_tau_violation"])
                cfg.fixed_tau_near_miss = float(fixed_cfg["fixed_tau_near_miss"])
                cfg.fixed_max_action_delta = float(fixed_cfg["fixed_max_action_delta"])
                cfg.fixed_alpha = float(fixed_cfg["fixed_alpha"])


# ============================================================
# Metadata
# ============================================================

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        safe: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, (str, int, float, bool)) or k is None:
                safe_key = str(k)
            else:
                safe_key = str(k)
            safe[safe_key] = _json_safe(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def save_run_metadata(run_dir: str, data: Dict[str, Any]) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)


def save_run_debug(run_dir: str, data: Dict[str, Any]) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    with (p / "run_debug.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    context_schedule = build_context_schedule(args)
    context = context_schedule[0]
    canonical_context = canonicalize_context(context)
    threshold_patch = load_threshold_patch(args.threshold_patch)
    method_cfg = get_method_config(args.method)

    print("\n=== TRAIN CONTINUOUS DIAGNOSTICS ===")
    print(f"env: {args.env}")
    print(f"method: {args.method}")
    print(f"regime: {args.regime}")
    print(f"seed: {args.seed}")
    print(f"total_steps: {args.total_steps}")
    print(f"max_episode_steps: {args.max_episode_steps}")
    print(f"run_dir: {args.run_dir}")
    print(f"initial_context: {context}")
    print(f"context_schedule: {context_schedule}")
    print(f"threshold_patch: {args.threshold_patch}")
    print(f"method_cfg: {method_cfg}")
    print("===================================\n")

    env = build_env(
        env_id=args.env,
        run_dir=str(run_dir),
        seed=args.seed,
        method=args.method,
        method_cfg=method_cfg,
        context=canonical_context,
        disable_longitudinal=args.disable_longitudinal,
        disable_lateral=args.disable_lateral,
        max_episode_steps=args.max_episode_steps,
    )

    thresholds: Optional[Dict[str, float]] = None
    threshold_map: Optional[Dict[str, Dict[str, float]]] = None
    fixed_cfg: Optional[Dict[str, float]] = None
    slot_weights = parse_slot_weights(args.slot_weights)

    if method_cfg["apply_thresholds"]:
        thresholds = attach_thresholds_to_env(
            env=env,
            context=context,
            threshold_patch=threshold_patch,
            slot_weights=slot_weights,
            max_tau_violation=args.max_tau_violation,
            max_tau_near_miss=args.max_tau_near_miss,
            verbose=args.verbose_thresholds,
            online_stats=OnlineThresholdStats(),
        )
        threshold_map = attach_context_schedule_to_env(
            env=env,
            context_schedule=context_schedule,
            threshold_patch=threshold_patch,
            slot_weights=slot_weights,
            max_tau_violation=args.max_tau_violation,
            max_tau_near_miss=args.max_tau_near_miss,
            verbose=args.verbose_thresholds,
        )
        print(
            f"[train_continuous.py] Initial thresholds: "
            f"tau_violation={thresholds['tau_violation']:.3f}, "
            f"tau_near_miss={thresholds['tau_near_miss']:.3f}"
        )
        print(f"[train_continuous.py] Scheduled contexts: {list(threshold_map.keys())}")
    else:
        print("[train_continuous.py] Unconstrained mode: threshold logic disabled.")

    if method_cfg["use_fixed_constraints"]:
        if method_cfg["fixed_strategy"] == "A":
            fixed_cfg = compute_fixed_baseline_A(threshold_patch=threshold_patch)
        elif method_cfg["fixed_strategy"] == "C":
            fixed_cfg = compute_fixed_baseline_C(threshold_patch=threshold_patch)
        else:
            raise ValueError(f"Unsupported fixed strategy: {method_cfg['fixed_strategy']}")
        apply_fixed_config_to_env(env, fixed_cfg)
        print(f"[train_continuous.py] Applied fixed config: {fixed_cfg}")

    print(f"[train_continuous.py] Action space: {env.action_space}")

    debug_payload = {
        "env": args.env,
        "method": args.method,
        "regime": args.regime,
        "method_cfg": method_cfg,
        "fixed_strategy": method_cfg.get("fixed_strategy"),
        "seed": args.seed,
        "total_steps": args.total_steps,
        "max_episode_steps": args.max_episode_steps,
        "initial_context": list(context),
        "context_schedule": [list(x) for x in context_schedule],
        "threshold_patch": args.threshold_patch,
        "resolved_thresholds": thresholds,
        "threshold_map": threshold_map,
        "fixed_config": fixed_cfg,
        "apply_thresholds": bool(method_cfg["apply_thresholds"]),
    }
    save_run_debug(str(run_dir), debug_payload)

    save_run_metadata(
        str(run_dir),
        {
            "env": args.env,
            "method": args.method,
            "regime": args.regime,
            "method_cfg": method_cfg,
            "fixed_strategy": method_cfg.get("fixed_strategy"),
            "fixed_config": fixed_cfg,
            "seed": args.seed,
            "total_steps": args.total_steps,
            "max_episode_steps": args.max_episode_steps,
            "initial_context": list(context),
            "context_schedule": [list(x) for x in context_schedule],
            "threshold_patch": args.threshold_patch,
            "resolved_thresholds": thresholds,
            "threshold_map": threshold_map,
            "algo": "SAC",
            "action_space": str(env.action_space),
            "continuous_action_config": {
                "type": "ContinuousAction",
                "longitudinal": not args.disable_longitudinal,
                "lateral": not args.disable_lateral,
            },
            "merge_continuous_patch": bool(args.env == "merge-v0"),
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

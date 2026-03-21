from __future__ import annotations

from typing import Tuple, Optional

import gymnasium as gym
import numpy as np

# IMPORTANT: import highway_env so that highway-v0 / merge-v0 are registered.
try:
    import highway_env  # noqa: F401
except Exception:
    highway_env = None

from src.context import MarkovContextScheduler
from src.safety import SafetyParams, ConformalCalibrator
from src.wrappers import (
    ContextNonstationaryWrapper,
    ObservationNoiseWrapper,
    SafetyShieldWrapper,
    SoftToHardBudgetWrapper,
    FixedKinematicsObsWrapper,
)


def _base_highway_config() -> dict:
    """
    Base config for highway-v0 tuned to produce more meaningful safety signal.
    """
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "absolute": False,
            "normalize": True,
            "order": "sorted",
        },
        "action": {
            "type": "ContinuousAction",
        },
        "lanes_count": 4,
        "vehicles_count": 50,
        "controlled_vehicles": 1,
        "duration": 60,
        "ego_spacing": 1.0,
        "vehicles_density": 1.5,
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": -0.05,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": True,
        "simulation_frequency": 15,
        "policy_frequency": 1,
    }


def _base_merge_config() -> dict:
    return {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "heading"],
            "absolute": False,
            "normalize": True,
            "order": "sorted",
        },
        "action": {
            "type": "ContinuousAction",
        },
        "collision_reward": -1.0,
        "right_lane_reward": 0.05,
        "high_speed_reward": 0.2,
        "merging_speed_reward": -0.5,
        "lane_change_reward": -0.05,
        "duration": 60,
        "simulation_frequency": 15,
        "policy_frequency": 1,
    }


def _make_base_env(env_id: str, seed: int):
    env_id = str(env_id).strip()

    if env_id == "highway-v0":
        env = gym.make("highway-v0", config=_base_highway_config())
    elif env_id == "merge-v0":
        env = gym.make("merge-v0", config=_base_merge_config())
    else:
        env = gym.make(env_id)

    try:
        env.reset(seed=int(seed))
    except TypeError:
        try:
            env.seed(int(seed))
        except Exception:
            pass

    try:
        env.action_space.seed(int(seed))
    except Exception:
        pass

    try:
        env.observation_space.seed(int(seed))
    except Exception:
        pass

    return env


def make_env(
    env_id: str,
    seed: int,
    action_space_type: str,
    p_stay: float,
    no_mpc: bool,
    no_conformal: bool,
    safety_params: SafetyParams,
    *,
    budget_C: float = 0.0,
    budget_T: int = 60,
    budget_delta: float = 0.0,
    no_budget: bool = False,
    baseline_rl: bool = False,
) -> Tuple[gym.Env, Optional[ConformalCalibrator], MarkovContextScheduler]:
    """
    Create the wrapped environment.

    Standard path:
      base env
      -> ContextNonstationaryWrapper
      -> ObservationNoiseWrapper
      -> SafetyShieldWrapper
      -> optional SoftToHardBudgetWrapper
      -> FixedKinematicsObsWrapper

    True baseline path (Option A):
      base env
      -> ContextNonstationaryWrapper
      -> ObservationNoiseWrapper
      -> FixedKinematicsObsWrapper

    Returns:
      env, calibrator, scheduler
    """
    if action_space_type not in {"discrete", "continuous"}:
        raise ValueError(
            f"action_space_type must be 'discrete' or 'continuous', got {action_space_type}"
        )

    base_env = _make_base_env(env_id, seed)

    scheduler = MarkovContextScheduler(p_stay=float(p_stay), seed=int(seed))
    env = ContextNonstationaryWrapper(base_env, scheduler)
    env = ObservationNoiseWrapper(env, seed=int(seed))

    calibrator: Optional[ConformalCalibrator] = None

    if not baseline_rl:
        if not bool(no_conformal):
            calibrator = ConformalCalibrator(alpha=0.1, window=200, seed=int(seed))

        env = SafetyShieldWrapper(
            env,
            params=safety_params,
            action_space_type=action_space_type,
            no_mpc=bool(no_mpc),
            no_conformal=bool(no_conformal),
            calibrator=calibrator,
        )

        if (not bool(no_budget)) and float(budget_C) > 0.0:
            env = SoftToHardBudgetWrapper(
                env,
                C=float(budget_C),
                T=int(budget_T),
                risk_delta=float(budget_delta),
                enabled=True,
            )

    # Keep observation shape fixed for SB3 compatibility.
    try:
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Box):
            if len(obs_space.shape) == 2:
                k = int(obs_space.shape[0])
                env = FixedKinematicsObsWrapper(env, K=k)
            elif len(obs_space.shape) == 1:
                # FixedKinematicsObsWrapper also supports 1D shapes in the updated version.
                env = FixedKinematicsObsWrapper(env)
    except Exception:
        pass

    return env, calibrator, scheduler
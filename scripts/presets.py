# scripts/presets.py
from __future__ import annotations
from typing import Dict, Any

"""
Experiment presets for highway-env tasks.

Design choice:
- merge-v0 → discrete algorithms (DQN / PPO)
- highway-v0 → continuous algorithms (SAC)

NOTE:
merge-v0 + continuous actions is intentionally NOT supported because
merge_env internally assumes discrete actions in its reward function.
"""

PRESETS: Dict[str, Dict[str, Any]] = {

    # ---------------------------------------------------
    # highway-v0 (DISCRETE) – general driving benchmark
    # ---------------------------------------------------
    "highway_discrete_default": {
        "env_id": "highway-v0",
        "action_space_type": "discrete",
        "sb3": {
            "dqn": {
                "learning_rate": 3e-4,
                "buffer_size": 100_000,
                "learning_starts": 10_000,
                "batch_size": 64,
                "gamma": 0.99,
                "train_freq": 4,
                "target_update_interval": 1_000,
            },
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
            },
        },
        "safety": {
            "horizon_n": 10,
            "epsilon": 0.5,
            "delta_nearmiss": 1.0,
            "d0": 2.0,
            "h": 1.2,
        },
        "nonstationarity": {
            "p_stay": 0.8,
        },
    },

    # ---------------------------------------------------
    # merge-v0 (DISCRETE) – merging traffic scenario
    # ---------------------------------------------------
    "merge_discrete_default": {
        "env_id": "merge-v0",
        "action_space_type": "discrete",
        "sb3": {
            "dqn": {
                "learning_rate": 2e-4,
                "buffer_size": 150_000,
                "learning_starts": 15_000,
                "batch_size": 64,
                "gamma": 0.99,
                "train_freq": 4,
                "target_update_interval": 1_500,
            },
            "ppo": {
                "learning_rate": 2.5e-4,
                "n_steps": 4096,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
            },
        },
        "safety": {
            "horizon_n": 10,
            "epsilon": 0.7,
            "delta_nearmiss": 1.2,
            "d0": 2.2,
            "h": 1.3,
        },
        "nonstationarity": {
            "p_stay": 0.85,
        },
    },

    # ---------------------------------------------------
    # highway-v0 (CONTINUOUS) – SAC baseline
    # ---------------------------------------------------
    "highway_continuous_default": {
        "env_id": "highway-v0",
        "action_space_type": "continuous",
        "sb3": {
            "sac": {
                "learning_rate": 3e-4,
                "buffer_size": 200_000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "train_freq": 1,
                "gradient_steps": 1,
            },
        },
        "safety": {
            "horizon_n": 10,
            "epsilon": 0.5,
            "delta_nearmiss": 1.0,
            "d0": 2.0,
            "h": 1.2,
        },
        "nonstationarity": {
            "p_stay": 0.8,
        },
    },
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Retrieve preset configuration.

    Raises helpful error if preset name is wrong.
    """
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. "
            f"Available presets: {list(PRESETS.keys())}"
        )

    return PRESETS[name].copy()
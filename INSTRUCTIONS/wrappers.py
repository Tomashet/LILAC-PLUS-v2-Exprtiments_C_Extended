# src/wrappers.py
from __future__ import annotations

import numpy as np
import gymnasium as gym


class FixedKinematicsObsWrapper(gym.ObservationWrapper):
    """
    Forces a fixed (K, F) observation shape by truncating/padding along axis 0.

    Works with highway-env Kinematics obs where obs is typically (N, F).
    Padding rows are zeros.
    """
    def __init__(self, env: gym.Env, K: int):
        super().__init__(env)
        self.K = int(K)

        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 2:
            raise TypeError(f"Expected Box( (N,F) ) obs space, got {space}")

        _, F = space.shape

        # Build a new Box with fixed (K, F). Use the underlying bounds if possible.
        low = np.full((self.K, F), float(space.low.min()), dtype=space.dtype)
        high = np.full((self.K, F), float(space.high.max()), dtype=space.dtype)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(self.K, F), dtype=space.dtype
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs)
        if obs.ndim != 2:
            return obs

        N, F = obs.shape
        out = np.zeros((self.K, F), dtype=obs.dtype)

        ncopy = min(N, self.K)
        out[:ncopy] = obs[:ncopy]
        return out
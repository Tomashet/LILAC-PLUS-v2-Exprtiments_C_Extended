from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym


class LilacObsAugmentWrapper(gym.Wrapper):
    """
    Appends a learned latent z to the observation:
      obs_aug = concat(obs_fixed_flat, z)

    Also buffers a per-episode context window of transitions for encoder training.
    The buffer stores tuples (s, a, r, s') as float32 with FIXED dimensions.

    This fixes shape instability when the wrapped env occasionally returns
    flattened observations of different lengths across episodes.
    """

    def __init__(self, env: gym.Env, latent_dim: int = 8, context_len: int = 32):
        super().__init__(env)
        self.latent_dim = int(latent_dim)
        self.context_len = int(context_len)

        self._z = np.zeros((self.latent_dim,), dtype=np.float32)

        # Episode transition ring buffer
        self._ctx: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None

        # FIXED base dimensions inferred from declared spaces, not from live samples.
        self._obs_dim = int(np.prod(self.env.observation_space.shape))
        self._act_dim = int(np.prod(self.env.action_space.shape))

        # Expose updated observation space
        low = -np.inf * np.ones((self._obs_dim + self.latent_dim,), dtype=np.float32)
        high = np.inf * np.ones((self._obs_dim + self.latent_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _fix_flat_dim(self, x: Any, target_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        out = np.zeros((target_dim,), dtype=np.float32)
        n = min(arr.shape[0], target_dim)
        out[:n] = arr[:n]
        return out

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        return self._fix_flat_dim(obs, self._obs_dim)

    def _flatten_action(self, action: Any) -> np.ndarray:
        return self._fix_flat_dim(action, self._act_dim)

    # ---- LILAC control surface (called via VecEnv.env_method) ----
    def set_z(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if z.shape[0] != self.latent_dim:
            raise ValueError(f"z has shape {z.shape}, expected ({self.latent_dim},)")
        self._z = z

    def get_z(self) -> np.ndarray:
        return self._z.copy()

    def pop_episode_context(self) -> np.ndarray:
        """
        Returns a fixed-shape context array:
          ctx: [K, (s,a,r,s')]
        If the episode is shorter than K, left-pad with zeros.
        """
        per_step = self._obs_dim + self._act_dim + 1 + self._obs_dim
        items = self._ctx[-self.context_len:]

        if len(items) == 0:
            return np.zeros((self.context_len, per_step), dtype=np.float32)

        ctx = np.zeros((self.context_len, per_step), dtype=np.float32)
        start = self.context_len - len(items)
        for i, (s, a, r, sp) in enumerate(items):
            row = np.concatenate(
                [s, a, np.array([r], dtype=np.float32), sp],
                axis=0,
            ).astype(np.float32, copy=False)
            ctx[start + i] = row

        # Clear for next episode
        self._ctx = []
        self._last_obs = None
        self._last_action = None
        return ctx

    def set_adjustment_risk(self, *args, **kwargs) -> None:
        """
        Proxy to inner wrappers (SafetyShieldWrapper) so callbacks can call env_method
        on the top wrapper.
        """
        if hasattr(self.env, "set_adjustment_risk"):
            return self.env.set_adjustment_risk(*args, **kwargs)
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_f = self._flatten_obs(obs)
        self._last_obs = obs_f
        self._last_action = None

        obs_aug = np.concatenate([obs_f, self._z], axis=0).astype(np.float32)

        info = dict(info) if info is not None else {}
        info["lilac_z_norm"] = float(np.linalg.norm(self._z))
        return obs_aug, info

    def step(self, action):
        obs_next, reward, terminated, truncated, info = self.env.step(action)

        obs_next_f = self._flatten_obs(obs_next)
        act_f = self._flatten_action(action)

        # Buffer transition with FIXED dimensions
        if self._last_obs is not None:
            s = self._last_obs
            a = act_f
            sp = obs_next_f
            self._ctx.append((s, a, float(reward), sp))
            if len(self._ctx) > (self.context_len * 4):
                self._ctx = self._ctx[-(self.context_len * 4):]
            self._last_obs = sp
        else:
            self._last_obs = obs_next_f

        obs_aug = np.concatenate([obs_next_f, self._z], axis=0).astype(np.float32)

        info = dict(info) if info is not None else {}
        info["lilac_z_norm"] = float(np.linalg.norm(self._z))
        return obs_aug, reward, terminated, truncated, info
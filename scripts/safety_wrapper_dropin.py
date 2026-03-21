from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np


def _context_to_id(ctx_tuple: tuple[str, ...]) -> str:
    return "|".join(str(x) for x in ctx_tuple)


@dataclass
class SafetyWrapperConfig:
    use_context_constraints: bool = False
    use_adjust_speed: bool = False
    use_soft_to_hard: bool = False

    max_action_delta: float = 0.15
    w_lateral: float = 6.0
    w_long_pos: float = 3.0
    w_long_neg: float = 2.0
    lambda_near_miss: float = 0.10
    lambda_violation: float = 0.25
    safe_longitudinal: float = -0.25
    safe_lateral: float = 0.0


class SafetyMethodWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, method: str, config: SafetyWrapperConfig) -> None:
        super().__init__(env)
        self.method = method
        self.config = config

        self.tau_violation: float = 2.0
        self.tau_near_miss: float = 8.0

        self.context_tuple: tuple[str, ...] = tuple()
        self.ctx_id: str = ""
        self.calibration_log_path: Optional[str] = None
        self.run_seed: Optional[int] = None

        self._prev_action: Optional[np.ndarray] = None
        self._episode_index = 0
        self._step_index = 0
        self._reset_episode_stats()

    def set_context_thresholds(self, tau_violation: float, tau_near_miss: float) -> None:
        self.tau_violation = float(tau_violation)
        self.tau_near_miss = float(max(tau_near_miss, tau_violation))

    def set_thresholds(self, tau_violation: float, tau_near_miss: float) -> None:
        self.set_context_thresholds(tau_violation=tau_violation, tau_near_miss=tau_near_miss)

    def set_context_metadata(self, context: Any) -> None:
        if isinstance(context, tuple):
            ctx = tuple(str(x) for x in context)
        elif isinstance(context, list):
            ctx = tuple(str(x) for x in context)
        else:
            ctx = (str(context),)
        self.context_tuple = ctx
        self.ctx_id = _context_to_id(ctx)

    def set_calibration_log_path(self, path: str) -> None:
        self.calibration_log_path = str(path)
        p = Path(self.calibration_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode_idx",
                        "step_idx",
                        "method",
                        "seed",
                        "ctx_id",
                        "ctx_tuple",
                        "proxy_cost",
                        "clearance",
                        "tau_violation",
                        "tau_near_miss",
                        "violation",
                        "near_miss",
                        "shielded",
                        "action_correction",
                        "reward_penalty",
                    ],
                )
                writer.writeheader()

    def set_run_seed(self, seed: int) -> None:
        self.run_seed = int(seed)

    def _reset_episode_stats(self) -> None:
        self.ep_violation_count = 0
        self.ep_near_miss_count = 0
        self.ep_shield_count = 0
        self.ep_reward_penalty_sum = 0.0
        self.ep_action_correction_sum = 0.0
        self.ep_step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._episode_index += 1
        self._step_index = 0
        self._reset_episode_stats()
        return obs, info

    def _ensure_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).copy()
        if arr.ndim == 0:
            arr = np.array([float(arr)], dtype=np.float32)
        if arr.shape[0] == 1:
            arr = np.array([arr[0], 0.0], dtype=np.float32)
        return arr[:2]

    def _proxy_cost(self, action: np.ndarray) -> float:
        a_lon = float(action[0])
        a_lat = float(action[1])
        cost = (
            self.config.w_lateral * abs(a_lat)
            + self.config.w_long_pos * max(a_lon, 0.0)
            + self.config.w_long_neg * max(-a_lon, 0.0)
        )
        return float(cost)

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        if self._prev_action is None:
            return action.copy()
        delta = action - self._prev_action
        delta = np.clip(delta, -self.config.max_action_delta, self.config.max_action_delta)
        return (self._prev_action + delta).astype(np.float32)

    def _safe_action(self, action: np.ndarray) -> np.ndarray:
        safe = action.copy()
        safe[0] = min(float(safe[0]), self.config.safe_longitudinal)
        safe[1] = self.config.safe_lateral
        return safe.astype(np.float32)

    def _compute_penalty(self, cost: float) -> float:
        penalty = 0.0
        if cost > self.tau_near_miss:
            penalty += self.config.lambda_near_miss * (cost - self.tau_near_miss)
        if cost > self.tau_violation:
            penalty += self.config.lambda_violation * (cost - self.tau_violation)
        return float(max(0.0, penalty))

    def _append_calibration_row(
        self,
        proxy_cost: float,
        violation: bool,
        near_miss: bool,
        shielded: bool,
        action_correction: float,
        reward_penalty: float,
    ) -> None:
        if not self.calibration_log_path:
            return
        row = {
            "episode_idx": int(self._episode_index),
            "step_idx": int(self._step_index),
            "method": self.method,
            "seed": "" if self.run_seed is None else int(self.run_seed),
            "ctx_id": self.ctx_id,
            "ctx_tuple": str(self.context_tuple),
            "proxy_cost": float(proxy_cost),
            "clearance": float(proxy_cost),
            "tau_violation": float(self.tau_violation),
            "tau_near_miss": float(self.tau_near_miss),
            "violation": int(bool(violation)),
            "near_miss": int(bool(near_miss)),
            "shielded": int(bool(shielded)),
            "action_correction": float(action_correction),
            "reward_penalty": float(reward_penalty),
        }
        with Path(self.calibration_log_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def step(self, action):
        original_action = self._ensure_action(action)
        applied_action = original_action.copy()

        if self.config.use_adjust_speed:
            applied_action = self._smooth_action(applied_action)

        cost_before_override = self._proxy_cost(applied_action)
        near_miss = cost_before_override > self.tau_near_miss
        violation = cost_before_override > self.tau_violation

        if near_miss:
            self.ep_near_miss_count += 1
        if violation:
            self.ep_violation_count += 1

        shielded = False
        if self.config.use_soft_to_hard and violation:
            shielded_action = self._safe_action(applied_action)
            if not np.allclose(shielded_action, applied_action):
                self.ep_shield_count += 1
                shielded = True
            applied_action = shielded_action

        obs, reward, terminated, truncated, info = self.env.step(applied_action)

        reward_penalty = 0.0
        if self.config.use_context_constraints:
            reward_penalty = self._compute_penalty(cost_before_override)
            reward = float(reward - reward_penalty)

        correction = float(np.linalg.norm(applied_action - original_action))
        self.ep_reward_penalty_sum += reward_penalty
        self.ep_action_correction_sum += correction
        self.ep_step_count += 1
        self._step_index += 1

        self._append_calibration_row(
            proxy_cost=cost_before_override,
            violation=violation,
            near_miss=near_miss,
            shielded=shielded,
            action_correction=correction,
            reward_penalty=reward_penalty,
        )

        self._prev_action = applied_action.copy()

        info = dict(info)
        info["proxy_cost"] = float(cost_before_override)
        info["tau_violation"] = float(self.tau_violation)
        info["tau_near_miss"] = float(self.tau_near_miss)
        info["shielded"] = bool(shielded)
        info["ctx_id"] = self.ctx_id
        info["ctx_tuple"] = str(self.context_tuple)

        if terminated or truncated:
            denom = max(1, self.ep_step_count)
            info["violation_count"] = int(self.ep_violation_count)
            info["near_miss_count"] = int(self.ep_near_miss_count)
            info["shield_count"] = int(self.ep_shield_count)
            info["action_correction_mean"] = float(self.ep_action_correction_sum / denom)
            info["reward_penalty_sum"] = float(self.ep_reward_penalty_sum)

        return obs, reward, terminated, truncated, info

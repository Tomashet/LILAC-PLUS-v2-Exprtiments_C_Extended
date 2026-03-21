from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np


ContextTuple = tuple[str, ...]


def _context_to_id(ctx_tuple: ContextTuple) -> str:
    return "|".join(str(x) for x in ctx_tuple)


def _normalize_context(context: Any) -> ContextTuple:
    if isinstance(context, tuple):
        return tuple(str(x) for x in context)
    if isinstance(context, list):
        return tuple(str(x) for x in context)
    return (str(context),)


@dataclass
class SafetyWrapperConfig:
    use_context_constraints: bool = False
    use_adjust_speed: bool = False
    use_soft_to_hard: bool = False
    use_fixed_constraints: bool = False
    fixed_strategy: Optional[str] = None

    max_action_delta: float = 0.15
    w_lateral: float = 6.0
    w_long_pos: float = 3.0
    w_long_neg: float = 2.0
    lambda_near_miss: float = 0.10
    lambda_violation: float = 0.25
    safe_longitudinal: float = -0.25
    safe_lateral: float = 0.0

    fixed_tau_violation: float = 2.0
    fixed_tau_near_miss: float = 8.0
    fixed_max_action_delta: float = 0.10
    fixed_alpha: float = 0.10

    initial_budget: float = 1.0
    budget_decay_on_violation: float = 0.05
    risk_tightening_scale: float = 0.25
    alpha_min: float = 0.01
    alpha_max: float = 0.20
    proactive_horizon: int = 1
    proactive_trigger_ratio: float = 0.85


@dataclass
class ProactiveContextState:
    current_context: ContextTuple
    predicted_context: ContextTuple
    risk_score: float
    predicted_risk_score: float
    adaptation_demand: float
    horizon: int = 1


class SafetyMethodWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, method: str, config: SafetyWrapperConfig) -> None:
        super().__init__(env)
        self.method = method
        self.config = config

        self.tau_violation: float = 2.0
        self.tau_near_miss: float = 8.0
        self._active_tau_violation: float = self.tau_violation
        self._active_tau_near_miss: float = self.tau_near_miss

        self.context_tuple: ContextTuple = tuple()
        self.ctx_id: str = ""
        self.calibration_log_path: Optional[str] = None
        self.run_seed: Optional[int] = None

        self._prev_action: Optional[np.ndarray] = None
        self._prev_proxy_cost: float = 0.0
        self._episode_index = 0
        self._step_index = 0

        self.initial_budget: float = float(config.initial_budget)
        self.remaining_budget: float = float(config.initial_budget)

        self.last_predicted_context: ContextTuple = tuple()
        self.last_risk_score: float = 0.0
        self.last_predicted_risk_score: float = 0.0
        self.last_adaptation_demand: float = 0.0
        self.last_adjustment_limit: float = float(config.max_action_delta)
        self.last_alpha: float = float(config.fixed_alpha)

        self.context_schedule: list[ContextTuple] = []
        self.context_threshold_map: dict[ContextTuple, dict[str, float]] = {}

        self._calibration_fieldnames = [
            "episode_idx",
            "step_idx",
            "method",
            "seed",
            "ctx_id",
            "ctx_tuple",
            "predicted_ctx_tuple",
            "proxy_cost",
            "clearance",
            "tau_violation",
            "tau_near_miss",
            "active_tau_violation",
            "active_tau_near_miss",
            "risk_score",
            "predicted_risk_score",
            "adaptation_demand",
            "adjustment_limit",
            "remaining_budget",
            "alpha",
            "violation",
            "near_miss",
            "shielded",
            "action_correction",
            "reward_penalty",
        ]

        self._reset_episode_stats()

    # -----------------------------
    # external configuration
    # -----------------------------
    def set_context_thresholds(self, tau_violation: float, tau_near_miss: float) -> None:
        self.tau_violation = float(tau_violation)
        self.tau_near_miss = float(max(tau_near_miss, tau_violation))
        self._active_tau_violation = self.tau_violation
        self._active_tau_near_miss = self.tau_near_miss

    def set_thresholds(self, tau_violation: float, tau_near_miss: float) -> None:
        self.set_context_thresholds(tau_violation=tau_violation, tau_near_miss=tau_near_miss)

    def set_context_metadata(self, context: Any) -> None:
        ctx = _normalize_context(context)
        self.context_tuple = ctx
        self.ctx_id = _context_to_id(ctx)

    def set_context_schedule(
        self,
        schedule: list[ContextTuple],
        threshold_map: Optional[Mapping[ContextTuple, Mapping[str, float]]] = None,
    ) -> None:
        normalized_schedule = [_normalize_context(ctx) for ctx in schedule]
        self.context_schedule = normalized_schedule
        if normalized_schedule:
            self.set_context_metadata(normalized_schedule[0])

        normalized_map: dict[ContextTuple, dict[str, float]] = {}
        if threshold_map is not None:
            for ctx, vals in threshold_map.items():
                nctx = _normalize_context(ctx)
                normalized_map[nctx] = {
                    "tau_violation": float(vals["tau_violation"]),
                    "tau_near_miss": float(max(vals["tau_near_miss"], vals["tau_violation"])),
                }
        self.context_threshold_map = normalized_map
        self._apply_episode_context(force=True)

    def set_calibration_log_path(self, path: str) -> None:
        self.calibration_log_path = str(path)
        p = Path(self.calibration_log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._calibration_fieldnames)
                writer.writeheader()

    def set_run_seed(self, seed: int) -> None:
        self.run_seed = int(seed)

    # -----------------------------
    # lifecycle helpers
    # -----------------------------
    def _reset_episode_stats(self) -> None:
        self.ep_violation_count = 0
        self.ep_near_miss_count = 0
        self.ep_shield_count = 0
        self.ep_reward_penalty_sum = 0.0
        self.ep_action_correction_sum = 0.0
        self.ep_step_count = 0
        self.remaining_budget = float(self.initial_budget)

    def _apply_episode_context(self, force: bool = False) -> None:
        if not self.context_schedule:
            return
        idx = max(0, self._episode_index - 1) % len(self.context_schedule)
        new_ctx = self.context_schedule[idx]
        if force or new_ctx != self.context_tuple:
            self.set_context_metadata(new_ctx)
        if not self.config.use_fixed_constraints:
            threshold_rec = self.context_threshold_map.get(new_ctx)
            if threshold_rec is not None:
                self.set_context_thresholds(
                    tau_violation=float(threshold_rec["tau_violation"]),
                    tau_near_miss=float(threshold_rec["tau_near_miss"]),
                )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = None
        self._prev_proxy_cost = 0.0
        self._episode_index += 1
        self._step_index = 0
        self._reset_episode_stats()
        self._apply_episode_context()
        info = dict(info)
        info["ctx_id"] = self.ctx_id
        info["ctx_tuple"] = str(self.context_tuple)
        return obs, info

    # -----------------------------
    # core safety helpers
    # -----------------------------
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

    def _safe_action(self, action: np.ndarray) -> np.ndarray:
        safe = action.copy()
        safe[0] = min(float(safe[0]), self.config.safe_longitudinal)
        safe[1] = self.config.safe_lateral
        return safe.astype(np.float32)

    def _risk_from_context(self, ctx: ContextTuple) -> float:
        density = ctx[0] if len(ctx) > 0 else "low"
        behavior = ctx[1] if len(ctx) > 1 else "calm"
        sensor = ctx[2] if len(ctx) > 2 else "clean"

        score = 0.0
        if density == "high":
            score += 1.0
        elif density in ("medium", "med"):
            score += 0.5

        if behavior in ("aggr", "aggressive"):
            score += 1.0
        elif behavior in ("mixed", "moderate"):
            score += 0.5

        if sensor in ("dropout", "noisy"):
            score += 1.0
        elif sensor in ("foggy", "blur"):
            score += 0.5

        return float(score)

    @staticmethod
    def _escalate_slot(value: str, ladder: tuple[str, ...]) -> str:
        if value not in ladder:
            return value
        idx = ladder.index(value)
        return ladder[min(len(ladder) - 1, idx + 1)]

    def _try_env_forecast_context(self) -> Optional[ContextTuple]:
        for attr in ("get_predicted_context", "predict_context", "forecast_context"):
            fn = getattr(self.env, attr, None)
            if callable(fn):
                try:
                    out = fn()
                    if out is not None:
                        return _normalize_context(out)
                except Exception:
                    pass
        return None

    def _predict_context(self, ctx: ContextTuple) -> ContextTuple:
        env_pred = self._try_env_forecast_context()
        if env_pred is not None:
            return env_pred

        # If a context schedule exists, forecast the next episode's context.
        if self.context_schedule:
            cur_idx = max(0, self._episode_index - 1) % len(self.context_schedule)
            next_idx = (cur_idx + 1) % len(self.context_schedule)
            next_ctx = self.context_schedule[next_idx]
            if next_ctx != ctx:
                return next_ctx

        # Fallback: pessimistic local forecast from current stress and budget.
        density = ctx[0] if len(ctx) > 0 else "low"
        behavior = ctx[1] if len(ctx) > 1 else "calm"
        sensor = ctx[2] if len(ctx) > 2 else "clean"

        forecast_pressure = 0.0
        if self._active_tau_violation > 1e-6:
            forecast_pressure += self._prev_proxy_cost / self._active_tau_violation
        budget_ratio = self.remaining_budget / max(1e-6, self.initial_budget)
        forecast_pressure += max(0.0, 1.0 - budget_ratio)

        predicted_density = density
        predicted_behavior = behavior
        predicted_sensor = sensor

        if forecast_pressure >= self.config.proactive_trigger_ratio:
            predicted_behavior = self._escalate_slot(predicted_behavior, ("calm", "mixed", "aggr"))
        if forecast_pressure >= 1.00:
            predicted_sensor = self._escalate_slot(predicted_sensor, ("clean", "foggy", "dropout"))
        if forecast_pressure >= 1.50 or budget_ratio < 0.50:
            predicted_density = self._escalate_slot(predicted_density, ("low", "medium", "high"))

        return (predicted_density, predicted_behavior, predicted_sensor)

    def _build_proactive_context(self) -> ProactiveContextState:
        current_ctx = self.context_tuple if self.context_tuple else ("low", "calm", "clean")
        predicted_ctx = self._predict_context(current_ctx)

        current_risk = self._risk_from_context(current_ctx)
        predicted_risk = self._risk_from_context(predicted_ctx)
        adaptation_demand = max(0.0, predicted_risk - current_risk)

        self.last_predicted_context = predicted_ctx
        self.last_risk_score = current_risk
        self.last_predicted_risk_score = predicted_risk
        self.last_adaptation_demand = adaptation_demand

        return ProactiveContextState(
            current_context=current_ctx,
            predicted_context=predicted_ctx,
            risk_score=current_risk,
            predicted_risk_score=predicted_risk,
            adaptation_demand=adaptation_demand,
            horizon=int(self.config.proactive_horizon),
        )

    def _update_context_thresholds(self, pcs: ProactiveContextState) -> None:
        if self.config.use_fixed_constraints:
            self._active_tau_violation = float(self.config.fixed_tau_violation)
            self._active_tau_near_miss = float(max(self.config.fixed_tau_near_miss, self.config.fixed_tau_violation))
            return

        tighten = self.config.risk_tightening_scale * pcs.predicted_risk_score
        active_v = max(0.5, float(self.tau_violation) - tighten)
        active_n = max(active_v, float(self.tau_near_miss) - 2.0 * tighten)
        self._active_tau_violation = float(active_v)
        self._active_tau_near_miss = float(active_n)

    def _compute_adjustment_limit(self, pcs: ProactiveContextState) -> float:
        if self.config.use_fixed_constraints:
            limit = float(self.config.fixed_max_action_delta)
        else:
            base = float(self.config.max_action_delta)
            limit = base / (1.0 + pcs.predicted_risk_score + pcs.adaptation_demand)
            limit = max(0.03, min(base, limit))
        self.last_adjustment_limit = float(limit)
        return float(limit)

    def _smooth_action_with_limit(self, action: np.ndarray, max_delta: float) -> np.ndarray:
        if self._prev_action is None:
            return action.copy()
        delta = action - self._prev_action
        delta = np.clip(delta, -max_delta, max_delta)
        return (self._prev_action + delta).astype(np.float32)

    def _compute_alpha(self, pcs: ProactiveContextState) -> float:
        if self.config.use_fixed_constraints:
            alpha = float(self.config.fixed_alpha)
        else:
            budget_ratio = max(0.0, self.remaining_budget / max(1e-6, self.initial_budget))
            alpha = 0.15 * budget_ratio / (1.0 + pcs.predicted_risk_score)
            alpha = max(self.config.alpha_min, min(self.config.alpha_max, alpha))
        self.last_alpha = float(alpha)
        return float(alpha)

    def _update_budget(self, violation: bool) -> None:
        if violation:
            self.remaining_budget = max(0.0, self.remaining_budget - float(self.config.budget_decay_on_violation))

    def _compute_penalty(self, cost: float) -> float:
        penalty = 0.0
        if cost > self._active_tau_near_miss:
            penalty += self.config.lambda_near_miss * (cost - self._active_tau_near_miss)
        if cost > self._active_tau_violation:
            penalty += self.config.lambda_violation * (cost - self._active_tau_violation)
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
            "predicted_ctx_tuple": str(self.last_predicted_context),
            "proxy_cost": float(proxy_cost),
            "clearance": float(proxy_cost),
            "tau_violation": float(self.tau_violation),
            "tau_near_miss": float(self.tau_near_miss),
            "active_tau_violation": float(self._active_tau_violation),
            "active_tau_near_miss": float(self._active_tau_near_miss),
            "risk_score": float(self.last_risk_score),
            "predicted_risk_score": float(self.last_predicted_risk_score),
            "adaptation_demand": float(self.last_adaptation_demand),
            "adjustment_limit": float(self.last_adjustment_limit),
            "remaining_budget": float(self.remaining_budget),
            "alpha": float(self.last_alpha),
            "violation": int(bool(violation)),
            "near_miss": int(bool(near_miss)),
            "shielded": int(bool(shielded)),
            "action_correction": float(action_correction),
            "reward_penalty": float(reward_penalty),
        }
        with Path(self.calibration_log_path).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._calibration_fieldnames)
            writer.writerow(row)

    # -----------------------------
    # gym step
    # -----------------------------
    def step(self, action):
        original_action = self._ensure_action(action)
        applied_action = original_action.copy()

        pcs = self._build_proactive_context()
        self._update_context_thresholds(pcs)

        if self.config.use_adjust_speed:
            max_delta = self._compute_adjustment_limit(pcs)
            applied_action = self._smooth_action_with_limit(applied_action, max_delta)
        else:
            self.last_adjustment_limit = float(
                self.config.fixed_max_action_delta if self.config.use_fixed_constraints else self.config.max_action_delta
            )

        cost_before_override = self._proxy_cost(applied_action)
        near_miss = cost_before_override > self._active_tau_near_miss
        violation = cost_before_override > self._active_tau_violation

        if near_miss:
            self.ep_near_miss_count += 1
        if violation:
            self.ep_violation_count += 1

        shielded = False
        if self.config.use_soft_to_hard:
            _ = self._compute_alpha(pcs)
            strict_mode = (pcs.predicted_risk_score > 1.0) or (self.remaining_budget < 0.3)
            if self.config.use_fixed_constraints:
                strict_mode = violation and (cost_before_override > self._active_tau_violation)
            if violation and strict_mode:
                shielded_action = self._safe_action(applied_action)
                if not np.allclose(shielded_action, applied_action):
                    self.ep_shield_count += 1
                    shielded = True
                applied_action = shielded_action
        else:
            self.last_alpha = float(self.config.fixed_alpha)

        obs, reward, terminated, truncated, info = self.env.step(applied_action)

        reward_penalty = 0.0
        if self.config.use_context_constraints:
            reward_penalty = self._compute_penalty(cost_before_override)
            reward = float(reward - reward_penalty)

        self._update_budget(violation)

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
        self._prev_proxy_cost = float(cost_before_override)

        info = dict(info)
        info["proxy_cost"] = float(cost_before_override)
        info["tau_violation"] = float(self._active_tau_violation)
        info["tau_near_miss"] = float(self._active_tau_near_miss)
        info["shielded"] = bool(shielded)
        info["ctx_id"] = self.ctx_id
        info["ctx_tuple"] = str(self.context_tuple)
        info["predicted_ctx_tuple"] = str(self.last_predicted_context)
        info["risk_score"] = float(self.last_risk_score)
        info["predicted_risk_score"] = float(self.last_predicted_risk_score)
        info["adaptation_demand"] = float(self.last_adaptation_demand)
        info["adjustment_limit"] = float(self.last_adjustment_limit)
        info["remaining_budget"] = float(self.remaining_budget)
        info["alpha"] = float(self.last_alpha)

        if terminated or truncated:
            denom = max(1, self.ep_step_count)
            info["violation_count"] = int(self.ep_violation_count)
            info["near_miss_count"] = int(self.ep_near_miss_count)
            info["shield_count"] = int(self.ep_shield_count)
            info["action_correction_mean"] = float(self.ep_action_correction_sum / denom)
            info["reward_penalty_sum"] = float(self.ep_reward_penalty_sum)

        return obs, reward, terminated, truncated, info

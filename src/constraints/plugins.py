from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .base import ConstraintPlugin, ContextMetrics


@dataclass
class CPSSPluginConfig:
    """Configuration for the CPSS-style constraint tightening.

    In this codebase, the environment-side wrapper listens to
    set_adjustment_risk(...) and tightens an internal slack parameter
    (epsilon override) when unsafe.
    """

    env_method_name: str = "set_adjustment_risk"
    unsafe_on_risk: float = 0.6


class CPSSConstraintPlugin(ConstraintPlugin):
    """Default plugin: pass risk/unsafe to the existing SafetyShieldWrapper hooks."""

    def __init__(self, cfg: CPSSPluginConfig):
        self.cfg = cfg

    def on_context_metrics(self, env_idx: int, metrics: ContextMetrics) -> Dict[str, float]:
        unsafe = bool(metrics.cp_flag or (metrics.risk >= float(self.cfg.unsafe_on_risk)))
        try:
            self.training_env.env_method(
                self.cfg.env_method_name,
                risk=float(metrics.risk),
                unsafe=unsafe,
                s_env=float(metrics.s_env),
                s_agent=float(metrics.s_agent),
                indices=[env_idx],
            )
        except Exception:
            pass

        return {
            "constraint/risk": float(metrics.risk),
            "constraint/unsafe": float(unsafe),
            "constraint/s_env": float(metrics.s_env),
            "constraint/s_uncert": float(metrics.s_uncert),
        }


@dataclass
class ProactiveForecastPluginConfig:
    """Proactive forecasted safety constraint.

    Tightens safety margin using predicted context shift and uncertainty.

    In this codebase, the environment-side SafetyShieldWrapper computes:
        eps_override = base_epsilon + adj_eps_scale * risk
    whenever unsafe=True.

    Therefore `risk` is interpreted as an extra epsilon margin and can be > 1.0.
    """

    env_method_name: str = "set_adjustment_risk"

    # Change-point handling
    cooldown_episodes: int = 2

    # Margin computation
    k_env: float = 1.0
    k_uncert: float = 0.25
    margin_threshold: float = 0.0

    # Extra conservatism after change-point
    cooldown_extra_margin: float = 0.5


class ProactiveForecastConstraintPlugin(ConstraintPlugin):
    """Proactive constraint tightening driven by forecasted nonstationarity."""

    def __init__(self, cfg: ProactiveForecastPluginConfig):
        self.cfg = cfg
        self._cooldown = None

    def on_training_start(self, model: Any, training_env: Any) -> None:
        super().on_training_start(model, training_env)
        self._cooldown = np.zeros(int(training_env.num_envs), dtype=np.int64)

    def on_context_metrics(self, env_idx: int, metrics: ContextMetrics) -> Dict[str, float]:
        assert self._cooldown is not None

        if bool(metrics.cp_flag):
            self._cooldown[env_idx] = int(self.cfg.cooldown_episodes)
        else:
            self._cooldown[env_idx] = max(0, int(self._cooldown[env_idx]) - 1)

        base_margin = (
            float(self.cfg.k_env) * float(metrics.s_env)
            + float(self.cfg.k_uncert) * float(metrics.s_uncert)
        )
        margin = base_margin + (
            float(self.cfg.cooldown_extra_margin) if (self._cooldown[env_idx] > 0) else 0.0
        )

        unsafe = bool(
            (margin > float(self.cfg.margin_threshold))
            or (self._cooldown[env_idx] > 0)
        )

        # Risk is interpreted as "extra epsilon margin" and may be > 1.
        risk = float(max(0.0, margin))

        try:
            self.training_env.env_method(
                self.cfg.env_method_name,
                risk=risk,
                unsafe=unsafe,
                s_env=float(metrics.s_env),
                s_agent=float(metrics.s_agent),
                indices=[env_idx],
            )
        except Exception:
            pass

        return {
            "constraint_pf/margin": float(margin),
            "constraint_pf/base_margin": float(base_margin),
            "constraint_pf/unsafe": float(unsafe),
            "constraint_pf/cooldown": float(self._cooldown[env_idx]),
            "constraint_pf/s_env": float(metrics.s_env),
            "constraint_pf/s_uncert": float(metrics.s_uncert),
        }


@dataclass
class AdjustSpeedPluginConfig:
    """Adjustment-speed constraint.

    Uses s_env from the context engine and estimates s_agent from an external
    adaptation-speed estimator driven by model updates.
    """

    env_method_name: str = "set_adjustment_risk"
    margin: float = 0.0
    temperature: float = 10.0
    unsafe_when_raw_gt: float = 0.0


class AdjustSpeedConstraintPlugin(ConstraintPlugin):
    """Unsafe if environment shift exceeds adaptation speed."""

    def __init__(self, cfg: AdjustSpeedPluginConfig, adapt_estimator: Any):
        self.cfg = cfg
        self.adapt_est = adapt_estimator

    def on_context_metrics(self, env_idx: int, metrics: ContextMetrics) -> Dict[str, float]:
        # Use the external adaptation estimator as s_agent.
        s_agent = float(self.adapt_est.speed())
        raw = float(metrics.s_env) - s_agent

        # Logistic risk score.
        denom = max(1e-6, float(self.cfg.temperature))
        risk = float(1.0 / (1.0 + np.exp(-(raw - float(self.cfg.margin)) / denom)))
        unsafe = bool(raw > float(self.cfg.unsafe_when_raw_gt))

        try:
            self.training_env.env_method(
                self.cfg.env_method_name,
                risk=risk,
                unsafe=unsafe,
                s_env=float(metrics.s_env),
                s_agent=float(s_agent),
                indices=[env_idx],
            )
        except Exception:
            pass

        return {
            "constraint_as/s_env": float(metrics.s_env),
            "constraint_as/s_agent": float(s_agent),
            "constraint_as/raw": float(raw),
            "constraint_as/risk": float(risk),
            "constraint_as/unsafe": float(unsafe),
        }

    def on_rollout_end(self) -> None:
        try:
            self.adapt_est.on_update(self.model)
        except Exception:
            pass
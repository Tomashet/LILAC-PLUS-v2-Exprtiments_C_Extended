from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ContextMetrics:
    """Standardized information emitted by the context engine.

    All fields are per-environment scalar summaries for the current episode boundary
    (or for the latest update interval).
    """

    # Predicted change magnitude (e.g., ||mu_pred - mu_post||)
    s_env: float
    # Predictive uncertainty magnitude (e.g., ||sigma_pred||)
    s_uncert: float
    # Optional: agent adaptation capacity proxy
    s_agent: float
    # Risk in [0,1]
    risk: float
    # Change-point detection
    cp_flag: bool
    # Debug extras
    cp_delta: float = 0.0
    cp_kl: float = 0.0


class ConstraintPlugin:
    """A thin adapter that turns context metrics into environment-side constraint signals.

    The plugin is intended to be used from SB3 callbacks. It must be VecEnv-safe:
    use env_method to communicate with wrappers.
    """

    def on_training_start(self, model: Any, training_env: Any) -> None:
        self.model = model
        self.training_env = training_env

    def on_context_metrics(self, env_idx: int, metrics: ContextMetrics) -> Dict[str, float]:
        """Consume context metrics and push constraint state into env wrappers.

        Returns a dict of scalars to be logged.
        """

        raise NotImplementedError

    def on_step_end(self) -> Dict[str, float]:
        """Optional per-step logging hook."""
        return {}

    def on_rollout_end(self) -> None:
        """Optional hook called at the end of a rollout/update."""
        return None

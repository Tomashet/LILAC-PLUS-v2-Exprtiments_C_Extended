from __future__ import annotations

from typing import Optional

from .base import ConstraintPlugin
from .plugins import (
    AdjustSpeedConstraintPlugin,
    AdjustSpeedPluginConfig,
    CPSSConstraintPlugin,
    CPSSPluginConfig,
    ProactiveForecastConstraintPlugin,
    ProactiveForecastPluginConfig,
)


def make_constraint(name: str, *, adapt_estimator=None) -> Optional[ConstraintPlugin]:
    """Factory for constraint plugins.

    Parameters
    ----------
    name:
        One of: 'none', 'cpss', 'proactive_forecast', 'adjust_speed'
    adapt_estimator:
        Required for 'adjust_speed'.
    """

    name = (name or "none").strip().lower()
    if name in ("none", "off", "no", "false"):
        return None
    if name in ("cpss",):
        return CPSSConstraintPlugin(CPSSPluginConfig())
    if name in ("proactive", "proactive_forecast", "forecast"):
        return ProactiveForecastConstraintPlugin(ProactiveForecastPluginConfig())
    if name in ("adjust", "adjust_speed", "as"):
        if adapt_estimator is None:
            raise ValueError("adjust_speed constraint requires adapt_estimator")
        return AdjustSpeedConstraintPlugin(AdjustSpeedPluginConfig(), adapt_estimator=adapt_estimator)
    raise ValueError(f"Unknown constraint plugin: {name}")

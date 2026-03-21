"""Constraint plugin interface.

This package defines a small abstraction layer that lets you combine a
context engine (e.g., LILAC PLUS) with multiple safety mechanisms.

The goal is to keep:
  - context inference/prediction (z, uncertainty, change-points)
  - constraint logic (CPSS, adjustment-speed, proactive forecast)

as independent modules that communicate via a standard set of metrics.
"""

from .base import ConstraintPlugin, ContextMetrics
from .factory import make_constraint

__all__ = ["ConstraintPlugin", "ContextMetrics", "make_constraint"]

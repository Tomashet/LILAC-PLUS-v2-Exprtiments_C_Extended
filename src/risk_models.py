# src/risk_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class EmpiricalRiskConfig:
    """Configuration for EmpiricalRiskModel."""
    # bin edges for clearance (meters). last bin captures values >= last edge.
    clearance_bins: Tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
    # Beta prior (Laplace smoothing)
    alpha: float = 1.0
    beta: float = 1.0


class EmpiricalRiskModel:
    """Online empirical estimate of P(violation | clearance_bin, action, ctx_id).

    This is intentionally lightweight for a clean soft→hard constraint demo.
    You can later swap this for a neural predictor p_psi(s,a,z).
    """

    def __init__(self, cfg: EmpiricalRiskConfig):
        self.cfg = cfg
        self.count: Dict[Tuple[int, int, int], int] = {}
        self.viol: Dict[Tuple[int, int, int], int] = {}

    def _bin_clearance(self, clearance: float) -> int:
        bins = self.cfg.clearance_bins
        # np.digitize returns 1..len(bins); shift to 0-index
        idx = int(np.digitize([float(clearance)], bins, right=False)[0]) - 1
        return max(0, min(idx, len(bins) - 1))

    def predict_proba(self, clearance: float, action: int, ctx_id: int) -> float:
        b = self._bin_clearance(clearance)
        key = (b, int(action), int(ctx_id))
        n = self.count.get(key, 0)
        v = self.viol.get(key, 0)
        a = float(self.cfg.alpha)
        bb = float(self.cfg.beta)
        return float((v + a) / (n + a + bb))

    def update(self, clearance: float, action: int, ctx_id: int, violation: int) -> None:
        b = self._bin_clearance(clearance)
        key = (b, int(action), int(ctx_id))
        self.count[key] = self.count.get(key, 0) + 1
        self.viol[key] = self.viol.get(key, 0) + int(bool(violation))

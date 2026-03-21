from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class FeasibilityConfig:
    margin: float = 0.0
    temperature: float = 10.0
    clip: float = 20.0


class FeasibilityMonitor:
    """
    unsafe iff S_env > S_agent + margin
    risk_score is a smooth sigmoid proxy in [0,1]
    """
    def __init__(self, cfg: FeasibilityConfig):
        self.cfg = cfg

    def unsafe(self, s_env: float, s_agent: float) -> bool:
        return bool(s_env > (s_agent + self.cfg.margin))

    def risk_score(self, s_env: float, s_agent: float) -> float:
        x = (s_env - (s_agent + self.cfg.margin)) * self.cfg.temperature
        x = max(-self.cfg.clip, min(self.cfg.clip, x))
        return float(1.0 / (1.0 + math.exp(-x)))
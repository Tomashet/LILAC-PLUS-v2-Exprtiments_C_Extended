from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Union
import numpy as np


@dataclass
class ShiftSpeedConfig:
    window: int = 200
    # "discrete" uses ctx_id switches; "l2" uses embedding deltas
    metric: str = "discrete"
    eps: float = 1e-8


class ShiftSpeedEstimator:
    """
    Estimates environment/context shift speed.

    - metric="discrete": z_t is an int ctx_id; speed = mean(ctx_t != ctx_{t-1})
    - metric="l2": z_t is a vector embedding; speed = mean(||z_t - z_{t-1}||_2)
    """
    def __init__(self, cfg: ShiftSpeedConfig):
        self.cfg = cfg
        self._hist = deque(maxlen=int(cfg.window))

    def update(self, z_t: Union[int, np.ndarray]) -> None:
        self._hist.append(z_t)

    def speed(self) -> float:
        if len(self._hist) < 2:
            return 0.0

        if self.cfg.metric == "discrete":
            # ctx_id switching frequency
            switches = 0
            prev = self._hist[0]
            for cur in list(self._hist)[1:]:
                switches += int(cur != prev)
                prev = cur
            return float(switches) / float(max(1, len(self._hist) - 1))

        if self.cfg.metric == "l2":
            z = [np.asarray(x, dtype=np.float32).reshape(-1) for x in self._hist]
            dz = [np.linalg.norm(z[i] - z[i - 1]) for i in range(1, len(z))]
            return float(np.mean(dz)) if dz else 0.0

        raise ValueError(f"Unknown metric: {self.cfg.metric}")
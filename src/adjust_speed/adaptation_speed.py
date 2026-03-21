from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch


@dataclass
class AdaptSpeedConfig:
    window_updates: int = 20
    eps: float = 1e-8


class AdaptationSpeedEstimator:
    """
    SB3-friendly proxy for "how fast the agent can adjust":
    measure mean L2 change in policy parameters per update.
    """
    def __init__(self, cfg: AdaptSpeedConfig):
        self.cfg = cfg
        self._delta_hist = deque(maxlen=int(cfg.window_updates))
        self._prev_params: np.ndarray | None = None

    def _flat_params(self, model) -> np.ndarray:
        with torch.no_grad():
            chunks = []
            # SB3 algorithms all have model.policy as a torch module
            for p in model.policy.parameters():
                chunks.append(p.detach().view(-1).cpu().numpy())
            if not chunks:
                return np.zeros((1,), dtype=np.float32)
            return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    def on_update(self, model) -> None:
        cur = self._flat_params(model)
        if self._prev_params is None:
            self._prev_params = cur
            return
        delta = float(np.linalg.norm(cur - self._prev_params))
        self._delta_hist.append(delta)
        self._prev_params = cur

    def speed(self) -> float:
        if not self._delta_hist:
            return 0.0
        return float(np.mean(self._delta_hist))
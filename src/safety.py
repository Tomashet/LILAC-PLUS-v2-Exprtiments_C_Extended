from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np


@dataclass
class SafetyParams:
    d0: float = 2.0
    h: float = 1.2
    delta_nearmiss: float = 1.0
    horizon_n: int = 10
    epsilon: float = 0.5


def clearance_margin(env, params: SafetyParams) -> float:
    road = getattr(env.unwrapped, "road", None)
    ego = getattr(env.unwrapped, "vehicle", None)
    if road is None or ego is None:
        return float("nan")

    ego_pos = np.array(getattr(ego, "position", [0.0, 0.0]), dtype=np.float32)
    ego_speed = float(getattr(ego, "speed", 0.0))
    safe_dist = params.d0 + params.h * ego_speed

    min_dist = float("inf")
    for v in getattr(road, "vehicles", []):
        if v is ego:
            continue
        pos = np.array(getattr(v, "position", [0.0, 0.0]), dtype=np.float32)
        dist = float(np.linalg.norm(pos - ego_pos))
        if dist < min_dist:
            min_dist = dist

    if not np.isfinite(min_dist):
        return float("nan")

    return float(min_dist - safe_dist)


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1
    window: int = 200
    seed: int = 0

    def __post_init__(self):
        self.errors: List[float] = []

    def update(self, err: float) -> None:
        if np.isfinite(err):
            self.errors.append(float(abs(err)))
            if len(self.errors) > self.window:
                self.errors = self.errors[-self.window:]

    def inflation(self) -> float:
        if len(self.errors) < 30:
            return 0.0
        return float(np.quantile(np.array(self.errors, dtype=np.float32), 1.0 - self.alpha))


@dataclass
class MPCLikeSafetyShield:
    params: SafetyParams
    action_space_type: str  # "discrete" | "continuous"
    no_mpc: bool = False
    no_conformal: bool = False
    calibrator: Optional[ConformalCalibrator] = None

    def _predict_clearance_proxy(self, cur_d: float, cur_speed: float, action, step_idx: int) -> float:
        dv = 0.0

        if self.action_space_type == "discrete":
            if isinstance(action, (int, np.integer)):
                if int(action) == 3:
                    dv = +1.5
                elif int(action) == 4:
                    dv = -2.0
        else:
            a = np.array(action, dtype=np.float32).reshape(-1)
            if a.size >= 2:
                dv = 2.0 * float(a[1])

        speed_k = max(0.0, cur_speed + (step_idx + 1) * dv)
        safe_dist_k = self.params.d0 + self.params.h * speed_k
        min_dist_approx = cur_d + (self.params.d0 + self.params.h * cur_speed)
        return float(min_dist_approx - safe_dist_k)

    def _actions_equal(self, a: Any, b: Any) -> bool:
        if self.action_space_type == "discrete":
            try:
                return int(a) == int(b)
            except Exception:
                return a == b

        try:
            aa = np.asarray(a, dtype=np.float32).reshape(-1)
            bb = np.asarray(b, dtype=np.float32).reshape(-1)
            if aa.shape != bb.shape:
                return False
            return bool(np.allclose(aa, bb, atol=1e-6, rtol=1e-6))
        except Exception:
            return a == b

    def filter_action(
        self,
        env,
        action,
        cur_ctx_id: int,
        *,
        eps_override: Optional[float] = None,
    ) -> Tuple[object, Dict]:
        if self.no_mpc:
            return action, {"shield_used": False, "shield_reason": "disabled"}

        cur_d = clearance_margin(env, self.params)
        ego = getattr(env.unwrapped, "vehicle", None)
        cur_speed = float(getattr(ego, "speed", 0.0)) if ego is not None else 0.0

        if not np.isfinite(cur_d):
            return action, {"shield_used": False, "shield_reason": "nan_clearance"}

        inflate = 0.0
        if (not self.no_conformal) and (self.calibrator is not None):
            inflate = self.calibrator.inflation()

        base_eps = float(self.params.epsilon) if eps_override is None else float(eps_override)
        eps = base_eps + inflate

        candidates = self._candidate_actions(action)
        best = action
        feasible = False

        for cand in candidates:
            ok = True
            for k in range(self.params.horizon_n):
                if self._predict_clearance_proxy(cur_d, cur_speed, cand, k) <= eps:
                    ok = False
                    break
            if ok:
                best = cand
                feasible = True
                break

        if feasible:
            used = not self._actions_equal(best, action)
            return best, {
                "shield_used": bool(used),
                "shield_reason": "feasible_alt" if used else "original_ok",
                "eps": eps,
                "inflate": inflate,
                "cur_d": cur_d,
            }

        fallback = self._fallback_action()
        return fallback, {
            "shield_used": True,
            "shield_reason": "fallback",
            "eps": eps,
            "inflate": inflate,
            "cur_d": cur_d,
        }

    def _candidate_actions(self, action) -> List[object]:
        if self.action_space_type == "discrete":
            return [action, 4, 1, 0, 2, 3]

        a = np.array(action, dtype=np.float32).reshape(-1)
        if a.size < 2:
            return [action]

        steer = float(a[0])
        return [
            np.array([steer, float(a[1])], dtype=np.float32),
            np.array([steer, -0.5], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.0, -0.2], dtype=np.float32),
        ]

    def _fallback_action(self) -> object:
        if self.action_space_type == "discrete":
            return 4
        return np.array([0.0, -1.0], dtype=np.float32)
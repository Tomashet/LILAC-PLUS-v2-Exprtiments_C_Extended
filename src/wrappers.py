# src/wrappers.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym

from src.context import MarkovContextScheduler, context_to_highway_config
from src.safety import SafetyParams, ConformalCalibrator


class ContextNonstationaryWrapper(gym.Wrapper):
    """
    Applies a Markov context switch once per episode and configures the underlying highway-env.

    Stores last_config for downstream wrappers (noise/dropout, logging).

    IMPORTANT: attaches ctx_id/ctx_tuple to both reset() AND step() info so step-level
    logging (SB3 callback) can visualize context switching.
    """

    def __init__(self, env: gym.Env, scheduler: MarkovContextScheduler):
        super().__init__(env)
        self.scheduler = scheduler
        self.last_config: Dict[str, Any] = {}
        self._first_reset = True

        # Current context metadata
        self._ctx_id: int = -1
        self._ctx_tuple: Any = None

    def reset(self, **kwargs):
        # Advance context once per new episode (not on the very first reset)
        if self._first_reset:
            ctx = self.scheduler.current()
            self._first_reset = False
        else:
            ctx = self.scheduler.step_episode()

        cfg = context_to_highway_config(ctx)
        self.last_config = cfg
        self._ctx_id = int(cfg.get("_ctx_id", -1))
        self._ctx_tuple = cfg.get("_ctx_tuple", None)

        # Configure underlying env if possible
        if hasattr(self.env.unwrapped, "configure"):
            self.env.unwrapped.configure(cfg)

        obs, info = self.env.reset(**kwargs)

        # Ensure ctx metadata is visible in info (reset)
        info = dict(info) if info is not None else {}
        info["ctx_id"] = self._ctx_id
        info["ctx_tuple"] = self._ctx_tuple
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure ctx metadata is visible in info (every step)
        info = dict(info) if info is not None else {}
        info["ctx_id"] = self._ctx_id
        info["ctx_tuple"] = self._ctx_tuple
        return obs, reward, terminated, truncated, info


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise and/or dropout to observations based on context config keys:
      - _ctx_obs_noise_std
      - _ctx_dropout_prob

    This wrapper expects the wrapped env (or another wrapper) to expose .last_config
    (ContextNonstationaryWrapper provides it).
    """

    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def observation(self, observation):
        obs = np.asarray(observation).astype(np.float32, copy=False)

        cfg = getattr(self.env, "last_config", {}) or {}
        noise_std = float(cfg.get("_ctx_obs_noise_std", 0.0))
        dropout_prob = float(cfg.get("_ctx_dropout_prob", 0.0))

        if noise_std > 0:
            obs = obs + self.rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)

        if dropout_prob > 0:
            mask = self.rng.random(obs.shape) < dropout_prob
            obs = obs.copy()
            obs[mask] = 0.0

        return obs


class SafetyShieldWrapper(gym.Wrapper):
    """
    Wraps an env with an (optional) safety shield.

    - Lazy-imports a shield class from src/safety.py and supports multiple class names.
    - Attaches 'shield_used'/'shield_reason' to info.
    - Defines a clear safety metric:
        violation := crashed/collision from highway-env info (if provided)

    NEW (Adjustment-speed constraint integration):
    - Supports set_adjustment_risk(risk, unsafe, s_env, s_agent)
    - If unsafe, tightens epsilon passed to shield (eps_override) when supported.
    """

    def __init__(
        self,
        env: gym.Env,
        params: SafetyParams,
        action_space_type: str,
        no_mpc: bool,
        no_conformal: bool,
        calibrator: Optional[ConformalCalibrator] = None,
    ):
        super().__init__(env)
        self.params = params
        self.action_space_type = action_space_type
        self.no_mpc = bool(no_mpc)
        self.no_conformal = bool(no_conformal)
        self.calibrator = calibrator
        self.shield = None

        # Adjustment-speed state (set by SB3 callback via env_method)
        self._adj_risk: float = 0.0
        self._adj_unsafe: bool = False
        self._adj_s_env: float = 0.0
        self._adj_s_agent: float = 0.0

        # How strongly to tighten eps when unsafe (simple default; tune later)
        self._adj_eps_scale: float = 1.0

        # Only try to build a shield if at least one feature is enabled
        if not (self.no_mpc and self.no_conformal):
            from src import safety as safety_mod

            # Try common class names (pick first that exists)
            ShieldCls = (
                getattr(safety_mod, "SafetyShield", None)
                or getattr(safety_mod, "Shield", None)
                or getattr(safety_mod, "SafetyLayer", None)
                or getattr(safety_mod, "MPCShield", None)
                or getattr(safety_mod, "MPCLikeSafetyShield", None)  # <-- correction
            )

            if ShieldCls is None:
                raise ImportError(
                    "Could not find a shield class in src/safety.py.\n"
                    "Tried: SafetyShield, Shield, SafetyLayer, MPCShield, MPCLikeSafetyShield.\n"
                    "Run:\n"
                    '  python -c "import src.safety as s; print([n for n in dir(s) if \'hield\' in n.lower()])"\n'
                    "and then update SafetyShieldWrapper to use the correct name."
                )

            try:
                self.shield = ShieldCls(
                    params=params,
                    action_space_type=action_space_type,
                    no_mpc=no_mpc,
                    no_conformal=no_conformal,
                    calibrator=calibrator,
                )
            except TypeError:
                # Fallback: try positional args
                self.shield = ShieldCls(params, action_space_type, no_mpc, no_conformal, calibrator)

    # ---- Adjustment-speed setter (called via VecEnv env_method) ----
    def set_adjustment_risk(
        self,
        *,
        risk: float,
        unsafe: bool,
        s_env: float = 0.0,
        s_agent: float = 0.0,
    ) -> None:
        self._adj_risk = float(risk)
        self._adj_unsafe = bool(unsafe)
        self._adj_s_env = float(s_env)
        self._adj_s_agent = float(s_agent)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _get_ctx_id_from_env_or_info(self, info: Dict[str, Any]) -> int:
        # Prefer info (ContextNonstationaryWrapper injects ctx_id each step)
        if "ctx_id" in info:
            try:
                return int(info.get("ctx_id", -1))
            except Exception:
                return -1
        # Fallback: try env attribute (if present)
        for obj in (self.env, getattr(self.env, "unwrapped", None)):
            if obj is not None and hasattr(obj, "_ctx_id"):
                try:
                    return int(getattr(obj, "_ctx_id"))
                except Exception:
                    pass
        return -1

    def step(self, action):
        shield_used = False
        shield_reason = ""
        proposed_action = action
        shield_meta: Dict[str, Any] = {"shield_used": False, "shield_reason": ""}

        # --- For conformal calibration: cache pre-step clearance proxy inputs ---
        cur_d_pre = np.nan
        cur_speed_pre = 0.0
        try:
            from src.safety import clearance_margin
            cur_d_pre = float(clearance_margin(self.env, self.params))
            ego = getattr(self.env.unwrapped, "vehicle", None)
            cur_speed_pre = float(getattr(ego, "speed", 0.0)) if ego is not None else 0.0
        except Exception:
            pass

        # Compute eps_override if unsafe
        eps_override: Optional[float] = None
        if self._adj_unsafe:
            # More conservative when unsafe. (Interpretation: raise eps threshold.)
            eps_override = float(self.params.epsilon) + float(self._adj_eps_scale) * float(self._adj_risk)

        # ---- Shield pre-step (may modify action) ----
        if self.shield is not None:
            # Try calling shield with the most informative signature first.
            try:
                # Some shields may accept ctx_id and eps_override
                # We don't have info yet (until env.step), but ctx_id is injected by the context wrapper
                # *after* env.step; so we get ctx_id by peeking at wrapped env attribute if possible.
                dummy_info: Dict[str, Any] = {}
                ctx_id = self._get_ctx_id_from_env_or_info(dummy_info)
                proposed_action, shield_meta = self.shield.filter_action(  # type: ignore[attr-defined]
                    self.env, action, ctx_id, eps_override=eps_override
                )
            except TypeError:
                # Backward compatibility: filter_action(env, action, ctx_id) or filter_action(env, action)
                try:
                    dummy_info = {}
                    ctx_id = self._get_ctx_id_from_env_or_info(dummy_info)
                    proposed_action, shield_meta = self.shield.filter_action(self.env, action, ctx_id)  # type: ignore[attr-defined]
                except TypeError:
                    try:
                        proposed_action, shield_used, shield_reason = self.shield.filter_action(self.env, action)  # type: ignore[attr-defined]
                        shield_meta = {
                            "shield_used": bool(shield_used),
                            "shield_reason": str(shield_reason),
                        }
                    except Exception:
                        shield_meta = {"shield_used": False, "shield_reason": "shield_error_fallback"}
                        proposed_action = action
            except Exception:
                # Last resort: callable shield interface
                try:
                    out = self.shield(action=action, env=self.env)  # type: ignore[misc]
                    proposed_action = out.get("action", action)
                    shield_meta = {
                        "shield_used": bool(out.get("shield_used", False)),
                        "shield_reason": str(out.get("shield_reason", "")),
                        "eps": out.get("eps", np.nan),
                        "inflate": out.get("inflate", np.nan),
                    }
                except Exception:
                    proposed_action = action
                    shield_meta = {"shield_used": False, "shield_reason": "shield_error_fallback"}

        # ---- Environment step ----
        obs, reward, terminated, truncated, info = self.env.step(proposed_action)
        info = dict(info) if info is not None else {}

        # --- Conformal calibration: update on clearance proxy residual (post-step) ---
        if (not self.no_conformal) and (self.calibrator is not None) and (self.shield is not None):
            try:
                from src.safety import clearance_margin
                cur_d_post = float(clearance_margin(self.env, self.params))
                pred = None
                if hasattr(self.shield, "_predict_clearance_proxy") and np.isfinite(cur_d_pre):
                    try:
                        pred = float(self.shield._predict_clearance_proxy(cur_d_pre, cur_speed_pre, proposed_action, 0))
                    except Exception:
                        pred = None
                if pred is not None and np.isfinite(cur_d_post) and np.isfinite(pred):
                    resid = float(cur_d_post - pred)
                    self.calibrator.update(resid)
                    info["calib_resid"] = resid
                else:
                    info["calib_resid"] = np.nan
            except Exception:
                info["calib_resid"] = np.nan

        # Now we have ctx_id from info (ContextNonstationaryWrapper adds it)
        ctx_id = self._get_ctx_id_from_env_or_info(info)

        # If the shield wants ctx_id but we didn't provide it pre-step, some users prefer
        # a *post-step* diagnostic call. We DO NOT do that here to avoid side effects.
        # (We keep things simple and safe.)

        # Populate shield logging fields
        info["shield_used"] = bool(shield_meta.get("shield_used", False))
        info["shield_reason"] = str(shield_meta.get("shield_reason", ""))

        # Optional fields if the shield provides them
        if "eps" in shield_meta:
            info["eps"] = shield_meta.get("eps", info.get("eps", np.nan))
        if "inflate" in shield_meta:
            info["inflate"] = shield_meta.get("inflate", info.get("inflate", np.nan))

        # --- Define safety violation signal ---
        crashed = bool(info.get("crashed", False) or info.get("collision", False))
        info["violation"] = crashed
        info.setdefault("near_miss", False)

        # --- Attach adjustment-speed diagnostics ---
        info["adj_risk"] = float(self._adj_risk)
        info["adj_unsafe"] = bool(self._adj_unsafe)
        info["adj_s_env"] = float(self._adj_s_env)
        info["adj_s_agent"] = float(self._adj_s_agent)
        info["adj_eps_override"] = float(eps_override) if eps_override is not None else np.nan
        info["ctx_id"] = ctx_id  # ensure consistent even if downstream overwrote

        return obs, reward, terminated, truncated, info



class SoftToHardBudgetWrapper(gym.Wrapper):
    """Soft→Hard budget wrapper (demonstration).

    Goal: Given a trajectory-level (soft) budget C on violations, enforce a per-state (hard)
    admissibility constraint by filtering actions online.

    Soft constraint (episode/window):
        sum_t c_t <= C, where c_t = 1{violation at t}.

    Hard per-state constraint:
        allow action a only if  p_hat(violation | state, action, ctx) <= b_t,
        where b_t is derived from remaining budget: b_t = B_rem / T_rem.

    Notes:
    - Designed for discrete action spaces (action_space.n available). For continuous actions,
      the wrapper logs budget diagnostics but does not search alternatives.
    - Uses an online empirical risk estimator conditioned on clearance bin, action, and context id.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        C: float,
        T: int,
        risk_delta: float = 0.0,
        min_bt: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(env)
        from src.budget import BudgetScheduler
        from src.risk_models import EmpiricalRiskModel, EmpiricalRiskConfig

        self.enabled = bool(enabled)
        self.budget = BudgetScheduler(C=float(C), T=int(T), min_bt=float(min_bt))
        self.risk = EmpiricalRiskModel(EmpiricalRiskConfig())
        self.risk_delta = float(risk_delta)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.budget.reset()
        info = dict(info) if info is not None else {}
        info["budget_bt"] = float(self.budget.bt())
        info["budget_rem"] = float(self.budget.B_rem)
        info["budget_filtered"] = 0
        info["budget_risk"] = np.nan
        info["budget_C"] = float(self.budget.C)
        info["budget_T"] = int(self.budget.T)
        return obs, info

    def _ctx_id(self, info: Dict[str, Any]) -> int:
        try:
            return int(info.get("ctx_id", -1))
        except Exception:
            return -1

    def _get_min_clearance(self) -> float:
        """Approximate clearance to nearest vehicle (meters).

        If the env exposes a more direct clearance metric in info (some shields do),
        this wrapper will use that; otherwise it computes a simple Euclidean distance.
        """
        # Prefer precomputed field if present
        try:
            # Some shields populate info["clearance"] downstream; we do a best-effort read
            # from wrapped env state if available.
            pass
        except Exception:
            pass

        try:
            road = self.env.unwrapped.road
            ego = self.env.unwrapped.vehicle
            min_dist = np.inf
            for veh in getattr(road, "vehicles", []):
                if veh is ego:
                    continue
                dist = float(np.linalg.norm(np.asarray(veh.position) - np.asarray(ego.position)))
                if dist < min_dist:
                    min_dist = dist
            return float(min_dist if np.isfinite(min_dist) else 1e6)
        except Exception:
            return 1e6

    def _risk_cons(self, clearance: float, a: int, ctx: int) -> float:
        return min(1.0, float(self.risk.predict_proba(clearance, int(a), int(ctx))) + self.risk_delta)

    def step(self, action):
        # local budget threshold
        bt = float(self.budget.bt())
        proposed = action
        filtered = False
        chosen_risk = np.nan

        # For discrete actions we can search safe alternatives
        if self.enabled and hasattr(self.action_space, "n"):
            clearance = self._get_min_clearance()
            # ctx_id is injected by ContextNonstationaryWrapper; before env.step it may not be in info,
            # but we can read it after env.step for learning. For filtering we default to -1.
            ctx = -1

            ra = self._risk_cons(clearance, int(proposed), ctx)
            if ra <= bt:
                chosen_risk = ra
            else:
                candidates = []
                for a in range(int(self.action_space.n)):
                    r = self._risk_cons(clearance, a, ctx)
                    if r <= bt:
                        candidates.append((r, a))
                if candidates:
                    chosen_risk, proposed = min(candidates, key=lambda x: x[0])
                    filtered = True
                else:
                    # nothing satisfies bt: choose min-risk action
                    best_r, best_a = min(
                        ((self._risk_cons(clearance, a, ctx), a) for a in range(int(self.action_space.n))),
                        key=lambda x: x[0],
                    )
                    chosen_risk, proposed = best_r, best_a
                    filtered = True

        obs, reward, terminated, truncated, info = self.env.step(proposed)
        info = dict(info) if info is not None else {}

        # --- Conformal calibration: update on clearance proxy residual (post-step) ---
        if (not self.no_conformal) and (self.calibrator is not None) and (self.shield is not None):
            try:
                from src.safety import clearance_margin
                cur_d_post = float(clearance_margin(self.env, self.params))
                pred = None
                if hasattr(self.shield, "_predict_clearance_proxy") and np.isfinite(cur_d_pre):
                    try:
                        pred = float(self.shield._predict_clearance_proxy(cur_d_pre, cur_speed_pre, proposed_action, 0))
                    except Exception:
                        pred = None
                if pred is not None and np.isfinite(cur_d_post) and np.isfinite(pred):
                    resid = float(cur_d_post - pred)
                    self.calibrator.update(resid)
                    info["calib_resid"] = resid
                else:
                    info["calib_resid"] = np.nan
            except Exception:
                info["calib_resid"] = np.nan

        # Observe violation signal (SafetyShieldWrapper defines info["violation"])
        v = int(bool(info.get("violation", False)))

        # Update risk model using realized ctx_id if available
        if self.enabled and hasattr(self.action_space, "n"):
            ctx = self._ctx_id(info)
            clearance = float(info.get("clearance", self._get_min_clearance()))
            self.risk.update(clearance, int(proposed), ctx, v)

        # Update remaining budget
        self.budget.step(v)

        # Attach diagnostics
        info["budget_bt"] = bt
        info["budget_rem"] = float(self.budget.B_rem)
        info["budget_filtered"] = int(filtered)
        info["budget_risk"] = float(chosen_risk) if np.isfinite(chosen_risk) else np.nan
        info["budget_C"] = float(self.budget.C)
        info["budget_T"] = int(self.budget.T)

        return obs, reward, terminated, truncated, info



class FixedKinematicsObsWrapper(gym.ObservationWrapper):
    """
    Force a fixed (K, F) observation by truncating/padding on the first axis.

    This solves SB3 buffer mismatch when highway-env returns different (N, F) depending on
    config/context (e.g., merge defaults to 5 vehicles but contexts use 10).
    """

    def __init__(self, env: gym.Env, K: int = 10):
        super().__init__(env)
        self.K = int(K)

        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 2:
            raise TypeError(f"Expected Box((N,F)) obs space, got {space}")

        _, F = space.shape
        self.F = int(F)

        low = np.full((self.K, self.F), -np.inf, dtype=space.dtype)
        high = np.full((self.K, self.F), np.inf, dtype=space.dtype)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.K, self.F),
            dtype=space.dtype,
        )

    def observation(self, observation):
        obs = np.asarray(observation)
        if obs.ndim != 2:
            return obs

        N, F = obs.shape
        out = np.zeros((self.K, F), dtype=np.float32)
        ncopy = min(N, self.K)
        out[:ncopy] = obs[:ncopy].astype(np.float32, copy=False)
        return out
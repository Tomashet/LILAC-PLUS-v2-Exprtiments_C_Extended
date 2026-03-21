from __future__ import annotations
from stable_baselines3.common.callbacks import BaseCallback


class AdjustSpeedSafetyCallback(BaseCallback):
    """
    Computes:
      - S_env: context shift speed (from info['ctx_id'] by default)
      - S_agent: adaptation speed proxy (policy parameter delta per update)
      - risk, unsafe

    Then pushes (risk, unsafe, s_env, s_agent) into the env wrapper via env_method:
      set_adjustment_risk(...)
    """
    def __init__(
        self,
        shift_estimator,
        adapt_estimator,
        feasibility_monitor,
        *,
        ctx_key: str = "ctx_id",
        env_method_name: str = "set_adjustment_risk",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.shift_est = shift_estimator
        self.adapt_est = adapt_estimator
        self.mon = feasibility_monitor
        self.ctx_key = ctx_key
        self.env_method_name = env_method_name

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos:
            info0 = infos[-1]  # last env in vec
            ctx = info0.get(self.ctx_key, None)
            if ctx is not None:
                self.shift_est.update(ctx)

        s_env = float(self.shift_est.speed())
        s_agent = float(self.adapt_est.speed())
        risk = float(self.mon.risk_score(s_env, s_agent))
        unsafe = bool(self.mon.unsafe(s_env, s_agent))

        # Push state into env wrapper if method exists (VecEnv-safe)
        try:
            self.training_env.env_method(
                self.env_method_name,
                risk=risk,
                unsafe=unsafe,
                s_env=s_env,
                s_agent=s_agent,
            )
        except Exception:
            pass

        # log to SB3 logger
        self.logger.record("adj/s_env", s_env)
        self.logger.record("adj/s_agent", s_agent)
        self.logger.record("adj/risk", risk)
        self.logger.record("adj/unsafe", float(unsafe))
        return True

    def _on_rollout_end(self) -> None:
        # PPO: called each rollout; still okay as a proxy update trigger.
        # For DQN/SAC, this may be less frequent; still produces useful signal.
        try:
            self.adapt_est.on_update(self.model)
        except Exception:
            pass
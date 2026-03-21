# src/lilac/callback.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback

from .modules import (
    EncoderConfig,
    ContextEncoder,
    LatentBayesFilter,
    RewardDecoder,
    kl_diag_gaussians,
)

from src.constraints.base import ContextMetrics
from src.constraints.base import ConstraintPlugin


@dataclass
class LilacConfig:
    latent_dim: int = 8
    context_len: int = 32
    lr: float = 3e-4
    beta_kl: float = 0.2
    beta_kl_anneal_steps: int = 50_000
    lambda_dec: float = 1.0
    train_every_steps: int = 200
    updates_per_train: int = 1
    warmup_episodes: int = 5
    margin: float = 0.0
    temperature: float = 10.0

    # --- Patch 2: change-point + uncertainty-aware risk ---
    changepoint_delta_thresh: float = 0.75
    changepoint_kl_thresh: float = 4.0
    changepoint_cooldown_episodes: int = 2
    var_inflate_on_cp: float = 10.0  # multiply diagonal variance on detected changepoint

    # uncertainty-aware risk weights
    risk_w_shift: float = 1.0
    risk_w_uncert: float = 0.5


class LilacLatentCallback(BaseCallback):
    """
    Trains a LILAC-style latent module online and feeds a per-episode latent z
    into the environment wrapper. Also sets CPSS adjustment-risk based on predicted
    latent shift magnitude.

    This implementation is deliberately simple and stable:
      - Train from per-episode context buffers (not from replay buffer).
      - Prior is an LSTM over episode latents.
      - Decoder predicts reward only (aux loss).
    """
    def __init__(self, cfg: LilacConfig, *, constraint: ConstraintPlugin | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.constraint = constraint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder: Optional[ContextEncoder] = None
        self.prior: Optional[LatentBayesFilter] = None
        self.decoder: Optional[RewardDecoder] = None
        self.opt: Optional[torch.optim.Optimizer] = None

        # Per-env Gaussian belief state (mu, logvar) representing the *prior at episode start*
        self._mu_prior: Optional[torch.Tensor] = None
        self._logvar_prior: Optional[torch.Tensor] = None

        # Last fused posterior (for change-point deltas)
        self._mu_post_prev: Optional[torch.Tensor] = None
        self._logvar_post_prev: Optional[torch.Tensor] = None

        # Change-point cooldown counter per env
        self._cp_cooldown: Optional[np.ndarray] = None

        # The latent actually fed to the policy this episode (mu_pred)
        self._z_exec: Optional[torch.Tensor] = None
        self._episode_count: Optional[np.ndarray] = None

        # pending: tuples (env_idx, ctx[K,feat], mu_prior[latent], logvar_prior[latent])
        self._pending_contexts: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        self._last_train_step: int = 0

    def _init_modules(self) -> None:
        assert self.training_env is not None
        n_envs = int(self.training_env.num_envs)

        # Determine obs_dim/act_dim from env spaces
        obs_dim_aug = int(np.prod(self.training_env.observation_space.shape))
        act_dim = int(np.prod(self.training_env.action_space.shape))
        obs_dim = obs_dim_aug - int(self.cfg.latent_dim)
        if obs_dim <= 0:
            raise RuntimeError(f"obs_dim computed as {obs_dim} (obs_aug={obs_dim_aug}, latent={self.cfg.latent_dim})")

        enc_cfg = EncoderConfig(obs_dim=obs_dim, act_dim=act_dim, latent_dim=self.cfg.latent_dim,
                                context_len=self.cfg.context_len, hidden=256)
        self.encoder = ContextEncoder(enc_cfg).to(self.device)
        # Patch 1: Replace RNN prior with Bayesian fusion filter
        self.prior = LatentBayesFilter(latent_dim=self.cfg.latent_dim, init_process_logvar=-2.0).to(self.device)
        self.decoder = RewardDecoder(obs_dim=obs_dim, act_dim=act_dim, latent_dim=self.cfg.latent_dim, hidden=256).to(self.device)

        params = list(self.encoder.parameters()) + list(self.prior.parameters()) + list(self.decoder.parameters())
        self.opt = torch.optim.Adam(params, lr=float(self.cfg.lr))

        # Initialize per-env belief state
        self._mu_prior = torch.zeros(n_envs, self.cfg.latent_dim, device=self.device)
        self._logvar_prior = torch.zeros(n_envs, self.cfg.latent_dim, device=self.device)
        self._mu_post_prev = torch.zeros(n_envs, self.cfg.latent_dim, device=self.device)
        self._logvar_post_prev = torch.zeros(n_envs, self.cfg.latent_dim, device=self.device)
        self._z_exec = torch.zeros(n_envs, self.cfg.latent_dim, device=self.device)
        self._episode_count = np.zeros(n_envs, dtype=np.int64)

        self._cp_cooldown = np.zeros(n_envs, dtype=np.int64)

        # Initialize env z to zeros
        self.training_env.env_method("set_z", np.zeros((self.cfg.latent_dim,), dtype=np.float32))

    def _beta(self) -> float:
        # simple linear anneal
        t = int(self.num_timesteps)
        if self.cfg.beta_kl_anneal_steps <= 0:
            return float(self.cfg.beta_kl)
        return float(self.cfg.beta_kl) * min(1.0, t / float(self.cfg.beta_kl_anneal_steps))

    def _on_training_start(self) -> None:
        self._init_modules()
        if self.constraint is not None:
            self.constraint.on_training_start(self.model, self.training_env)

    def _maybe_collect_episode_contexts(self, dones: np.ndarray) -> None:
        # For each env where episode ended, pull context from wrapper.
        if dones is None:
            return
        idxs = np.where(dones)[0].tolist()
        if not idxs:
            return
        ctxs = self.training_env.env_method("pop_episode_context", indices=idxs)
        # Snapshot the prior used for this episode (stored in _mu_prior/_logvar_prior)
        mu_prior_np = self._mu_prior.detach().cpu().numpy()
        logvar_prior_np = self._logvar_prior.detach().cpu().numpy()
        for i, ctx in zip(idxs, ctxs):
            if isinstance(ctx, np.ndarray):
                self._pending_contexts.append((int(i), ctx, mu_prior_np[int(i)].copy(), logvar_prior_np[int(i)].copy()))
            self._episode_count[i] += 1

    def _train_latent(self) -> Dict[str, float]:
        assert self.encoder and self.prior and self.decoder and self.opt
        if len(self._pending_contexts) == 0:
            return {}

        # Train on up to a small batch of contexts
        batch = self._pending_contexts[:32]
        self._pending_contexts = self._pending_contexts[32:]

        env_idxs = [b[0] for b in batch]
        ctx = np.stack([b[1] for b in batch], axis=0)  # [B,K,feat]
        mu_prior = torch.from_numpy(np.stack([b[2] for b in batch], axis=0)).float().to(self.device)
        logvar_prior = torch.from_numpy(np.stack([b[3] for b in batch], axis=0)).float().to(self.device)
        ctx_t = torch.from_numpy(ctx).float().to(self.device)
        ctx_flat = ctx_t.reshape(ctx_t.shape[0], -1)

        # Posterior
        mu_q, logvar_q = self.encoder(ctx_flat)
        z_q = ContextEncoder.sample(mu_q, logvar_q)

        # Prior for this episode is the predicted belief at episode start (snapshotted)
        mu_p, logvar_p = mu_prior, logvar_prior

        # Decoder loss: use the LAST transition in context (s,a,r,s') to predict reward
        # context row layout: [s, a, r, sp]
        per_step = ctx_t.shape[-1]
        # split sizes
        # infer dims
        obs_dim = self.encoder.cfg.obs_dim
        act_dim = self.encoder.cfg.act_dim
        last = ctx_t[:, -1, :]
        s = last[:, :obs_dim]
        a = last[:, obs_dim:obs_dim+act_dim]
        r = last[:, obs_dim+act_dim]
        r_hat = self.decoder(s, a, z_q)
        dec_loss = F.mse_loss(r_hat, r)

        kl = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p).mean()
        beta = self._beta()
        loss = self.cfg.lambda_dec * dec_loss + beta * kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.prior.parameters()) + list(self.decoder.parameters()), 5.0)
        self.opt.step()

        # For stability, store latest posterior mean as "previous posterior" for CP deltas
        self._mu_post_prev[env_idxs] = mu_q.detach()
        self._logvar_post_prev[env_idxs] = logvar_q.detach()

        return {
            "lilac/loss": float(loss.item()),
            "lilac/dec_loss": float(dec_loss.item()),
            "lilac/kl": float(kl.item()),
            "lilac/beta": float(beta),
        }

    @torch.no_grad()
    def _update_filter_predict_set(self, env_idx: int, ctx: np.ndarray,
                                  mu_prior_np: np.ndarray, logvar_prior_np: np.ndarray) -> Dict[str, float]:
        """Patch 1+2: Bayesian fusion + change-point aware risk + set next z.

        Inputs are the episode context (for measurement), and the *prior used for
        that episode* (snapshotted at episode start). We:
          1) compute measurement q(z|ctx)
          2) fuse prior + measurement to get posterior
          3) predict prior for next episode
          4) detect change-point (delta or KL spike) and inflate uncertainty
          5) compute uncertainty-aware risk and feed it to CPSS
          6) set z for next episode to predicted mean
        """
        assert self.encoder and self.prior
        # Measurement from this episode
        ctx_t = torch.from_numpy(ctx).float().to(self.device).unsqueeze(0)  # [1,K,feat]
        mu_q, logvar_q = self.encoder(ctx_t.reshape(1, -1))

        mu_p = torch.from_numpy(mu_prior_np).float().to(self.device).unsqueeze(0)
        logvar_p = torch.from_numpy(logvar_prior_np).float().to(self.device).unsqueeze(0)

        # Fuse to posterior
        mu_post, logvar_post = LatentBayesFilter.fuse(mu_p, logvar_p, mu_q, logvar_q)

        # Predict next prior
        mu_pred, logvar_pred = self.prior.predict(mu_post, logvar_post)

        # Change-point detection
        mu_prev = self._mu_post_prev[env_idx:env_idx+1]
        delta = torch.linalg.norm(mu_post - mu_prev, dim=-1).item()
        kl_meas_prior = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p).item()
        is_cp = (delta > float(self.cfg.changepoint_delta_thresh)) or (kl_meas_prior > float(self.cfg.changepoint_kl_thresh))

        # Cooldown handling
        if is_cp:
            self._cp_cooldown[env_idx] = int(self.cfg.changepoint_cooldown_episodes)
        else:
            self._cp_cooldown[env_idx] = max(0, int(self._cp_cooldown[env_idx]) - 1)

        # Inflate variance on CP to reflect uncertainty (helps cautious constraints)
        if is_cp:
            inflate = float(self.cfg.var_inflate_on_cp)
            logvar_pred = torch.clamp(logvar_pred + np.log(max(1e-6, inflate)), -10.0, 5.0)

        # Update stored states
        self._mu_post_prev[env_idx:env_idx+1] = mu_post
        self._logvar_post_prev[env_idx:env_idx+1] = logvar_post
        self._mu_prior[env_idx:env_idx+1] = mu_pred
        self._logvar_prior[env_idx:env_idx+1] = logvar_pred
        self._z_exec[env_idx:env_idx+1] = mu_pred

        # Risk computation: shift + uncertainty
        s_env = torch.linalg.norm(mu_pred - mu_post, dim=-1).item()
        sigma = torch.exp(0.5 * logvar_pred)
        s_uncert = torch.linalg.norm(sigma, dim=-1).item()

        # Agent capacity proxy (keep simple for now)
        s_agent = 0.0

        # Convert to [0,1] risk via logistic
        raw = float(self.cfg.risk_w_shift) * s_env + float(self.cfg.risk_w_uncert) * s_uncert
        margin = float(self.cfg.margin)
        risk = float(1.0 / (1.0 + np.exp(-(raw - margin) / max(1e-6, float(self.cfg.temperature)))))

        unsafe = bool((self._cp_cooldown[env_idx] > 0) or (raw > (s_agent + margin)))

        z_next = mu_pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
        self.training_env.env_method("set_z", z_next, indices=[env_idx])
        # Delegate constraint action to plugin if provided; otherwise use legacy hook.
        metrics = ContextMetrics(
            s_env=float(s_env),
            s_uncert=float(s_uncert),
            s_agent=float(s_agent),
            risk=float(risk),
            cp_flag=bool(is_cp),
            cp_delta=float(delta),
            cp_kl=float(kl_meas_prior),
        )
        if self.constraint is not None:
            try:
                out = self.constraint.on_context_metrics(env_idx, metrics)
            except Exception:
                out = {}
        else:
            out = {}
            try:
                self.training_env.env_method(
                    "set_adjustment_risk",
                    risk=float(risk),
                    unsafe=bool(unsafe),
                    s_env=float(s_env),
                    s_agent=float(s_agent),
                    indices=[env_idx],
                )
            except Exception:
                pass

        return {
            "lilac/s_env": float(s_env),
            "lilac/s_uncert": float(s_uncert),
            "lilac/risk": float(risk),
            "lilac/cp_delta": float(delta),
            "lilac/cp_kl": float(kl_meas_prior),
            "lilac/cp_flag": float(1.0 if is_cp else 0.0),
            "lilac/cp_cooldown": float(self._cp_cooldown[env_idx]),
            **{f"plugin/{k}": float(v) for k, v in out.items() if isinstance(v, (int, float))},
        }

    def _on_step(self) -> bool:
        if self.encoder is None:
            return True

        # Dones array
        dones = None
        if "dones" in self.locals:
            dones = np.array(self.locals["dones"], dtype=bool)
        else:
            # fallback: infer from infos (vector env)
            infos = self.locals.get("infos", [])
            if infos:
                dones = np.array([bool(i.get("terminal_observation") is not None) for i in infos], dtype=bool)

        if dones is not None:
            self._maybe_collect_episode_contexts(dones)
            # Patch 1+2: set next z and risk at episode boundary using Bayesian fusion
            for env_i in np.where(dones)[0].tolist():
                env_i = int(env_i)
                # Find the latest pending context for this env (it was just appended)
                # We search from the end to avoid mixing multiple finished episodes.
                ctx_tuple = None
                for j in range(len(self._pending_contexts) - 1, -1, -1):
                    if self._pending_contexts[j][0] == env_i:
                        ctx_tuple = self._pending_contexts[j]
                        break
                if ctx_tuple is not None and int(self._episode_count[env_i]) >= int(self.cfg.warmup_episodes):
                    _, ctx, mu_p_np, logvar_p_np = ctx_tuple
                    stats = self._update_filter_predict_set(env_i, ctx, mu_p_np, logvar_p_np)
                    for k, v in stats.items():
                        self.logger.record(k, v)

        # train periodically
        if (int(self.num_timesteps) - int(self._last_train_step)) >= int(self.cfg.train_every_steps):
            for _ in range(int(self.cfg.updates_per_train)):
                stats = self._train_latent()
                for k,v in stats.items():
                    self.logger.record(k, v)
            self._last_train_step = int(self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        # Allow constraint plugins to update internal adaptation-speed proxies.
        if self.constraint is not None:
            try:
                self.constraint.on_rollout_end()
            except Exception:
                pass

# src/lilac/modules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class EncoderConfig:
    obs_dim: int
    act_dim: int
    latent_dim: int = 8
    context_len: int = 32
    hidden: int = 256


class ContextEncoder(nn.Module):
    """
    q(z | context) where context is a short window of transitions.
    We keep it simple and stable:
      - Flatten [K, (s,a,r,sp)] into one vector.
      - Output Gaussian params (mu, logvar).
    """
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        per_step = cfg.obs_dim + cfg.act_dim + 1 + cfg.obs_dim
        in_dim = cfg.context_len * per_step
        self.net = _mlp(in_dim, cfg.hidden, 2 * cfg.latent_dim)

    def forward(self, ctx_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(ctx_flat)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar

    @staticmethod
    def sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class LatentPriorLSTM(nn.Module):
    """
    p(z_{i+1} | z_{1:i}) implemented as an LSTM over episode latents.
    We output Gaussian params (mu, logvar) for the next episode latent.
    """
    def __init__(self, latent_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.lstm = nn.LSTMCell(self.latent_dim, self.hidden)
        self.head = nn.Linear(self.hidden, 2 * self.latent_dim)

    def init_state(self, batch: int, device: torch.device):
        h = torch.zeros(batch, self.hidden, device=device)
        c = torch.zeros(batch, self.hidden, device=device)
        return h, c

    def step(self, z_prev: torch.Tensor, state):
        h, c = self.lstm(z_prev, state)
        out = self.head(h)
        mu, logvar = torch.chunk(out, 2, dim=-1)
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar, (h, c)


class LatentBayesFilter(nn.Module):
    """A lightweight, online-friendly latent dynamics model.

    This module implements a *random-walk* Gaussian prior with diagonal covariance
    and a closed-form Bayesian fusion (precision-weighted) update.

    It is intended to replace an RNN/LSTM prior when we care about:
      - low latency
      - stability under abrupt regime changes
      - convenience for online deployment

    State is maintained externally (per-env) as (mu, logvar).

    Predict step:
        mu_pred = mu
        logvar_pred = logvar + process_logvar

    Update step (fusion of prior and measurement):
        p(z)  = N(mu_p, diag(var_p))
        q(z)  = N(mu_q, diag(var_q))
        post  = N(mu,   diag(var))
        var   = (var_p^{-1} + var_q^{-1})^{-1}
        mu    = var * (mu_p/var_p + mu_q/var_q)

    We parameterize the (diagonal) process noise as a learnable vector so the
    model can adapt its smoothness to the domain.
    """

    def __init__(self, latent_dim: int = 8, init_process_logvar: float = -2.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.process_logvar = nn.Parameter(torch.full((self.latent_dim,), float(init_process_logvar)))

    def predict(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logvar_pred = torch.clamp(logvar + self.process_logvar, -10.0, 5.0)
        return mu, logvar_pred

    @staticmethod
    def fuse(mu_p: torch.Tensor, logvar_p: torch.Tensor,
             mu_q: torch.Tensor, logvar_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precision-weighted fusion of two diagonal Gaussians."""
        var_p = torch.exp(logvar_p)
        var_q = torch.exp(logvar_q)
        prec_p = 1.0 / (var_p + 1e-8)
        prec_q = 1.0 / (var_q + 1e-8)
        var = 1.0 / (prec_p + prec_q + 1e-8)
        mu = var * (mu_p * prec_p + mu_q * prec_q)
        logvar = torch.log(var + 1e-8)
        logvar = torch.clamp(logvar, -10.0, 5.0)
        return mu, logvar


class RewardDecoder(nn.Module):
    """Auxiliary decoder: predict reward from (s,a,z)."""
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim + latent_dim, hidden, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a, z], dim=-1)
        return self.net(x).squeeze(-1)


def kl_diag_gaussians(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                      mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    KL(q||p) for diagonal Gaussians.
    Returns per-sample KL (shape [B]).
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    ).sum(dim=-1)
    return kl

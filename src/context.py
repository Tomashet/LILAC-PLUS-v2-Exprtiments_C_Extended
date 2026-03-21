# src/context.py

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ContextKey = Tuple[Any, ...]


# ============================================================
# Context utilities
# ============================================================

def canonicalize_context(ctx: Any) -> ContextKey:
    if isinstance(ctx, tuple):
        return ctx
    if isinstance(ctx, list):
        return tuple(ctx)

    if isinstance(ctx, str):
        s = ctx.strip()

        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (tuple, list)):
                    return tuple(val)
            except Exception:
                pass

        if "|" in s:
            return tuple(part.strip() for part in s.split("|"))

        if "," in s:
            return tuple(part.strip() for part in s.split(","))

        return (s,)

    return (ctx,)


def context_to_str(ctx: Any) -> str:
    return str(canonicalize_context(ctx))


# ============================================================
# Threshold record
# ============================================================

@dataclass
class ThresholdRecord:
    tau_violation: float
    tau_near_miss: float

    @classmethod
    def from_any(cls, x: Any) -> "ThresholdRecord":
        if isinstance(x, ThresholdRecord):
            return x

        if isinstance(x, Mapping):
            tv = x.get("tau_violation", x.get("violation", x.get("tau_v")))
            tn = x.get("tau_near_miss", x.get("near_miss", x.get("tau_nm")))
            if tv is None or tn is None:
                raise ValueError(f"Cannot parse threshold record from mapping: {x}")
            return cls(float(tv), float(tn))

        if isinstance(x, (tuple, list)) and len(x) >= 2:
            return cls(float(x[0]), float(x[1]))

        raise ValueError(f"Unsupported threshold record format: {x}")


# ============================================================
# Patch I/O
# ============================================================

def _normalize_patch(raw: Mapping[Any, Any]) -> Dict[ContextKey, ThresholdRecord]:
    out: Dict[ContextKey, ThresholdRecord] = {}
    for k, v in raw.items():
        out[canonicalize_context(k)] = ThresholdRecord.from_any(v)
    return out


def load_threshold_patch(path: Optional[str | Path]) -> Dict[ContextKey, ThresholdRecord]:
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return _normalize_patch(raw)


def save_threshold_patch(
    patch: Mapping[Any, Any],
    path: str | Path,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable: Dict[str, Dict[str, float]] = {}
    for ctx, rec in _normalize_patch(patch).items():
        serializable[str(ctx)] = {
            "tau_violation": float(rec.tau_violation),
            "tau_near_miss": float(rec.tau_near_miss),
        }

    with p.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# ============================================================
# Similarity
# ============================================================

@dataclass
class SimilarityConfig:
    slot_weights: Optional[Sequence[float]] = None
    exact_match_bonus: float = 1e-6

    def weights_for(self, n: int) -> List[float]:
        if self.slot_weights is None:
            return [1.0] * n
        w = list(self.slot_weights)
        if len(w) < n:
            w.extend([1.0] * (n - len(w)))
        return w[:n]


def context_similarity(
    a: ContextKey,
    b: ContextKey,
    cfg: Optional[SimilarityConfig] = None,
) -> float:
    a = canonicalize_context(a)
    b = canonicalize_context(b)

    n = max(len(a), len(b))
    aa = list(a) + [None] * (n - len(a))
    bb = list(b) + [None] * (n - len(b))

    cfg = cfg or SimilarityConfig()
    weights = cfg.weights_for(n)
    denom = max(sum(weights), 1e-12)

    score = 0.0
    for x, y, w in zip(aa, bb, weights):
        if x == y:
            score += w

    sim = score / denom
    if a == b:
        sim += cfg.exact_match_bonus

    return float(min(1.0, max(0.0, sim)))


# ============================================================
# Helpers
# ============================================================

def _clip(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return float(min(max(x, lo), hi))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    sw = float(sum(weights))
    if sw <= 0:
        return float(sum(values) / max(1, len(values)))
    return float(sum(v * w for v, w in zip(values, weights)) / sw)


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    q: float,
) -> float:
    if not values:
        raise ValueError("No values passed to _weighted_quantile")
    if len(values) != len(weights):
        raise ValueError("values and weights length mismatch")

    q = min(1.0, max(0.0, q))
    pairs = sorted(zip(values, weights), key=lambda x: x[0])

    total_w = sum(max(0.0, w) for _, w in pairs)
    if total_w <= 0:
        idx = int(round((len(values) - 1) * q))
        return float(sorted(values)[idx])

    target = q * total_w
    csum = 0.0
    for val, w in pairs:
        csum += max(0.0, w)
        if csum >= target:
            return float(val)

    return float(pairs[-1][0])


# ============================================================
# Online stats for future meta-learning
# ============================================================

@dataclass
class RunningStat:
    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0
    ewma: Optional[float] = None
    alpha: float = 0.05

    def update(self, x: float, weight: float = 1.0) -> None:
        if weight <= 0:
            return

        prev_count = self.count
        self.count += weight

        delta = x - self.mean
        self.mean += (weight / self.count) * delta
        delta2 = x - self.mean
        self.m2 += weight * delta * delta2

        if self.ewma is None:
            self.ewma = float(x)
        else:
            self.ewma = (1.0 - self.alpha) * self.ewma + self.alpha * float(x)

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return max(0.0, self.m2 / max(1.0, self.count - 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class OnlineThresholdStats:
    violation: Dict[ContextKey, RunningStat] = field(default_factory=dict)
    near_miss: Dict[ContextKey, RunningStat] = field(default_factory=dict)

    def update(
        self,
        context: Any,
        violation_value: Optional[float] = None,
        near_miss_value: Optional[float] = None,
        weight: float = 1.0,
    ) -> None:
        ctx = canonicalize_context(context)

        if violation_value is not None:
            self.violation.setdefault(ctx, RunningStat()).update(float(violation_value), weight=weight)

        if near_miss_value is not None:
            self.near_miss.setdefault(ctx, RunningStat()).update(float(near_miss_value), weight=weight)

    def get_bias(self, context: Any) -> Dict[str, Optional[float]]:
        ctx = canonicalize_context(context)
        v = self.violation.get(ctx)
        n = self.near_miss.get(ctx)
        return {
            "violation_anchor": None if v is None else v.ewma,
            "near_miss_anchor": None if n is None else n.ewma,
            "violation_count": 0.0 if v is None else v.count,
            "near_miss_count": 0.0 if n is None else n.count,
        }


# ============================================================
# Inference config
# ============================================================

@dataclass
class BoundedInferenceConfig:
    top_k: int = 5
    min_similarity: float = 0.20

    violation_quantile: float = 0.35
    near_miss_quantile: float = 0.40
    mean_blend: float = 0.25

    # Conservative cold-start priors
    prior_tau_violation: float = 2.0
    prior_tau_near_miss: float = 8.0

    # Caps
    min_tau_violation: float = 1e-6
    min_tau_near_miss: float = 1e-6
    max_tau_violation: Optional[float] = None
    max_tau_near_miss: Optional[float] = None

    # Online adaptation hook
    online_anchor_blend: float = 0.10
    min_online_samples: float = 10.0

    verbose: bool = True


def make_safe_inference_config(
    max_tau_violation: Optional[float] = 10.0,
    max_tau_near_miss: Optional[float] = 40.0,
    verbose: bool = True,
) -> BoundedInferenceConfig:
    return BoundedInferenceConfig(
        top_k=5,
        min_similarity=0.20,
        violation_quantile=0.35,
        near_miss_quantile=0.40,
        mean_blend=0.25,
        prior_tau_violation=2.0,
        prior_tau_near_miss=8.0,
        min_tau_violation=1e-6,
        min_tau_near_miss=1e-6,
        max_tau_violation=max_tau_violation,
        max_tau_near_miss=max_tau_near_miss,
        online_anchor_blend=0.10,
        min_online_samples=10.0,
        verbose=verbose,
    )


# ============================================================
# Inferer
# ============================================================

class BoundedSimilarityInferer:
    def __init__(
        self,
        patch: Mapping[Any, Any],
        similarity_cfg: Optional[SimilarityConfig] = None,
        inference_cfg: Optional[BoundedInferenceConfig] = None,
        online_stats: Optional[OnlineThresholdStats] = None,
    ) -> None:
        self.patch = _normalize_patch(patch)
        self.similarity_cfg = similarity_cfg or SimilarityConfig()
        self.cfg = inference_cfg or BoundedInferenceConfig()
        self.online_stats = online_stats

    def _cold_start_prior(self, metric: str) -> float:
        if metric == "tau_violation":
            val = self.cfg.prior_tau_violation
            val = max(val, self.cfg.min_tau_violation)
            if self.cfg.max_tau_violation is not None:
                val = min(val, self.cfg.max_tau_violation)
            return float(val)

        val = self.cfg.prior_tau_near_miss
        val = max(val, self.cfg.min_tau_near_miss)
        if self.cfg.max_tau_near_miss is not None:
            val = min(val, self.cfg.max_tau_near_miss)
        return float(val)

    def get(self, context: Any) -> ThresholdRecord:
        ctx = canonicalize_context(context)

        if ctx in self.patch:
            rec = self.patch[ctx]
            if self.cfg.verbose:
                print(
                    f"[context.py] Using exact thresholds for {ctx}: "
                    f"tau_violation={rec.tau_violation:.3f}, "
                    f"tau_near_miss={rec.tau_near_miss:.3f}"
                )
            return rec

        rec = ThresholdRecord(
            tau_violation=self._infer_one(ctx, metric="tau_violation"),
            tau_near_miss=self._infer_one(ctx, metric="tau_near_miss"),
        )

        if rec.tau_near_miss < rec.tau_violation:
            rec.tau_near_miss = rec.tau_violation

        if self.cfg.verbose:
            print(
                f"[context.py] Using inferred thresholds for {ctx}: "
                f"tau_violation={rec.tau_violation:.3f}, "
                f"tau_near_miss={rec.tau_near_miss:.3f}"
            )

        return rec

    def _neighbors(self, ctx: ContextKey) -> List[Tuple[float, ContextKey, ThresholdRecord]]:
        out: List[Tuple[float, ContextKey, ThresholdRecord]] = []
        for k, rec in self.patch.items():
            sim = context_similarity(ctx, k, self.similarity_cfg)
            if sim >= self.cfg.min_similarity:
                out.append((sim, k, rec))
        out.sort(key=lambda x: x[0], reverse=True)
        return out[: self.cfg.top_k]

    def _infer_one(self, ctx: ContextKey, metric: str) -> float:
        prior = self._cold_start_prior(metric)
        neighbors = self._neighbors(ctx)

        # No known contexts at all, or no close neighbors: use conservative prior
        if not neighbors:
            return prior

        values = [float(getattr(rec, metric)) for _, _, rec in neighbors]
        weights = [float(sim) for sim, _, _ in neighbors]

        q = self.cfg.violation_quantile if metric == "tau_violation" else self.cfg.near_miss_quantile
        q_est = _weighted_quantile(values, weights, q)
        mean_est = _weighted_mean(values, weights)
        est = (1.0 - self.cfg.mean_blend) * q_est + self.cfg.mean_blend * mean_est

        # Confidence-controlled shrinkage toward prior
        avg_similarity = sum(weights) / max(1, len(weights))
        neighbor_fraction = min(1.0, len(neighbors) / max(1.0, float(self.cfg.top_k)))
        confidence = min(1.0, avg_similarity * neighbor_fraction * 1.5)

        est = confidence * est + (1.0 - confidence) * prior

        # Gentle online adjustment hook
        if self.online_stats is not None:
            bias = self.online_stats.get_bias(ctx)
            if metric == "tau_violation":
                online_anchor = bias.get("violation_anchor")
                online_count = float(bias.get("violation_count", 0.0) or 0.0)
            else:
                online_anchor = bias.get("near_miss_anchor")
                online_count = float(bias.get("near_miss_count", 0.0) or 0.0)

            if online_anchor is not None and online_count >= self.cfg.min_online_samples:
                est = (
                    (1.0 - self.cfg.online_anchor_blend) * est
                    + self.cfg.online_anchor_blend * float(online_anchor)
                )

        if metric == "tau_violation":
            est = max(est, self.cfg.min_tau_violation)
            if self.cfg.max_tau_violation is not None:
                est = min(est, self.cfg.max_tau_violation)
        else:
            est = max(est, self.cfg.min_tau_near_miss)
            if self.cfg.max_tau_near_miss is not None:
                est = min(est, self.cfg.max_tau_near_miss)

        return float(est)


# ============================================================
# Public API
# ============================================================

def infer_thresholds_for_context(
    context: Any,
    threshold_patch: Mapping[Any, Any],
    similarity_cfg: Optional[SimilarityConfig] = None,
    inference_cfg: Optional[BoundedInferenceConfig] = None,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> ThresholdRecord:
    inferer = BoundedSimilarityInferer(
        patch=threshold_patch,
        similarity_cfg=similarity_cfg,
        inference_cfg=inference_cfg,
        online_stats=online_stats,
    )
    return inferer.get(context)


def get_context_thresholds(
    context: Any,
    threshold_patch: Mapping[Any, Any],
    similarity_cfg: Optional[SimilarityConfig] = None,
    inference_cfg: Optional[BoundedInferenceConfig] = None,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> Dict[str, float]:
    rec = infer_thresholds_for_context(
        context=context,
        threshold_patch=threshold_patch,
        similarity_cfg=similarity_cfg,
        inference_cfg=inference_cfg,
        online_stats=online_stats,
    )
    return {
        "tau_violation": float(rec.tau_violation),
        "tau_near_miss": float(rec.tau_near_miss),
    }


get_thresholds = get_context_thresholds
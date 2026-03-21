# src/common.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.context import (
    BoundedInferenceConfig,
    OnlineThresholdStats,
    SimilarityConfig,
    canonicalize_context,
    get_context_thresholds,
    load_threshold_patch,
    make_safe_inference_config,
)


def load_json(path: Optional[str | Path], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path is None:
        return {} if default is None else default

    p = Path(path)
    if not p.exists():
        return {} if default is None else default

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_threshold_patch(threshold_patch_path: Optional[str | Path]) -> Dict[Any, Any]:
    return load_threshold_patch(threshold_patch_path)


def build_similarity_config(cfg_dict: Optional[Mapping[str, Any]] = None) -> SimilarityConfig:
    cfg_dict = dict(cfg_dict or {})
    return SimilarityConfig(
        slot_weights=cfg_dict.get("slot_weights"),
        exact_match_bonus=float(cfg_dict.get("exact_match_bonus", 1e-6)),
    )


def build_bounded_inference_config(cfg_dict: Optional[Mapping[str, Any]] = None) -> BoundedInferenceConfig:
    cfg_dict = dict(cfg_dict or {})

    base = make_safe_inference_config(
        max_tau_violation=cfg_dict.get("max_tau_violation", 10.0),
        max_tau_near_miss=cfg_dict.get("max_tau_near_miss", 40.0),
        verbose=bool(cfg_dict.get("verbose", True)),
    )

    for key, value in cfg_dict.items():
        if hasattr(base, key):
            setattr(base, key, value)

    return base


def resolve_context_thresholds(
    context: Any,
    threshold_patch: Mapping[Any, Any],
    similarity_cfg_dict: Optional[Mapping[str, Any]] = None,
    bounded_cfg_dict: Optional[Mapping[str, Any]] = None,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> Dict[str, float]:
    similarity_cfg = build_similarity_config(similarity_cfg_dict)
    inference_cfg = build_bounded_inference_config(bounded_cfg_dict)

    thresholds = get_context_thresholds(
        context=context,
        threshold_patch=threshold_patch,
        similarity_cfg=similarity_cfg,
        inference_cfg=inference_cfg,
        online_stats=online_stats,
    )

    if thresholds["tau_near_miss"] < thresholds["tau_violation"]:
        thresholds["tau_near_miss"] = thresholds["tau_violation"]

    return thresholds


def attach_thresholds_to_env(
    env: Any,
    context: Any,
    threshold_patch: Mapping[Any, Any],
    similarity_cfg_dict: Optional[Mapping[str, Any]] = None,
    bounded_cfg_dict: Optional[Mapping[str, Any]] = None,
    online_stats: Optional[OnlineThresholdStats] = None,
) -> Dict[str, float]:
    thresholds = resolve_context_thresholds(
        context=context,
        threshold_patch=threshold_patch,
        similarity_cfg_dict=similarity_cfg_dict,
        bounded_cfg_dict=bounded_cfg_dict,
        online_stats=online_stats,
    )

    if hasattr(env, "set_context_thresholds"):
        env.set_context_thresholds(
            tau_violation=thresholds["tau_violation"],
            tau_near_miss=thresholds["tau_near_miss"],
        )
    elif hasattr(env, "set_thresholds"):
        env.set_thresholds(
            tau_violation=thresholds["tau_violation"],
            tau_near_miss=thresholds["tau_near_miss"],
        )
    else:
        # last-resort plain attributes
        setattr(env, "tau_violation", thresholds["tau_violation"])
        setattr(env, "tau_near_miss", thresholds["tau_near_miss"])

    setattr(env, "resolved_context_key", canonicalize_context(context))
    setattr(env, "resolved_tau_violation", thresholds["tau_violation"])
    setattr(env, "resolved_tau_near_miss", thresholds["tau_near_miss"])

    return thresholds
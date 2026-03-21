# scripts/audit_threshold_inference.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.context import (
    BoundedInferenceConfig,
    BoundedSimilarityInferer,
    SimilarityConfig,
    ThresholdRecord,
    canonicalize_context,
    context_similarity,
    load_threshold_patch,
    make_safe_inference_config,
)


ContextKey = Tuple[Any, ...]


# ============================================================
# Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact vs inferred thresholds for contexts."
    )

    parser.add_argument(
        "--threshold_patch",
        type=str,
        default=None,
        help="Path to calibrated threshold patch JSON.",
    )

    parser.add_argument(
        "--contexts_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a list of contexts. "
            'Example: [["low","calm","clean"], ["high","aggr","dropout"]]'
        ),
    )

    parser.add_argument(
        "--context_density",
        type=str,
        default=None,
        help="Single context density for one-off audit.",
    )
    parser.add_argument(
        "--context_behavior",
        type=str,
        default=None,
        help="Single context behavior for one-off audit.",
    )
    parser.add_argument(
        "--context_sensor",
        type=str,
        default=None,
        help="Single context sensor condition for one-off audit.",
    )

    parser.add_argument(
        "--slot_weights",
        type=str,
        default=None,
        help='Comma-separated slot weights, e.g. "1.0,1.0,1.0"',
    )

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_similarity", type=float, default=0.20)
    parser.add_argument("--max_tau_violation", type=float, default=10.0)
    parser.add_argument("--max_tau_near_miss", type=float, default=40.0)

    parser.add_argument(
        "--prior_tau_violation",
        type=float,
        default=2.0,
        help="Cold-start prior for tau_violation.",
    )
    parser.add_argument(
        "--prior_tau_near_miss",
        type=float,
        default=8.0,
        help="Cold-start prior for tau_near_miss.",
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional path to save audit rows as CSV.",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def parse_slot_weights(text: Optional[str]) -> Optional[List[float]]:
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",")]


def load_contexts_from_json(path: str) -> List[ContextKey]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("contexts_json must contain a JSON list.")

    contexts: List[ContextKey] = []
    for item in raw:
        contexts.append(canonicalize_context(item))
    return contexts


def build_single_context(args: argparse.Namespace) -> Optional[ContextKey]:
    vals = [args.context_density, args.context_behavior, args.context_sensor]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise ValueError(
            "If using single-context mode, provide all of "
            "--context_density, --context_behavior, and --context_sensor."
        )
    return canonicalize_context((args.context_density, args.context_behavior, args.context_sensor))


def get_default_demo_contexts() -> List[ContextKey]:
    return [
        ("low", "calm", "clean"),
        ("low", "aggr", "clean"),
        ("medium", "calm", "dropout"),
        ("high", "aggr", "dropout"),
    ]


def determine_source(
    ctx: ContextKey,
    patch: Dict[ContextKey, ThresholdRecord],
    neighbors: List[Tuple[float, ContextKey, ThresholdRecord]],
) -> str:
    if ctx in patch:
        return "exact"
    if neighbors:
        return "inferred_neighbors"
    return "cold_start_prior"


# ============================================================
# Audit logic
# ============================================================

def get_neighbors_for_display(
    ctx: ContextKey,
    patch: Dict[ContextKey, ThresholdRecord],
    similarity_cfg: SimilarityConfig,
    top_k: int,
    min_similarity: float,
) -> List[Tuple[float, ContextKey, ThresholdRecord]]:
    rows: List[Tuple[float, ContextKey, ThresholdRecord]] = []
    for k, rec in patch.items():
        sim = context_similarity(ctx, k, similarity_cfg)
        if sim >= min_similarity:
            rows.append((sim, k, rec))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top_k]


def format_context(ctx: ContextKey) -> str:
    return str(tuple(ctx))


def audit_one_context(
    ctx: ContextKey,
    inferer: BoundedSimilarityInferer,
    patch: Dict[ContextKey, ThresholdRecord],
    similarity_cfg: SimilarityConfig,
    inference_cfg: BoundedInferenceConfig,
) -> Dict[str, Any]:
    neighbors = get_neighbors_for_display(
        ctx=ctx,
        patch=patch,
        similarity_cfg=similarity_cfg,
        top_k=inference_cfg.top_k,
        min_similarity=inference_cfg.min_similarity,
    )

    rec = inferer.get(ctx)
    source = determine_source(ctx, patch, neighbors)

    avg_neighbor_similarity = (
        sum(sim for sim, _, _ in neighbors) / len(neighbors) if neighbors else 0.0
    )

    row: Dict[str, Any] = {
        "context": format_context(ctx),
        "source": source,
        "tau_violation": float(rec.tau_violation),
        "tau_near_miss": float(rec.tau_near_miss),
        "num_neighbors": len(neighbors),
        "avg_neighbor_similarity": float(avg_neighbor_similarity),
        "prior_tau_violation": float(inference_cfg.prior_tau_violation),
        "prior_tau_near_miss": float(inference_cfg.prior_tau_near_miss),
    }

    for i in range(inference_cfg.top_k):
        if i < len(neighbors):
            sim, nctx, nrec = neighbors[i]
            row[f"neighbor_{i+1}_context"] = format_context(nctx)
            row[f"neighbor_{i+1}_similarity"] = float(sim)
            row[f"neighbor_{i+1}_tau_violation"] = float(nrec.tau_violation)
            row[f"neighbor_{i+1}_tau_near_miss"] = float(nrec.tau_near_miss)
        else:
            row[f"neighbor_{i+1}_context"] = ""
            row[f"neighbor_{i+1}_similarity"] = ""
            row[f"neighbor_{i+1}_tau_violation"] = ""
            row[f"neighbor_{i+1}_tau_near_miss"] = ""

    return row


def print_audit_row(row: Dict[str, Any], top_k: int) -> None:
    print("=" * 80)
    print(f"Context:   {row['context']}")
    print(f"Source:    {row['source']}")
    print(
        f"Resolved:  tau_violation={row['tau_violation']:.3f}, "
        f"tau_near_miss={row['tau_near_miss']:.3f}"
    )
    print(
        f"Prior:     tau_violation={row['prior_tau_violation']:.3f}, "
        f"tau_near_miss={row['prior_tau_near_miss']:.3f}"
    )
    print(
        f"Neighbors: {row['num_neighbors']} "
        f"(avg_similarity={row['avg_neighbor_similarity']:.3f})"
    )

    for i in range(top_k):
        c = row.get(f"neighbor_{i+1}_context", "")
        if c == "":
            continue
        s = row.get(f"neighbor_{i+1}_similarity", "")
        tv = row.get(f"neighbor_{i+1}_tau_violation", "")
        tn = row.get(f"neighbor_{i+1}_tau_near_miss", "")
        print(
            f"  #{i+1}: {c} | sim={float(s):.3f} | "
            f"tau_violation={float(tv):.3f}, tau_near_miss={float(tn):.3f}"
        )


def save_rows_to_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    patch = load_threshold_patch(args.threshold_patch)

    similarity_cfg = SimilarityConfig(
        slot_weights=parse_slot_weights(args.slot_weights),
        exact_match_bonus=1e-6,
    )

    inference_cfg = make_safe_inference_config(
        max_tau_violation=args.max_tau_violation,
        max_tau_near_miss=args.max_tau_near_miss,
        verbose=False,
    )
    inference_cfg.top_k = args.top_k
    inference_cfg.min_similarity = args.min_similarity
    inference_cfg.prior_tau_violation = args.prior_tau_violation
    inference_cfg.prior_tau_near_miss = args.prior_tau_near_miss

    inferer = BoundedSimilarityInferer(
        patch=patch,
        similarity_cfg=similarity_cfg,
        inference_cfg=inference_cfg,
        online_stats=None,
    )

    contexts: List[ContextKey] = []

    single_ctx = build_single_context(args)
    if single_ctx is not None:
        contexts.append(single_ctx)

    if args.contexts_json is not None:
        contexts.extend(load_contexts_from_json(args.contexts_json))

    if not contexts:
        contexts = get_default_demo_contexts()

    # deduplicate while preserving order
    seen = set()
    deduped: List[ContextKey] = []
    for ctx in contexts:
        cctx = canonicalize_context(ctx)
        if cctx not in seen:
            seen.add(cctx)
            deduped.append(cctx)
    contexts = deduped

    print("\n=== THRESHOLD INFERENCE AUDIT ===")
    print(f"threshold_patch: {args.threshold_patch}")
    print(f"num_patch_contexts: {len(patch)}")
    print(f"contexts_to_audit: {len(contexts)}")
    print(f"top_k: {inference_cfg.top_k}")
    print(f"min_similarity: {inference_cfg.min_similarity}")
    print(
        f"priors: tau_violation={inference_cfg.prior_tau_violation}, "
        f"tau_near_miss={inference_cfg.prior_tau_near_miss}"
    )
    print("=================================\n")

    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        row = audit_one_context(
            ctx=ctx,
            inferer=inferer,
            patch=patch,
            similarity_cfg=similarity_cfg,
            inference_cfg=inference_cfg,
        )
        rows.append(row)
        print_audit_row(row, inference_cfg.top_k)

    if args.out_csv:
        save_rows_to_csv(rows, args.out_csv)
        print(f"\nSaved audit CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
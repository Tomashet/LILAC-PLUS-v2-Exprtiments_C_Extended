#!/usr/bin/env python3
"""
Generate Stage B thesis figures from stageB_grouped_summary.csv.

Supports both grouped-summary schemas:
1) nested schema, e.g. violation_count_mean_mean / violation_count_mean_std
2) flat schema, e.g. violation_count_mean / violation_count_std
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "adjust_speed_only",
    "lilac_context",
    "lilac_speed",
    "lilac_soft2hard",
]

REGIME_ORDER = [
    "stationary",
    "nonstationary_seen",
    "nonstationary_unseen",
]

METHOD_LABELS = {
    "adjust_speed_only": "Adjust Speed Only",
    "lilac_context": "LiLAC Context",
    "lilac_speed": "LiLAC Speed",
    "lilac_soft2hard": "LiLAC Soft→Hard",
}

REGIME_LABELS = {
    "stationary": "Stationary",
    "nonstationary_seen": "Nonstationary Seen",
    "nonstationary_unseen": "Nonstationary Unseen",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="artifacts/stageB_summary/stageB_grouped_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/stageB_figures",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {path}")

    if "method" not in df.columns or "regime" not in df.columns:
        raise ValueError("Summary CSV must contain 'method' and 'regime' columns.")

    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    return df.sort_values(["method", "regime"]).reset_index(drop=True)


def resolve_cols(df: pd.DataFrame, base: str) -> tuple[str, str | None]:
    """
    Resolve a metric base name to (value_col, error_col).

    Supports:
    - nested schema: base_mean / base_std where base is already e.g. 'violation_count_mean'
      => violation_count_mean_mean / violation_count_mean_std
    - flat schema:   base / corresponding std
      => violation_count_mean / violation_count_std
    """
    candidates = [
        (f"{base}_mean", f"{base}_std"),
        (base, base.replace("_mean", "_std") if "_mean" in base else f"{base}_std"),
    ]

    for value_col, error_col in candidates:
        if value_col in df.columns:
            return value_col, error_col if error_col in df.columns else None

    raise KeyError(f"Could not resolve metric columns for base '{base}'. Available columns: {list(df.columns)}")


def grouped_bar_plot(
    df: pd.DataFrame,
    metric_base: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    value_col, error_col = resolve_cols(df, metric_base)

    x = np.arange(len(METHOD_ORDER))
    width = 0.24
    offsets = np.linspace(-width, width, len(REGIME_ORDER))

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, regime in enumerate(REGIME_ORDER):
        sub = df[df["regime"] == regime].set_index("method").reindex(METHOD_ORDER)

        vals = pd.to_numeric(sub[value_col], errors="coerce").values
        if error_col is not None and error_col in sub.columns:
            errs = pd.to_numeric(sub[error_col], errors="coerce").fillna(0.0).values
        else:
            errs = np.zeros(len(sub))

        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            yerr=errs,
            capsize=4,
            label=REGIME_LABELS[regime],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def scatter_reward_vs_violations(df: pd.DataFrame, out_path: Path) -> None:
    reward_col, _ = resolve_cols(df, "r_mean")
    viol_col, _ = resolve_cols(df, "violation_count_mean")

    fig, ax = plt.subplots(figsize=(9, 7))

    markers = {
        "stationary": "o",
        "nonstationary_seen": "s",
        "nonstationary_unseen": "^",
    }

    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime].copy()

        x = pd.to_numeric(sub[viol_col], errors="coerce")
        y = pd.to_numeric(sub[reward_col], errors="coerce")

        ax.scatter(
            x,
            y,
            marker=markers[regime],
            s=110,
            label=REGIME_LABELS[regime],
        )

        for _, row in sub.iterrows():
            xv = pd.to_numeric(pd.Series([row[viol_col]]), errors="coerce").iloc[0]
            yv = pd.to_numeric(pd.Series([row[reward_col]]), errors="coerce").iloc[0]
            if pd.isna(xv) or pd.isna(yv):
                continue
            ax.annotate(
                METHOD_LABELS.get(row["method"], str(row["method"])),
                (xv, yv),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

    ax.set_xlabel("Mean violation count")
    ax.set_ylabel("Mean reward")
    ax.set_title("Stage B: Reward vs Violations")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def unseen_focus_plot(df: pd.DataFrame, out_path: Path) -> None:
    value_col, error_col = resolve_cols(df, "violation_count_mean")

    sub = (
        df[df["regime"] == "nonstationary_unseen"]
        .set_index("method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )

    x = np.arange(len(sub))
    vals = pd.to_numeric(sub[value_col], errors="coerce").values

    if error_col is not None and error_col in sub.columns:
        errs = pd.to_numeric(sub[error_col], errors="coerce").fillna(0.0).values
    else:
        errs = np.zeros(len(sub))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, vals, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in sub["method"]], rotation=15, ha="right")
    ax.set_ylabel("Mean violation count")
    ax.set_title("Stage B: Unseen-Regime Violation Comparison")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_csv)

    grouped_bar_plot(
        df=df,
        metric_base="violation_count_mean",
        ylabel="Mean violation count",
        title="Stage B: Violations by Method and Regime",
        out_path=out_dir / "fig_stageB_violations_by_method_regime.pdf",
    )

    grouped_bar_plot(
        df=df,
        metric_base="near_miss_count_mean",
        ylabel="Mean near-miss count",
        title="Stage B: Near Misses by Method and Regime",
        out_path=out_dir / "fig_stageB_near_misses_by_method_regime.pdf",
    )

    grouped_bar_plot(
        df=df,
        metric_base="shield_count_mean",
        ylabel="Mean shield count",
        title="Stage B: Shield Use by Method and Regime",
        out_path=out_dir / "fig_stageB_shield_use_by_method_regime.pdf",
    )

    grouped_bar_plot(
        df=df,
        metric_base="r_mean",
        ylabel="Mean reward",
        title="Stage B: Reward by Method and Regime",
        out_path=out_dir / "fig_stageB_reward_by_method_regime.pdf",
    )

    scatter_reward_vs_violations(
        df=df,
        out_path=out_dir / "fig_stageB_reward_vs_violations.pdf",
    )

    unseen_focus_plot(
        df=df,
        out_path=out_dir / "fig_stageB_unseen_violation_focus.pdf",
    )

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["baseline", "context", "adjust_speed", "full"]
REGIME_ORDER = ["seen", "unseen"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures from compact-grid aggregated summary."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="artifacts/compact_summary/compact_grid_aggregated_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/compact_summary/figures",
    )
    return parser.parse_args()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    df = df.sort_values(["method", "regime"]).reset_index(drop=True)
    return df


def save_bar_plot(
    df: pd.DataFrame,
    out_path: Path,
    y_col_mean: str,
    y_col_std: str,
    ylabel: str,
    title: str,
) -> None:
    methods = METHOD_ORDER
    regimes = REGIME_ORDER

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, regime in enumerate(regimes):
        sub = (
            df[df["regime"] == regime]
            .set_index("method")
            .reindex(methods)
            .reset_index()
        )
        means = sub[y_col_mean].to_numpy(dtype=float)
        stds = sub[y_col_std].to_numpy(dtype=float)

        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=regime,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for regime, marker in zip(REGIME_ORDER, ["o", "s"]):
        sub = (
            df[df["regime"] == regime]
            .set_index("method")
            .reindex(METHOD_ORDER)
            .reset_index()
        )

        x = sub["r_mean_mean_over_seeds"].to_numpy(dtype=float)
        y = sub["violation_count_mean_mean_over_seeds"].to_numpy(dtype=float)

        ax.scatter(x, y, s=80, marker=marker, label=regime)

        for _, row in sub.iterrows():
            ax.annotate(
                row["method"],
                (row["r_mean_mean_over_seeds"], row["violation_count_mean_mean_over_seeds"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Reward (mean across seeds)")
    ax.set_ylabel("Violations (mean across seeds)")
    ax.set_title("Safety–Performance Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_regime_shift_plot(df: pd.DataFrame, out_path_reward: Path, out_path_violation: Path) -> None:
    rows = []
    for method in METHOD_ORDER:
        seen = df[(df["method"] == method) & (df["regime"] == "seen")]
        unseen = df[(df["method"] == method) & (df["regime"] == "unseen")]
        if seen.empty or unseen.empty:
            continue

        seen_r = float(seen["r_mean_mean_over_seeds"].iloc[0])
        unseen_r = float(unseen["r_mean_mean_over_seeds"].iloc[0])
        seen_v = float(seen["violation_count_mean_mean_over_seeds"].iloc[0])
        unseen_v = float(unseen["violation_count_mean_mean_over_seeds"].iloc[0])

        rows.append(
            {
                "method": method,
                "delta_reward": unseen_r - seen_r,
                "delta_violations": unseen_v - seen_v,
            }
        )

    delta_df = pd.DataFrame(rows).set_index("method").reindex(METHOD_ORDER).reset_index()

    x = np.arange(len(delta_df))

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, delta_df["delta_reward"].to_numpy(dtype=float))
    ax1.set_xticks(x)
    ax1.set_xticklabels(delta_df["method"], rotation=20)
    ax1.set_ylabel("Unseen - Seen reward")
    ax1.set_title("Regime Robustness: Reward Change")
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_path_reward, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, delta_df["delta_violations"].to_numpy(dtype=float))
    ax2.set_xticks(x)
    ax2.set_xticklabels(delta_df["method"], rotation=20)
    ax2.set_ylabel("Unseen - Seen violations")
    ax2.set_title("Regime Robustness: Violation Change")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_path_violation, dpi=200, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    args = parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_csv)

    save_bar_plot(
        df,
        out_dir / "fig_reward_by_method_regime.png",
        "r_mean_mean_over_seeds",
        "r_mean_std_over_seeds",
        "Reward",
        "Reward by Method and Regime",
    )

    save_bar_plot(
        df,
        out_dir / "fig_violations_by_method_regime.png",
        "violation_count_mean_mean_over_seeds",
        "violation_count_mean_std_over_seeds",
        "Violation count",
        "Violations by Method and Regime",
    )

    save_bar_plot(
        df,
        out_dir / "fig_near_misses_by_method_regime.png",
        "near_miss_count_mean_mean_over_seeds",
        "near_miss_count_mean_std_over_seeds",
        "Near-miss count",
        "Near-Misses by Method and Regime",
    )

    save_bar_plot(
        df,
        out_dir / "fig_shield_by_method_regime.png",
        "shield_count_mean_mean_over_seeds",
        "shield_count_mean_std_over_seeds",
        "Shield count",
        "Shield / Hard Interventions by Method and Regime",
    )

    save_bar_plot(
        df,
        out_dir / "fig_action_correction_by_method_regime.png",
        "action_correction_mean_mean_mean_over_seeds",
        "action_correction_mean_mean_std_over_seeds",
        "Action correction mean",
        "Action Correction by Method and Regime",
    )

    save_bar_plot(
        df,
        out_dir / "fig_penalty_by_method_regime.png",
        "reward_penalty_sum_mean_mean_over_seeds",
        "reward_penalty_sum_mean_std_over_seeds",
        "Penalty sum",
        "Reward Penalty by Method and Regime",
    )

    save_tradeoff_plot(
        df,
        out_dir / "fig_tradeoff_reward_vs_violations.png",
    )

    save_regime_shift_plot(
        df,
        out_dir / "fig_delta_reward_unseen_minus_seen.png",
        out_dir / "fig_delta_violations_unseen_minus_seen.png",
    )

    print("\nSaved figures to:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
})

METHODS = ["baseline_sac", "lilac_none"]
REGIMES = ["stationary", "nonstationary_seen", "nonstationary_unseen"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        default="artifacts/stageA_summary/stageA_aggregated_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/stageA_figures",
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], METHODS, ordered=True)
    df["regime"] = pd.Categorical(df["regime"], REGIMES, ordered=True)
    return df.sort_values(["method", "regime"]).reset_index(drop=True)


def grouped_bar_plot(df, mean_col, std_col, ylabel, title, filename):
    x = np.arange(len(REGIMES))
    width = 0.35

    fig, ax = plt.subplots()

    for i, method in enumerate(METHODS):
        sub = df[df["method"] == method].set_index("regime").reindex(REGIMES).reset_index()
        means = sub[mean_col].to_numpy(dtype=float)
        stds = sub[std_col].to_numpy(dtype=float)

        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def tradeoff_plot(df, filename):
    fig, ax = plt.subplots()

    markers = {
        "stationary": "o",
        "nonstationary_seen": "s",
        "nonstationary_unseen": "^",
    }

    for regime in REGIMES:
        sub = df[df["regime"] == regime].set_index("method").reindex(METHODS).reset_index()

        x = sub["r_mean_mean_over_seeds"].to_numpy(dtype=float)
        y = sub["violation_count_mean_mean_over_seeds"].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=90,
            marker=markers[regime],
            label=regime,
        )

        for _, row in sub.iterrows():
            ax.annotate(
                row["method"],
                (row["r_mean_mean_over_seeds"], row["violation_count_mean_mean_over_seeds"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Violations")
    ax.set_title("Stage A: Safety–Performance Tradeoff")
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def robustness_plot(df, metric_col, ylabel, title, filename):
    rows = []
    for method in METHODS:
        stationary = df[(df["method"] == method) & (df["regime"] == "stationary")]
        seen = df[(df["method"] == method) & (df["regime"] == "nonstationary_seen")]
        unseen = df[(df["method"] == method) & (df["regime"] == "nonstationary_unseen")]

        if stationary.empty or seen.empty or unseen.empty:
            continue

        s = float(stationary[metric_col].iloc[0])
        ns = float(seen[metric_col].iloc[0])
        nu = float(unseen[metric_col].iloc[0])

        rows.append({
            "method": method,
            "seen_minus_stationary": ns - s,
            "unseen_minus_stationary": nu - s,
        })

    delta_df = pd.DataFrame(rows).set_index("method").reindex(METHODS).reset_index()

    x = np.arange(len(delta_df))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, delta_df["seen_minus_stationary"], width, label="nonstationary_seen - stationary")
    ax.bar(x + width / 2, delta_df["unseen_minus_stationary"], width, label="nonstationary_unseen - stationary")

    ax.set_xticks(x)
    ax.set_xticklabels(delta_df["method"], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = load_data(args.summary_csv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped_bar_plot(
        df,
        "r_mean_mean_over_seeds",
        "r_mean_std_over_seeds",
        "Reward",
        "Stage A: Reward by Method and Regime",
        out / "fig_stageA_reward_by_regime.pdf",
    )

    grouped_bar_plot(
        df,
        "violation_count_mean_mean_over_seeds",
        "violation_count_mean_std_over_seeds",
        "Violation count",
        "Stage A: Violations by Method and Regime",
        out / "fig_stageA_violations_by_regime.pdf",
    )

    grouped_bar_plot(
        df,
        "near_miss_count_mean_mean_over_seeds",
        "near_miss_count_mean_std_over_seeds",
        "Near-miss count",
        "Stage A: Near-Misses by Method and Regime",
        out / "fig_stageA_nearmisses_by_regime.pdf",
    )

    tradeoff_plot(
        df,
        out / "fig_stageA_tradeoff_reward_vs_violations.pdf",
    )

    robustness_plot(
        df,
        "r_mean_mean_over_seeds",
        "Reward change from stationary",
        "Stage A: Reward Robustness",
        out / "fig_stageA_delta_reward.pdf",
    )

    robustness_plot(
        df,
        "violation_count_mean_mean_over_seeds",
        "Violation change from stationary",
        "Stage A: Violation Robustness",
        out / "fig_stageA_delta_violations.pdf",
    )

    print("Stage A thesis figures saved to:", out)


if __name__ == "__main__":
    main()
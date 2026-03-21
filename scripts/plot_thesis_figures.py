from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Global plotting style
# -----------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8,5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "pdf.fonttype": 42,
})

# Consistent method order
METHODS = ["baseline","context","adjust_speed","full"]
REGIMES = ["seen","unseen"]

# Consistent colors
METHOD_COLORS = {
    "baseline": "#444444",
    "context": "#1f77b4",
    "adjust_speed": "#2ca02c",
    "full": "#d62728",
}

REGIME_MARKERS = {
    "seen": "o",
    "unseen": "s"
}

# -----------------------------
# Argument parser
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary_csv",
        default="artifacts/compact_summary/compact_grid_aggregated_summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="artifacts/thesis_figures",
    )
    return parser.parse_args()


# -----------------------------
# Load data
# -----------------------------

def load_data(path):
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], METHODS)
    df["regime"] = pd.Categorical(df["regime"], REGIMES)
    return df.sort_values(["method","regime"])


# -----------------------------
# Bar plot helper
# -----------------------------

def bar_plot(df, metric, metric_std, ylabel, filename):

    width = 0.35
    x = np.arange(len(METHODS))

    fig, ax = plt.subplots()

    for i, regime in enumerate(REGIMES):

        subset = df[df["regime"]==regime].set_index("method").loc[METHODS]

        means = subset[metric]
        stds = subset[metric_std]

        ax.bar(
            x + (i-0.5)*width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=regime,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel(ylabel)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


# -----------------------------
# Tradeoff plot
# -----------------------------

def tradeoff_plot(df, filename):

    fig, ax = plt.subplots()

    for regime in REGIMES:

        subset = df[df["regime"]==regime]

        x = subset["r_mean_mean_over_seeds"]
        y = subset["violation_count_mean_mean_over_seeds"]

        ax.scatter(
            x,
            y,
            s=80,
            marker=REGIME_MARKERS[regime],
            label=regime,
        )

        for _,row in subset.iterrows():
            ax.annotate(
                row["method"],
                (row["r_mean_mean_over_seeds"],row["violation_count_mean_mean_over_seeds"]),
                xytext=(5,5),
                textcoords="offset points",
            )

    ax.set_xlabel("Reward")
    ax.set_ylabel("Violations")
    ax.set_title("Safety–Performance Tradeoff")

    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close()


# -----------------------------
# Regime difference plot
# -----------------------------

def regime_delta_plot(df, metric, filename, ylabel):

    rows = []

    for m in METHODS:

        seen = df[(df.method==m)&(df.regime=="seen")]
        unseen = df[(df.method==m)&(df.regime=="unseen")]

        if len(seen)==0 or len(unseen)==0:
            continue

        delta = float(unseen[metric]) - float(seen[metric])

        rows.append((m,delta))

    methods = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots()

    ax.bar(methods,values)

    ax.set_ylabel(ylabel)
    ax.set_title("Regime Change Effect (Unseen − Seen)")

    fig.tight_layout()
    fig.savefig(filename,bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():

    args = parse_args()

    df = load_data(args.summary_csv)

    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    # Reward
    bar_plot(
        df,
        "r_mean_mean_over_seeds",
        "r_mean_std_over_seeds",
        "Reward",
        out/"fig_reward_by_method_regime.pdf",
    )

    # Violations
    bar_plot(
        df,
        "violation_count_mean_mean_over_seeds",
        "violation_count_mean_std_over_seeds",
        "Violations",
        out/"fig_violations_by_method_regime.pdf",
    )

    # Near misses
    bar_plot(
        df,
        "near_miss_count_mean_mean_over_seeds",
        "near_miss_count_mean_std_over_seeds",
        "Near Misses",
        out/"fig_nearmiss_by_method_regime.pdf",
    )

    # Shield
    bar_plot(
        df,
        "shield_count_mean_mean_over_seeds",
        "shield_count_mean_std_over_seeds",
        "Shield Interventions",
        out/"fig_shield_by_method_regime.pdf",
    )

    # Action correction
    bar_plot(
        df,
        "action_correction_mean_mean_mean_over_seeds",
        "action_correction_mean_mean_std_over_seeds",
        "Action Correction",
        out/"fig_action_correction_by_method_regime.pdf",
    )

    # Tradeoff
    tradeoff_plot(
        df,
        out/"fig_reward_vs_violations_tradeoff.pdf"
    )

    # Regime deltas
    regime_delta_plot(
        df,
        "r_mean_mean_over_seeds",
        out/"fig_delta_reward.pdf",
        "Reward Difference"
    )

    regime_delta_plot(
        df,
        "violation_count_mean_mean_over_seeds",
        out/"fig_delta_violations.pdf",
        "Violation Difference"
    )

    print("Thesis figures saved to:", out)


if __name__ == "__main__":
    main()

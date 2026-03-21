#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["baseline", "cpss_only", "lilac_only", "full"]


def grouped_bar_plot(
    df: pd.DataFrame,
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    p_values = sorted(df["p_stay"].unique())
    methods = METHOD_ORDER

    x = np.arange(len(methods))
    width = 0.35 if len(p_values) == 2 else max(0.15, 0.8 / max(1, len(p_values)))

    plt.figure(figsize=(9, 5))
    for i, p in enumerate(p_values):
        sub = df[df["p_stay"] == p].copy()
        sub["method"] = pd.Categorical(sub["method"], categories=methods, ordered=True)
        sub = sub.sort_values("method")

        means = sub[value_col].to_numpy()
        stds = sub[std_col].fillna(0.0).to_numpy()

        offset = (i - (len(p_values) - 1) / 2.0) * width
        plt.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=fr"$p_{{stay}}={p:.2f}$")

    plt.xticks(x, methods, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def switch_curve_plot(
    df: pd.DataFrame,
    metric: str,
    p_stay: float,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for method in METHOD_ORDER:
        sub = df[(df["method"] == method) & (df["p_stay"] == p_stay)].sort_values("steps_since_switch")
        if sub.empty:
            continue
        plt.plot(sub["steps_since_switch"], sub[metric], marker="o", label=method)

    plt.xlabel("Steps since context switch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate figures from experiment summary CSVs.")
    parser.add_argument("--summary_dir", type=str, default="artifacts/summary")
    parser.add_argument("--fig_dir", type=str, default="artifacts/figures")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary_dir = Path(args.summary_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(summary_dir / "main_summary_by_method_pstay.csv")
    sw_grouped = pd.read_csv(summary_dir / "switch_aligned_grouped.csv")

    grouped_bar_plot(
        df=main_df,
        value_col="violation_rate_mean",
        std_col="violation_rate_std",
        ylabel="Violation rate",
        title="Violation rate by method and nonstationarity level",
        out_path=fig_dir / "fig_violation_rate_by_method.pdf",
    )

    grouped_bar_plot(
        df=main_df,
        value_col="near_miss_rate_mean",
        std_col="near_miss_rate_std",
        ylabel="Near-miss rate",
        title="Near-miss rate by method and nonstationarity level",
        out_path=fig_dir / "fig_near_miss_rate_by_method.pdf",
    )

    grouped_bar_plot(
        df=main_df,
        value_col="shield_used_rate_mean",
        std_col="shield_used_rate_std",
        ylabel="Shield use rate",
        title="Shield use by method and nonstationarity level",
        out_path=fig_dir / "fig_shield_use_by_method.pdf",
    )

    for p in sorted(sw_grouped["p_stay"].unique()):
        pstay_tag = str(p).replace(".", "p")

        switch_curve_plot(
            df=sw_grouped,
            metric="violation_rate",
            p_stay=p,
            out_path=fig_dir / f"fig_switch_violation_curve_pstay_{pstay_tag}.pdf",
            ylabel="Violation rate",
            title=fr"Post-switch violation rate ($p_{{stay}}={p:.2f}$)",
        )

        switch_curve_plot(
            df=sw_grouped,
            metric="near_miss_rate",
            p_stay=p,
            out_path=fig_dir / f"fig_switch_near_miss_curve_pstay_{pstay_tag}.pdf",
            ylabel="Near-miss rate",
            title=fr"Post-switch near-miss rate ($p_{{stay}}={p:.2f}$)",
        )

        switch_curve_plot(
            df=sw_grouped,
            metric="shield_used_rate",
            p_stay=p,
            out_path=fig_dir / f"fig_switch_shield_curve_pstay_{pstay_tag}.pdf",
            ylabel="Shield use rate",
            title=fr"Post-switch shield use ($p_{{stay}}={p:.2f}$)",
        )

    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()
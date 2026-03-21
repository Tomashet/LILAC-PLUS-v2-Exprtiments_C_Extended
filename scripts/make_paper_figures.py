# scripts/make_paper_figures.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV safely; return None for missing/empty/bad files."""
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_example_run(runs_dir: str, substring: str) -> str:
    candidates = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    for p in candidates:
        if substring in os.path.basename(p):
            return p
    raise RuntimeError(
        f"Could not find a run directory containing '{substring}'. "
        f"Available runs (first 20): {[os.path.basename(x) for x in candidates[:20]]}"
    )


def plot_context_switch(run_dir: str, out_dir: str, max_steps: int = 20000) -> None:
    path = os.path.join(run_dir, "train_monitor.csv")
    df = safe_read_csv(path)
    if df is None:
        raise FileNotFoundError(f"Missing or empty: {path}")
    if "timestep" not in df.columns or "ctx_id" not in df.columns:
        raise ValueError(f"{path} must contain columns: timestep, ctx_id")

    df = df[["timestep", "ctx_id"]].dropna()
    df = df[df["timestep"] <= max_steps]

    plt.figure(figsize=(9, 2.7))
    plt.plot(df["timestep"], df["ctx_id"], drawstyle="steps-post")
    plt.xlabel("Timesteps")
    plt.ylabel("Context ID")
    plt.title("Markov context switching (timeline)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_context_switch.png"), dpi=200)
    plt.close()


def plot_violation_rate(run_dir: str, out_dir: str, window: int = 2000) -> None:
    path = os.path.join(run_dir, "train_monitor.csv")
    df = safe_read_csv(path)
    if df is None:
        raise FileNotFoundError(f"Missing or empty: {path}")
    if "timestep" not in df.columns or "violation" not in df.columns:
        raise ValueError(f"{path} must contain columns: timestep, violation")

    df = df[["timestep", "violation"]].dropna()
    df["violation"] = df["violation"].astype(float)

    y = df["violation"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=(6.6, 4.6))
    plt.plot(df["timestep"].values, y.values)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Violation rate (rolling mean, window={window})")
    plt.title("Violation rate vs training steps")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_violation_rate.png"), dpi=200)
    plt.close()


def infer_condition(run_dir: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns (condition_label, p_stay) where condition_label in {"stationary","nonstationary"}.
    Uses folder naming if present; otherwise uses config.json(p_stay).
    """
    name = os.path.basename(run_dir).lower()

    # Prefer run-name tags if present
    if "_stationary" in name:
        return "stationary", None
    if "_nonstationary" in name:
        return "nonstationary", None
    if "pst1.00" in name or "pst1.0" in name:
        return "stationary", 1.0
    if "pst0.85" in name:
        return "nonstationary", 0.85

    cfg = load_config(run_dir)
    p_stay = cfg.get("p_stay", None)
    if p_stay is None:
        # fallback if config saved differently
        p_stay = (cfg.get("preset_cfg", {}) or {}).get("nonstationarity", {}).get("p_stay", None)

    if p_stay is None:
        return None, None

    p_stay = float(p_stay)
    if p_stay >= 0.999:
        return "stationary", p_stay
    return "nonstationary", p_stay


def plot_stationary_vs_nonstationary_reward(runs_dir: str, out_dir: str) -> None:
    run_dirs = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]

    # Collect curves
    stationary = []
    nonstationary = []

    for r in run_dirs:
        progress = os.path.join(r, "progress.csv")
        df = safe_read_csv(progress)
        if df is None:
            continue
        if "time/total_timesteps" not in df.columns:
            continue
        if "rollout/ep_rew_mean" not in df.columns:
            continue

        cond, _ = infer_condition(r)
        if cond == "stationary":
            stationary.append((r, df))
        elif cond == "nonstationary":
            nonstationary.append((r, df))

    plt.figure(figsize=(7.6, 4.8))

    # Plot individual runs faintly
    for _, df in stationary:
        plt.plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"], alpha=0.25)
    for _, df in nonstationary:
        plt.plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"], alpha=0.25)

    # Plot mean curves if possible
    def plot_mean(group, label):
        if len(group) == 0:
            return

        # Build common grid on overlapping timesteps
        mins = [g[1]["time/total_timesteps"].min() for g in group]
        maxs = [g[1]["time/total_timesteps"].max() for g in group]
        tmin = int(max(mins))
        tmax = int(min(maxs))

        if tmax <= tmin:
            return  # no overlap, already plotted individual lines

        import numpy as np
        grid = np.linspace(tmin, tmax, 200)

        ys = []
        for _, df in group:
            x = df["time/total_timesteps"].to_numpy()
            y = df["rollout/ep_rew_mean"].to_numpy()
            ys.append(np.interp(grid, x, y))

        y_mean = np.mean(np.stack(ys, axis=0), axis=0)
        plt.plot(grid, y_mean, linewidth=2.5, label=label)

    plot_mean(stationary, "Stationary (p_stay=1.0)")
    plot_mean(nonstationary, "Nonstationary (Markov switching)")

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward mean (rollout/ep_rew_mean)")
    plt.title("Stationary vs Nonstationary performance")
    if stationary or nonstationary:
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_stationary_vs_nonstationary_reward.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="paper_figures")
    ap.add_argument("--example_run", default="merge-v0_discrete_dqn_seed0",
                    help="Substring to pick example run for figs 1–2")
    ap.add_argument("--window", type=int, default=2000, help="Rolling window for violation rate")
    ap.add_argument("--max_steps", type=int, default=20000, help="Max steps to show in context plot")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    example_run_dir = find_example_run(args.runs_dir, args.example_run)

    plot_context_switch(example_run_dir, args.out_dir, max_steps=args.max_steps)
    plot_violation_rate(example_run_dir, args.out_dir, window=args.window)
    plot_stationary_vs_nonstationary_reward(args.runs_dir, args.out_dir)

    print(f"Saved figures to: {args.out_dir}")
    print(" - fig1_context_switch.png")
    print(" - fig2_violation_rate.png")
    print(" - fig3_stationary_vs_nonstationary_reward.png")


if __name__ == "__main__":
    main()
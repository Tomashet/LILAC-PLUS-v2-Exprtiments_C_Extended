import argparse
import glob
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_progress(run_dir):
    # SB3 CSV logger writes progress.csv in the run directory
    path = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # SB3 sometimes uses "time/total_timesteps" column name
    # and reward as "rollout/ep_rew_mean"
    if "time/total_timesteps" not in df.columns:
        return None
    if "rollout/ep_rew_mean" not in df.columns:
        return None
    return df[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs", help="Folder that contains run subfolders")
    ap.add_argument("--env_contains", default="", help="Only include runs whose name contains this (e.g. merge-v0)")
    ap.add_argument("--algo_contains", default="", help="Only include runs whose name contains this (e.g. dqn/ppo/sac)")
    ap.add_argument("--stationary_threshold", type=float, default=0.999,
                    help="p_stay >= threshold treated as stationary")
    args = ap.parse_args()

    run_dirs = [p for p in glob.glob(os.path.join(args.runs_dir, "*")) if os.path.isdir(p)]
    if args.env_contains:
        run_dirs = [r for r in run_dirs if args.env_contains in os.path.basename(r)]
    if args.algo_contains:
        run_dirs = [r for r in run_dirs if args.algo_contains in os.path.basename(r)]

    stationary_runs = []
    nonstationary_runs = []

    for r in run_dirs:
        cfg = load_config(r)
        p_stay = cfg.get("p_stay", None)
        if p_stay is None:
            # Sometimes stored under preset_cfg/nonstationarity in some versions;
            # try a fallback.
            p_stay = (cfg.get("preset_cfg", {}) or {}).get("nonstationarity", {}).get("p_stay", None)

        dfp = load_progress(r)
        if dfp is None or p_stay is None:
            continue

        p_stay = float(p_stay)
        if p_stay >= args.stationary_threshold:
            stationary_runs.append((r, dfp))
        else:
            nonstationary_runs.append((r, dfp))

    if not stationary_runs and not nonstationary_runs:
        raise RuntimeError(
            "No valid runs found with progress.csv and config.json(p_stay). "
            "Make sure SB3 is logging progress.csv and you trained at least one run."
        )

    plt.figure()

    # Plot individual curves lightly, then mean curve per group
    def plot_group(runs, label):
        if not runs:
            return
        # Align timesteps by exact timesteps present; simplest is to interpolate onto a common grid.
        # We'll make a grid from min to max of all curves with 200 points.
        import numpy as np
        mins = [df["time/total_timesteps"].min() for _, df in runs]
        maxs = [df["time/total_timesteps"].max() for _, df in runs]
        tmin, tmax = int(max(mins)), int(min(maxs))
        if tmax <= tmin:
            # If ranges don't overlap, just plot individual curves
            for rdir, df in runs:
                plt.plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"], alpha=0.35)
            return

        grid = np.linspace(tmin, tmax, 200)

        ys = []
        for rdir, df in runs:
            x = df["time/total_timesteps"].to_numpy()
            y = df["rollout/ep_rew_mean"].to_numpy()
            # plot light individual
            plt.plot(x, y, alpha=0.25)
            # interpolate for mean
            ys.append(np.interp(grid, x, y))

        y_mean = np.mean(np.stack(ys, axis=0), axis=0)
        plt.plot(grid, y_mean, linewidth=2.5, label=label)

    plot_group(stationary_runs, "Stationary (p_stay=1.0)")
    plot_group(nonstationary_runs, "Nonstationary (Markov switching)")

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward mean (SB3 rollout/ep_rew_mean)")
    title_bits = ["Stationary vs Nonstationary"]
    if args.env_contains:
        title_bits.append(args.env_contains)
    if args.algo_contains:
        title_bits.append(args.algo_contains)
    plt.title(" | ".join(title_bits))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
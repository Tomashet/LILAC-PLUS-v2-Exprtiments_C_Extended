import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--window", type=int, default=2000,
                    help="Rolling window in timesteps (rows). Increase for smoother curves.")
    args = ap.parse_args()

    path = os.path.join(args.run_dir, "train_monitor.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    if "violation" not in df.columns:
        raise ValueError("train_monitor.csv has no violation column. Check your callback logging.")

    df = df[["timestep", "violation"]].dropna()
    df["violation"] = df["violation"].astype(float)

    # Rolling violation rate (per-step)
    viol_rate = df["violation"].rolling(args.window, min_periods=1).mean()

    plt.figure()
    plt.plot(df["timestep"].values, viol_rate.values)
    plt.xlabel("Timesteps")
    plt.ylabel(f"Violation rate (rolling mean, window={args.window})")
    plt.title(f"Violation rate vs steps: {os.path.basename(args.run_dir)}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
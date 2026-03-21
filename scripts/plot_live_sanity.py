# scripts/plot_live_sanity.py
# Live plot (4 panels): ctx_id, rolling violation rate, adj_risk, adj_unsafe
import argparse
import os
import time

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to train_monitor.csv (or pass --run_dir)")
    ap.add_argument("--run_dir", default="", help="runs/<run_dir> containing train_monitor.csv")
    ap.add_argument("--window", type=int, default=3000, help="How many recent rows to plot")
    ap.add_argument("--pause", type=float, default=0.8, help="Seconds between refresh")
    ap.add_argument("--viol_roll", type=int, default=200, help="Rolling window for violation rate")
    args = ap.parse_args()

    csv_path = args.csv or os.path.join("runs", args.run_dir, "train_monitor.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")

    plt.ion()
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    title = os.path.basename(os.path.dirname(csv_path)) or csv_path

    while True:
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                time.sleep(args.pause)
                continue

            df = df.tail(args.window)

            t = df.get("timestep", pd.Series(range(len(df)))).to_numpy()
            ctx = df.get("ctx_id", pd.Series([-1] * len(df))).to_numpy()
            viol = df.get("violation", pd.Series([0] * len(df))).astype(float).to_numpy()
            adj_risk = df.get("adj_risk", pd.Series([0.0] * len(df))).astype(float).to_numpy()
            adj_unsafe = df.get("adj_unsafe", pd.Series([0] * len(df))).astype(int).to_numpy()

            viol_roll = pd.Series(viol).rolling(args.viol_roll, min_periods=1).mean().to_numpy()

            ax1.cla(); ax2.cla(); ax3.cla(); ax4.cla()

            # 1) Context
            ax1.plot(t, ctx)
            ax1.set_ylabel("ctx_id")
            ax1.set_title(f"{title}  (live)")

            # 2) Violation rate
            ax2.plot(t, viol_roll)
            ax2.set_ylabel(f"viol rate (roll={args.viol_roll})")
            ax2.set_ylim(-0.05, 1.05)

            # 3) Adjustment-speed risk
            ax3.plot(t, adj_risk)
            ax3.set_ylabel("adj_risk")
            ax3.set_ylim(-0.05, 1.05)

            # 4) Unsafe trigger (binary)
            ax4.step(t, adj_unsafe, where="post")
            ax4.set_ylabel("adj_unsafe")
            ax4.set_xlabel("timestep")
            ax4.set_ylim(-0.1, 1.1)

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(args.pause)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
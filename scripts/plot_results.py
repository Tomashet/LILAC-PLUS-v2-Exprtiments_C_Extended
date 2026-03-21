from __future__ import annotations
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from src.logging_utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    plots_dir = os.path.join(args.run_dir, "plots")
    ensure_dir(plots_dir)

    train_csv = os.path.join(args.run_dir, "train_monitor.csv")
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
        df["viol_roll"] = df["violation"].rolling(200, min_periods=50).mean()
        plt.figure(); plt.plot(df["timestep"], df["viol_roll"])
        plt.xlabel("timestep"); plt.ylabel("rolling violation rate"); plt.title("Training: rolling violations")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_violations.png")); plt.close()

        df["clear_roll"] = df["clearance"].rolling(200, min_periods=50).mean()
        plt.figure(); plt.plot(df["timestep"], df["clear_roll"])
        plt.xlabel("timestep"); plt.ylabel("rolling clearance"); plt.title("Training: rolling clearance")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_clearance.png")); plt.close()

    eval_csv = os.path.join(args.run_dir, "eval_metrics.csv")
    if os.path.exists(eval_csv):
        df = pd.read_csv(eval_csv)
        plt.figure(); plt.plot(df["episode"], df["viol_rate_step"])
        plt.xlabel("episode"); plt.ylabel("violation rate"); plt.title("Evaluation: violations by episode")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "eval_violations.png")); plt.close()

        plt.figure(); plt.plot(df["episode"], df["return"])
        plt.xlabel("episode"); plt.ylabel("return"); plt.title("Evaluation: return by episode")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "eval_return.png")); plt.close()

        grp = df.groupby(["density","aggr","noise"])[["viol_rate_step","return"]].mean().reset_index()
        plt.figure(); plt.scatter(grp["return"], grp["viol_rate_step"])
        plt.xlabel("mean return"); plt.ylabel("mean violation rate"); plt.title("Context-wise return vs violations")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "context_pareto.png")); plt.close()

    print(f"Saved plots to: {plots_dir}")
if __name__ == "__main__":
    main()

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    args = ap.parse_args()

    path = os.path.join(args.run_dir, "train_monitor.csv")
    df = pd.read_csv(path)

    df = df[["timestep", "ctx_id"]].dropna()
    df = df[df["timestep"] <= args.max_steps]

    plt.figure(figsize=(10,3))
    plt.plot(df["timestep"], df["ctx_id"], drawstyle="steps-post")

    plt.xlabel("Timesteps")
    plt.ylabel("Context ID")
    plt.title("Markov Context Switching")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
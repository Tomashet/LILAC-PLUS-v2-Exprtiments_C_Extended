import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_DIR = Path("runs")
OUT_DIR = Path("runs/thesis_artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for run in RUNS_DIR.glob("*"):
    config_file = run / "config.json"
    eval_file = run / "eval_metrics.csv"

    if not config_file.exists() or not eval_file.exists():
        continue

    config = json.load(open(config_file))
    df = pd.read_csv(eval_file)

    rows.append({
        "method": config.get("constraint", "baseline"),
        "p_stay": config.get("p_stay", 1.0),
        "reward": df["episode_reward"].mean(),
        "violations": df["violation_rate"].mean(),
        "collisions": df["collision_rate"].mean()
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "all_eval_rows.csv", index=False)

table = df.groupby("method").mean().reset_index()
table.to_csv(OUT_DIR / "main_table.csv", index=False)

plt.figure()
for m in df.method.unique():
    sub = df[df.method == m]
    plt.plot(sub.p_stay, sub.violations, marker="o", label=m)

plt.xlabel("p_stay (nonstationarity)")
plt.ylabel("violation rate")
plt.legend()
plt.savefig(OUT_DIR / "fig_nonstationarity_sweep.png")
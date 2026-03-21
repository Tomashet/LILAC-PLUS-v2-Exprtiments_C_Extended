from __future__ import annotations
import argparse
import glob
import os
import pandas as pd
import numpy as np

METRICS = ["viol_rate_step", "near_rate_step", "return", "shield_rate_step"]

def summarize_run(run_dir: str):
    path = os.path.join(run_dir, "eval_metrics.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {m: float(df[m].mean()) for m in METRICS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*", help="Glob for run dirs (must contain eval_metrics.csv)")
    ap.add_argument("--out", default="table_results.tex")
    ap.add_argument("--caption", default="Safety and performance under episodic nonstationarity (mean over episodes, mean$\\pm$std over seeds).")
    ap.add_argument("--label", default="tab:results_nonstationary")
    args = ap.parse_args()

    run_dirs = [d for d in glob.glob(args.pattern) if os.path.isdir(d)]
    rows = []
    for d in run_dirs:
        summ = summarize_run(d)
        if summ is None:
            continue
        name = os.path.basename(d)
        method = name.split("_seed")[0] if "_seed" in name else name
        rows.append({"method": method, **summ})

    if not rows:
        raise RuntimeError(f"No runs found matching {args.pattern} with eval_metrics.csv")

    df = pd.DataFrame(rows)
    agg = df.groupby("method")[METRICS].agg(["mean", "std"]).reset_index()

    def fmt(mean, std):
        if np.isnan(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    lines = []
    lines += [r"\begin{table}[t]", r"\centering", r"\small"]
    lines += [r"\caption{" + args.caption + r"}", r"\label{" + args.label + r"}"]
    lines += [r"\begin{tabular}{lcccc}", r"\toprule"]
    lines += [r"Method & Viol.\ rate $\downarrow$ & Near-miss $\downarrow$ & Return $\uparrow$ & Shield use $\uparrow$ \\", r"\midrule"]

    for _, r in agg.iterrows():
        m = r["method"]
        v = fmt(r[("viol_rate_step","mean")], r[("viol_rate_step","std")])
        n = fmt(r[("near_rate_step","mean")], r[("near_rate_step","std")])
        ret = fmt(r[("return","mean")], r[("return","std")])
        sh = fmt(r[("shield_rate_step","mean")], r[("shield_rate_step","std")])
        lines.append(f"{m} & {v} & {n} & {ret} & {sh} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote LaTeX table to: {args.out}")
    print(agg)

if __name__ == "__main__":
    main()

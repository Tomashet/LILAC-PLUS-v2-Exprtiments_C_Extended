from __future__ import annotations
import argparse
import glob
import os
from typing import Dict, Tuple
import pandas as pd
import numpy as np

METRICS = ["viol_rate_step", "near_rate_step", "return", "shield_rate_step"]

# Fixed paper-friendly ordering
ABL_ORDER = ["full", "no_mpc", "no_conformal", "no_mpc_no_conformal"]
ENV_ORDER = ["highway", "merge"]
ALG_ORDER_DISCRETE = ["dqn", "ppo"]
ALG_ORDER_CONT = ["sac"]

def parse_method(method: str) -> Dict[str, str]:
    env = "highway" if method.startswith("highway") else ("merge" if method.startswith("merge") else "other")
    action_type = "continuous" if "continuous" in method else "discrete"
    algo = "sac" if "_sac_" in method else ("ppo" if "_ppo_" in method else "dqn")
    ablation = "full"
    for ab in ABL_ORDER:
        if method.endswith("_" + ab):
            ablation = ab
            break
    return {"env": env, "action_type": action_type, "algo": algo, "ablation": ablation}

def summarize_runs(pattern: str) -> pd.DataFrame:
    run_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    rows = []
    for d in run_dirs:
        path = os.path.join(d, "eval_metrics.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        name = os.path.basename(d)
        method = name.split("_seed")[0] if "_seed" in name else name
        row = {"method": method}
        for m in METRICS:
            row[m] = float(df[m].mean())
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No eval_metrics.csv found under {pattern}")
    return pd.DataFrame(rows)

def method_sort_key(method: str) -> Tuple[int,int,int,int,str]:
    p = parse_method(method)
    env_i = ENV_ORDER.index(p["env"]) if p["env"] in ENV_ORDER else 99
    act_i = 0 if p["action_type"] == "discrete" else 1
    if p["action_type"] == "discrete":
        algo_i = ALG_ORDER_DISCRETE.index(p["algo"]) if p["algo"] in ALG_ORDER_DISCRETE else 99
    else:
        algo_i = ALG_ORDER_CONT.index(p["algo"]) if p["algo"] in ALG_ORDER_CONT else 99
    abl_i = ABL_ORDER.index(p["ablation"]) if p["ablation"] in ABL_ORDER else 99
    return (env_i, act_i, algo_i, abl_i, method)

def to_latex_table(agg: pd.DataFrame, caption: str, label: str) -> str:
    def fmt(mean, std):
        if np.isnan(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} $\\pm$ {std:.3f}"

    lines = []
    lines += [r"\\begin{table}[t]", r"\\centering", r"\\small"]
    lines += [r"\\caption{" + caption + r"}", r"\\label{" + label + r"}"]
    lines += [r"\\begin{tabular}{lcccc}", r"\\toprule"]
    lines += [r"Method & Viol.\\ rate $\\downarrow$ & Near-miss $\\downarrow$ & Return $\\uparrow$ & Shield use $\\uparrow$ \\\\", r"\\midrule"]

    for _, r in agg.iterrows():
        m = r["method"]
        v = fmt(r[("viol_rate_step","mean")], r[("viol_rate_step","std")])
        n = fmt(r[("near_rate_step","mean")], r[("near_rate_step","std")])
        ret = fmt(r[("return","mean")], r[("return","std")])
        sh = fmt(r[("shield_rate_step","mean")], r[("shield_rate_step","std")])
        lines.append(f"{m} & {v} & {n} & {ret} & {sh} \\\\")
    lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}"]
    return "\\n".join(lines) + "\\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*_seed*", help="Glob for run dirs")
    ap.add_argument("--out_dir", default="paper_tables", help="Where to write .tex files")
    ap.add_argument("--label_prefix", default="tab:results", help="Prefix for LaTeX labels")
    ap.add_argument("--caption_prefix", default="Results under episodic nonstationarity.", help="Prefix for captions")
    ap.add_argument("--split_by_env", action="store_true", help="Also write per-env discrete/continuous tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = summarize_runs(args.pattern)

    agg = df.groupby("method")[METRICS].agg(["mean","std"]).reset_index()
    agg = agg.sort_values("method", key=lambda s: s.map(method_sort_key))

    meta_df = agg["method"].map(parse_method).apply(pd.Series)
    agg2 = pd.concat([agg.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    disc = agg2[agg2["action_type"] == "discrete"].copy()
    cont = agg2[agg2["action_type"] == "continuous"].copy()

    metric_cols = ["method"] + [c for c in disc.columns if isinstance(c, tuple)]

    disc_tex = to_latex_table(
        disc[metric_cols],
        caption=args.caption_prefix + " Discrete control (DQN/PPO).",
        label=args.label_prefix + "_discrete",
    )
    cont_tex = to_latex_table(
        cont[metric_cols],
        caption=args.caption_prefix + " Continuous control (SAC).",
        label=args.label_prefix + "_continuous",
    )

    with open(os.path.join(args.out_dir, "results_discrete.tex"), "w", encoding="utf-8") as f:
        f.write(disc_tex)
    with open(os.path.join(args.out_dir, "results_continuous.tex"), "w", encoding="utf-8") as f:
        f.write(cont_tex)

    if args.split_by_env:
        for env in ENV_ORDER:
            disc_e = disc[disc["env"] == env]
            cont_e = cont[cont["env"] == env]
            if len(disc_e):
                tex = to_latex_table(
                    disc_e[metric_cols],
                    caption=args.caption_prefix + f" {env} (discrete).",
                    label=args.label_prefix + f"_{env}_discrete",
                )
                with open(os.path.join(args.out_dir, f"results_{env}_discrete.tex"), "w", encoding="utf-8") as f:
                    f.write(tex)
            if len(cont_e):
                tex = to_latex_table(
                    cont_e[metric_cols],
                    caption=args.caption_prefix + f" {env} (continuous).",
                    label=args.label_prefix + f"_{env}_continuous",
                )
                with open(os.path.join(args.out_dir, f"results_{env}_continuous.tex"), "w", encoding="utf-8") as f:
                    f.write(tex)

    print("Wrote paper-friendly tables to:", args.out_dir)

if __name__ == "__main__":
    main()

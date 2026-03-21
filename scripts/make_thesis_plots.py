"""Generate thesis plots from runs/.

This script aggregates SB3 training logs across multiple runs and produces
publication-friendly plots (using matplotlib defaults).

Outputs (in out_dir):
  - learning_curves_return.png
  - learning_curves_violation.png
  - intervention_rate.png
  - risk_calibration.png
  - summary_table.csv

Assumptions:
  - Each run directory contains a train_monitor.csv and optionally config.json
  - Column names may vary; the script is robust to missing columns.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RunMeta:
    run_dir: str
    constraint: str
    lilac: bool
    p_stay: Optional[float]
    seed: Optional[int]
    total_steps: Optional[int]


def safe_read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # also allow case-insensitive match
    lower_map = {x.lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_run(run_path: Path) -> Tuple[Optional[RunMeta], Optional[pd.DataFrame]]:
    monitor_path = run_path / "train_monitor.csv"
    if not monitor_path.exists():
        return None, None

    try:
        df = pd.read_csv(monitor_path)
    except Exception:
        return None, None

    cfg = safe_read_json(run_path / "config.json")
    # Best-effort extraction.
    constraint = str(cfg.get("constraint", cfg.get("constraints", "unknown")))
    lilac = bool(cfg.get("lilac", cfg.get("use_lilac", False)))
    p_stay = cfg.get("p_stay")
    seed = cfg.get("seed")
    total_steps = cfg.get("total_steps")

    meta = RunMeta(
        run_dir=run_path.name,
        constraint=constraint,
        lilac=lilac,
        p_stay=float(p_stay) if p_stay is not None else None,
        seed=int(seed) if seed is not None else None,
        total_steps=int(total_steps) if total_steps is not None else None,
    )
    return meta, df


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) == 0:
        return x
    w = min(window, len(x))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="valid")


def condition_label(meta: RunMeta) -> str:
    base = meta.constraint if meta.constraint else "unknown"
    if meta.lilac:
        base = f"lilac+{base}"
    else:
        base = f"base+{base}"
    if meta.p_stay is not None:
        base += f"_p{meta.p_stay:g}"
    return base


def aggregate_learning_curves(
    runs: List[Tuple[RunMeta, pd.DataFrame]],
    metric_col: str,
    x_col: Optional[str],
    window: int,
) -> Dict[str, pd.DataFrame]:
    """Return label -> dataframe with columns [x, mean, stderr]."""
    grouped: Dict[str, List[pd.DataFrame]] = {}
    for meta, df in runs:
        label = condition_label(meta)
        grouped.setdefault(label, []).append(df)

    out: Dict[str, pd.DataFrame] = {}
    for label, dfs in grouped.items():
        series_list = []
        x_list = []
        for df in dfs:
            y = df[metric_col].to_numpy(dtype=float)
            if window > 1:
                y = moving_average(y, window)
            series_list.append(y)

            if x_col and x_col in df.columns:
                x = df[x_col].to_numpy(dtype=float)
                if window > 1:
                    x = x[window - 1 :]
            else:
                x = np.arange(len(y), dtype=float)
            x_list.append(x)

        # Align by min length
        min_len = min(len(s) for s in series_list)
        if min_len <= 1:
            continue
        Y = np.stack([s[:min_len] for s in series_list], axis=0)
        X = x_list[0][:min_len]
        mean = Y.mean(axis=0)
        stderr = Y.std(axis=0, ddof=1) / math.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)
        out[label] = pd.DataFrame({"x": X, "mean": mean, "stderr": stderr})
    return out


def plot_with_bands(curves: Dict[str, pd.DataFrame], title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    for label, df in sorted(curves.items()):
        x = df["x"].to_numpy()
        m = df["mean"].to_numpy()
        s = df["stderr"].to_numpy()
        plt.plot(x, m, label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def risk_calibration_plot(
    runs: List[Tuple[RunMeta, pd.DataFrame]],
    risk_col: str,
    viol_col: str,
    bins: int,
    out_path: Path,
) -> None:
    rows = []
    for meta, df in runs:
        if risk_col not in df.columns or viol_col not in df.columns:
            continue
        r = pd.to_numeric(df[risk_col], errors="coerce")
        v = pd.to_numeric(df[viol_col], errors="coerce")
        tmp = pd.DataFrame({"risk": r, "viol": v}).dropna()
        if tmp.empty:
            continue
        tmp["label"] = condition_label(meta)
        rows.append(tmp)
    if not rows:
        return
    data = pd.concat(rows, ignore_index=True)

    plt.figure()
    for label, g in data.groupby("label"):
        g = g.copy()
        # If violations are counts, convert to indicator.
        if g["viol"].max() > 1.0:
            g["viol"] = (g["viol"] > 0).astype(float)
        g["bin"] = pd.cut(g["risk"], bins=bins, include_lowest=True)
        b = g.groupby("bin", observed=True).agg(risk_mean=("risk", "mean"), viol_rate=("viol", "mean"), n=("viol", "size"))
        if len(b) < 2:
            continue
        plt.plot(b["risk_mean"].to_numpy(), b["viol_rate"].to_numpy(), marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Risk calibration (predicted risk vs empirical violation rate)")
    plt.xlabel("Predicted risk (binned mean)")
    plt.ylabel("Empirical violation probability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs and generate thesis plots.")
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_dir", default="runs/thesis_plots")
    ap.add_argument("--window", type=int, default=20, help="Moving average window over episodes.")
    ap.add_argument("--bins", type=int, default=10, help="Bins for risk calibration.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: List[Tuple[RunMeta, pd.DataFrame]] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        meta, df = load_run(d)
        if meta is None or df is None:
            continue
        loaded.append((meta, df))

    if not loaded:
        raise SystemExit(f"No runs found under {runs_root} (expected train_monitor.csv files).")

    # Pick standard columns.
    sample_df = loaded[0][1]
    x_col = find_col(sample_df, ["timesteps", "total_timesteps", "timestep", "steps"])  # may be absent
    ret_col = find_col(sample_df, ["episode_return", "return", "ep_return", "reward", "ep_reward"])
    viol_col = find_col(sample_df, ["violations", "episode_violations", "violation_count", "collisions", "cost", "episode_cost"])
    shield_col = find_col(sample_df, ["shield_used", "shield_interventions", "interventions"])
    risk_col = find_col(sample_df, ["lilac/risk", "risk", "adj_risk"])

    # Fallbacks if some columns are missing.
    if ret_col is None:
        # If nothing obvious, pick first numeric column.
        numeric = [c for c in sample_df.columns if pd.api.types.is_numeric_dtype(sample_df[c])]
        ret_col = numeric[0] if numeric else sample_df.columns[0]
    if viol_col is None:
        # Try to locate any 'violation' substring.
        for c in sample_df.columns:
            if "viol" in c.lower() or "collision" in c.lower() or "cost" in c.lower():
                viol_col = c
                break
    if shield_col is None:
        for c in sample_df.columns:
            if "shield" in c.lower() and "use" in c.lower():
                shield_col = c
                break

    # Build learning curves.
    curves_ret = aggregate_learning_curves(loaded, ret_col, x_col, args.window)
    plot_with_bands(
        curves_ret,
        title="Learning curves (return)",
        xlabel=x_col or "episode",
        ylabel=ret_col,
        out_path=out_dir / "learning_curves_return.png",
    )

    if viol_col is not None:
        curves_viol = aggregate_learning_curves(loaded, viol_col, x_col, args.window)
        plot_with_bands(
            curves_viol,
            title="Learning curves (violations/cost)",
            xlabel=x_col or "episode",
            ylabel=viol_col,
            out_path=out_dir / "learning_curves_violation.png",
        )

    if shield_col is not None:
        curves_shield = aggregate_learning_curves(loaded, shield_col, x_col, args.window)
        plot_with_bands(
            curves_shield,
            title="Shield intervention rate (per episode)",
            xlabel=x_col or "episode",
            ylabel=shield_col,
            out_path=out_dir / "intervention_rate.png",
        )

    if risk_col is not None and viol_col is not None:
        risk_calibration_plot(loaded, risk_col, viol_col, bins=args.bins, out_path=out_dir / "risk_calibration.png")

    # Summary table.
    summary_rows = []
    for meta, df in loaded:
        row = {
            "run_dir": meta.run_dir,
            "label": condition_label(meta),
            "seed": meta.seed,
            "p_stay": meta.p_stay,
            "lilac": meta.lilac,
            "constraint": meta.constraint,
        }
        for name, col in [("return", ret_col), ("viol", viol_col), ("shield", shield_col), ("risk", risk_col)]:
            if col is None or col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if x.empty:
                continue
            row[f"{name}_last"] = float(x.iloc[-1])
            row[f"{name}_mean"] = float(x.mean())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_table.csv", index=False)

    print(f"Saved plots and summary to: {out_dir}")


if __name__ == "__main__":
    # Avoid any global style changes; rely on matplotlib defaults.
    matplotlib.rcParams.update({"figure.max_open_warning": 0})
    main()

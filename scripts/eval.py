from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, SAC
from src.safety import SafetyParams
from src.logging_utils import append_csv
from .common import make_env

def load_model(run_dir: str, action_space_type: str):
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_path = os.path.join(run_dir, "models", "final_model.zip")
    algo = cfg.get("algo", "sac" if action_space_type == "continuous" else "dqn")
    if action_space_type == "continuous":
        return SAC.load(model_path), cfg
    return (DQN.load(model_path) if algo == "dqn" else PPO.load(model_path)), cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="highway-v0")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action_space_type", choices=["discrete","continuous"], default="")
    # Optional overrides for soft→hard budget wrapper (otherwise loaded from run config)
    ap.add_argument("--budget_C", type=float, default=None)
    ap.add_argument("--budget_T", type=int, default=None)
    ap.add_argument("--budget_delta", type=float, default=None)
    ap.add_argument("--no_budget", action="store_true")
    args = ap.parse_args()

    action_space_type = args.action_space_type or ("continuous" if "continuous" in os.path.basename(args.run_dir) else "discrete")
    model, cfg = load_model(args.run_dir, action_space_type)
    safety_params = SafetyParams(**cfg.get("safety_params", {}))

    budget_C_eval = float(args.budget_C) if args.budget_C is not None else float(cfg.get("budget_C", 0.0))
    budget_T_eval = int(args.budget_T) if args.budget_T is not None else int(cfg.get("budget_T", 60))
    budget_delta_eval = float(args.budget_delta) if args.budget_delta is not None else float(cfg.get("budget_delta", 0.0))
    no_budget_eval = bool(args.no_budget) if args.no_budget else bool(cfg.get("no_budget", True))

    env, _, _ = make_env(
        args.env,
        args.seed,
        action_space_type,
        float(cfg.get("p_stay", 0.8)),
        bool(cfg.get("no_mpc", False)),
        bool(cfg.get("no_conformal", False)),
        safety_params,
        budget_C=budget_C_eval,
        budget_T=budget_T_eval,
        budget_delta=budget_delta_eval,
        no_budget=no_budget_eval,
    )

    out_csv = os.path.join(args.run_dir, "eval_metrics.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    for ep in range(args.episodes):
        if hasattr(env, "next_episode"):
            env.next_episode()
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ep_ret = 0.0; steps = 0; viol = 0; near = 0; shield = 0; budget_filtered = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward); steps += 1
            viol += int(bool(info.get("violation", False)))
            near += int(bool(info.get("near_miss", False)))
            shield += int(bool(info.get("shield_used", False)))
            budget_filtered += int(bool(info.get("budget_filtered", False)))

        ctx = info.get("ctx_tuple") or ("","","")
        append_csv(out_csv, {
            "episode": ep, "return": ep_ret, "steps": steps,
            "viol_steps": viol, "near_steps": near,
            "viol_rate_step": viol/max(steps,1), "near_rate_step": near/max(steps,1),
            "shield_rate_step": shield/max(steps,1),
            "budget_filtered_rate_step": budget_filtered/max(steps,1),
            "budget_C": budget_C_eval,
            "budget_T": budget_T_eval,
            "ctx_id": info.get("ctx_id", -1),
            "density": ctx[0], "aggr": ctx[1], "noise": ctx[2],
        })

    env.close()
    df = pd.read_csv(out_csv)
    print(f"Wrote: {out_csv}")
    print(df[["viol_rate_step","near_rate_step","return","shield_rate_step"]].mean())
if __name__ == "__main__":
    main()

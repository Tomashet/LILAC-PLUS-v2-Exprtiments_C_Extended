import time
import numpy as np

from scripts.common import make_env
from src.safety import SafetyParams

def run(env_id="merge-v0", p_stay=0.85, episodes=10, render=False):
    safety_params = SafetyParams(horizon_n=10, epsilon=0.5)
    env, _, _ = make_env(
        env_id=env_id,
        seed=0,
        action_space_type="discrete",
        p_stay=p_stay,
        no_mpc=True,
        no_conformal=True,
        safety_params=safety_params,
    )

    if render:
        # highway-env supports render(), but only if env supports it; keep it optional
        pass

    ctx_seen = []
    for ep in range(episodes):
        obs, info = env.reset()
        ctx_id = info.get("ctx_id", -1)
        ctx_tuple = info.get("ctx_tuple", None)
        ctx_seen.append(ctx_id)
        print(f"EP {ep:02d} | ctx_id={ctx_id} | ctx={ctx_tuple}")

        done = False
        steps = 0
        while not done and steps < 200:
            a = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            steps += 1

            if render:
                try:
                    env.render()
                    time.sleep(0.03)
                except Exception:
                    pass

    env.close()
    print("Context IDs:", ctx_seen)
    print("Unique contexts:", len(set(ctx_seen)))

if __name__ == "__main__":
    run()
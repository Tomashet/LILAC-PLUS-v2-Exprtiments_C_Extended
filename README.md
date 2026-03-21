# Proactive Context-Dependent Safety Experiments (highway-env + SB3) — with LILAC-style Context Prediction

This repository is a **runnable experiment package** for **episodic nonstationarity** in `highway-env`, featuring:

- **Discrete control**: SB3 **DQN** and **PPO**
- **Continuous control**: SB3 **SAC**
- **Episode-level nonstationarity** via a **Markov context scheduler**
- **Proactive Context-Based Safety (CPSS)**: safety shield + soft-to-hard budget style constraints
- **(NEW)** **LILAC PLUS** latent context module for SAC:
  - infers a latent context `z` from recent transitions (`q(z|context)`)
  - predicts latent evolution using an **online Bayesian filter** (prior/posterior fusion)
  - detects **change-points** and inflates uncertainty after regime switches
  - converts predicted shift + uncertainty into a **risk signal** for constraint tightening

> Goal: combine **context prediction** (LILAC-style) with **constraints for nonstationary environments** (CPSS), and evaluate on `highway-v0`.

---

## 1) Setup

### 1.1 Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 1.2 Install dependencies
Install from `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

If you don’t have a requirements file in your copy of the repo, you can install the typical stack:
```bash
pip install stable-baselines3 gymnasium highway-env numpy torch matplotlib pandas tensorboard
```

> **Tip:** Do **not** enable rendering during training; it slows everything down drastically.

---

## 2) Quick sanity check (environment)

Run a short training run (SAC) to confirm everything imports:

```bash
python -m scripts.train_continuous --env highway-v0 --total_steps 20000 --seed 0 --run_dir smoke_sac
```

This should create a run directory under `runs/` (see below).

---

## 3) Continuous Control — SAC (Baseline CPSS)

This runs **SAC + CPSS** without the LILAC latent module.

```bash
python -m scripts.train_continuous   --env highway-v0   --total_steps 200000   --seed 0   --p_stay 0.8   --adjust_speed   --run_dir CPSS_baseline_sac
```

Key options:
- `--p_stay` controls how quickly contexts switch (higher = more stationary)
- `--adjust_speed` enables the CPSS “adaptation-speed risk” logic (baseline version)

---

## 4) Continuous Control — SAC + LILAC-style Context Prediction + CPSS (Recommended)

This is the main new pipeline: **learn latent context + predict shift + tighten constraints proactively**.

```bash
python -m scripts.train_continuous   --env highway-v0   --total_steps 200000   --seed 0   --p_stay 0.8   --lilac   --lilac_latent_dim 8   --lilac_context_len 32   --lilac_warmup_episodes 5   --run_dir LILAC_CPSS_sac
```

### 4.1 Unified constraint plugins (recommended)

When `--lilac` is enabled, you can select a constraint 
plugin via `--constraint`:

- `--constraint cpss` (default): passes LILAC risk to the existing CPSS shield hook (`set_adjustment_risk`).
- `--constraint proactive_forecast`: experV4-style proactive tightening: interprets the context-engine output as an **extra safety margin** (epsilon override) computed from predicted shift + uncertainty (+ change-point cooldown).
- `--constraint adjust_speed`: combines LILAC shift with an 
  “adaptation speed” proxy measured from model updates.
- `--constraint none`: disables constraint tightening signals.

Example (default CPSS):

```bash
python -m scripts.train_continuous \
  --env highway-v0 --total_steps 200000 --seed 0 --p_stay 0.8 \
  --lilac --constraint cpss \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir LILACPLUS_CPSS_sac
```

Example (forecast-style tightening):

```bash
python -m scripts.train_continuous \
  --env highway-v0 --total_steps 200000 --seed 0 --p_stay 0.8 \
  --lilac --constraint proactive_forecast \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir LILACPLUS_forecast_sac
```

Example (adjustment-speed safety constraint):

```bash
python -m scripts.train_continuous \
  --env highway-v0 --total_steps 200000 --seed 0 --p_stay 0.8 \
  --lilac --constraint adjust_speed \
  --adj_adapt_window 20 \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir LILACPLUS_adjustspeed_sac
```

### Important note
Do **not** pass `--adjust_speed` together with `--lilac`.

- `--adjust_speed` = “classic” CPSS risk estimation
- `--lilac` = LILAC-style risk estimation (derived from predicted latent shift)

Both write to the same shield signal (`set_adjustment_risk`), so they should not be enabled simultaneously.

---

## 5) Discrete Control — DQN / PPO (Optional)

Baseline discrete runs are still supported:

```bash
# DQN
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 --p_stay 0.8 --run_dir dqn_baseline

# PPO
python -m scripts.train_discrete --env highway-v0 --algo ppo --total_steps 200000 --seed 0 --p_stay 0.8 --run_dir ppo_baseline
```

---

## 6) Logging and Outputs

Runs are stored in:
```
runs/<run_dir>/
```

Typical files you will find:
- `train_monitor.csv` (episode metrics)
- `config.json` (run configuration)
- `tb/` (TensorBoard logs, if enabled)
- plots created by the figures script (if you use it)

### TensorBoard
If TensorBoard logs exist, run:

```bash
tensorboard --logdir runs
```

Then open the printed local URL in your browser.

### LILAC-specific logs
When `--lilac` is enabled, you should see metrics like:
- `lilac/loss`, `lilac/dec_loss`, `lilac/kl`, `lilac/beta`
- `lilac/s_env` (predicted latent shift magnitude)
- `lilac/risk` (adjustment-speed risk passed to the CPSS shield)

### CPSS shield metrics
Your `train_monitor.csv` should include (names may vary by config):
- `adj_risk`, `adj_unsafe`, `adj_s_env`, `adj_s_agent`, `adj_eps_override`
- `shield_used`, `eps` and related shield diagnostics

---

## 7) Recommended training schedule (so you don’t wait forever)

- **Smoke test:** `20k–50k` steps (minutes)
- **Dev/debug:** `100k–200k` steps (tens of minutes)
- **Paper-quality:** `500k–1M` steps (1–3 hours typical on CPU)

If things are slow:
- reduce `--total_steps`
- increase vectorized envs if your script supports it
- ensure `render` is OFF

---

## 8) Full step-by-step runbook (training + diagnostics + tests)

This is the recommended workflow for building reliable thesis results.

### 8.1 Stage 1 — Smoke tests (catch wiring bugs fast)

**(A) Baseline SAC, stationary**

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 20000 --seed 0 \
  --constraints off \
  --run_dir S0_baseline_stationary
```

**Pass criteria**
- finishes without crash
- `runs/S0_baseline_stationary*/train_monitor.csv` exists and has rows
- no NaNs in reward/return columns

**(B) Nonstationary env, no constraints, no LILAC**

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 20000 --seed 0 --p_stay 0.8 \
  --constraints off \
  --run_dir S1_nonstationary_no_constraints
```

**(C) Shield + conformal calibration residual (ExpertV4 parity)**

Run with conformal enabled (i.e., do *not* pass `--no_conformal`).

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 30000 --seed 0 --p_stay 0.8 \
  --constraints on \
  --run_dir S2_shield_conformal_only
```

**Diagnostics to check**
- `calib_resid` appears in logs (if you log per-step info)
- `inflate` becomes non-zero after enough residuals
- `shield_used` is sometimes > 0

**(D) LILAC PLUS on, constraints off**

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 30000 --seed 0 --p_stay 0.8 \
  --lilac --constraint none \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir S3_lilac_only
```

**Diagnostics to check**
- `lilac/dec_loss`, `lilac/s_env`, `lilac/s_uncert`, `lilac/cp_flag` exist (TensorBoard / logs)
- losses are finite

**(E) Full integration: LILAC PLUS + constraint plugin + shield**

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 50000 --seed 0 --p_stay 0.8 \
  --lilac --constraint proactive_forecast \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir S4_lilac_plus_proactive
```

**Pass criteria**
- `adj_risk` / `adj_unsafe` (or equivalent) appear
- `eps_override` increases on risk spikes / after change-points

### 8.2 Stage 2 — Development runs (debug behavior)

After smoke tests pass, run 200k steps and inspect curves:

```bash
python -m scripts.train_continuous \
  --env highway-v0 --algo sac --total_steps 200000 --seed 0 --p_stay 0.8 \
  --lilac --constraint proactive_forecast \
  --lilac_latent_dim 8 --lilac_context_len 32 --lilac_warmup_episodes 5 \
  --run_dir D1_sac_200k_lilac_proactive
```

Repeat with multiple seeds (`--seed 0,1,2`).

### 8.3 Stage 3 — Thesis-quality ablations

Recommended grid:
- steps: `500k` (or `1M` for final)
- seeds: `0..4`
- nonstationarity: `p_stay ∈ {0.95, 0.8, 0.6}`
- conditions:
  - baseline (no constraints, no LILAC)
  - shield-only (ExpertV4 shield + conformal)
  - LILAC PLUS only
  - LILAC PLUS + proactive_forecast
  - LILAC PLUS + adjust_speed
  - LILAC PLUS + cpss

---

## 9) New helper scripts (grid runner + thesis plots)

### 9.1 Run a thesis grid

This runs multiple jobs sequentially and writes a manifest JSON.

```bash
python -m scripts.run_thesis_grid \
  --lilac \
  --env highway-v0 --algo sac \
  --total_steps 500000 \
  --seeds 0,1,2 \
  --p_stays 0.95,0.8,0.6 \
  --constraints none,proactive_forecast,adjust_speed,cpss \
  --tag THESIS
```

Dry-run (prints commands only):

```bash
python -m scripts.run_thesis_grid --lilac --dry_run
```

### 9.2 Generate thesis plots

After runs finish:

```bash
python -m scripts.make_thesis_plots \
  --runs_root runs \
  --out_dir runs/thesis_plots \
  --window 20 \
  --bins 10
```

Outputs (in `runs/thesis_plots/`):
- `learning_curves_return.png`
- `learning_curves_violation.png`
- `intervention_rate.png`
- `risk_calibration.png`
- `summary_table.csv`

---

## 10) Recommended diagnostics (what to check)

### 10.1 Safety / constraints
- violation rate (per episode)
- time-to-first-violation
- shield intervention rate (`shield_used`)
- epsilon terms: base `eps`, conformal `inflate`, plugin-driven `eps_override`
- CPSS budget terms (if using CPSS): remaining budget / spent budget

### 10.2 LILAC PLUS
- `s_env` (predicted shift)
- `s_uncert` (uncertainty)
- `cp_flag` (change-point)
- `risk` (risk passed to constraint)

### 10.3 Conformal calibration
- `calib_resid` (clearance proxy residual)
- inflation `inflate` trending with residual distribution


---

## 8) How the LILAC + CPSS coupling works (high level)

When `--lilac` is enabled, the code does:

1. Collect recent transitions as a **context window** (length `--lilac_context_len`)
2. Encode latent context:
   - `q(z|context)` (Gaussian) → sample `z`
3. Predict next-episode latent using an **LSTM prior**:
   - `p(z_next | z_history)`
4. Convert predicted shift to an **adaptation-speed risk**:
   - `s_env = || z_next - z ||`
   - `risk = sigmoid((s_env - margin) * temp)`
5. Send this risk to the shield via:
   - `env_method("set_adjustment_risk", risk=..., unsafe=..., s_env=..., s_agent=...)`
6. The CPSS shield **tightens epsilon** when unsafe and optionally overrides actions

This matches the paper’s idea of learning structured nonstationarity via latent variables, but adds your safety constraint mechanism. (See: “Deep Reinforcement Learning amidst Continual Structured Non-Stationarity”, ICML 2021.)

---

## 9) Troubleshooting

### “Training is stuck / extremely slow”
Common causes:
- rendering enabled
- too many envs on a weak CPU
- Windows antivirus scanning the run directory

Fix:
- disable render
- try `--total_steps 20000` to confirm pipeline works
- move repo to a shorter path (Windows)

### “Module not found / imports fail”
- confirm you are in the venv
- run from repo root
- ensure `pip install -r requirements.txt` succeeded

### “No TensorBoard logs”
That’s OK — some configs only write CSV by default. Use:
- `runs/<run_dir>/train_monitor.csv`

---

## 10) Reproducing the main result (suggested ablations)

Run 3 seeds each:

1) **SAC + CPSS baseline**
2) **SAC + LILAC + CPSS**
3) (Optional) **SAC only** (no constraints) to show the safety/return trade-off

Example:
```bash
python -m scripts.run_experiments --env highway-v0 --algo sac --seeds 0,1,2 --p_stay 0.8 --constraints on --lilac off
python -m scripts.run_experiments --env highway-v0 --algo sac --seeds 0,1,2 --p_stay 0.8 --constraints on --lilac on
```

(If `scripts/run_experiments.py` differs in your repo version, use the individual `train_*` commands above.)

---

## 11) File map (where things live)

- `scripts/train_continuous.py` — SAC entry point (includes `--lilac` flags)
- `src/lilac/` — LILAC-style latent module implementation
  - `modules.py` — encoder, LSTM prior, decoder, KL utilities
  - `wrappers.py` — observation augmentation with `z`
  - `callback.py` — trains latent module and updates CPSS risk

---

If you want, I can also add a **one-command “smoke test” script** and a **`make_figures.py`** section to this README that produces the main plots automatically from the `runs/` directory.

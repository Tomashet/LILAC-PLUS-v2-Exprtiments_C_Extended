# LILAC+ research experiments: safety constraints under changing conditions

Research code by Tim Tomashevskiy for exploring context-based constraints (CB), adjustment-speed constraints (AS), and soft-to-hard-inspired action filtering (SH) with Stable-Baselines3 and highway-env.

**Status:** experimental research snapshot. The current continuous runner is a SAC demonstration of context-dependent safety-control logic. It is not a verified reproduction of the collision reductions in the linked papers, nor a complete end-to-end learned LILAC+ pipeline.

## Start here

- `scripts/train_continuous.py`: current SAC entry point and method configurations.
- `src/safety_wrapper.py`: action-cost proxy, context-dependent thresholds, action smoothing, and fallback logic.
- `src/context.py`: context-key normalization and threshold inference.
- `src/lilac/`: separate latent-context modules retained for research; **not connected to the current continuous runner**.
- [Implementation limits](docs/implementation-status.md): what the current experiments do and do not establish.
- [Repository selection review](docs/repository-review.md): why this snapshot was selected for cleanup.

## Installation

Python 3.12 was used for the application-readiness checks. Create an isolated environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For CPU-only PyTorch, install it first from the official wheel index, then install the remaining dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-demo.txt
```

`requirements-demo.txt` specifies the demo's core environment versions. It is not a full transitive lockfile. The original `requirements.txt` is retained for historical experiments; its SB3 extras can pull unrelated Atari dependencies.

## Short execution check

Run from the repository root. The output directory must not already exist, preventing old artifacts from being mistaken for new results.

```bash
python -m scripts.test_pipeline_smoke --steps 120 --run_root runs/application_smoke --threshold_patch artifacts/calibration/recommended_thresholds_patch.json
```

This performs threshold audits and short SAC runs for an adaptive configuration and a calibrated fixed baseline. It checks for a saved model, run metadata, completed episodes, and finite episode returns. It verifies execution, **not convergence or a safety guarantee**.

## Run a baseline and an adaptive configuration

```bash
python -m scripts.train_continuous --env highway-v0 --method unconstrained --regime stationary --total_steps 120 --max_episode_steps 20 --seed 0 --run_dir runs/demo_baseline
python -m scripts.train_continuous --env highway-v0 --method cb+as+sh --regime nonstationary_seen --total_steps 120 --max_episode_steps 20 --seed 0 --run_dir runs/demo_adaptive
```

For longer exploratory runs, increase `--total_steps`; choose budgets according to available compute. No training duration or performance level is promised by this example.

Output paths are used literally by this runner. Each run writes `run_config.json`, `run_debug.json`, `train_monitor.csv`, `calibration_monitor.csv`, and `model.zip`.

### Interpreting the output

- `r` in the Monitor file is episode return, including configured penalties.
- `violation_count` and `near_miss_count` refer to **threshold exceedances of an action-magnitude proxy**, not collision counts or geometric near misses.
- `shield_count` counts action overrides; `action_correction_mean` measures their magnitude.
- Context regime names refer to the configured schedule. In this runner they change controller metadata and thresholds; they do not by themselves change road layouts, traffic dynamics, or sensor observations.

See [the implementation notes](docs/implementation-status.md) before using these metrics in a scientific comparison.

## Checks

```bash
python -m pip install pytest
python -m pytest tests -q
```

The regression tests cover calibrated threshold records used by fixed baselines. They do not validate the paper claims.

## Related research

- [Safe Continual Reinforcement Learning under Nonstationarity via Adaptive Safety Constraints](https://arxiv.org/abs/2605.18842)
- [From Cumulative Constraints to Adaptive Runtime Safety Control for Nonstationary Reinforcement Learning](https://arxiv.org/abs/2605.18841)
- [Proactive Context-Forecasted Safety Constraints for Nonstationary Reinforcement Learning](https://arxiv.org/abs/2609.08080), CoLLAs 2026

These links provide research context. This snapshot's provenance against each publication still needs to be established; do not attribute a paper's numerical results to the short demo.

## Historical files

Previous commands and experimental runbooks are preserved in [legacy-runbook.md](docs/legacy-runbook.md). Some use obsolete CLI flags and should not be treated as current instructions. Existing artifacts and run directories are retained as historical records, without changing or regenerating their reported numbers.

## License

MIT; see [LICENSE](LICENSE).

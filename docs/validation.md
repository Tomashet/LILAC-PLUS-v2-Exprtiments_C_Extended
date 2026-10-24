# Cleanup validation

Validation was performed in an isolated Python 3.12 environment with CPU PyTorch 2.14.1, NumPy 1.26.4, Gymnasium 0.29.1, highway-env 1.8.2, and Stable-Baselines3 2.3.2.

- `python -m pytest tests/test_fixed_thresholds.py -q`: 3 tests passed.
- Threshold audits completed both with and without the retained calibration patch.
- `highway-v0` SAC runs completed for `unconstrained`, `cb+as+sh`, and calibrated `fixed_full_A`.
- Each run used seed 0, 120 environment steps, and a 20-step episode cap. Each saved model was reloaded successfully and recorded 20 gradient updates.
- Both adaptive/fixed smoke outputs contained 6 completed episodes with finite returns.
- All Python files under `src/` and `scripts/` parsed successfully after repairing the legacy launcher's indentation.
- `git diff --check` passed.
- A common-pattern credential scan found no matches in inspected tracked files under 2 MB. Only the current shallow checkout was scanned; this is not an exhaustive secret or history audit.

The first continuous merge-v0 check failed because its reward expects `speed_index` on a vehicle type that continuous control does not supply. The quick-start was changed to highway-v0; merge-v0 was not silently patched to change reward behavior.

These short runs verify basic training, logging, serialization, and calibrated-baseline execution. They do not validate learning convergence, collision reduction, physical distribution-shift robustness, or the paper's safety theorem. Historical data was not regenerated or relabeled.

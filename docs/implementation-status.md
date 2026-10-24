# Implementation status and limits

## Current continuous entry point

`scripts/train_continuous.py` uses SAC, `SafetyMethodWrapper`, and externally specified context tuples. It does not import or train the latent encoder in `src/lilac/`. The old README's `--lilac`, `--constraint`, and `--p_stay` examples do not describe this entry point.

The wrapper's `_proxy_cost` is a weighted magnitude of the proposed continuous action. `violation` and `near_miss` are exceedances of configured thresholds on that proxy, computed before any safety override. They are not measurements of collisions or distance to another vehicle. The calibration column named `clearance` also currently contains this proxy. Treat it as a legacy column name, not physical clearance.

The schedule is available to the wrapper, which can use the next scheduled context as a forecast. This is oracle information, not evidence of learned forecasting. Schedule updates change context metadata and controller settings; the current runner does not apply density, behavior, or sensor categories to highway-env dynamics or observations. Therefore `nonstationary_unseen` is not, by itself, a held-out physical-environment generalization experiment.

The budget is a heuristic controller state that decreases on proxy violations. This implementation alone does not establish the finite-horizon cumulative-cost guarantee in the CPSS theory. The fallback braking/steering action is heuristic and is not verified recursively feasible.

The `lilac_base` method name does not establish that a learned LILAC model is running. Use the imports and actual call path when describing the implementation.

## Readiness work completed on this branch

- Replaced obsolete front-page commands with commands using the current parser.
- Preserved the previous README as a historical runbook.
- Corrected a fixed-baseline crash: Python evaluated a dictionary fallback on `ThresholdRecord` even when the requested attribute existed. Both mappings and records now go through the existing normalizer.
- Updated the smoke runner to supply the required method argument, complete short episodes, reject reused output directories, and verify finite returns and newly produced artifacts.
- Added a demo dependency list that avoids unrelated SB3 Atari extras.

## Compatibility finding

The continuous `merge-v0` path failed the short execution check with highway-env 1.8.2: its reward function expects a discrete-action vehicle with `speed_index`. The existing compatibility patch only converts the reward argument and does not supply that vehicle interface. The supported smoke command therefore uses `highway-v0`. Do not describe continuous merge-v0 as validated.

## Before claiming full research reproduction

Map a pinned commit and configuration set to the paper, establish physically meaningful task and safety metrics, verify that context changes affect the intended environment components, identify oracle versus learned information, and run the relevant multi-seed experiment. This cleanup does not silently change any of these research semantics or replace the original artifacts.

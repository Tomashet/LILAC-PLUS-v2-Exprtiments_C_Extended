# Application repository review

## Selection

The connected Tomashet account exposes 29 owned repositories. The inventory includes public and private work. Selection is based on code and documentation inspection, not repository size or recency alone.

**Primary candidate:** `LILAC-PLUS-v2-Exprtiments_C_Extended`, after the reviewed cleanup. It has the broadest inspected public experiment implementation: CB/AS/SH method configurations, context threshold inference, SAC training, stage-based analysis, and retained results. It needs an honest front page because its old documentation describes a different pipeline.

**Alternative small demonstration:** `experV4`. It has a compact DQN/PPO/SAC structure and geometry-based safety code. It uses oracle-context/proxy components, and its continuous action handling and metric propagation need runtime review before it can be promoted as a reproduction package.

This selection does not mean the selected snapshot reproduces all thesis or conference claims. See implementation-status.md.

## Inventory and disposition

| Repository or group | Inspection finding | Recommended role |
| --- | --- | --- |
| LILAC-PLUS-v2-Exprtiments_C_Extended | 523 tree entries; 448 tracked files in checkout; broad current method/analysis structure; obsolete README and broken smoke command | Primary cleanup target; keep experimental limitations visible |
| LILAC-PLUS-v0 | 348 tree entries; same legacy README; continuous CLI differs; smoke omits required method | Historical predecessor, not the application link |
| LILAC-PLUS (private) | 3,082 tree entries, many stored run outputs and copies | Research archive; retain privately |
| experV4 | Compact earlier proactive-safety experiments; oracle contexts acknowledged | Secondary demo candidate; provenance and execution still need checking |
| ExpertV5 | README advertises plot_run.py and plot_compare.py, absent from fetched tree | Historical adjustment-speed experiments; repair before highlighting |
| ExpertV5-main (private) | Separate similarly named snapshot; not deeply inspected | Retain pending comparison, not an automatic duplicate deletion |
| ExpertV6_-CPSS- | Several empty module files, including rollout/sampling.py and training/runner.py | Incomplete CPSS development snapshot |
| ExpertV6-CPSS- | GitHub reports size zero | Check before archiving; not a showcase candidate |
| Soft2Hard-NS (private) | Budget code and patched experiments; documentation mixes estimated-risk filtering with strong enforcement claims | Preserve as development work; reconcile theory and implementation |
| Soft2HardNs (private), SofToHard (public) | Further soft-to-hard versions; SofToHard has source/config/script structure | Historical candidates requiring deeper comparison |
| Soft2Hard | GitHub reports size zero | Not a showcase candidate |
| NS-CONSTRAINTS (private), NS-CONSTRAINTS3, NS-CONSTRAINTS5 (private) | Small earlier snapshots; NS-CONSTRAINTS3 contains t2c_prosafe | Retain; investigate only if needed to establish provenance |
| safe-nonstationary-driving-experiments, safe-nonstationary-driving-experiments_v4, safe-nonstationary-driving-experiments_v4-main (private) | Earlier experiment snapshots; v4 root has scripts/src/configs | Preserve as source history; no private application links |
| FSRL, Safe-Rl-Benchmark | SafeRLBench-style library structure, distinct from current thesis experiments | Establish upstream attribution and original contributions before showcasing |
| SafeRLBaselines | Root mainly README and Safe-RL directory | Establish original contributions before showcasing |
| gymnasiumT | Gymnasium library structure with upstream-style docs/tests | Dependency/reference work, unless a specific original contribution is documented |
| gymnasium (private) | GitHub reports size zero | Not a showcase candidate |
| gym_custom_env11 (private) | Warehouse robot environments/training, notebook, sprites, IDE files | Possible secondary environment example after cleanup |
| CAS771 | GitHub reports size zero | Coursework/history, not current primary candidate |
| IBM_TRIAL | Only two tree entries; README 47 bytes | Not a primary research showcase |
| networking-intelligence-scripts (private) | Separate automation project | Outside this RL application review |
| tomashet.github.io | Research website | Use as a navigation hub, not as the code sample |

## Findings that affect scientific interpretation

The extended runner currently uses action-cost threshold exceedances rather than collision counts, reads scheduled future contexts, and changes controller metadata without applying those categories to physical traffic or sensors. Its latent modules exist separately but are not wired into the current SAC entry point. These are reasons to document scope accurately, not to silently relabel old results or rewrite research behavior for an application.

The inspection also found an eager-default bug in calibrated fixed-baseline configuration, a missing required method in the smoke runner, and an indentation error in the legacy run_all launcher. The cleanup fixes these execution problems while preserving original result artifacts.

## Cleanup policy used

Changes are confined to a review branch. Historical repositories, original stored results, repository visibility, and main branches remain unchanged. Existing generated outputs were not deleted. A common credential-pattern scan of the selected checkout found no matches; the shallow checkout does not constitute a complete history audit.

## Remaining portfolio work

After reviewing this branch, establish the exact paper-to-commit mapping and choose whether to curate this as an explicitly scoped experiment demo or locate the full paper reproduction code among the retained snapshots. Only then consider a shorter public repository name and profile pinning. Renaming, deleting, archiving, or making private code public is not part of this cleanup.

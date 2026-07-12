# C-ACT code and protocol review

**Review status:** code paths are substantially hardened, but the end-to-end protocol is not yet empirically complete. The release runner now fails closed when sealed task cards, real E2 rollouts, E1c paired data, or D_audit evidence are missing.

## Completed code-quality fixes

- E2 direct rollouts now use isolated stores, a common `matched_cell_id`, explicit world-snapshot hashes, and real coverage/HRR/EAHR fields. The selector rejects incomplete cells, duplicate methods, mixed snapshots, failed rollouts, and constraint violations.
- `CactMemory` and the runner propagate branch mode, parent identity, target opportunity, prefix assignment/trace, kappa override, and snapshot hash. Paired branches are checked for identical prefixes before a preference label is emitted.
- E1c generation and pairwise training are fail-closed at exactly 320 rows, with leakage-safe 240/80 parent-episode splitting, unique `pair_id` values, binary preferences, validation-selected L2/threshold, and no fabricated outcomes.
- VLM lifecycle management now verifies health, handles occupied ports safely, cleans owned processes, and routes E5 through a multi-GPU port pool when configured.
- Missing online metrics remain `null` instead of silently becoming zero; E5 result rows carry the true `run_id` used to read logs.
- Frozen-store hardlinks are opt-in only (`CACT_FROZEN_HARDLINK=1` plus `CACT_ALLOW_UNSAFE_HARDLINK=1`); the default is a safe copy.
- `validate_e2_audit.py` validates the direct selected-policy audit and the sealed paired audit before E3.

## Verification performed

- All repository Python files compile with `py_compile`.
- `tests/test_protocol_v2_smoke.py`, `tests/test_controller_ledger.py`, and `tests/test_extreme_stress.py` pass.
- Synthetic checks pass for pairwise 320-row training, E2 selection, branch initialization, and E1c dry-run planning.
- `tests/test_runner_integrity.py` could not run because this Windows environment does not have `pytest`; install the pinned test dependencies on Ubuntu and rerun it before release.

## Remaining empirical blockers (intentional)

1. The repository does not contain the sealed task-card registry required by the protocol. `cact_calib.yaml` has 12 generic entries and does not provide all required fields (template hash, generator version, inventory, constraints, predicate, trigger, budget, and randomization state) for a disjoint 8-task D_select/8-task D_audit design. `run_all.sh` now stops at M0 unless `CACT_TASK_CARDS` points to validated JSON/YAML cards.
2. Real matched-risk E2 output still must be collected with the actual Minecraft/VLM stack and a `world_snapshot_manifest`; no synthetic table is accepted.
3. Real E1c output still must contain 320 non-tied paired opportunities from sealed branches. The generator checks prefix identity, target reachability, snapshot identity, and observed outcomes.
4. D_audit still requires externally supplied direct rollouts for the four selected methods and at least 200 sealed paired audit opportunities. The validator is present; the generator is deliberately not allowed to invent these artifacts.
5. The complete protocol has not been executed on an Ubuntu host with Java 21, Minecraft, CUDA, and the target VLM. Local smoke tests are not evidence of scientific results.

## Method and experiment-design improvements recommended before submission

- Make the primary claim decision-local: "reuse only when the current opportunity is certified safe and useful." Treat lifecycle/Thompson/sanitizer mechanisms as implementation details or ablations, not independent novelty claims.
- Register a single immutable task-card manifest with 8 D_select and 8 disjoint D_audit cards, plus a separate E3/E4 registry. Hash every template and world snapshot; never derive task labels from observed outcomes.
- Select E2 using coverage first subject to absolute HRR and incremental EAHR constraints, then report hierarchical bootstrap confidence intervals over task-seed cells (selection and final reporting must use disjoint cells).
- Report applicability boundaries: unsupported/fallback rate, worst task stratum, and performance conditional on a valid certificate. Do not advertise an OOD guarantee unless an actual held-out distribution is added.
- Keep D_pair-train parent-episode grouping and add a preregistered rule for ties/over-sampling. If ties reduce usable rows, collect additional sealed candidates until exactly 320 non-ties are obtained rather than relabeling ties.
- For D_audit, freeze the selected policy before collecting audit data, include all four methods in every matched cell, and publish the audit schema and snapshot hashes with the paper artifact.
- Add a reproducibility manifest containing code commit, dependency lock, Java/Minecraft/VLM versions, GPU model, CUDA settings, seeds, and all artifact hashes. Report throughput and peak memory on Ubuntu separately from scientific metrics.

**Submission decision:** after the empirical artifacts above are collected and validated, rerun the full release script and the missing pytest suite. Until then, the project is code-ready for server execution but not result-ready for a paper claim.
## Deep-pass additions

- Parallel execution now checkpoints only successful/skipped episodes and raises on any failed/timeout/error batch; a resume cannot silently skip a failed episode.
- Frozen protocol runs can require a canonical `task_id|world_seed -> snapshot_hash` manifest. `run_all.sh` enables this requirement by default and passes the manifest into E3/E4; missing hashes fail closed.
- Hydra branch-prefix JSON is quoted as a string, preventing a trace from being parsed as a structured override.
- E2 and D_audit validators now require run/store identifiers, return code, matched-cell identity, finite [0,1] metrics, and paired snapshot agreement.
- The main episode process returns nonzero for reset, timeout, exception, or environment-logger failures; genuine task failure remains a recorded outcome rather than infrastructure success.
- Task-card validation accepts a sealed card manifest/list, rejects duplicate task IDs, and reports the actual card count.

A remaining environment-level risk is that `preferred_spawn_biome` must be verified from the runtime observation/world snapshot, not inferred from configuration alone. Task cards should therefore include a post-reset world assertion and its hash.
- Paper sources were synchronized with the executable grid: six methods, E3 = 1,728 episodes, E4 = 240 episodes, and E5 = 2,600 episodes under the current five-stream/ten-round configuration. Unsupported prospective-result wording was removed from the LaTeX abstract, and the synchronized paper files are now included in the release manifest.
- E2/D_audit matched cells additionally require a common episode ID across methods, preventing a nominal cell key from masking non-paired episodes.
**Audit scope note:** the configured cross-model reviewer endpoint was unavailable in this session, so the findings above are a local static/code-and-protocol audit. They are not a substitute for an independent reviewer pass.
- Ubuntu performance was improved by bounded VLM health polling, safe process cleanup, multi-GPU port routing, reduced thread oversubscription, and fail-fast manifests; quantitative GPU/CPU profiling remains a server-side task because this host has no CUDA.
## One-command deployment additions

- `protocol_inputs/task_cards.json` now contains a sealed 8-task D_select / 8-task D_audit registry derived from frozen benchmark templates; `validate_task_cards.py --require-sealed` checks it before claiming runs.
- `build_task_card_registry.py` regenerates the registry deterministically without reading outcomes.
- `collect_world_snapshots.py` hashes real Minecraft save trees with a documented canonical-tree rule and refuses missing/empty worlds.
- `run_e2_audit_rollouts.py` collects the four selected direct audit methods and invokes the real paired-branch collector for the >=200-row audit.
- `setup_and_run.sh` checks dependencies, prepares the registry/manifest, enables real E2/E1c/D_audit collection, and delegates to `run_all.sh`. It still requires either an existing manifest or a real world-root template; no world state is invented.

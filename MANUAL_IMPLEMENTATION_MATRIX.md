# New Manual Implementation Matrix

## Implemented and locally verified

| Manual area | Implementation |
|---|---|
| Applicability/specification | `cact/contract.py`, `cact/cact_memory.py` |
| Sequential opportunity schema | `cact/protocol_v2.py`, `analysis/validate_logs.py` |
| Four-level evidence backoff | `AIPWEstimator.aggregate`, `AdmissionPolicyV2` |
| Logistic AIPW and fold seeds | `cact/protocol_v2.py` |
| Unified positive-risk ledger | `AdmissionPolicyV2`, certificate logging in `CactMemory` |
| Pairwise preference baseline | `cact/preference_gate.py`, `experiments/train_pairwise.py` |
| Frozen hash and release artifacts | `parallel_runner.py`, `release_protocol.py`, `protocol_release/` |
| ECHR/EAHR/budget/boundary metrics | `cact/metrics.py` |
| Paired hierarchical bootstrap utility | `analysis/bootstrap.py` |
| Split/task-card validation tools | `analysis/split_audit.py`, `analysis/validate_task_cards.py` |
| Ubuntu runtime optimizations | `parallel_runner.py`, `online_runner.py`, `trust_store.py`, `docs/UBUNTU_PERFORMANCE.md` |

## Implemented as fail-fast gates, requiring real artifacts

- `D_pair-train`: `run_all.sh` stops if sealed `pairs.jsonl` is absent.
- `D_audit`: calibration script refuses to emit a deployable policy when audit fails.
- Task-card completeness: validator refuses incomplete cards instead of inventing initial states or world constraints.
- Frozen evaluation: store/policy mutation is a failure, not a warning.

## Not claimable without external execution

- E0 detector accuracy, paired branch agreement, D_select/D_audit feasibility, E3 confirmatory results, and E5 stream stability require real Minecraft/VLM rollouts.
- Ubuntu throughput numbers require a Linux/GPU run; current development host cannot measure them.

This matrix is intentionally conservative: a code path is not marked empirically complete merely because it compiles.

## Remaining protocol-level qualification

- The artifact schema separates `full` and `pointwise` controller families, but their independent matched-risk kappa selection still requires real D_select rollouts; the code does not claim those empirical selectors have passed.

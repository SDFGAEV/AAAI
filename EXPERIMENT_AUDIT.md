# C-ACT Implementation Audit

**Date:** 2026-07-11  
**Scope:** new protocol manual, current repository HEAD, local static/protocol checks

## Verdict: WARN (implementation complete; execution data not yet complete)

### Implemented

- Four-level evidence hierarchy `g0→g3`, logistic nuisance AIPW, episode-level folds, cluster uncertainty, and parent heterogeneity penalty.
- Applicability contract masking, eligibility/censor fields, H1–H6 outcome slots, propensity logging, and frozen artifact/store hash checks.
- Unified positive incremental-risk ledger with no-credit fallback rule and episode reset.
- Six-method registry including `PairwisePreferenceGate`, `C-ACT-Pointwise`, and ledger-enabled `C-ACT`.
- Pairwise preference model/trainer with schema validation and fail-fast missing-artifact behavior.
- E0–E5 runner, protocol release manifest, substrate manifest, task registry, deviation log, and new paper draft.

### Verified

- All core Python modules compile with `py_compile`.
- `tests/test_protocol_v2_smoke.py` passes.
- E0 task grid parses correctly.
- `protocol_release` regenerates with the new manual hash and eight benchmark registries.
- Git whitespace check passes.

### Blocking/qualification items

1. No real Minecraft/VLM outcomes have been collected in this audit.
2. `run_all.sh` intentionally stops if sealed `D_pair-train` `pairs.jsonl` files are absent; preference labels are never synthesized.
3. Full pytest is environment-dependent and was not used as the acceptance gate because the bundled runtime lacks the project's optional test dependencies.

No numerical result or safety claim should be made until E0 passes, D_pair-train exists, D_audit passes once, and E3/E5 frozen artifacts are available.

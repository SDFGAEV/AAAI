# Trust Before Reuse: Risk-Budgeted Counterfactual Admission of Self-Evolved Knowledge

## Abstract

Self-evolving agents turn interaction traces into persistent knowledge, but retrieval is not evidence that reuse will improve the current decision. We present C-ACT, a decision-time admission controller between a fixed top-1 retriever and the same base planner. C-ACT combines applicability masking, episode-clustered cross-fitted AIPW, four-level evidence backoff, and a deterministic positive incremental-risk ledger. The ledger charges only admitted windows by the positive upper incremental-risk estimate; fallback neither earns credit nor spends budget, and a new episode resets the ledger. A separate PairwisePreferenceGate is trained only from sealed paired-branch preference data.

We evaluate six preregistered methods—NoKnowledge, NoGate, FixedBayes, PairwisePreferenceGate, C-ACT-Pointwise, and C-ACT—under E0–E5, including ID-seed, compositional-OOD, contract-boundary, ledger-stress, four ablations, and five controlled streams. All frozen runs share the same candidate, planner, world seed, budget, and prompt except the raw knowledge block. We report harmful-reuse rate, risk–coverage, ledger exhaustion, unsupported fallback, boundary false admission, success, and cost. The paper makes no universal safety claim: conclusions are restricted to eligible, supported, retrieval-conditional contexts and the declared rollout horizon.

## 1. Problem and method

At an eligible decision checkpoint, (A=1) injects the retrieved self-evolved knowledge and (A=0) executes the same base planner without that block. A single local intervention window is used; a second intervention censors the opportunity. Contract fields answer only applicability—scope, preconditions, hard non-applicable boundaries, postconditions, provenance, and raw text. They are never used as outcome or harm proxies.

The estimator uses known propensity (e(X)=0.5) during randomized logging, five episode-level folds with three preregistered fold seeds, L2-regularized logistic nuisance models, and AIPW pseudo-outcomes for (Y), (H(1)), and (H(1)-H(0)). Evidence is queried from the finest supported hierarchy (g^0) (source×type×task-group×failure-type×risk-tier×resource-scarcity×boundary) through (g^1), (g^2), and (g^3) (source×type). Arm support requires (n_1,n_0ge12) and ESS≥24; parent backoff receives a between-child heterogeneity penalty.

For a selected context group, C-ACT admits only when applicability holds, benefit LCB≥0.05, absolute-risk UCB≤0.10, incremental-risk UCB≤0.02, support is deployable, and the episode ledger has sufficient remaining budget. The charge is (c_t=[\overline r^{inc}_t]_+). Negative estimates do not create credit. Episode-level harm is measured directly from non-overlapping H1–H4 environment events; H5 efficiency harm and H6 paired-audit harm are reported separately.

## 2. Fit, selection, audit, and frozen evaluation

D_fit trains nuisance models; D_select chooses the sole (κ) from ({0,.5,1,1.5,2,2.5,3}); D_audit is opened once and cannot be reused to tune the controller. A missing or failed D_pair-train artifact stops the pipeline rather than synthesizing preference labels. `protocol_release/manifest.json`, `substrate_manifest.json`, task registries, seed registries, raw JSONL, hashes, exclusions, and deviations are required before E3.

E3 uses 36 conditions×8 seeds×six methods. E4 contains exactly four variants: full C-ACT, no Contract, no Adaptive Calibration, and no Active Base Logging. E5 uses five independent controlled streams with shared candidate/evidence updates followed by method-specific frozen retention and exclusive hard-transfer evaluation. Frozen processes cannot write knowledge, evidence, thresholds, or policy artifacts; store and policy hashes are checked before and after each run.

## 3. Claims and limitations

The confirmatory claims are: (C1) relative to NoGate, C-ACT reduces harmful reuse without unacceptable success/coverage loss; and (C2) applicability boundaries and adaptive risk–coverage improve boundary discrimination under held-out audit. These are empirical, local, and retrieval-conditional claims. The protocol does not identify long-horizon individual causal effects, repair retrieval failure, or guarantee safety on unsupported OOD contexts. All reported values must be traceable to versioned artifacts; no result is inserted before the corresponding audit passes.

# Trust Before Reuse: Contract-Aware Adaptive Counterfactual Trust for Self-Evolved Knowledge

## Abstract

Self-evolving agents accumulate skills, remedies, and procedural memories, but retrieval does not imply that reusing a memory will help in the current state. A memory can be relevant and historically successful while being harmful under a different resource configuration or safety boundary. We study this decision-time admission problem and present **C-ACT**, a governance layer that decides whether an already-retrieved knowledge item may influence the next action.

C-ACT makes three design commitments. First, each item is represented by an auditable Knowledge Contract with preconditions, postconditions, and hard non-applicability boundaries. Second, the gate estimates reuse value against a randomized base-policy arm with recorded assignment propensity, rather than comparing reuse success with an uncontrolled historical average. Third, risk thresholds are selected on a calibration split and then frozen for held-out evaluation; probationary reuse is supervised and cannot bypass contract or postcondition checks. The design is intentionally modular: C-ACT does not generate knowledge and makes no claim about a particular upstream memory generator.

We instantiate the protocol in Minecraft and evaluate six preregistered methods: NoKnowledge, NoGate, FixedBayes, PairwisePreferenceGate, C-ACT-Pointwise, and C-ACT on disjoint in-distribution, compositional-OOD, contract-boundary, ablation, and long-horizon stream splits. The primary endpoints are harmful-reuse rate and Coverage@Risk≤10%, with task success as a secondary endpoint. All comparisons are paired by task and world seed and analyzed with episode-clustered bootstrap confidence intervals. Numerical results are intentionally omitted from this preregistered draft; the submission version will populate tables only from the versioned artifact manifest produced by the protocol.

## 1. Introduction

Open-world agents increasingly improve by writing experience back into a knowledge store. The difficult step is not only generating or retrieving a memory, but deciding whether the memory should be allowed to change behavior now. A recipe correction may be useful when the required resource is available and unsafe when the resource is scarce. A failure remedy may repair one task family and cause a different failure when its precondition is absent. Similarity and historical success therefore provide evidence of relevance, not evidence that reuse causes improvement in the current context.

We formulate knowledge admission as a binary decision between two policies: reuse the retrieved item or execute the same base policy without it. The comparison must be randomized or otherwise propensity-aware, and its outcome must be defined by the environment rather than by the gate's own posterior. This framing separates three concerns that are often conflated: whether a memory is applicable, whether it improves the next outcome, and whether it crosses a safety boundary.

Our contribution is a compact, auditable protocol and implementation of C-ACT:

1. **Knowledge Contracts** expose preconditions, postconditions, and hard non-applicability boundaries before statistical gating.
2. **Propensity-aware admission** estimates the probability that reuse is beneficial relative to a recorded base-policy arm and separately labels harmful reuse from environment outcomes.
3. **Held-out risk calibration** selects thresholds on a calibration split, freezes them for evaluation, and reports a risk–coverage curve rather than an unsupported safety guarantee.
4. **A leakage-resistant evaluation protocol** separates calibration, ID generalization, compositional OOD, contract-boundary, ablation, and long-horizon stream tasks.

The paper makes no claim that C-ACT is a knowledge generator, an exact reproduction of XENON, or a finite-sample safety certificate. It is a decision-time admission layer whose empirical value is tested by the protocol below.

## 2. Problem formulation

Let (u) be a retrieved knowledge item and (c) the current context. Let (A\in\{0,1\}) denote the intervention: (A=1) injects (u), while (A=0) executes the same base policy without (u). The dispatcher records (e(X)=P(A=1\mid X)) for every randomized decision. For an episode-level outcome (Y), the target effect is

\[
\Delta(u,c)=E[Y(1)-Y(0)\mid u,c].
\]

The gate uses the posterior probability

\[
\pi_\delta(u,c)=P(\Delta(u,c)>\delta\mid D),
\]

with preregistered (delta=0.05), together with a harm-risk upper estimate (R_h(u,c)). The estimator uses recorded propensities (inverse-propensity or doubly robust form, clipped only by a preregistered positivity rule) and clusters uncertainty at the episode level. Decisions within one episode are not treated as independent samples.

Harm is an observable label, not a posterior event. An episode is harmful when it incurs irreversible resource loss, unrecoverable termination, progress regression beyond the declared budget, a safety-boundary violation, or failure to recover within the fixed horizon. The definition is evaluated from environment state and task verifier outputs. Success is likewise defined by the task verifier and contract postcondition; a process return code is never used as task success.

## 3. C-ACT

### 3.1 Knowledge Contracts

Each memory is stored with a typed contract: preconditions, expected postconditions, applicable task groups, and hard non-applicable contexts. Precondition and safety checks run before the statistical gate. Postconditions run after execution and are logged separately, allowing contract violations to be measured independently of the harm model.

### 3.2 Evidence and gate

C-ACT maintains evidence for reuse outcomes, base-arm outcomes, and harm labels. The implementation may use conjugate Beta components for online state, but all posterior updates retain propensity and episode identifiers; non-integer effective counts are never silently truncated. A candidate is admitted only if its contract is applicable, its calibrated uplift probability exceeds (	au), its harm risk is within (h), and no hard interaction conflict is present. Probation is entered only when the certified rule fails but an independent probation rule passes; probation always invokes supervision and postcondition verification.

### 3.3 Calibration and freezing

Thresholds ((\tau,\delta,h)) are selected on `CAL-FIT` and `CAL-TUNE` by maximizing coverage subject to empirical upper risk ≤0.10. The calibration artifact stores the data hash, task IDs, selection rule, effective sample size, and bootstrap seed. Held-out runs load this artifact explicitly. Frozen experiments use read-only snapshots and run-specific trust stores; any state mutation invalidates the run.

## 4. Experimental protocol

### 4.1 Task splits

The protocol uses immutable task IDs and an automated overlap report. `CAL-FIT` (8 tasks) and `CAL-TUNE` (4 tasks) are never reused for evaluation. `E1-ID` contains 8 held-out templates; `E2-COMP` contains 8 unseen target/resource compositions; `E3-BOUND` contains 8 applicable/non-applicable pairs. `E4-ABL` uses 8 dedicated templates. `E5-LONG` has five independent streams; each of ten rounds contains 16 shared accumulation, 4 shared calibration, and 8 frozen evaluation episodes per online method. Stream evaluation and hard-transfer tasks are disjoint from accumulation and calibration.

### 4.2 Baselines

The main comparison contains NoKnowledge, NoGate, FixedBayes, PairwisePreferenceGate, C-ACT-Pointwise, and C-ACT. NoKnowledge never injects retrieved knowledge. NoGate always reuses an available candidate. FixedBayes shares the evidence representation but uses fixed thresholds and no contract/adaptive calibration. PairwisePreferenceGate is trained only from sealed paired branches, while C-ACT-Pointwise uses the pointwise controller family. An external method is named only after an exact protocol reproduction; otherwise it is reported as a style ablation.

### 4.3 Metrics and statistics

Primary endpoints are HRR and Coverage@Risk≤10%; SR/HardSR, KUS, contract satisfaction, resource-conflict rate, chain-failure rate, knowledge-pollution rate, retention SR, safety drift, and ECE are secondary or exploratory. Every metric reports its numerator, denominator, effective sample size, and 95% cluster-bootstrap interval. The two preregistered contrasts are C-ACT vs NoGate and C-ACT vs FixedBayes; Holm correction is applied to the four primary tests. E5 round slopes and AULC are reported separately and are not pooled with frozen evaluation.

### 4.4 Integrity checks

Before any full run, deterministic fixtures validate the metric implementation, schema validation fails on missing primary fields, baseline behavior tests match the definitions above, task overlap is zero, and frozen snapshot hashes are unchanged. Pipeline failures (OOM, crash, timeout) are retried under a recorded rule and are not counted as task failures.

## 5. Results reporting template

The following table is populated only from the artifact manifest after M0–M4 complete.

| Method | SR | HRR ↓ | Coverage@Risk≤10% ↑ | KUS | 95% CI | N episodes |
|---|---:|---:|---:|---:|---|---:|
| Base-Only | — | n/a | n/a | n/a | — | — |
| NoGate | — | — | — | — | — | — |
| FixedBayes | — | — | — | — | — | — |
| ACT | — | — | — | — | — | — |
| C-ACT-Full | — | — | — | — | — | — |

No direction, significance statement, or safety claim is inserted until the corresponding raw JSONL, derived metric file, manifest hash, and statistical report exist.

## 6. Limitations

The evidence is limited to Minecraft and to the upstream knowledge representation available to the agent. Propensity-aware estimates require positivity; contexts with unsupported arms are reported as unsupported rather than extrapolated. Contract extraction can be conservative and may reject useful knowledge. The protocol measures empirical risk under a declared horizon and budget; it does not establish a universal or finite-sample safety guarantee. Long-horizon results are exploratory unless the five independent streams provide stable estimates.

## 7. Reproducibility and release checklist

The release must contain task splits, method configs, calibration artifact, seeds, code commit, schema version, raw JSONL, metric fixtures, overlap report, snapshot hashes, bootstrap seeds, and the exact table-generation command. The final paper will cite only values traceable to this manifest.


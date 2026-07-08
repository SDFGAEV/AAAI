"""
C-ACT: Contracted Adaptive Counterfactual Trust
================================================

A decision-time admission layer that certifies whether retrieved
self-evolved knowledge is allowed to affect the next action.

Core modules:
  contract          — Knowledge Contract schema, extraction, verification
  trust_store       — Bayesian 3-Beta posterior store with lifecycle
  trust_gate        — Per-group adaptive counterfactual calibration gate
  context_bucket    — Adaptive hierarchical context encoding
  empirical_bayes   — Level-aware empirical Bayes priors
  lifecycle_manager — 6-state knowledge lifecycle state machine
  temporal_decay    — Drift-aware lazy temporal decay
  active_logging    — Adaptive force-base logging with propensity
  thompson_probe    — Safe Thompson exploration for cold-start
  interaction_gate  — Pairwise knowledge interaction conflict detection
  decision_controller — Main C-ACT decision orchestration (7-step flow)
  cact_memory       — Integration layer wrapping XENON DecomposedMemory
  metrics           — Evaluation metrics (SR, HRR, Cov@Risk, CSR, CVR, KPR)
  run_experiment    — Full E0–E5 experiment orchestrator
"""

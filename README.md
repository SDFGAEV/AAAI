# C-ACT: Contextual Admission via Counterfactual Treatment Effects

**Decision-time knowledge admission for self-evolving Minecraft agents.**

When a self-evolving agent retrieves a piece of experience-derived knowledge, should it actually be allowed to influence behavior? C-ACT answers this by treating knowledge reuse as a **risk-budgeted counterfactual treatment decision**.

## Architecture

```
XENON (ADG + FAM) → fixed retriever → C-ACT → same base planner
                                    ↑
                         ADMIT / FALLBACK
```

C-ACT sits between a frozen retriever and a frozen base planner. It intercepts the top-1 candidate before it enters the prompt and decides whether the candidate is allowed to act. The decision is based on four layers:

1. **Applicability** — is the knowledge relevant to the current state?
2. **Evidence support** — is there enough randomized use/base data at this context level?
3. **Pointwise risk** — does the estimated benefit exceed the minimum uplift, and do absolute/incremental harm risks stay within budget?
4. **Cumulative budget** — has the per-episode positive incremental risk budget been exhausted?

## Method

- **Estimator**: Episode-clustered cross-fitted AIPW with hierarchical evidence backoff (g⁰→g¹→g²→g³)
- **Policy selection**: κ-grid nested policy family with D_fit / D_select / D_audit isolation
- **Controller**: Unified evidence–budget state with no-credit rule and episode reset
- **Labels**: Pre-registered H1–H4 harm detectors from environment events (not posterior proxies)

## Experiments

| Stage | Purpose | Scale |
|-------|---------|-------|
| E0 | Substrate validation | 24 episodes |
| E1 | Knowledge construction + randomized logging + preference baseline | 240–480 episodes + 320 pairs |
| E2 | Policy selection + sealed audit + paired branch audit | 976–1216 episodes + 200 pairs |
| E3 | Three-tier frozen evaluation (ID / Compositional OOD / Contract-boundary) | 1728 episodes |
| E4 | Four core ablations | 240 episodes |
| E5 | Multi-stream controlled online growth (5 streams × 10 rounds) | ~2400 episodes |

Six methods: NoKnowledge, NoGate, FixedBayes, PairwisePreferenceGate, C-ACT-Pointwise, C-ACT.

## Quick Start

```bash
# Server setup
pip install -r requirements_server.txt
bash setup_ubuntu.sh

# Run full pipeline
bash experiments/run_all.sh --workers 4

# Run specific stage
bash experiments/run_all.sh --only E3 --workers 4
```

## Requirements

- Python 3.10+
- CUDA 12+ GPU with ≥24GB VRAM
- Java 21 (Minecraft server)
- Qwen2.5-VL-7B, VPT 2x, STEVE-1, MineCLIP checkpoints

## Citation

```bibtex
@misc{cact2026,
  title={Trust Before Reuse: Risk-Budgeted Counterfactual Admission of Self-Evolved Knowledge in Minecraft Agents},
  author={},
  year={2026},
  note={AAAI 2027 submission}
}
```

## License

Research code. See LICENSE for details.

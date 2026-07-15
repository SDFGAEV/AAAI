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
| E1 | Knowledge construction + randomized logging + preference baseline | 240 episodes + 320 paired rows |
| E2 | Policy selection + sealed audit + paired branch audit | 960 direct rollouts + real audit artifacts |
| E3 | Three-tier frozen evaluation (ID / Compositional OOD / Contract-boundary) | 1728 episodes |
| E4 | Four core ablations | 240 episodes |
| E5 | Multi-stream controlled online growth (5 streams x 10 rounds) | 2,400 episodes |

Six methods: NoKnowledge, NoGate, FixedBayes, PairwisePreferenceGate, C-ACT-Pointwise, C-ACT.

## Quick Start (Ubuntu)

The repository does not ship a one-command server installer because PyTorch/CUDA,
Minecraft/Java, and the VLM checkpoint must match the target host. Use a clean
Python 3.10+ environment, install the MineRL dependencies, then install the
project/runtime packages required by your CUDA build:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r minerl/requirements.txt
python -m pip install hydra-core omegaconf pyyaml scipy shortuuid rich pytest
# Install the CUDA-matched PyTorch/Transformers/VLM stack separately.
java -version                         # must report Java 21+
python experiments/health_check.py
```

Before a claiming run, the setup script uses the sealed task-card registry.
For XENON native mode (`DefaultWorldGenerator(force_reset=True)`), it derives a
procedural snapshot ID from the declared seed and pinned generator provenance; no
pre-existing save directory is required. Filesystem-backed worlds may instead set
`CACT_WORLD_ROOT_TEMPLATE` or provide `CACT_WORLD_SNAPSHOT_MANIFEST`. E2/E1c/D_audit
are collected from real rollouts; missing evidence still stops the pipeline:

```bash
export CACT_TASK_CARDS="/data/cact/task_cards.json"
# Optional for XENON procedural mode; filesystem mode may set this explicitly.
# export CACT_WORLD_SNAPSHOT_MANIFEST="/data/cact/world_snapshot_manifest.json"
export CACT_WORKERS=4
# Optional multi-GPU VLM pool:
export CACT_GPUS=0,1,2,3
bash setup_and_run.sh
```

`CACT_REQUIRE_TASK_CARDS=0` is for non-claiming dry runs only. Stage-specific
runs should call `experiments/parallel_runner.py` directly with the corresponding
benchmark, seeds, methods, and snapshot manifest; `run_all.sh` itself does not
implement `--only` or positional `--workers` flags.

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

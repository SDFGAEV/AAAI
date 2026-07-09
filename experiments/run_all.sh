#!/bin/bash
# ============================================================================
# C-ACT Full Experiment Pipeline (E0 → E5)
#
# Episode budget: E0(24) + E1(192) + E2(144) + E3(2016) + E4(480) + E5(1600) = 4456
#
# Usage:
#   bash experiments/run_all.sh                    # Full pipeline
#   bash experiments/run_all.sh --from E2          # Resume from E2
#   bash experiments/run_all.sh --only E3          # Run only E3
#   bash experiments/run_all.sh --workers 8        # 8 parallel workers
#
# Each stage gates on the previous: if a stage fails, the pipeline stops.
# Results are checkpointed — resume with --from to skip completed stages.
# ============================================================================

set -euo pipefail

# ── Config ──
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-4}"
VLM_PORT="${VLM_PORT:-12345}"
PLAN_MODEL="${PLAN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
RESUME="${RESUME:-}"

# Stage flags
RUN_E0=true; RUN_E1=true; RUN_E2=true; RUN_E3=true; RUN_E4=true; RUN_E5=true

# ── Parse args ──
FROM_STAGE=""
ONLY_STAGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_STAGE="$2"; shift 2 ;;
        --only) ONLY_STAGE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --vlm_port) VLM_PORT="$2"; shift 2 ;;
        --plan_model) PLAN_MODEL="$2"; shift 2 ;;
        --resume) RESUME="--resume"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Handle --from/--only
if [[ -n "$ONLY_STAGE" ]]; then
    RUN_E0=false; RUN_E1=false; RUN_E2=false; RUN_E3=false; RUN_E4=false; RUN_E5=false
    eval "RUN_${ONLY_STAGE}=true"
elif [[ -n "$FROM_STAGE" ]]; then
    case "$FROM_STAGE" in
        E0) ;;
        E1) RUN_E0=false ;;
        E2) RUN_E0=false; RUN_E1=false ;;
        E3) RUN_E0=false; RUN_E1=false; RUN_E2=false ;;
        E4) RUN_E0=false; RUN_E1=false; RUN_E2=false; RUN_E3=false ;;
        E5) RUN_E0=false; RUN_E1=false; RUN_E2=false; RUN_E3=false; RUN_E4=false ;;
    esac
fi

# ── Runner helper ──
run_stage() {
    local stage="$1"; shift
    echo ""
    echo "============================================================"
    echo "  STAGE $stage: $*"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    if $PYTHON experiments/parallel_runner.py "$@" --workers "$WORKERS" \
        --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL" $RESUME; then
        echo "[$stage] PASSED"
        return 0
    else
        echo "[$stage] FAILED — pipeline stopped"
        exit 1
    fi
}

# ── Pre-flight ──
echo "[Pipeline] Running health check..."
$PYTHON experiments/health_check.py || {
    echo "[Pipeline] Health check failed. Fix issues before running."
    exit 1
}

# ── Stage execution ──
STAGE_START=$(date +%s)

# ════════════════════════════════════════════════════════════════════
# E0: Sanity Check — 6 tasks × 2 seeds × 2 methods = 24 episodes
# Verify system runs: reset, success detection, contract extraction, logging
# ════════════════════════════════════════════════════════════════════
if $RUN_E0; then
    run_stage "E0" \
        --benchmark cact_calib \
        --seeds 1001-1002 \
        --methods NoKnowledge XENON-Original
fi

# ════════════════════════════════════════════════════════════════════
# E1: Knowledge Accumulation — 24 tasks × 8 seeds = 192 episodes
# Run with NoGate policy + active base logging (mixed logging)
# Output: knowledge contracts, initial use/base/harm stats, propensity logs
# ════════════════════════════════════════════════════════════════════
if $RUN_E1; then
    run_stage "E1" \
        --benchmark cact_train \
        --seeds 2001-2008 \
        --methods XENON-Original
fi

# ════════════════════════════════════════════════════════════════════
# E2: Adaptive Calibration — 12 tasks × 8 seeds + 48 probes = 144 episodes
# Learn per-group thresholds τ*, δ*, h*, interaction thresholds
# ════════════════════════════════════════════════════════════════════
if $RUN_E2; then
    echo "=== E2: Collecting calibration data ==="
    $PYTHON experiments/parallel_runner.py \
        --benchmark cact_calib \
        --seeds 3001-3008 \
        --methods ACT \
        --workers "$WORKERS" --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"

    echo "=== E2: Running calibration optimization ==="
    $PYTHON <<'PYEOF'
from cact.trust_store import TrustStore
from cact.trust_gate import TrustGate
import json, os

store = TrustStore(store_path="cact_ckpt/trust_store")
gate = TrustGate()
calib = store.get_calibration_data()
if calib:
    results = gate.calibrate_all_groups(calib)
    os.makedirs("exp_results", exist_ok=True)
    gate.save_calibration("exp_results/calibration.json")
    print(f"Calibration done. Groups: {list(calib.keys())}")
    for g, r in results.items():
        print(f"  {g}: tau={r.get('tau','?')} delta={r.get('delta','?')} harm={r.get('harm','?')} n={r.get('n_calib','?')}")
else:
    print("WARNING: No calibration data collected — using fixed defaults")
PYEOF
    echo "[E2] PASSED"
fi

# ════════════════════════════════════════════════════════════════════
# E3: Strict Frozen Main Evaluation
# 36 tasks × 8 seeds × 7 methods = 2016 episodes
# Frozen: no learning, no posterior updates, no probes, no new knowledge
# ════════════════════════════════════════════════════════════════════
if $RUN_E3; then
    run_stage "E3" \
        --benchmark cact_p3 \
        --seeds 4001-4008 \
        --methods NoKnowledge XENON-Original BankCuration \
                  LifecycleSuccessGate FixedBayes ACT C-ACT-Full
fi

# ════════════════════════════════════════════════════════════════════
# E4: Ablation Study — 12 tasks × 5 seeds × 8 variants = 480 episodes
# 7 component ablations + OracleGate (upper bound)
# Uses hardest sub-population: tech_tree + failure_recovery + interaction_stress
# ════════════════════════════════════════════════════════════════════
if $RUN_E4; then
    for variant in C-ACT-NoContract C-ACT-NoActiveCalib C-ACT-NoDecay \
                   C-ACT-NoAttribution C-ACT-NoInteraction C-ACT-NoLevelPrior \
                   C-ACT-NoSanitizer OracleGate; do
        echo "--- E4: $variant ---"
        $PYTHON experiments/parallel_runner.py \
            --benchmark cact_ablation \
            --seeds 4001-4005 \
            --methods "$variant" \
            --workers "$WORKERS" --vlm_port "$VLM_PORT"
    done
fi

# ════════════════════════════════════════════════════════════════════
# E5: Online Knowledge-Growth Evaluation
# 10 rounds × 40 episodes × 4 methods = 1600 episodes
# Per round: 20 accumulation + 8 calibration + 12 frozen evaluation
#            (6 retention + 6 hard-transfer)
# ════════════════════════════════════════════════════════════════════
if $RUN_E5; then
    echo "=== E5: Online Evolution (10 rounds) ==="
    $PYTHON experiments/online_runner.py \
        --rounds 10 \
        --methods Online-SuccessLifecycle Online-FixedBayes Online-ACT Online-C-ACT \
        --seed 5001 \
        --workers "$WORKERS" --vlm_port "$VLM_PORT"
fi

STAGE_END=$(date +%s)
ELAPSED=$((STAGE_END - STAGE_START))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Episode budget: 24 + 192 + 144 + 2016 + 480 + 1600 = 4456"
echo "  Total wall time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "  Results: $PROJ/exp_results/"
echo "============================================================"

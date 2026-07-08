#!/bin/bash
# ============================================================================
# C-ACT Full Experiment Pipeline (E0 → E5)
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

# E0: Sanity Check (~24 episodes, < 5 min)
if $RUN_E0; then
    run_stage "E0" \
        --benchmark cact_calib \
        --seeds 1001-1002 \
        --methods NoKnowledge XENON-Original \
        --print_grid 2>/dev/null  # dry-run then real
    $PYTHON experiments/parallel_runner.py \
        --benchmark cact_calib \
        --seeds 1001-1002 \
        --methods NoKnowledge XENON-Original \
        --workers 2 --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"
fi

# E1: Knowledge Accumulation (~192 episodes, ~1.5h @ 4 workers)
if $RUN_E1; then
    run_stage "E1" \
        --benchmark cact_train \
        --seeds 2001-2008 \
        --methods XENON-Original
fi

# E2: Adaptive Calibration (~120 episodes, ~1h @ 4 workers)
if $RUN_E2; then
    run_stage "E2" \
        --benchmark cact_calib \
        --seeds 3001-3008 \
        --methods ACT
fi

# E3: Main Evaluation (~2016 episodes, ~4-5h @ 4 workers)
if $RUN_E3; then
    run_stage "E3" \
        --benchmark cact_p3 \
        --seeds 4001-4008 \
        --methods NoKnowledge XENON-Original BankCuration \
                  LifecycleSuccessGate FixedBayes ACT C-ACT-Full
fi

# E4: Ablation (~480 episodes, ~1h @ 4 workers)
if $RUN_E4; then
    run_stage "E4" \
        --benchmark cact_p3 \
        --seeds 4001-4005 \
        --methods C-ACT-Full
    # Note: E4 requires variant configs — run with cact_method overrides:
    # (simplified here; actual E4 needs per-variant cact_method changes)
fi

# E5: Online Evolution — 4 methods x 3 seeds x 10 rounds
# Uses online_runner.py with persistent trust stores
if $RUN_E5; then
    echo "=== E5: Online Evolution (10 rounds) ==="
    $PYTHON experiments/online_runner.py \
        --benchmark_accum cact_train \
        --benchmark_test cact_p3 \
        --seeds 5001-5003 \
        --rounds 10 \
        --methods Online-NoGate Online-BankCuration Online-ACT Online-C-ACT \
        --workers "$WORKERS" --vlm_port "$VLM_PORT"
fi

STAGE_END=$(date +%s)
ELAPSED=$((STAGE_END - STAGE_START))

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Total wall time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
echo "  Results: $PROJ/exp_results/"
echo "============================================================"

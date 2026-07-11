#!/usr/bin/env bash
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORKERS="${CACT_WORKERS:-4}"
VLM_PORT="${CACT_VLM_PORT:-12345}"
PLAN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RESULTS="$PROJ/exp_results"
CAL_STORE="$RESULTS/calibration_store"
POLICY_FILE="$RESULTS/v2_policy.json"
PREFERENCE_FILE="$RESULTS/d_pair_train_preference.json"
mkdir -p "$RESULTS"
cd "$PROJ"
# Avoid BLAS/thread oversubscription when multiple Minecraft workers run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
run() { "$PYTHON" experiments/parallel_runner.py "$@" --workers "$WORKERS" --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"; }
run_serial() { "$PYTHON" experiments/parallel_runner.py "$@" --workers 1 --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"; }

echo "[M0] health check and protocol release"
"$PYTHON" experiments/health_check.py
"$PYTHON" experiments/release_protocol.py --label protocol-candidate

# E0: six substrate tasks × two seeds × NoKnowledge/NoGate.
run_serial --benchmark cact_e0 --task_indices 0,1,2,3,4,5 --seeds 1001-1002 --methods NoKnowledge NoGate

# E1a freezes the shared knowledge store; E1b collects randomized opportunities.
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2001-2005 --methods C-ACT --store_path "$CAL_STORE"
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2101-2105 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect

# E2: direct matched-risk policy selection.  The selector requires a real
# rollout table containing Base + 7 Full + 7 Pointwise candidates on every
# identical task/world/episode cell; no offline replay or fabricated labels.
if [[ -z "${E2_DIRECT_JSONL:-}" || ! -f "${E2_DIRECT_JSONL}" ]]; then
  echo "STOP: provide E2_DIRECT_JSONL from the matched-risk rollout (15 methods/cell)." >&2
  exit 2
fi
"$PYTHON" experiments/e2_direct_select.py --input "$E2_DIRECT_JSONL" \
  --out "$RESULTS/e2_direct_selection.json"
run_serial --benchmark cact_calib --task_indices 0,1,2,3,4,5 --seeds 3001-3008 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect
run_serial --benchmark cact_calib --task_indices 6,7,8,9,10,11 --seeds 3011-3018 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect
# E1c: D_pair-train must be a real sealed paired-branch artifact.
PAIR_GLOB="$RESULTS/cact_pair_train/*/pairs.jsonl"
if compgen -G "$PAIR_GLOB" > /dev/null; then
  "$PYTHON" experiments/train_pairwise.py --input "$PAIR_GLOB" --out "$PREFERENCE_FILE"
else
  echo "STOP: missing D_pair-train pairs.jsonl; do not fabricate preference labels." >&2
  exit 2
fi
export CACT_PREFERENCE_PATH="$PREFERENCE_FILE"

"$PYTHON" experiments/calibrate_v2.py --fit-glob "$RESULTS/cact_logs/cact_train_C-ACT_seed*_task[0-9]*/opportunities.jsonl" --select-glob "$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task[0-5]/opportunities.jsonl" --audit-glob "$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task[6-9]/opportunities.jsonl" "$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task1[01]/opportunities.jsonl" --out "$POLICY_FILE"

# E3: 36 conditions × 8 seeds × six preregistered methods, strict frozen.
run --benchmark cact_p3 --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 --seeds 4001-4008 --methods NoKnowledge NoGate FixedBayes PairwisePreferenceGate C-ACT-Pointwise C-ACT --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E4: exactly four core ablations × five seeds.
run --benchmark cact_ablation --task_indices 0,1,2,3,4,5,6,7,8,9,10,11 --seeds 5001-5005 --methods C-ACT C-ACT-NoContract C-ACT-NoAdaptiveTau C-ACT-NoActiveCalib --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E5: five independent controlled streams, ten rounds each.
for seed in 6001 6002 6003 6004 6005; do
  "$PYTHON" experiments/online_runner.py --rounds 10 --seed "$seed" --workers "$WORKERS" --vlm_port "$VLM_PORT" --protocol_path "$POLICY_FILE" --methods Online-NoGate Online-FixedBayes Online-C-ACT-Pointwise Online-C-ACT
done
echo "C-ACT manual protocol complete. Results: $RESULTS"

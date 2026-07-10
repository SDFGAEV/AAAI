#!/usr/bin/env bash
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="python"
WORKERS="4"
VLM_PORT="12345"
PLAN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RESULTS="$PROJ/exp_results"
CAL_STORE="$RESULTS/calibration_store"
POLICY_FILE="$RESULTS/v2_policy.json"
mkdir -p "$RESULTS"
cd "$PROJ"
run() { "$PYTHON" experiments/parallel_runner.py "$@" --workers "$WORKERS" --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"; }
run_serial() { "$PYTHON" experiments/parallel_runner.py "$@" --workers 1 --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"; }

echo "[M0] health check and protocol release"
"$PYTHON" experiments/health_check.py
"$PYTHON" experiments/release_protocol.py --label protocol-candidate

# E0: six substrate tasks × two seeds × NoKnowledge/NoGate.
run_serial --benchmark cact_e0 --task_indices 0,1,2,3,4,5 --seeds 1001-1002 --methods NoKnowledge NoGate

# E1a freezes the shared knowledge store; E1b collects randomized opportunities.
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2001-2005 --methods C-ACT-Full --store_path "$CAL_STORE"
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2101-2105 --methods C-ACT-Full --store_path "$CAL_STORE" --protocol_path collect

# E2: disjoint six-template D_select and six-template D_audit.
run_serial --benchmark cact_calib --task_indices 0,1,2,3,4,5 --seeds 3001-3008 --methods C-ACT-Full --store_path "$CAL_STORE" --protocol_path collect
run_serial --benchmark cact_calib --task_indices 6,7,8,9,10,11 --seeds 3011-3018 --methods C-ACT-Full --store_path "$CAL_STORE" --protocol_path collect
"$PYTHON" experiments/calibrate_v2.py --fit-glob "$RESULTS/cact_logs/cact_train_C-ACT-Full_seed*_task[0-9]*/opportunities.jsonl" --select-glob "$RESULTS/cact_logs/cact_calib_C-ACT-Full_seed*_task[0-5]/opportunities.jsonl" --audit-glob "$RESULTS/cact_logs/cact_calib_C-ACT-Full_seed*_task[6-9]/opportunities.jsonl" "$RESULTS/cact_logs/cact_calib_C-ACT-Full_seed*_task1[01]/opportunities.jsonl" --out "$POLICY_FILE"

# E3: 36 conditions × 8 seeds × six preregistered methods, strict frozen.
run --benchmark cact_p3 --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 --seeds 4001-4008 --methods NoKnowledge NoGate SuccessLifecycle FixedBayes ACT C-ACT-Full --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E4: exactly four core ablations × five seeds.
run --benchmark cact_ablation --task_indices 0,1,2,3,4,5,6,7,8,9,10,11 --seeds 5001-5005 --methods C-ACT-Full C-ACT-NoContract C-ACT-NoAdaptiveTau C-ACT-NoActiveCalib --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E5: five independent controlled streams, ten rounds each.
for seed in 6001 6002 6003 6004 6005; do
  "$PYTHON" experiments/online_runner.py --rounds 10 --seed "$seed" --workers "$WORKERS" --vlm_port "$VLM_PORT" --protocol_path "$POLICY_FILE" --methods Online-SuccessLifecycle Online-FixedBayes Online-ACT Online-C-ACT
done
echo "C-ACT manual protocol complete. Results: $RESULTS"

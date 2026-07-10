#!/usr/bin/env bash
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="python"
WORKERS="4"
VLM_PORT="12345"
PLAN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RESULTS="$PROJ/exp_results"
CAL_STORE="$RESULTS/calibration_store"
CAL_FILE="$RESULTS/calibration.json"

mkdir -p "$RESULTS"
cd "$PROJ"

run() {
  echo "[RUN] $*"
  "$PYTHON" experiments/parallel_runner.py "$@" \
    --workers "$WORKERS" --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"
}
run_serial() {
  echo "[RUN-SERIAL] $*"
  "$PYTHON" experiments/parallel_runner.py "$@" \
    --workers 1 --vlm_port "$VLM_PORT" --plan_model "$PLAN_MODEL"
}

echo "[M0] health check"
"$PYTHON" experiments/health_check.py

run_serial --benchmark cact_calib --task_indices 0,1,2,3,4,5,6,7 \
    --seeds 3001-3003 --methods ACT \
    --store_path "$CAL_STORE" --active_calib_rate 0.20
run_serial --benchmark cact_calib --task_indices 8,9,10,11 \
    --seeds 3011-3013 --methods ACT \
    --store_path "$CAL_STORE" --active_calib_rate 0.20

"$PYTHON" - "$CAL_STORE" "$CAL_FILE" <<'PY'
import sys
from cact.trust_store import TrustStore
from cact.trust_gate import TrustGate
store, out = sys.argv[1:]
ts = TrustStore(store_path=store)
gate = TrustGate()
gate.calibrate_all_groups(ts.get_calibration_data())
gate.save_calibration(out)
print("calibration:", out)
PY

run --benchmark cact_p3 --task_indices 12,13,14,15,16,17,18,19 \
    --seeds 4001-4005 --methods Base-Only NoGate FixedBayes ACT C-ACT-Full \
    --calibration_path "$CAL_FILE"
run --benchmark cact_p3 --task_indices 20,21,22,23,24,25,26,27 \
    --seeds 4101-4105 --methods Base-Only NoGate FixedBayes ACT C-ACT-Full \
    --calibration_path "$CAL_FILE"
run --benchmark cact_p3 --task_indices 28,29,30,31,32,33,34,35 \
    --seeds 4201-4205 --methods Base-Only NoGate FixedBayes ACT C-ACT-Full \
    --frozen --snapshot_path "$CAL_STORE" --calibration_path "$CAL_FILE"

run --benchmark cact_ablation --task_indices 0,1,2,3,4,5,6,7,8,9,10,11 \
    --seeds 4301-4305 --methods C-ACT-Full C-ACT-NoContract C-ACT-NoAdaptiveTau C-ACT-NoActiveCalib C-ACT-NoLifecycle \
    --frozen --snapshot_path "$CAL_STORE" --calibration_path "$CAL_FILE"

for seed in 5001 5002 5003; do
  "$PYTHON" experiments/online_runner.py --rounds 10 --seed "$seed" \
      --workers "$WORKERS" --vlm_port "$VLM_PORT" \
      --methods Online-SuccessLifecycle Online-FixedBayes Online-ACT Online-C-ACT
done

echo "Protocol runs complete. Results: $RESULTS"

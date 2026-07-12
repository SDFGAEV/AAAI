#!/usr/bin/env bash
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORKERS="${CACT_WORKERS:-4}"
VLM_PORT="${CACT_VLM_PORT:-12345}"
VLM_PORTS="${CACT_VLM_PORTS:-}"
PLAN_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RESULTS="$PROJ/exp_results"
CAL_STORE="$RESULTS/calibration_store"
POLICY_FILE="$RESULTS/v2_policy.json"
PREFERENCE_FILE="$RESULTS/d_pair_train_preference.json"
CACT_TASK_CARDS="${CACT_TASK_CARDS:-$PROJ/protocol_inputs/task_cards.json}"
mkdir -p "$RESULTS"
cd "$PROJ"
# Avoid BLAS/thread oversubscription when multiple Minecraft workers run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CACT_REQUIRE_WORLD_SNAPSHOT_HASH="${CACT_REQUIRE_WORLD_SNAPSHOT_HASH:-0}"

# ── Multi-GPU VLM pool ───────────────────────────────────────────────
# Usage: CUDA_VISIBLE_DEVICES is ignored; instead set CACT_GPUS="0,1,2,3".
# Each GPU gets one VLM server on port 12345 + gpu_index.
# Workers are distributed round-robin across ports.
# On exit (or error), all VLM servers are killed automatically.
VLM_PID_LIST=()
_gpu_list() {
  local gpus="${CACT_GPUS:-0}"
  echo "$gpus" | tr ',' '\n'
}
_start_vlm_pool() {
  local idx=0
  for gpu in $(_gpu_list); do
    local port=$((VLM_PORT + idx))
    echo "[VLM] GPU $gpu → port $port"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" app.py --port "$port" --plan_model "$PLAN_MODEL" \
      > "$RESULTS/vlm_gpu${gpu}_port${port}.log" 2>&1 &
    VLM_PID_LIST+=($!)
    idx=$((idx + 1))
  done
  # Build comma-separated port list for parallel_runner.
  local ports=()
  for ((i=0; i<idx; i++)); do ports+=($((VLM_PORT + i))); done
  VLM_PORTS=$(IFS=,; echo "${ports[*]}")
  export VLM_PORTS
  # Wait for every VLM to accept /health (parallel — not serial).
  for ((i=0; i<idx; i++)); do
    local p=$((VLM_PORT + i))
    echo "[VLM] waiting for port $p ..."
    (for _ in $(seq 1 180); do
      if curl -s --connect-timeout 1 "http://127.0.0.1:$p/health" > /dev/null 2>&1; then echo "ready:$p"; exit 0; fi
      sleep 0.5
    done
    echo "fail:$p") &
  done
  wait
  for ((i=0; i<idx; i++)); do
    local p=$((VLM_PORT + i))
    if ! curl -s --connect-timeout 1 "http://127.0.0.1:$p/health" > /dev/null 2>&1; then
      echo "[VLM] FATAL: port $p never became healthy"
      exit 1
    fi
    echo "[VLM] port $p ready"
  done
}
_kill_vlm_pool() {
  for pid in "${VLM_PID_LIST[@]:-}"; do
    kill "$pid" 2>/dev/null || true
    # Wait briefly, then force-kill if still alive.
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  done
}
trap _kill_vlm_pool EXIT

# Start multi-GPU VLM pool.  Single-GPU path stays backwards-compatible.
if [ -n "${CACT_GPUS:-}" ] && [ "${CACT_GPUS:-}" != "0" ]; then
  _start_vlm_pool
fi

# ── Runners ────────────────────────────────────────────────────────────
_runner_args() {
  RUNNER_ARGS=()
  if [ -n "${VLM_PORTS:-}" ]; then
    RUNNER_ARGS+=(--vlm_ports "$VLM_PORTS")
  else
    RUNNER_ARGS+=(--vlm_port "$VLM_PORT")
  fi
  if [ -n "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]; then
    RUNNER_ARGS+=(--world_snapshot_manifest "$CACT_WORLD_SNAPSHOT_MANIFEST")
  fi
}
run() {
  _runner_args
  "$PYTHON" experiments/parallel_runner.py "$@" --workers "$WORKERS" \
    --plan_model "$PLAN_MODEL" "${RUNNER_ARGS[@]}"
}
run_serial() {
  _runner_args
  "$PYTHON" experiments/parallel_runner.py "$@" --workers 1 \
    --plan_model "$PLAN_MODEL" "${RUNNER_ARGS[@]}"
}
run_online() {
  local extra=()
  if [ -n "${VLM_PORTS:-}" ]; then
    extra+=(--vlm_ports "$VLM_PORTS")
  else
    extra+=(--vlm_port "$VLM_PORT")
  fi
  local manifest_args=()
  if [ -n "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]; then
    manifest_args+=(--world_snapshot_manifest "$CACT_WORLD_SNAPSHOT_MANIFEST")
  fi
  "$PYTHON" experiments/online_runner.py "$@" --workers "$WORKERS" \
    --protocol_path "$POLICY_FILE" "${manifest_args[@]}" "${extra[@]}"
}

echo "[M0] compile .pyc (reduces subprocess startup latency)"
"$PYTHON" -OO -m compileall -q cact/ experiments/ src/optimus1/ 2>/dev/null || true

echo "[M0] health check and protocol release"
"$PYTHON" experiments/health_check.py
"$PYTHON" experiments/release_protocol.py --label protocol-candidate
if [[ "${CACT_REQUIRE_TASK_CARDS:-1}" == "1" ]]; then
  if [[ -z "${CACT_TASK_CARDS:-}" ]]; then
    echo "STOP: CACT_TASK_CARDS must name the sealed task-card JSON/YAML files." >&2
    exit 2
  fi
  read -r -a TASK_CARD_FILES <<< "$CACT_TASK_CARDS"
  "$PYTHON" analysis/validate_task_cards.py "${TASK_CARD_FILES[@]}" --require-sealed \
    --out "$RESULTS/task_card_validation.json"
fi

# E0: six substrate tasks × two seeds × NoKnowledge/NoGate.
run_serial --benchmark cact_e0 --task_indices 0,1,2,3,4,5 --seeds 1001-1002 --methods NoKnowledge NoGate

# E1a freezes the shared knowledge store; E1b collects randomized opportunities.
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2001-2005 --methods C-ACT --store_path "$CAL_STORE"
run_serial --benchmark cact_train --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --seeds 2101-2105 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect

# D_select/D_audit opportunity logging is required before a provisional
# artifact can parameterize the 15-policy direct E2 rollouts.
run_serial --benchmark cact_calib --task_indices 0,1,2,3,4,5 --seeds 3001-3008 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect
run_serial --benchmark cact_calib --task_indices 6,7,8,9,10,11 --seeds 3011-3018 --methods C-ACT --store_path "$CAL_STORE" --protocol_path collect
FIT_GLOB="$RESULTS/cact_logs/cact_train_C-ACT_seed*_task*/opportunities.jsonl"
SELECT_GLOB="$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task[0-5]/opportunities.jsonl"
AUDIT_GLOB1="$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task[6-9]/opportunities.jsonl"
AUDIT_GLOB2="$RESULTS/cact_logs/cact_calib_C-ACT_seed*_task1[01]/opportunities.jsonl"
PROVISIONAL_POLICY="$RESULTS/v2_policy_provisional.json"
"$PYTHON" experiments/calibrate_v2.py --fit-glob "$FIT_GLOB" --select-glob "$SELECT_GLOB" \
  --audit-glob "$AUDIT_GLOB1" "$AUDIT_GLOB2" --out "$PROVISIONAL_POLICY"

# E2: direct matched-risk policy selection.  Generate the table only when
# explicitly enabled; otherwise require a precomputed, audited input.
if [[ -z "${E2_DIRECT_JSONL:-}" ]]; then
  if [[ "${CACT_AUTO_GENERATE_E2:-0}" != "1" ]]; then
    echo "STOP: set E2_DIRECT_JSONL or CACT_AUTO_GENERATE_E2=1 for real E2 rollouts." >&2
    exit 2
  fi
  E2_DIRECT_JSONL="$RESULTS/e2_select_rollouts.jsonl"
  "$PYTHON" experiments/run_e2_select_rollouts.py \
    --benchmark cact_p3 --task-indices "${CACT_E2_TASK_INDICES:-0,1,2,3,4,5,6,7}" \
    --seeds "${CACT_E2_SEEDS:-3001-3008}" --workers "$WORKERS" --vlm-port "$VLM_PORT" \
    --snapshot-path "$CAL_STORE" --world-snapshot-manifest "${CACT_WORLD_SNAPSHOT_MANIFEST:?set CACT_WORLD_SNAPSHOT_MANIFEST}" \
    --protocol-path "$PROVISIONAL_POLICY" --out "$E2_DIRECT_JSONL"
fi
"$PYTHON" experiments/e2_direct_select.py --input "$E2_DIRECT_JSONL" \
  --out "$RESULTS/e2_direct_selection.json"

# E1c: generate the real sealed paired-branch artifact when requested.
PAIR_GLOB="$RESULTS/cact_pair_train/*/pairs.jsonl"
PAIR_FILES=()
if compgen -G "$PAIR_GLOB" > /dev/null; then
  mapfile -t PAIR_FILES < <(compgen -G "$PAIR_GLOB" | sort)
fi
if (( ${#PAIR_FILES[@]} == 0 )); then
  if [[ "${CACT_AUTO_GENERATE_E1C:-0}" != "1" ]]; then
    echo "STOP: missing E1c pairs.jsonl; set CACT_AUTO_GENERATE_E1C=1 for real branch collection." >&2
    exit 2
  fi
  "$PYTHON" experiments/generate_pair_train.py \
    --pilot-task-indices "${CACT_E1C_TASK_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11}" \
    --pilot-seeds "${CACT_E1C_SEEDS:-2101-2115}" --workers 1 \
    --world-snapshot-manifest "${CACT_WORLD_SNAPSHOT_MANIFEST:?set CACT_WORLD_SNAPSHOT_MANIFEST}" \
    --out "$RESULTS/cact_pair_train/generated/pairs.jsonl"
  PAIR_FILES=("$RESULTS/cact_pair_train/generated/pairs.jsonl")
fi
if (( ${#PAIR_FILES[@]} != 1 )); then
  echo "STOP: expected exactly one sealed E1c pairs.jsonl, found ${#PAIR_FILES[@]}." >&2
  exit 2
fi
"$PYTHON" experiments/train_pairwise.py --input "${PAIR_FILES[0]}" --out "$PREFERENCE_FILE"
export CACT_PREFERENCE_PATH="$PREFERENCE_FILE"

"$PYTHON" experiments/calibrate_v2.py --fit-glob "$FIT_GLOB" --select-glob "$SELECT_GLOB" \
  --audit-glob "$AUDIT_GLOB1" "$AUDIT_GLOB2" --direct-selection "$RESULTS/e2_direct_selection.json" --out "$POLICY_FILE"

# D_audit: direct selected-policy rollout plus sealed paired audit.
if [[ "${CACT_REQUIRE_E2_AUDIT:-1}" == "1" ]]; then
  if [[ -z "${E2_AUDIT_JSONL:-}" || -z "${E2_AUDIT_PAIRS_JSONL:-}" || ! -f "$E2_AUDIT_JSONL" || ! -f "$E2_AUDIT_PAIRS_JSONL" ]]; then
    if [[ "${CACT_AUTO_GENERATE_E2_AUDIT:-0}" != "1" ]]; then
      echo "STOP: provide E2_AUDIT_JSONL/E2_AUDIT_PAIRS_JSONL or set CACT_AUTO_GENERATE_E2_AUDIT=1." >&2
      exit 2
    fi
    E2_AUDIT_JSONL="$RESULTS/e2_audit_rollouts.jsonl"
    E2_AUDIT_PAIRS_JSONL="$RESULTS/e2_audit_pairs.jsonl"
    "$PYTHON" experiments/run_e2_audit_rollouts.py \
      --benchmark cact_p3 --task-indices "${CACT_DAUDIT_TASK_INDICES:-8,9,10,11,12,13,14,15}" \
      --seeds "${CACT_DAUDIT_SEEDS:-3011-3018}" --snapshot-path "$CAL_STORE" \
      --world-snapshot-manifest "${CACT_WORLD_SNAPSHOT_MANIFEST:?set CACT_WORLD_SNAPSHOT_MANIFEST}" \
      --protocol-path "$POLICY_FILE" --policy-path "$POLICY_FILE" \
      --out-rollouts "$E2_AUDIT_JSONL" --out-pairs "$E2_AUDIT_PAIRS_JSONL" --workers "$WORKERS"
  fi
  "$PYTHON" experiments/validate_e2_audit.py --rollouts "$E2_AUDIT_JSONL" \
    --pairs "$E2_AUDIT_PAIRS_JSONL" --out "$RESULTS/e2_audit_validation.json"
fi

# E3: 36 conditions × 8 seeds × six preregistered methods, strict frozen.
run --benchmark cact_p3 --task_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 --seeds 4001-4008 --methods NoKnowledge NoGate FixedBayes PairwisePreferenceGate C-ACT-Pointwise C-ACT --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E4: exactly four core ablations × five seeds.
run --benchmark cact_ablation --task_indices 0,1,2,3,4,5,6,7,8,9,10,11 --seeds 5001-5005 --methods C-ACT C-ACT-NoContract C-ACT-NoAdaptiveTau C-ACT-NoActiveCalib --frozen --snapshot_path "$CAL_STORE" --protocol_path "$POLICY_FILE"

# E5: five independent controlled streams, ten rounds each.
for seed in 6001 6002 6003 6004 6005; do
  run_online --rounds 10 --seed "$seed" \
    --methods Online-NoGate Online-FixedBayes Online-C-ACT-Pointwise Online-C-ACT
done
echo "C-ACT manual protocol complete. Results: $RESULTS"

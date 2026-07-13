#!/usr/bin/env bash
set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$PROJ"

command -v "$PYTHON" >/dev/null || { echo "STOP: Python 3.10+ not found" >&2; exit 2; }
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("STOP: Python 3.10+ required")
for name in ("torch", "transformers", "hydra", "omegaconf", "fastapi", "uvicorn", "requests", "psutil", "numpy", "gym", "Pyro4"):
    try: __import__(name)
    except Exception as exc: raise SystemExit(f"STOP: missing Python package {name}: {exc}")
print("Python dependency check: PASS")
PY

if ! command -v java >/dev/null; then
  echo "STOP: Java 21+ is required" >&2; exit 2
fi
JAVA_MAJOR="$(java -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p' | head -n1)"
if [[ -z "$JAVA_MAJOR" || "$JAVA_MAJOR" -lt 21 ]]; then
  echo "STOP: Java 21+ required, detected ${JAVA_MAJOR:-unknown}" >&2; exit 2
fi

if [[ ! -f "$PROJ/protocol_inputs/task_cards.json" ]]; then
  "$PYTHON" experiments/build_task_card_registry.py --out "$PROJ/protocol_inputs/task_cards.json"
fi
export CACT_TASK_CARDS="${CACT_TASK_CARDS:-$PROJ/protocol_inputs/task_cards.json}"

# XENON regenerates each world from (task_id, seed); no filesystem manifest is
# required. A manifest remains opt-in for MineDojo-style save directories.
if [[ -n "${CACT_WORLD_ROOT_TEMPLATE:-}" ]]; then
  if [[ -z "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]]; then
    export CACT_WORLD_SNAPSHOT_MANIFEST="$PROJ/protocol_inputs/world_snapshot_manifest.json"
  fi
  if [[ ! -f "$CACT_WORLD_SNAPSHOT_MANIFEST" ]]; then
    export CACT_SNAPSHOT_TASK_INDICES="${CACT_SNAPSHOT_TASK_INDICES:-0-35}"
    export CACT_SNAPSHOT_SEEDS="${CACT_SNAPSHOT_SEEDS:-3001-3008,3011-3018,4001-4008,5001-5005,6001-6005}"
    "$PYTHON" experiments/collect_world_snapshots.py \
      --world-root-template "$CACT_WORLD_ROOT_TEMPLATE" \
      --task-indices "$CACT_SNAPSHOT_TASK_INDICES" --seeds "$CACT_SNAPSHOT_SEEDS" \
      --out "$CACT_WORLD_SNAPSHOT_MANIFEST"
  fi
elif [[ -n "${CACT_WORLD_SNAPSHOT_MANIFEST:-}" ]]; then
  [[ -f "$CACT_WORLD_SNAPSHOT_MANIFEST" ]] || { echo "STOP: explicit CACT_WORLD_SNAPSHOT_MANIFEST not found" >&2; exit 2; }
else
  unset CACT_WORLD_SNAPSHOT_MANIFEST
  echo "[XENON] procedural worlds: snapshot identity derives from seed + generator provenance"
fi

export CACT_AUTO_GENERATE_E2="${CACT_AUTO_GENERATE_E2:-1}"
export CACT_AUTO_GENERATE_E1C="${CACT_AUTO_GENERATE_E1C:-1}"
export CACT_AUTO_GENERATE_E2_AUDIT="${CACT_AUTO_GENERATE_E2_AUDIT:-1}"
export CACT_REQUIRE_TASK_CARDS="${CACT_REQUIRE_TASK_CARDS:-1}"
export CACT_REQUIRE_E2_AUDIT="${CACT_REQUIRE_E2_AUDIT:-1}"
export CACT_WORKERS="${CACT_WORKERS:-4}"

exec bash "$PROJ/experiments/run_all.sh"

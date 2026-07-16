#!/usr/bin/env bash
set -euo pipefail

# XENON's official runtime is MCP-Reborn/Minecraft 1.16.5.  The server uses
# the published XENON image because its Java 8 runtime is required by this
# Forge build; local installations can fall back to a native Java 8 runtime.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
port=0; env_mode=0; run_dir="run"; performance_dir=""; gpu=0; max_mem="4G"; replaceable=0
while (($#)); do
  case "$1" in
    -port) port="$2"; shift 2;;
    -env) env_mode=1; shift;;
    -runDir) run_dir="$2"; shift 2;;
    -performanceDir) performance_dir="$2"; shift 2;;
    -gpu) gpu="$2"; shift 2;;
    -maxMem) max_mem="$2"; shift 2;;
    -replaceable) replaceable=1; shift;;
    -seed) shift 2;; # world seed is carried in Mission XML, not a JVM flag
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$port" =~ ^[0-9]+$ ]] || { echo "invalid -port: $port" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid -gpu: $gpu" >&2; exit 2; }
mkdir -p "$SCRIPT_DIR/$run_dir"
# Minecraft resolves assets relative to --gameDir.  Reuse the immutable
# downloaded 1.16 asset tree without copying it for every episode.
if [[ ! -e "$SCRIPT_DIR/$run_dir/assets" ]]; then
  ln -s ../run/assets "$SCRIPT_DIR/$run_dir/assets"
fi

image="${CACT_MINECRAFT_IMAGE:-sjlee1218/xenon:latest}"
if command -v docker >/dev/null 2>&1 && docker image inspect "$image" >/dev/null 2>&1; then
  # Label each container for exact per-episode cleanup.
  run_label="${CACT_RUN_ID:-run_${run_dir}}"
  exec docker run --rm -t --label "cact.run_id=$run_label" --user "$(id -u):$(id -g)" --gpus "device=$gpu" \
    -v "$SCRIPT_DIR:/app/mcp" --network host --shm-size=4g \
    -e HOME=/tmp -e DISPLAY=:99 "$image" bash -lc \
    "set -e; cd /app/mcp; Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX >/tmp/xvfb-$$.log 2>&1 & xvfb_pid=\$!; trap 'kill \$xvfb_pid 2>/dev/null || true' EXIT; exec java -Xmx$max_mem -jar build/libs/mcprec-6.13.jar --gameDir '$run_dir' --envPort=$port"
fi

command -v java >/dev/null 2>&1 || { echo "Java 8 or XENON Docker image is required" >&2; exit 127; }
exec xvfb-run -a java -Xmx"$max_mem" -jar "$SCRIPT_DIR/build/libs/mcprec-6.13.jar" --gameDir "$SCRIPT_DIR/$run_dir" --envPort="$port"
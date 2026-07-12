# Runtime profiling and hardening report

Date: 2026-07-12

## Scope

Static review covered the C-ACT runner, VLM service, batch proxy, ServerAPI,
XENON environment wrapper, and MineRL/Minecraft Java launcher. GPU/Java process
profiling was not possible on this Windows host because nvidia-smi, perf,
and a local Minecraft runtime are unavailable.

## Measured checks

| Target | Result |
|---|---:|
| Procedural snapshot identity generation | 288 cells in 0.003044 s |
| Batch proxy synthetic concurrency | 64 requests, 4 batches, mean batch size 16, 0.1394 s |
| Python compilation | PASS |
| Protocol smoke test | PASS |
| Controller ledger | 600/600 PASS |
| Extreme stress suite | 992/992 PASS |

Artifacts:

- world_identity.prof
- world_identity_profile.txt
- identity_benchmark.json
- batch_proxy_benchmark.json

## Fixes applied

- VLM binds to loopback by default; non-loopback binding requires an API token.
- Optional X-CACT-Token authentication is propagated through ServerAPI and batch proxy.
- Request body and image payload limits reject oversized or malformed input.
- Client-controlled image paths are sandboxed under CACT_IMAGE_ROOT.
- VLM retry loops are bounded by CACT_MAX_RETRIES.
- Batch proxy uses ThreadingHTTPServer and threading.Event instead of a
  single-threaded server and 5 ms busy-wait polling.
- ServerAPI uses a shared HTTP connection pool.
- batch_chat initializes the agent on first use.
- XENON environment mutable inventory defaults were removed; seed reset uses the
  modern Gym API when available.
- MineRL/Minecraft startup now has a bounded, non-blocking Linux log wait
  (MINERL_STARTUP_TIMEOUT) and configurable shutdown grace periods
  (MINERL_KILL_GRACE, MINERL_EXIT_GRACE).

## Server recommendations

Set these before a production run:

~~~bash
export CACT_HOST=127.0.0.1
export CACT_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CACT_HTTP_MAX_CONCURRENCY=64
export CACT_MAX_REQUEST_BYTES=$((16*1024*1024))
export CACT_MAX_IMAGE_BYTES=$((8*1024*1024))
export MINERL_STARTUP_TIMEOUT=180
export MINERL_KILL_GRACE=0.5
export MINERL_EXIT_GRACE=1.0
~~~

For a remote VLM endpoint, place TLS/authentication in a reverse proxy and
never expose the unauthenticated app directly.

## Remaining server-only measurements

Run on Ubuntu with the real stack:

- nvidia-smi dmon / torch.profiler for GPU utilization and peak memory;
- /usr/bin/time -v for Java/Python RSS and context switches;
- startup latency and episode reset latency;
- Java process count after a forced timeout;
- throughput with CACT_WORKERS=1,2,4,8 and with/without the batch proxy.


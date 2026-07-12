# Instrumentation changelog

No profiling-only instrumentation was inserted into production code. The
runtime fixes were correctness, security, and throughput fixes identified by
static inspection and synthetic benchmarks.

| File | Change type | Profiling-specific change |
|---|---|---|
| profile_output/world_identity.prof | created | cProfile artifact for procedural identity generation |
| profile_output/world_identity_profile.txt | created | Top cumulative cProfile functions |
| profile_output/identity_benchmark.json | created | Wall-clock identity benchmark |
| profile_output/batch_proxy_benchmark.json | created | Synthetic concurrent proxy throughput benchmark |
| profile_output/PROFILE_REPORT.md | created | Results, limitations, and server-only profiling plan |


# Ubuntu/GPU 运行优化说明

## 推荐环境变量

```bash
export PYTHON=python3
export CACT_WORKERS=4                 # 按 Minecraft 实例和 GPU 显存压测调整
export CACT_VLM_PORT=12345
export CACT_VLM_STARTUP_WAIT=8       # 已预热的常驻 VLM 可降低
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

正式 E3/E5 已将子进程 stdout/stderr 改为每-run 文件，避免 Python 主进程缓存完整输出；TrustStore 使用紧凑 JSON 和原子替换，减少格式化 I/O。冻结评估可在确认 E0 通过且 frozen store 不写入后启用：

```bash
export CACT_FROZEN_HARDLINK=1
```

该选项通过 hardlink 复用只读冻结快照，若任何 frozen 代码尝试写 store，会被哈希检查发现；未确认前保持默认关闭。

## 语义不变性

- `CACT_WORKERS` 只改变并行度，不改变任务、seed、候选或方法。
- shared update 阶段仍强制 `workers=1`，防止共享 store 竞争写入。
- frozen 评估仍验证 store 与 policy artifact 前后哈希。
- `CACT_DURABLE_WRITES=1` 可为关键 checkpoint 启用每次 fsync；默认追求吞吐。
- 运行器记录每个子进程的 stdout/stderr 路径，失败可复盘。

## 压测建议

先用 E0 做 `CACT_WORKERS=1,2,4` 小规模对比，记录 runner 的 `elapsed_sec`、GPU utilization、Minecraft server CPU 和 VLM batch latency；只有吞吐提高且 E0 结果不变时才扩大到 E3/E5。


## Frozen-store hardlink safety

`CACT_FROZEN_HARDLINK=1` is not sufficient by itself. Hardlinks are enabled only when `CACT_ALLOW_UNSAFE_HARDLINK=1` is also set; the safe default is a real copy. This prevents an accidental frozen write from mutating the calibration source store.
## Release gates and multi-GPU E5

The release runner validates sealed task cards before spending GPU time. Set
`CACT_TASK_CARDS` to a whitespace-separated list of JSON/YAML card files (or
set `CACT_REQUIRE_TASK_CARDS=0` only for a non-claiming dry run). E2 additionally
requires `CACT_WORLD_SNAPSHOT_MANIFEST`, and E1c/D_audit require real sealed
artifacts; these gates must not be bypassed for paper results.

For a VLM pool, set `CACT_GPUS=0,1,2,3`. The runner exports the corresponding
ports to E5 via `--vlm_ports`; do not launch a second server on those ports.
For frozen E3/E4 runs, the same `CACT_WORLD_SNAPSHOT_MANIFEST` is required by default; its keys must be `task_id|world_seed`, and each hash is propagated into the episode logs.

# 参考材料学习笔记

## 核心主线

本项目不是简单复现 Voyager，也不是只减少 LLM 调用次数。核心 claim 应收紧为：

LLM 不在实时动作选择路径上。

也就是说，Controller 每一帧计算动作时不等待 LLM 返回；LLM 只在后台低频异步生成或更新结构化规范，例如价值矩阵 `V`、约束 `C`、任务目标 DAG `G`。实时动作由 Controller 本地完成。

## 架构原则

- LLM Worker 负责语义推理和结构化规范生成，允许调用 LLM API，允许读写 `V/C/G` 共享快照。
- Controller 负责实时控制，只读取已验证的 `V/C/G` 快照，不调用 LLM，不写 `V/C/G`。
- Mineflayer 层尽量复用 Voyager 原实现，不在 Node 子进程里重写决策逻辑。
- LLM 输出不应是“直接控制游戏的自然语言动作序列”，而应是可校验、可复现的结构化规范。

## Controller 判断目标推进

Controller 判断“动作是否推进任务目标”必须是纯结构化逻辑，不应再调用 LLM。

示例流程：

1. 从任务 DAG 中找到当前可执行叶子任务，也就是所有 `needs` 都已完成的节点。
2. 检查候选动作的 `target` 是否匹配可执行任务的 `target_item`。
3. 检查动作类型是否匹配目标类型，例如 `mine` 对方块，`craft` 对合成品，`kill` 对实体。

间接推进动作，例如“走向森林”，不应由 goal bonus 处理，也不应调用 LLM 判断。它应由价值矩阵中的 exploration 维度驱动。goal bonus 只给直接产生目标物品或完成直接依赖的动作。

## 改造计划

基于 Voyager 改造时的边界：

- 复用 `voyager/env/bridge.py`
- 复用 `voyager/env/mineflayer/*`
- 复用 `voyager/control_primitives/*.js`
- 复用 `voyager/utils/*.py`
- 重写或替换 Python 决策层
- 将原来的 per-task LLM loop 改成 per-frame Controller loop

建议新增模块：

- `agents/llm_worker.py`：低频异步生成/诊断/更新结构化规范
- `agents/controller.py`：实时意图选择与动作选择
- `agents/value_matrix.py`：多维价值矩阵、特征权重、置信度
- `agents/constraint_engine.py`：硬约束和软约束求值
- `agents/feedback.py`：追踪数据、异常检测、证据分级
- `agents/entity_registry.py`：实体特征与继承关系

## 实验关注点

实验不要只证明 LLM call 更少。更关键的是证明：

- 威胁响应更快
- 异常恢复更快
- 动态环境下更稳定
- Controller 单帧决策延迟稳定低于实时阈值
- LLM Worker 延迟不会阻塞动作选择

批注中也指出，实验难跑时不必强行做显著性统计；有统计更好，但主线是实验设计要能支撑架构 claim。

## Git 规范

- 项目统一优先使用 JS，不主动引入 TS；已有 TS 文件不必强改。
- Commit 注释统一使用中文。
- Commit 格式：`type(scope): 简要说明`
- 常用 type：
  - `feat`：新增功能
  - `update`：局部代码修改
  - `fix`：修复问题
  - `docs`：文档变更
  - `style`：格式调整
  - `refactor`：重构
  - `perf`：性能优化
  - `test`：测试
  - `chore`：配置、依赖、脚本等杂项
  - `revert`：回滚
  - `release`：发布版本

协作流程上，从 `develop` 拉取并创建个人分支开发，与 xzh 核对无误后再 push/合并到 `develop`；`main` 保持稳定版本。

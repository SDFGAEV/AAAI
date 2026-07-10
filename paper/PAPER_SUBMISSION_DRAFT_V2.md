# C-ACT: Contextual Admission via Counterfactual Treatment Effects

## Abstract

开放世界智能体通常能够检索到相关知识，却无法判断当前上下文是否满足知识的适用边界。我们提出 C-ACT，一种位于固定检索器与同一基础规划器之间的上下文知识准入层。C-ACT 将一次知识复用视为局部处理干预：在满足预注册资格条件的机会窗口内，以固定概率随机分配 reuse 或 base，记录结果、风险、资源冲突和可审计性字段，并使用按 episode 聚类的交叉拟合 AIPW 估计条件增益与风险。策略不是通过全局平均收益选择，而是在 source×type、task-group×failure-type×risk-tier 以及 resource-scarcity×boundary 的层级上进行支持度检查和回退；在最坏分层的绝对风险与增量风险上施加上置信界约束，选择最大覆盖率的准入策略。

本文只主张两点： (i) 在数据支持的上下文中，C-ACT 能提供可复现的、可审计的知识准入；(ii) 当支持不足或风险约束无法满足时，C-ACT 会显式回退，从而给出应用边界与风险—覆盖率权衡。本文不声称有限样本下的普适安全保证，也不将 contract 满足误写成价值或伤害贡献。

## 1. Problem and Scope

给定固定 retriever 输出的 top-1 知识 (K)、任务上下文 (X) 和基础规划器 (P)，C-ACT 在一个局部干预窗口内决定是否让 (P) 使用 (K)。第二次干预在协议中被删失；结果变量为该窗口内的子目标完成 (Y)，主要风险标签为 H1–H4（错误知识复用、资源冲突、不可逆状态恶化、链式失败），H5 为资源/成本风险，H6 为审计异常。Contract 只用于 applicability：scope、precondition、hard non-applicable boundary 和 provenance；它不被当作 (Y) 或 (H) 的代理标签。

机会只有在以下条件同时满足时进入主估计：固定 top-1 retriever、reuse/base 两分支可执行、全局安全 shield 通过、无高风险上下文、局部窗口可屏蔽其他知识、任务尚未完成且无重复动作。资格不足的机会仍可记录，但不能进入 AIPW 主估计。

## 2. Method

### 2.1 Randomized opportunity logging

在 D_fit、D_select 和 D_audit 中，reuse assignment 使用 (e(X)=0.5)；实现允许在 ([0.2,0.8]) 内配置，但会记录 reuse/base propensity 并验证 positivity。每条机会写入 `cact.v2` JSONL：episode/opportunity/round/stream/task/world、knowledge/source/type、retrieval rank/score、raw-text hash、上下文分层字段、assignment、propensities、randomization seed、窗口与删失标记、(Y)、H1–H6、成本及 label source。

### 2.2 Episode-clustered cross-fitted AIPW

我们按 episode 划分 5 个折，绝不拆分同一 episode。D_fit 仅用于拟合 outcome nuisance models；D_select 用于策略选择，D_audit 用于一次性封存审计。对 (Z\in\{Y,H,H(1)\})，使用 cross-fitted AIPW pseudo-outcomes：

\[
\phi_Y=\hat m_1(X)-\hat m_0(X)+\frac{A(Y-\hat m_1(X))}{e(X)}-\frac{(1-A)(Y-\hat m_0(X))}{1-e(X)},
\]

并分别得到 (Delta_Y=E[phi_Y])、绝对风险 (R_\mathrm{abs}=E[\psi_{H(1)}]) 和增量风险 (R_\mathrm{inc}=E[\phi_H])。分层支持要求 reuse/base 各至少 12 个样本且 ESS 至少 24；不满足时运行时按 g2→g1→g0 回退，并报告 UnsupportedFallbackRate 与 MeanBackoffDepth。

### 2.3 Calibrated admission policy

在 (κ\in\{0,0.5,1,1.5,2,2.5,3\}) 上选择最大覆盖率策略，要求每个被准入分层满足

\[
\mathrm{LCB}(\Delta_Y)\ge \delta=0.05,\quad
\mathrm{UCB}(R_\mathrm{abs})\le \epsilon_\mathrm{abs}=0.10,\quad
\mathrm{UCB}(R_\mathrm{inc})\le \epsilon_\mathrm{inc}=0.02.
\]

D_audit 只做一次封存审计；审计失败不允许写出可部署策略。冻结评估阶段只读取快照和策略 artifact，并验证其哈希未发生变化。

## 3. Experimental Protocol

- **E0 substrate validation:** 验证任务、retriever top-1、两分支可执行性、局部窗口和日志 schema。
- **E1a/E1b:** 注册 applicability contract；在独立机会集合中验证随机 assignment、propensity、窗口和删失字段。
- **E2:** 按 D_fit/D_select/D_audit 进行 policy selection、sealed audit 和 paired branch audit。
- **E3 main:** 在三层冻结任务集上比较 NoKnowledge、NoGate、SuccessLifecycle、FixedBayes、ACT 和 C-ACT。报告 ID success rate、Comp-OOD HardSR、Boundary FAR、HRR、Coverage、UnsupportedFallbackRate、调用数和 token 成本。
- **E4 ablations:** 仅包含 NoContract、NoAdaptiveTau、NoActiveCalib 和完整 C-ACT 四个预注册变体。
- **E5 streams:** 至少 5 个独立受控 stream，报告跨 stream 均值、标准差和最坏分层结果。

所有方法共享同一 retriever、基础规划器、任务、世界种子和计算预算；差异仅来自准入层。结果文件必须能够由 `analysis/validate_logs.py` 与 `calibrate_v2.py` 重新解析，不允许手工填充表格。

## 4. Claims and Limitations

我们的主要结论限定为：在支持充分且审计通过的上下文中，C-ACT 的准入决策满足预注册的经验风险约束；在支持不足或边界冲突时，系统会回退并报告适用性缺口。由于样本量、任务分布和局部干预窗口有限，本文不提供有限样本普适安全定理，不声称跨环境因果外推，也不声称 C-ACT 是最早提出此类方法的工作。


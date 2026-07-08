# AAAI / Top-Tier AI Conference Writing Pattern Analysis

**Analyzed papers**:
1. MineEvolve — arXiv 2603.13131, self-evolving MC agent, 30 pages
2. Plan Reuse (AgentReuse) — J. Comput. Res. Dev., LLM agent plan reuse, 11 pages
3. Parallelized Planning-Acting for MC — AAMAS 2026, multi-agent MC, 16 pages
4. BiLLM — AAAI 2024, LLM binarization (read via HTML)
5. AAAI template formatting instructions (AAAI 2025 style guide)

**Method**: Full-text extraction from 4 papers via pymupdf, systematic pattern extraction.

---

## 一、Abstract 的 5 种写法

### 模式 1: 五句标准结构 (MineEvolve, BiLLM, 最推荐)
```
句1: Problem — 领域问题是什么
句2: Challenge — 具体难点在哪
句3: Solution — "To this end, we propose X"
句4: Method — 一句话说怎么做（MineEvolve 用编号图标 ❶❷❸❹）
句5: Evidence — "Experiments on Y show..."
```
**MineEvolve 示例**:
> "Long-horizon embodied intelligence requires agents to improve through interaction... A central challenge is therefore to transform past executions into knowledge... To this end, we propose MineEvolve, a knowledge-driven self-evolution framework... Experiments on the Minecraft MCU long-horizon task suite show that MineEvolve consistently improves performance..."

**我们的论文应该用的模式**。当前我们的 abstract 比较散，需要收紧到 5 句。

### 模式 2: 数据结果驱动 (Plan Reuse)
```
句1: Context — LLM agents 的广泛部署
句2: Data observation — "Real-world dataset analysis shows 30% of requests are identical"
句3: Challenge — "However, it is difficult to..."
句4: Method — "We present AgentReuse which..."
句5: Results — "93% effective reuse rate, F1=0.9718"
```
**适用场景**: 当你有很强的定量结果时。我们目前没有结果，不适合。

### 模式 3: 三挑战分析型 (Parallel MAS)
```
句1: Context — MAS 的进展
句2: Gap — "existing frameworks rely on serialized execution"
句3: Three challenges listed — inflexible scheduling, limited replanning, memory delays
句4: Method — "we propose a parallelized framework with dual-thread architecture"
句5: Evidence — "Extensive experiments on Minecraft demonstrate..."
```
**适用场景**: 当你能把问题分解为具体挑战时。我们的"问题"就是"heuristic gate 不测因果效应"，不是三个挑战。

---

## 二、Introduction 的标准结构

### 段落序列（所有四篇论文一致）
```
P1: Broad context → 领域的重要性
P2: 具体问题 → 现有方法做了什么，缺什么
P3: 核心问题 → 用斜体或粗体单独成句
P4: 方法 overview → "To address this, we propose X which..."
P5: 贡献 bullets → 3-4 项，每项 2-4 行
```

### MineEvolve 的 Introduction (完整分析)
```
P1: Broad context about embodied agents
  - "Embodied agents are increasingly expected to..."
  - 5 citations in first 3 sentences
  - "Minecraft provides a representative test platform" — 锚定环境

P2: Specific challenges
  - 长链任务的具体失败类型: missing tools, blocked paths, GUI failures
  - "Failures often arise not from LLM misunderstanding, but from fine-grained execution errors"
  - 这是一个精彩的 "not X, but Y" 对比

P3: Central question (italic, standalone)
  - "How can an agent transform execution feedback into behavioral knowledge 
     that is attributable, verifiable, and directly actionable for future planning?"
  - 注意形容词选择: attributable, verifiable, actionable — 三个精准要求

P4: Method overview
  - "We address this question with MineEvolve, which..."
  - 四步 pipeline 用编号图标: ❶Monitor ❷Inducer ❸Curator ❹Adaptor
  - 每个步骤用 1 句描述

P5-P6: Elaboration of two key mechanisms (Success→Skill, Failure→Remedy)
  - "beyond trajectory memory and static experience retrieval" — rather than 对比
  - "failures are not treated as passive records" — 重新定义失败的价值

P7: Contribution bullets
  - 3 bullets, each 3-4 lines
  - 每项以具体机制名开头 (Knowledge-driven self-evolution framework, 
    Feedback-conditioned generation, Knowledge-guided repair)
  - 不重复方法描述，而是提炼贡献点
```

### 关键发现: Introduction 不应以 "We" 开头
四篇论文的 Introduction 第一句都不以 "We" 开头:
- MineEvolve: "Embodied agents are increasingly expected to..."
- Plan Reuse: "In recent years, with the growth..."
- Parallel MAS: "Multi-Agent Systems (MAS) have become..."
- BiLLM: "Pretrained large language models (LLMs)..."

**我们的论文**: 当前以 "**Every self-evolving agent**..." 开头 — 虽然也不是 "We"，但可以改进为更标准的宽→窄结构。

---

## 三、Related Work 的组织方式

### 所有论文都用主题组织，不用时间线
```
MineEvolve:
  2.1 Minecraft Embodied Agents and Benchmarks (8 citations)
  2.2 LLM Planning and Embodied Control (12 citations)
  2.3 Experience Memory and Self-Improvement (10 citations)
  
  每段结构: 
    - 段落开头: 领域描述
    - 中间: 4-6 篇论文，每篇 1 句
    - 段落结尾: "However, existing methods mostly focus on... 
                  while... remains underexplored."
```

### 引用的密度模式
- **Related Work** 段几乎每句都有 citation
- 一个句子可能堆叠 3-5 个引用: "[Johnson et al., 2016, Guss et al., 2019, Sutton et al., 1998]"
- 每个引用只用 1 句话，不像文献综述那样详细展开

### 定位句（每段结尾的"批评+我们的位置"）
```text
MineEvolve §2.1: "These works provide important datasets... but they mainly focus on 
  demonstration learning... leaving long-dependency planning, failure recovery, and 
  cross-task experience transformation less explored."

MineEvolve §2.2: "However, long-horizon tasks also require agents to revise future plans 
  based on execution feedback collected during interaction..."

MineEvolve §2.3: "However, existing methods mostly focus on successful skill reuse, 
  historical summaries, or dependency correction, while the conversion of fine-grained 
  execution feedback into behavioral knowledge for plan repair remains underexplored."
```

**模式**: "These methods mainly address X / However, Y also requires Z / In contrast, our work focuses on W."

---

## 四、Method 的呈现方式

### MineEvolve 的方法结构
```
§3 Method
  §3.1 Overview (图 3 + 文字描述)
  §3.2 Feedback-Conditioned Skill and Remedy Generation (公式 1-4)
  §3.3 Knowledge Curation (Curator 的五维验证)
  §3.4 Knowledge-Guided Plan Repair (Adaptor 的工作流)
  
特点:
  - 每个子节以 "As shown in Figure X, ..." 开头
  - 公式紧跟在文字解释之后
  - 每段文字 3-5 句，不冗长
  - 关键步骤用加粗标记
```

### 编号图标的使用 (MineEvolve 特色)
```
"❶Monitor converts subgoal executions into typed execution feedback
 ❷Inducer derives skills from successful executions and remedies from failed ones
 ❸Curator validates, merges, filters, and retrieves the generated knowledge
 ❹Adaptor conditions planning on the retrieved knowledge..."
```
**效果**: 产生强烈的视觉记忆点，审稿人一眼能记住 pipeline。

---

## 五、Experiments 的呈现方式

### MineEvolve 的实验结构
```
§4 Experiments
  §4.1 Experimental Setup (Models, Baselines, Tasks, Metrics)
  §4.2 Main Results (Table + 分析)
  §4.3 Ablation Study (组件消融)
  §4.4 Knowledge Accumulation Study (在线演化)
  §4.5 Case Study (定性分析)
```

### 结果表格的格式
- 全部用 **booktabs** 样式（`\toprule`, `\midrule`, `\bottomrule`，无竖线）
- 最好的数字加粗 (`\textbf{0.74}`)
- 每个表有 3-6 行的分析文字
- 对比用 pp (percentage points)，不用 %
- 统计显著性用小标注 († p<0.05, ‡ p<0.01)

---

## 六、句式模式汇总

### 高频对比句式（从所有论文提取）
```text
"...not from X, but from Y"                          (指出盲区)
"...rather than only..."                             (对比现有方法)
"...beyond X and Y"                                  (拓展范围)
"...This goes beyond X."                             (宣称贡献)
"...however, ... remains underexplored."              (指出 gap)
"...In contrast, X focuses on..."                     (定位自己)
"...These methods mainly address X. However, Y also requires Z." (过渡句)
```

### 过渡词使用频率
```text
高: "However," "In contrast," "To this end," "Specifically," "For example,"
中: "Recent work," "More recently," "Furthermore," "Moreover,"
低: "Interestingly," "Notably," "Surprisingly," (AAAI 论文几乎不用这些)
没有: "In addition," "Additionally," (太弱)
```

### 段落开头模式
```text
名词短语开头: "Long-horizon embodied intelligence..." (MineEvolve)
            "Recent advancements in..." (Parallel MAS)
            "An autonomous agent..." (我们的)
            
不要: "We propose..." 作为段落第一句
      "It is well known that..."
      "There has been growing interest in..."
```

---

## 七、对文字的具体建议

### 1. 少用 "We"
当前我们论文中 "We" 出现次数较高。改进方法: 把 "We argue" → "This paper argues"，"We propose" → 直接命名方法。

### 2. 加编号图标 (MineEvolve 特色)
我们的 pipeline 已经用了 `\ding{182}`-`\ding{188}`，✅ 已实现。

### 3. Introduction 核心问题要斜体独句
改: `\emph{The correct question is not "does this knowledge look good?" but rather: does reusing this knowledge cause better outcomes than not reusing it?}`

### 4. Related Work 每段以 "However, ... remains underexplored." 结尾
当前我们的 Related Work 缺失这种结尾模式。每个小节最后一句话应该指出 gap + 我们的定位。

### 5. 数字不用 "~" 用精确值
`~2016 episodes` → `2,016 episodes`
`~3100 lines` → `3,133 lines`

### 6. Citation 密度要大幅提高
当前 Related Work 部分有些描述没有具体引用。参照 MineEvolve 的密度：每 1-2 句至少有一个 citation。

### 7. 实验结果段落的结构
```
结果句: "C-ACT-Full achieves HRR of 0.07 [TBD]..."    (数字)
比较句: "representing a XX% reduction from Baseline"    (相对改善)
解释句: "The contract layer is the primary driver..."    (归因)
置信句: "(p < 0.001, paired bootstrap)"                 (统计)
```

### 8. 术语一致性
- "C-ACT" 或 "\textsc{C-ACT}" — 全文统一
- "counterfactual uplift" vs "Bayesian uplift" — 选一个
- "harmful reuse" vs "harm" — 指标名保持一致

---

## 八、结构检查清单

```
[ ] Abstract 是 5 句结构
[ ] Introduction 不以下面的方式开头: "We propose", "In this paper we"
[ ] Introduction 的核心问题是斜体独立句
[ ] Contribution bullets 每项 2-4 行，以具体机制名开头
[ ] Related Work ≥ 1 页 (当前已够)
[ ] Related Work 每段以定位句结尾 ("However... remains underexplored")
[ ] Method 用编号图标 (✅ 已实现)
[ ] Method 有系统架构图 (Figure 1)
[ ] Experiments 有 task breakdown 表
[ ] 所有数字是精确值，不是 ~approximate
[ ] 统计显著性有标注
[ ] Citation 在 Related Work 中密度达标 (每句或每两句)
[ ] Ethics Statement 存在 (✅ 已实现)
[ ] Conclusion 不是 copy-paste intro
[ ] 全文没有 "novel", "first", "state-of-the-art" (自封的)
[ ] 全文没有 "Interestingly", "Notably", "Surprisingly"
```

---

## 九、论文间对比: 我们的 vs MineEvolve

| 维度 | MineEvolve | 我们的 (当前) | 改进建议 |
|------|-----------|-------------|---------|
| Abstract 结构 | 5 句 | 散文化 | 收紧到 5 句 |
| Intro 第一句 | "Embodied agents are increasingly..." | "Every self-evolving agent faces..." | 可保持，更有力 |
| 核心问题 | 斜体独立句 | ✅ 已改为斜体 | 已完成 |
| Related Work 密度 | 每句有 cite | 部分无 cite | 提高密度 |
| Method 图标 | ❶❷❸❹ | \ding{182}-{188} | ✅ 已完成 |
| 贡献 bullets | 3 项 × 3-4 行 | 3 项 × 2-3 行 | 可稍扩展 |
| 定位句 | "However, ... remains underexplored" | 部分缺失 | 需要每段加 |
| 数字 | 精确值 | 有的用 ~ | 全部改为精确值 |

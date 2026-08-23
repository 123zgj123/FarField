<div align="center">

# FarField

**大胆探索。只有同一科学对象撑得住的结果，才能晋升。**

本机自动研究循环：远场搜索、先登记再观察的两臂实验，以及模型改不了的证据合同。

[English](README.md) · [简体中文](README.zh-CN.md) · [引用](#引用)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-003a70)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-9a7b2e)](LICENSE)
[![CI](https://github.com/123zgj123/FarField/actions/workflows/tests.yml/badge.svg)](https://github.com/123zgj123/FarField/actions/workflows/tests.yml)
[![Dependencies](https://img.shields.io/badge/stdlib%20only-no%20pip%20deps-5c6b7a)](pyproject.toml)

[Guijia Zhang](https://123zgj123.github.io)

</div>

## 演示

DeepSeek-V4-Pro 上的英文控制台实况。题目：*combining diffusion models with agent-tool authorization security*。本场留下 `diffusion model × quasipolynomial time algorithm`，任务内造 GENERATED `labeled_traces` 世界，处理臂 4 / 对照臂 10 得到 `supports`，确认数为 0。

<p align="center">
  <a href="https://123zgj123.github.io/FarField/demo.html">
    <img src="assets/farfield-demo-poster.png" alt="播放 FarField 控制台演示（78 秒）" width="920">
  </a>
</p>

## 摘要

自动研究系统已经能写出像样的想法、实验，甚至一篇像论文的草稿。真正开放的问题不是生成，而是：**什么时候，一个 AI 给出的结果值得相信。**

FarField 是一套可检查的本机循环。给它一个题目，它在概念图上搜索，提出可证伪的假设，在**看到任何数字之前登记两臂实验**，并输出别人可以复跑的方案。模型可以写主张、机制和脚本。图上的查重、已有工作检查、世界绑定，以及 treatment / control 的算术，都由普通 Python 完成——这些门上没有模型。

一次确认只允许在这条谓词成立时发生：**绑定、构造、杠杆和探针度量必须是同一个科学对象。** schema 相同不是科学对象相同。一部小说可以扮成任何序列形主张；一台编出来的自动机不能顶替缺失的 I/O 轨迹。合成或构造数据可以削弱假设，不能确认假设。

终点不是会议 PDF，而是一份研究包。

## 贡献

1. **远场搜索是位置，不是信念。** 算子在概念图上移动；距离是搜索启发，从来不是新颖性证明。
2. **先登记，再观察。** treatment、control、指标、竞争解释和预期方向在执行前固定。
3. **同一对象合同。** `WORLD` 支持要求主张那个对象的已冻结夹具、该世界已声明的杠杆、度量该对象的探针，以及一份毁掉结构后失去原判的安慰剂。未完成的获取和生成顶替不能爬梯。
4. **证据不对称。** 负面证据比正面证据更容易成立。排名、文笔和评审分数不能提升结果。

<p align="center">
  <img src="assets/overview.png" alt="FarField 循环：题目、检索、假设、筛选、已登记两臂实验、研究包" width="920">
</p>

## 为什么做 FarField

很多自动研究系统已经能够生成看起来合理的想法、实验，甚至一篇像论文的草稿。更难的问题是：

> **什么时候，一个 AI 生成的结果真的值得相信？**

自动研究里很容易出现几类失败：

- 模型把自己的解释当成支持自己假设的证据；
- 模型先看到了实验数字，再修改实验或重新定义成功标准；
- 系统用“为了这条假设临时生成的数据”来确认这条假设；
- 一个想法因为模型评审打分高，就被当成科学上更可信；
- 一次超时、脚本错误或数据缺失，被错误解释成“假设失败”。

FarField 围绕一条更严格的原则设计：

> **大胆探索，谨慎相信。**

搜索阶段可以激进，证据阶段不可以。

因此，FarField 把**假设生成**和**科学判断**明确拆开：

- far-field 搜索负责提出远离当前局部邻域的概念组合；
- 模型负责写主张、机制、预测和实验实现；
- 可执行检查负责判断假设能不能进入后续研究；
- treatment、control、测量方式和预期方向必须在实验前登记；
- synthetic / generated data 可以削弱假设，但不能确认假设；
- 活假设之间的排序只是参考，不能直接接受、拒绝或提升科学结论。

FarField 的终点不是一篇会议 PDF，而是一份别人可以独立检查、执行和复现的研究方案。

## 研究循环

| 阶段 | FarField 做什么 |
| --- | --- |
| **检索** | 检索相关工作，并从概念共现图上采样较远、尚未形成局部邻接关系的概念组合。 |
| **假设** | 模型提出可证伪的主张、机制、预测，以及可能使它失败的条件。 |
| **筛选** | 确定性的图检查和文本检查丢掉已经存在的组合，以及结构上不合格的假设。 |
| **已有工作检查** | 在投入实验成本之前，检查主张或核心机制是否已经出现在已有文献中。 |
| **登记** | 在看到任何结果之前，固定 competing explanation、treatment、control、指标和预期方向。 |
| **实验** | 两臂使用同一个测量函数，只在被检验的机制上存在差异。 |
| **证据** | 科学结果与超时、缺文件、数据不匹配等运行故障分别记录。 |
| **方案** | 留下的假设被整理成 `RESEARCH_PACKET.md` 和 `protocol.json`，供独立复跑。 |

在临时构造数据上的短实验只能算一致性检查，不能算确认。

要支持一条假设，需要与当前结果独立冻结的数据，并对同一个已登记实验进行复跑。

## Far-field 探索

FarField 不只是让模型回答一句：

> “请 brainstorm 一些 novel ideas。”

它把三个问题分开：

```text
去哪里搜索
    ↓
在那里提出什么假设
    ↓
这个假设是否能活下来
```

搜索阶段先在概念空间中远离题目的局部邻域，再从落点附近取回真实概念。

当前使用的 far-field 操作包括：

- **directional** —— 从种子方向沿受控方向向外移动；
- **interpolate** —— 朝较远概念区域的中心移动；
- **low-density** —— 优先搜索概念空间中的稀疏区域；
- **analogy** —— 使用远距离概念之间的类比向量偏移寻找新的落点。

搜索阶段只决定：

> **值得去哪里看。**

它不决定：

> **那里找到的东西是不是真的。**

每次跳转都会记录使用的 operator、落点、取回的概念、与原始邻域的距离等信息，使探索路径可以检查和复现。

“在 embedding 空间里更远”也不等于“科学上一定更新”。

Far-field distance 只是搜索启发式。候选假设之后仍然需要通过图检查、已有工作检查和实验。

## 假设只是提案，不是证据

模型可以写：

- 主张；
- 机制；
- 预测；
- falsifier；
- 已失败的方向；
- 为什么失败；
- 如何重新表述问题；
- 一个可能的审稿人反对意见。

这些信息有助于组织研究过程，但它们都不算科学证据。

FarField 使用一条简单规则：

> **模型生成的文字可以提出一个主张，但不能证明这个主张成立。**

假设只有经过外部检查和实验之后，才能获得更高的科学状态。

## 证据模型

FarField 会区分实验数据是从哪里来的。

### Frozen world

Frozen world 是在当前实验结果之外独立冻结的数据集。

它的数据内容、来源、schema 和使用范围都会绑定到实验记录中。

如果数据集与主张匹配，那么在 frozen world 上得到的成功实验结果可以支持假设。

### Generated world

Generated world 是为了当前已登记实验构造的数据。

它可以暴露：

- 机制根本跑不通；
- 假设在最基本情形下就是错的；
- treatment 并没有产生预期效果。

但它不能独立确认“导致这份数据被生成的那条假设”。

### Synthetic world

Synthetic world 是用于一致性检查、调试和轻量实验的小型人工构造数据。

它遵守同样的不对称证据规则：

```text
synthetic / generated result
        |
        +-- weakens       -> 可以成为负面科学证据
        |
        +-- supports      -> 不能算确认
```

这种不对称是刻意设计的：

> **一个系统应该比证明自己的想法更容易推翻自己的想法。**

### 前向可模拟的世界

冻结让世界变得真实，但光有字节还不可分析：一段小说的 token 流可以装扮成任何"序列形状"主张的世界。因此每个 schema 族都带一座**运行时所有的动力学桥**——只包裹冻结字节，从不改写它们：

- **杠杆（lever）**是机制作用于这个世界的唯一通道（删边、删 token、碱基突变、行噪声、删转移……）；
- **可观测量（observable）**是运行时自己计算的测量——模型没有写过的裁判；
- **前向模拟**沿强度斜坡把冻结数据滚动起来；轨迹对（world digest、杠杆、种子）确定，任何审计者都能逐位重算。

写诊断之前，运行时先对绑定世界做一次侦察（每根杠杆各模拟一遍），把实测的剂量-响应读数交给设计者。诊断必须把机制绑到一根已声明的杠杆、把预测量绑到一个已声明的可观测量——或者诚实地回答 `none`，这会**解绑世界**：探针仍然运行，但降级为一致性检查，无法 corroborate。桥没有声明的杠杆会被直接拒绝：模型可以选择把手，不能发明把手。主张点名的 `operations` 必须是该 schema 的能力子集，空 `pass` 不再放行。登记完成后，运行时再对选中的杠杆做一次前向模拟；预报若显示该杠杆惰性，必须重写诊断，不能烧一次探针才发现。`manifest.domains` 只描述实例身份（Pride 只留 prose / english / narrative），`sketch/hash/cache` 属于 schema 提示，只排序、不守门；异象对象（agent-tool 对小说）直接拒绝。佛罗伦萨等 `undirected_named_graph` 与无向图共用动力学桥。WORLD 上的 `supports` 还要过安慰剂消费门：同一脚本在结构被毁掉的副本上必须失去判决（用位移对 margin，不用 |z|），否则不得 corroborate。`REGISTER_PROTOCOL` 写在第一次 `run_probe` 之前；谱系若点名了另一个科学对象，已绑上的冻结件会被反向解绑。

模拟器产出的一切都是**侦察证据**：它会上链（`SIMULATE_WORLD`），审计者能看到设计者登记前看到过什么；它能暴露惰性杠杆、杀掉代理绑定，并在 uninformative 之后喂给重新设计——但任何晋升门都不计入它。`supports` 依然只能来自登记过的两臂探针及其 host / 执行器复现。

绑定、构造、杠杆和探针度量是同一条谓词：必须是主张的那个对象。catalog 未命中时，任务内按主张自己的族构造一份 GENERATED 世界供模拟（图、轨迹、公式、表、流、序列）；`io` 仍标获取未完成。构造世界可以削弱，不能确认，也不再用公式自动机顶替别的对象。下场 mission 会执行 wishlist 上已有公开 URL 的 freeze 配方，新主张可以绑上新冻件，旧卡不改判。竞争解释必须是这个世界里另一根已声明杠杆，不是英文词表。

### 证据身份与复现层级

“实验被重跑过”并不是一件事。FarField 给每一次执行分配一个 **EvidenceID**——对主张、实验脚本、world 和实际读取的字节做哈希——并把重跑放在一个复现阶梯上：

| 层级 | 含义 |
| --- | --- |
| `E0` synthetic | 为当前主张构造或生成的数据；可以削弱，不能确认 |
| `E1` 同实例复现 | 宿主用同一协议重跑了**同一份字节** |
| `E2` 规模复现 | 同一来源、不同切片或全量数据 |
| `E3` 独立复现 | 另一份独立冻结的数据集 |
| `E4` 外部复现 | 其他环境提交两臂数字 |

只有 **EvidenceID 完全一致的 `E1` 复现**才能确认探针结果。在全量母数据上的运行是规模复现：它有自己的 EvidenceID，属于同一个 replication group——它检验主张能否泛化，而不是重现同一份证据。

确认是**逢缺即拒（fail-closed）**的：缺 digest、缺 EvidenceID、协议不完整，都是拒绝而不是警告——“无法证明是同一份证据”和“确实是另一份证据”是同一个答案。

主张遵守同样的纪律：研究线（anchor × 远概念）不因措辞改写而中断，但对主张的任何实质修改都是新的 **hypothesis revision**。新 revision 继承这条线的上下文，但**绝不继承真值状态**——支持“机制 A”的证据不能在主张悄悄变成“机制 B”之后继续生效。

每一次认知层面的动作——冻结数据、登记假设、登记协议、执行实验、确认层的每一次重放、提升结论——都会追加进一条 digest 链式事件日志。这条链证明动作的**先后顺序**：数据在假设看到它之前就被冻结，假设在它的证据之前就已登记，协议在确认它的那次运行之前就已登记。修改、删除或调换任何一条事件都会破坏其后所有 digest。

这条链是**门，不是日记**。任何主张要写成 `corroborated` 或 `verified`，晋升步骤都会先审计账本：digest 必须逐条验证通过，该假设修订必须在其证据之前登记在案，协议登记必须先于声称确认它的执行事件。没有链、链被破坏、顺序颠倒，都拒绝晋升——证据也许是真的，但无法证明的顺序不是可信来源（provenance）。研究状态自己的晋升日志在每次写入前也接受同样的审计。

阶梯上每一级都严格难于下一级。`corroborated` 要求上述 E1 宿主复现。`verified` 还额外要求**执行器所有的复测**（executor-owned replication）：由可信运行时——而不是模型——重放**已登记的**那份脚本，seed 由证据身份自指派生（`seed_i = H(experiment_digest ‖ world_digest ‖ i)`，digest 覆盖脚本自身字节，写脚本时原理上无法预知），每次执行和聚合都记入事件链，统计由执行器计算。只有当逐 seed 分离度的**整个 95% 置信区间**越过预登记的 margin 时才判 supports——均值过线而区间跨线，按名字记为 uninformative；所有 seed 下产出完全相同数字的脚本记为*不变*（invariant），它与无视 seed 的脚本无法区分，因此不能确认。晋升时还会反查账本：那 N 次 seed 执行和点名它们的聚合事件必须真实在链上。贯穿始终的原则是：**模型可以生成计算，但永远不能生成审计它的证据**——执行身份、原始测量、复测数据、digest、顺序、晋升证明，全部只由运行时产生。

## 两臂实验

实验必须在执行之前登记。

概念上，每个 probe 都遵循类似结构：

```python
control = measure(
    world,
    mechanism_enabled=False,
)

treatment = measure(
    world,
    mechanism_enabled=True,
)
```

两臂：

- 使用相同的数据；
- 使用相同的测量函数；
- 使用相同的指标；
- 只在被检验的机制上存在差异。

实验执行前，FarField 会固定：

- competing explanation；
- treatment arm；
- control arm；
- measurement；
- expected direction；
- compute tier。

实验脚本是在这些选择已经登记之后才生成的。

因此，模型不能在看到结果之后再重新定义：

> “其实这个数字也可以算成功。”

## 实验执行

模型生成的实验代码不会直接执行。

运行前会先检查：

- treatment 和 control 是否都存在；
- 两臂是否使用相同测量；
- 实验是否符合已登记设计；
- import 是否在允许范围内；
- 是否使用了被禁止的运行能力；
- 是否超过计算预算；
- 输出是否包含有限且有效的 treatment / control 数值。

同时，FarField 会明确区分**运行失败**和**科学失败**。

例如：

```text
timeout
missing file
invalid script
incompatible dataset
runner failure
```

不会被记录成：

```text
weakens
```

科学结果单独记录为：

```text
supports
weakens
uninformative
```

一个脚本超时，并不能说明科学假设是错的。

## 已有工作检查

一个想法可能很有意思，但已经有人做过。

因此，在继续投入实验资源之前，FarField 会检索候选概念附近的文献，并检查：

- 核心组合是否已经存在；
- 主张是否已经被提出；
- 关键机制是否已经在已有工作中出现。

如果一个结论已经被文献明确建立，正确的行为通常不是重新“发现”它，而是读那篇论文。

因此 FarField 会把：

```text
closed_by_prior
```

和：

```text
weakened
```

视为不同结果。

前者表示“这个问题已经有已有工作”，后者才表示“实验给出了反对该假设的证据”。

## 研究状态

FarField 可以在不同任务之间保留结构化研究状态。

状态会区分例如：

```text
known
open
contradicted
rejected
promising
```

模型提出过一个假设，并不意味着这个假设就会进入长期知识。

有证据支持的发现、持续开放的问题，以及已经确认的失败方向，可以影响之后的搜索。

失败路线也可以作为 negative memory 保留下来，避免系统不断重复同一个死路。

这让 FarField 可以积累研究经验，同时避免把历史模型输出直接当成事实。

## 证据提升

研究结论会被保守地提升。

一条研究线可能经历：

```text
speculative
    ↓
corroborated
    ↓
verified
```

遇到负面证据后，也可能走向：

```text
weakened
closed_by_prior
```

状态提升依赖实验和外部证据，而不是：

- 模型置信度；
- 模型写得是否有说服力；
- reviewer 是否喜欢这个想法；
- idea ranking 是否很高。

Synthetic result 本身不能把一条假设提升到确认状态。

## 排序只是参考

FarField 可以对仍然存活的研究想法做排序，帮助决定下一步优先看什么。

排序可以参考：

- 当前实验信号；
- 已有工作的空缺；
- 可执行性；
- 潜在价值；
- 研究兴趣。

但：

> **一个排名高的假设，不代表它更接近真理。**

模型 reviewer、Elo 或其他 preference score 不能覆盖：

- 图检查；
- prior-art closure；
- 实验登记；
- treatment / control 算术；
- evidence promotion policy。

FarField 会记录 ranking，但不会把 ranking 当作科学证据。

## 研究策略演化

FarField 可以根据历史任务学习：

> 哪些探索 operator 更容易产生最后真正经受住证据检验的研究路径？

系统会记录类似这样的轨迹：

```text
operator
    ↓
hypothesis
    ↓
screening
    ↓
experiment
    ↓
evidence
    ↓
promotion
```

之后可以拟合新的 routing policy，调整不同搜索 operator 在未来任务中的使用概率。

但新的 routing policy 不会因为在生成它的数据上表现好，就直接覆盖当前策略。

候选策略还需要在 held-out missions 上优于 incumbent policy。

只有基于真实证据的结果才能影响 routing。

下面这些不能作为 routing reward：

- synthetic support；
- generated-world support；
- Elo；
- 模型主观评分；
- 文本写得是否漂亮。

这样，FarField 不只是积累更多历史，而是尝试学习：

> **怎样搜索，才更容易得到最后能够经受外部证据检验的研究方向。**

## Verifier 演化

FarField 也可以从反复出现的失败模式中提出新的文本检查规则。

但一条新规则不会一生成就获得“杀死假设”的权限。

它首先进入 **shadow** 状态：

```text
candidate verifier
       ↓
shadow judge
       ↓
跨任务观察
       ↓
held-out validation
       ↓
lethal judge
```

Shadow judge 会记录：

> “如果我现在有权限，我会拒绝哪些假设？”

但它暂时不会真的拒绝这些假设。

只有当它在后续独立任务中持续预测到了真实失败，并且没有错误拒绝已有支持证据的假设时，才可能获得更高权限。

因此 FarField 会区分：

> **提出一个 verifier**

和：

> **给予这个 verifier 决策权**

这两件事。

## Skills

成功的研究过程可以被蒸馏成可复用 skill。

例如一个 skill 可以描述：

- 什么情况下某种实验设计有用；
- 如何构造 treatment / control；
- 两臂应该只在哪个机制上不同；
- 应该使用什么测量；
- 哪些失败模式值得特别检查。

这些 skill 可以帮助后续模型调用，但 skill 本身不是科学证据。

蒸馏出来的 guidance 不能：

- 绕过实验登记；
- 在看过结果后修改 expected direction；
- 绕过 verifier；
- 直接提升假设；
- 改写 treatment / control 的结果解释。

可执行权限仍然属于 runtime。

## 本地、可检查、可复现

FarField 刻意保持本地运行和可检查。

核心 runtime 使用：

- Python 3.11+
- 标准库 Python；
- 本地事件与研究产物；
- 冻结实验数据；
- 可复放的实验记录；
- 可选的 OpenAI-compatible 模型后端。

模型负责适合生成和开放式判断的部分。

确定性代码负责应该可以复现的部分。

简单来说：

> **模型提出，代码检查，实验决定。**

## 环境

- Python 3.11 或更新
- 核心运行时、测试和控制台不需要第三方 Python 包
- 现场模型调用需要 OpenAI-compatible API key
- 密钥通过 `$FARFIELD_LLM_KEY_FILE` 提供

不要：

- 把 API key 提交到 git；
- 把 API key 放进命令行参数。

## 安装

克隆仓库：

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField
```

不需要：

```text
pip install
```

把 `src/` 加到 `PYTHONPATH` 即可。

运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
python3 -m unittest discover -s tests -q
```

测试不需要 API key。

仓库自带一份较小的概念图：

```text
concepts/attn-concepts-s1/
```

大小约 8 MB，可以直接运行研究流程。

生产环境使用的更大概念图超过 GitHub 的单文件大小限制。

如果你有已经记录的原始页面，可以使用：

```bash
python3 scripts/build_concepts.py --replay
```

重新构建。

如果没有，FarField 会自动使用仓库中自带的小图。

## 快速开始

先建立本地密钥目录：

```bash
mkdir -p ~/.config/farfield
chmod 700 ~/.config/farfield
```

把模型供应商密钥写入：

```text
~/.config/farfield/llm.key
```

并限制权限：

```bash
chmod 600 ~/.config/farfield/llm.key
```

配置一个 OpenAI-compatible endpoint：

```bash
export FARFIELD_LLM_KEY_FILE=$HOME/.config/farfield/llm.key
export FARFIELD_LLM_BASE_URL=https://api.deepseek.com/v1
export FARFIELD_LLM_MODEL=deepseek-v4-pro
```

运行一次研究任务：

```bash
PYTHONPATH=src python3 -m farfield research /tmp/farfield-research \
  --topic "compress genomic sequence collections with succinct data structures"
```

启动本地控制台：

```bash
PYTHONPATH=src python3 scripts/console.py
```

然后访问：

```text
http://localhost:8765/
```

默认只绑定：

```text
127.0.0.1
```

走查见上文 [演示](#演示)。

从其他机器访问时，优先使用 SSH 端口转发。

只有在你明确接受服务和密钥相关流量出现在局域网时，才设置 `FARFIELD_HOST`。

环境变量见：

```text
.env.example
```

查看完整命令：

```bash
PYTHONPATH=src python3 -m farfield --help
```

其他命令包括：

```text
freeze
execute
run
attest-run
campaign
```

## 冻结实验数据

FarField 可以把实验绑定到冻结的外部数据上。

例如：

```bash
PYTHONPATH=src python3 -m farfield freeze \
  --id example-world \
  --file /path/to/data.txt \
  --schema text_stream \
  --slice all \
  --title "Example frozen dataset" \
  --source "Original dataset provenance"
```

冻结后的实验对象会记录：

- 数据内容；
- schema；
- slice 规则；
- 数据来源；
- 对应 id。

之后的实验不能在看到结果以后偷偷换数据集。

## 执行已经登记的实验

编译后的研究方案中包含：

```text
protocol.json
```

可以直接执行：

```bash
PYTHONPATH=src python3 -m farfield execute \
  /path/to/candidate-folder
```

FarField 会执行已登记的 treatment / control 实验，并记录两臂测量值。

`execute` 默认重建冻结的母数据，因此这次运行会被记录为 `E2` 规模复现，使用它自己的 EvidenceID。传 `--on-slice` 才是重跑探针见过的同一份字节——只有这种 `E1` 同实例复现能确认探针的 EvidenceID。

对于更重的实验，可以使用：

```bash
PYTHONPATH=src python3 -m farfield run \
  /path/to/candidate-folder \
  --runner host-heavy
```

如果实验无法在本机完成，也可以交给外部 runner。

## 外部复现

有些实验需要：

- GPU；
- 特定软件环境；
- 集群；
- 外部机器；
- 其他无法放进本地轻量 runtime 的资源。

这种情况下，FarField 可以接收外部执行产生的 treatment / control 数值，但不会允许外部 runner 直接宣布科学结论。

例如准备：

```json
{
  "treatment": 123.0,
  "control": 157.0
}
```

然后记录：

```bash
PYTHONPATH=src python3 -m farfield attest-run \
  /path/to/candidate-folder \
  --metrics metrics.json \
  --environment "A100 GPU, CUDA 13, PyTorch 3.x" \
  --runner "external replication"
```

已经登记的实验定义仍然是权威来源。

外部 runner 提供：

> **measurement**

而不是：

> **scientific verdict**

## 输出

一次研究运行会产生可以检查的本地文件，而不只是聊天记录。

常见输出包括：

```text
RESEARCH_PACKET.md
protocol.json
experiment.py
metrics.json
evidence_snapshot/
```

### `RESEARCH_PACKET.md`

给人阅读的研究方案，通常包括：

- 研究问题；
- 存活假设；
- 相关文献；
- 实验设计；
- 当前证据；
- 开放 falsifier；
- 下一步。

### `protocol.json`

机器可读的实验登记协议。

其中保存独立复跑所需要的实验定义。

### `experiment.py`

可执行的 treatment / control 实验。

### `metrics.json`

实验得到的两臂测量值。

### `evidence_snapshot/`

当前运行使用的冻结文献或实验依据。

任务运行状态默认写到：

```text
var/
```

并由 git 忽略。

## 目录

```text
src/farfield/       核心运行时与研究循环
tests/              标准库 unittest 测试
worlds/             冻结实验数据
corpora/            引文图资源
concepts/           概念共现图
scripts/            控制台与重建脚本
examples/           小型运行时 / 账本示例
webui/              本地研究控制台
assets/             README 配图
.agents/skills/     可选研究与实验钩子
```

公开仓库包含：

- 可运行 runtime；
- 测试；
- 小型可复现资源；
- 示例 frozen fixtures。

仓库不包含：

- API key；
- 本机生成的 mission 目录；
- 大型私有或本地 corpus；
- 超过 GitHub 文件大小限制的概念图；
- 内部开发笔记。

## 设计原则

FarField 尽量遵守几条简单规则。

### 1. 模型提案不是结果

LLM 可以提出假设和代码。

它不能宣布自己的假设已经被科学建立。

### 2. 先登记，再观察

实验比较方式和 expected direction 必须在 treatment / control 数值出现之前固定。

### 3. 使用对称的 control

Treatment 和 control 应该只在被检验的机制上存在差异。

### 4. 负面证据比正面证据更容易成立

Synthetic / generated data 可以暴露一条坏假设。

但它不能独立确认这条假设。

### 5. 运行失败不是科学证据

Timeout、脚本错误或基础设施问题不应该被解释成科学 falsification。

### 6. 排名不是验证

“这个 idea 值不值得做”和“这个 claim 是否成立”是两个不同问题。

### 7. 探索和写回使用不同门槛

一个方向可以值得探索，但不一定值得进入：

- long-term memory；
- routing policy；
- skill；
- verifier；
- verified knowledge。

FarField 允许低成本、弱证据的探索存在，但对影响未来系统行为的写回更加保守。

## FarField 不是什么

FarField 不打算：

- 自动生成一篇 submission-ready 的会议论文；
- 把 LLM 的置信度或措辞当作实验依据；
- 因为 LLM reviewer 喜欢一个想法就接受它；
- 用为了当前假设临时生成的数据确认当前假设；
- 看过结果以后悄悄重写实验；
- 把每次成功轨迹都直接变成永久系统行为；
- 用模型自评替代外部复现；
- 把远距离 embedding 当成科学 novelty 的证明。

它的目标更窄：

> **把一个开放研究方向变成可证伪、受证据约束，并且能够由其他研究者检查和复跑的实验方案。**

## 当前状态

FarField 目前仍然是一个实验性的研究系统。

这个仓库的目标，是让研究循环、证据规则和失败方式都可以被检查，而不是把系统包装成一个已经完成的“自主科学家”。

当前仍然存在一些限制：

- 能直接执行的研究问题受已有 frozen worlds 覆盖范围限制；
- 很多研究仍然需要外部 GPU、模拟器、真实系统或物理实验；
- far-field distance 是搜索启发式，而不是 scientific novelty 的保证；
- routing 和 verifier evolution 需要足够多跨任务证据之后才真正有意义；
- 本机 actor label 目前主要是逻辑角色，还不是完整的安全隔离边界。

这些限制会明确暴露，而不是用模型生成的置信度掩盖。

## 引用

如果你在研究中使用 FarField，请通过 [`CITATION.cff`](CITATION.cff) 引用本仓库。

## 许可证

MIT，见 [LICENSE](LICENSE)。

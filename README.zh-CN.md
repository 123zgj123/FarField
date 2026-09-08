<div align="center">

# FarField

**研究由状态驱动，结论由证据约束。**

一个实验性的本地科研运行时：以 ScientificState 统一调度文献检索、假设生成、预注册实验、验证与研究整理。

[English](README.md) · [简体中文](README.zh-CN.md) · [系统设计](ARCHITECTURE.md) · [验证记录](VALIDATION.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-003a70)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-9a7b2e)](LICENSE)
[![CI](https://github.com/123zgj123/FarField/actions/workflows/tests.yml/badge.svg)](https://github.com/123zgj123/FarField/actions/workflows/tests.yml)
[![Core dependencies](https://img.shields.io/badge/core-Python%20stdlib-5c6b7a)](pyproject.toml)

[Guijia Zhang](https://123zgj123.github.io)

</div>

> **这是研究软件，不是已经验证的自主科学家。** 当前可检验的是运行机制和证据准入规则；自主科学发现、通用实验执行以及有效的在线研究策略自改进尚未得到证明。详见[验证与局限](VALIDATION.md)。

## FarField 能做什么

用户提出 research intent 后，系统显式维护问题、假设、竞争解释、证据、矛盾与研究债务。OpenWorld 根据当前状态选择可执行行动，由现有 research workers 完成工作。

目标是探索科学推理的变换方式，而不是把概念之间的距离当成科研价值。

- **让研究过程可检查。** 科学事件、证据身份、belief 更新与未解决问题均可追踪和回放。
- **保留多分支探索。** 历史轨迹变换、假设重构与自由探索共同进入推测前沿。
- **先实验，再晋升结论。** 预注册双臂实验、精确 world 绑定、复现和适用的 heldout 检验约束结论强度。
- **区分失败的含义。** 执行受阻、策略拒绝、科学负结果和不确定结果不会混为一谈。
- **单独验证策略改进。** 没有可信的 heldout 评估，研究策略候选不能安装。

它适合希望在明确文献来源、冻结实验数据与可执行实验条件下，构建和审计自动科研流程的研究者。它不是“任意输入一个题目就自动得到可信论文”的工具。

## 唯一的科研控制循环

```text
ScientificState → Frontier → Scheduler → Action → Existing Research Worker
       ↑                                                   ↓
       └── Scientific Event ← Trusted Kernel ← CandidateResult
```

行动顺序取决于状态，而不是固定 pipeline。CLI 和控制台都调用 `run_research`；worker 不启动另一套独立科研任务。

| 行动 | 复用的现有能力 |
|---|---|
| SURVEY | 文献检索、引文与来源核验、最接近的已有工作、原文片段 |
| THEORIZE | 假设生成、探索、重构、诊断 |
| PROBE | 诊断、预注册 probe 构建、实际执行 |
| VERIFY | 精确对象与产物检查、复现、显式 heldout world |
| SYNTHESIZE | 研究简报、研究包、事实表、论文规划与草稿 |

Trusted Kernel、事件溯源、provenance 与 WORLD/GENERATED 分离仍是信任边界。模型可以提出 claim 和脚本，不能自行宣布生成内容已经成为科学证据。

轨迹抽取、后续跳转条件、belief 记账、实验选择与策略准入的详细说明见[系统设计](ARCHITECTURE.md)。

## 快速开始

需要 **Python 3.11+**。核心运行时与离线测试只依赖标准库；具体实验程序可能需要独立执行环境和额外依赖。

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField

# 离线验证，不需要 API Key。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src FARFIELD_LLM_MODE=replay \
  python3.11 -m unittest discover -s tests -q
```

### 启动有预算边界的真实研究会话

将服务商密钥保存到仓库之外的私有本地文件。不要写入源码，也不要把密钥本身放进命令参数。[.env.example](.env.example) 列出了配置变量；以下示例显式导出环境变量。

```bash
mkdir -p "$HOME/.config/farfield"
chmod 700 "$HOME/.config/farfield"
# 请先自行将服务商密钥私密保存到 ~/.config/farfield/llm.key。
chmod 600 "$HOME/.config/farfield/llm.key"

export FARFIELD_LLM_KEY_FILE="$HOME/.config/farfield/llm.key"
export FARFIELD_LLM_BASE_URL=https://api.openai.com/v1
export FARFIELD_LLM_MODEL=gpt-5.6-sol

PYTHONPATH=src python3.11 -m farfield research /tmp/farfield-research \
  --topic "Does sepal geometry predict iris species?" \
  --world fisher-iris --horizon 20 --api-calls 30
```

该命令允许**真实且可能计费的模型调用**和联网文献检索。请使用你的服务商账户可访问的模型。缺少凭证、限流、文献不足或不支持的实验绑定均可能阻止继续执行，系统不会补造证据。

位置参数是**项目根目录**，不是已有 mission 的目录。每次 CLI 调用都会在 `/tmp/farfield-research/var/missions/` 下创建新的时间戳 workspace。Iris fixture 使示例有明确数据对象，但不保证产生新颖假设或成功实验。`--horizon` 限制调度步数，`--api-calls` 限制模型调用次数，不是金额预算。

需要独立验证时，可重复指定 `--heldout-world <world-id>`，前提是目录中存在科学上适用的 fixture。对同一份数据重跑只算复现，不算独立重复验证。

### 恢复已有研究状态

当前 CLI 会新建 mission。继续已有研究应通过 Python API，指定**原 workspace、相同 topic 与兼容的 world**，并沿用上述环境配置：

```python
from pathlib import Path
from farfield.extras.openworld.runtime import run_research

for event in run_research(
    "Does sepal geometry predict iris species?",
    workspace=Path("/path/to/existing/var/missions/<mission-id>"),
    world="fisher-iris",
    horizon=20,
    api_calls=30,
):
    print(event)
```

运行 Python 时设置 `PYTHONPATH=src`。状态从科学事件中恢复，未解决行动进入新的有界会话。这是可恢复运行，不是无人值守的常驻服务。

### 本地控制台

```bash
PYTHONPATH=src python3.11 scripts/console.py
```

访问 `http://localhost:8765/`。控制台使用同一科研入口，默认只监听 loopback；远程使用建议通过 SSH 转发。

## 什么可以算作证据？

**大胆生成，谨慎晋升。**

假设与论文解读保持推测性质。生成的实验计划、想象场景、reviewer 意见与模型自洽性都不能成为 WORLD 证据。

可晋升结果需要对应的文献与证伪检查、针对精确绑定对象的可执行 probe、已准入测量、验证，以及结论范围要求的复现或迁移检查。描述性关联不等于因果机制。

Probe 明确记录五类结果：

| 状态 | 含义 |
|---|---|
| `probe_execution_blocked` | 未取得可用实验结果 |
| `probe_policy_refused` | 执行或准入不被允许 |
| `probe_scientifically_negative` | 实际测量提供反对假设的证据 |
| `probe_positive` | 在该 probe 的范围内测得支持 |
| `probe_inconclusive` | 尚不足以区分解释 |

Positive probe 不自动等于 verified finding。无证据的结论升级、缺失 world digest、协议字节变化与 EvidenceID 不一致均应拒绝通过。详见[证据与晋升](ARCHITECTURE.md#evidence-and-promotion)。

## 研究产物

每个 mission workspace 内：

```text
SCIENTIFIC_STATE.json   问题、理论、证据、belief 与 frontier
SCIENTIFIC_EVENTS.json 可回放的科学事件日志
RESEARCH_STATUS.md     已验证范围与剩余缺口
literature/            文献来源、论文状态与轨迹
llm_cache/             模型请求与响应记录
worlds/<world-id>/     已绑定的 fixture 数据与 manifest
research_workers/<evidence-id>/candidates/<card-id>/
  protocol.json        注册协议与身份信息
  experiment.py        可执行源码
  probe.json           测量、结论、来源与范围
  metrics.json         执行指标
  research_note.md     调度到整理行动后产生的研究说明
  PAPER_FACTS.json     调度到整理行动后产生的事实表
  paper/               可选论文草稿
```

只有对应行动实际产出后才有相应文件。受阻任务可能没有测量值。`research_complete: false` 表示研究仍开放，不表示已经成功完成论文。运行目录与凭证不上传公共仓库。

复现已有注册协议：

```bash
PYTHONPATH=src python3.11 -m farfield execute /path/to/candidate-folder --on-slice
```

这只复现原注册实例，不证明结果能够迁移到新 world。

## 当前验证与局限

[验证记录](VALIDATION.md) 将需求映射到测试，并区分离线回归与真实服务观察。

- **离线验证：** 主测试集 1,202 项，其中跳过 6 项；另有 30 项合同测试、2 项历史回归通过。
- **真实集成：** 有界 Iris 任务检索到真实论文、生成结构化假设，并进入诊断和 probe 构建。最后记录的会话在执行前拒绝了指标不一致的方案，随后生成失败重构分支；没有产生 WORLD probe evidence 或已验证发现。
- **轨迹覆盖：** 已有抽取和结构检索，但尚无广泛、可靠的轨迹库，也没有学得的 operator 选择分布。
- **实验覆盖：** world 目录与数据获取能力有限；注册身份并不能完整证明任意生成代码实现了文字所述的科学机制。
- **估计性质：** belief 是确定性记账，不是校准后的 Bayesian 置信度；调度优先级是 heuristic，不是实测 information gain。
- **自改进：** 已实现本地策略准入约束，尚未证明在线 research-policy RSI 能改善科研能力。

历史 far-jump 实验未证明优于匹配的随机搜索。Embedding 距离既不是新颖性证明，也不是科学价值信号。

## 目录与兼容性

| 路径 | 用途 |
|---|---|
| [src/farfield/extras/openworld/](src/farfield/extras/openworld/) | 科研控制器、状态、调度器与 worker adapter |
| [src/farfield/](src/farfield/) | 可信运行时与现有研究能力 |
| [tests/](tests/) | 离线、合同、parity 与回归测试 |
| [worlds/](worlds/) | 小型可核验实验 fixture |
| [scripts/](scripts/) / [webui/](webui/) | 本地控制台与维护工具 |
| [corpora/](corpora/) / [concepts/](concepts/) | 文献与历史图资产 |

历史 Python pipeline 函数保留为回归参考。`--legacy-pipeline` 已废弃且转入同一控制器，不恢复旧的阶段式流程。旧版双语 `RESEARCH_PACKET*.md` 是历史产物格式，不是当前命令保证输出的文件。

CLI help 中仍保留部分未转发给当前控制器的旧参数。尤其不要依赖 `--no-host-execute` 或旧 human-gate 参数来禁止执行。无真实服务调用的检查请使用离线测试；真实研究可能调度预注册实验执行。

另见 [CHANGELOG.md](CHANGELOG.md)、[CONTRIBUTING.md](CONTRIBUTING.md) 与 `PYTHONPATH=src python3.11 -m farfield --help`。

## 引用

研究中使用 FarField 时，请引用 [CITATION.cff](CITATION.cff)。

## 许可证

[MIT](LICENSE)。

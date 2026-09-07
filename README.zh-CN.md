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

## 近况

- **2026.09** — 喷雾停在 `plan_executable`（已编译、实习生能跑的 brief+plan），不停在「发现」。`SYNTHETIC` / `GENERATED` 支持可以关喷雾，仍然不能佐证、蒸馏或晋升。诊断之后默认走 `--world-sim`；结构洞会在本场对**同一对**做一次 compile refine，再重新登记诊断。`--host-execute` 默认开，记下的是 `protocol_executed`，不是发现。远跳仍是搜索启发：入档负结果还在（post-T 实现率不高于匹配随机）。
- **2026.08** — 证据合同：只有 attested 的 `WORLD` 探针可以佐证。`--world auto` 必须 domain 正命中才绑目录 freeze；弱 schema 匹配不能顶替。工作记忆按概念对隔离。
- **演示视频** — 控制台走查稍后单独上传。下方海报为占位。

<p align="center">
  <a href="demo.html">
    <img src="assets/farfield-demo-poster.png" alt="FarField 控制台（演示录像待上传）" width="920">
  </a>
</p>

## 从这里开始

| 你想… | 去 |
|---|---|
| 理解科学合同 | [为什么做 FarField](#为什么做-farfield) 与 [证据合同](#证据合同) |
| 安装并跑一场任务 | [安装](#安装) 与 [快速开始](#快速开始) |
| 打开本机控制台 | [控制台](#控制台) |
| 复跑已编译协议 | [执行协议](#执行协议) |
| 引用本软件 | [引用](#引用) |

## 摘要

自动研究系统已经能写出像样的想法、实验，甚至一篇像论文的草稿。真正开放的问题不是生成，而是：**什么时候，一个 AI 给出的结果值得相信。**

FarField 是一套可检查的本机循环。给它一个题目，它从已验证文献恢复发展轨迹并做远场跳跃，提出可证伪的假设，在**看到任何数字之前登记两臂实验**，并输出别人可以复跑的方案。模型可以写主张、机制和脚本。图上的查重、已有工作检查、世界绑定，以及 treatment / control 的算术，都由普通 Python 完成——这些门上没有模型。

一次确认只允许在这条谓词成立时发生：**绑定、构造、杠杆和探针度量必须是同一个科学对象。** schema 相同不是科学对象相同。合成或构造数据可以削弱假设，不能确认假设。

终点不是会议 PDF，而是一份研究包。

## 为什么做 FarField

自动研究里很容易出现三类失败：

1. 模型把自己的解释当成支持自己假设的证据；
2. 先看到数字，再改实验或重新定义成功标准；
3. 用「为这条假设临时生成的数据」来确认这条假设。

FarField 围绕一条更严格的原则：

> **大胆探索，谨慎相信。**

搜索阶段可以激进，证据阶段不可以。排名、Elo、评审分数是同事意见，不能接受、拒绝或提升科学结论。

<p align="center">
  <img src="assets/overview.png" alt="FarField 循环：题目、检索、假设、筛选、已登记两臂实验、研究包" width="920">
</p>

## 研究循环

| 阶段 | 做什么 |
|---|---|
| **检索** | 题目变成研究位置；已验证论文恢复发展轨迹；远跳把观察到的演变套到当前位置。概念图只做 scout / baseline / 旧图 continue 的 oracle，不发明 `pair[1]`。距离是搜索启发，不是新颖性证明。 |
| **假设** | 模型写主张、机制、预测和已放弃的路径。这些是提案，不是证据。 |
| **筛选** | 确定性图检查与文本检查拒绝已知组合和结构不合格的主张。 |
| **已有工作** | 检索到的文献可以在烧计算之前关闭主张（`closed_by_prior`）。 |
| **打磨** | 同一对、同一 freeze。文本门杀和 `no_handle` 走 `idea_rounds`（产品默认最多两轮）。world-sim 点名的结构洞再花一次 compile refine，然后重新登记诊断。不是新的远跳。 |
| **登记** | 竞争解释、两臂、指标、预期方向和计算档位在执行前固定。 |
| **实验** | 两臂调用同一个 `measure`，只差一个机制开关。产品 `experiment_rounds=-1` 允许在首次登记后再加试两次。 |
| **证据** | `supports` / `weakens` / `uninformative` 是科学结果。超时和缺文件不是。20 秒 WORLD `supports` 只到 speculative，要等宿主按同一 EvidenceID 重跑。 |
| **方案** | 完整研究方案在 `RESEARCH_PACKET.md`（中文）和 `RESEARCH_PACKET_EN.md`（英文），主张、文献、实验设计和下一步都写在文件里。每条可继续的 idea 另有独立文件夹 `ideas/<idea-name>/`（中英 `RESEARCH_BRIEF` + `EXPERIMENT_PLAN`）以及 `AGENT_PACKET.md`（开工指令）和 `protocol.json`。已被削弱或无法继续的卡不占 `ideas/`。 |

远探喷雾在**计划可执行**时停止：WORLD 支持、已编译 brief+plan 的 uninformative / SYNTHETIC / GENERATED 支持、邻域里已有该计划，或跳数上限。剩下的工作是 `farfield execute`，不是再开一场 `farfield research`。一条可执行的 SYNTHETIC 计划只说明自洽检查做完了——去 freeze，不是发现。

## 证据合同

```text
synthetic / generated result
        |
        +-- weakens    →  可以成为负面科学证据
        |
        +-- supports   →  一致性检查，不是确认
```

| 种类 | 含义 |
|---|---|
| `WORLD` | 已冻结夹具拷进 `data/` 且脚本真正读过。匹配宿主 EvidenceID 且机制被隔离后可以爬梯。 |
| `GENERATED` | 为本次登记构造。可以削弱；不能佐证、蒸馏或晋升。可执行计划**会**停喷雾。 |
| `SYNTHETIC` | 编造的一致性检查。同样：可以停喷雾，不能变成发现。 |
| `WORLD_SIM` | idea 演练（`farfield world-sim`，默认开）。想象的 best / median / worst。能改计划结构，不能改真，不能挡住 `farfield execute`。 |

`--world auto` 按**本任务课题**和本场已检索论文构造一个 GENERATED 世界。它**不会**自动绑 Pride、phage、A2A 或任何目录亲戚。schema 提示不能顶替 freeze。产品路径上只有显式 `--world <id>` 才绑目录夹具。未命中对佐证而言是 `WORLD_INCOMPATIBLE`：绝不静默顶替。构造出的探针可以削弱，不能佐证。前向模拟把已登记杠杆逐步执行到吸收、平台或时域上限。那是主张分析，不能爬梯。

「world」三层不要混：夹具（显式 `--world <id>` 的 `worlds/`，以及使命根 `workspace/world/` 的构造实例）、动力学（`dynworld` 杠杆）、idea 演练（`ideas/<name>/world-sim/`，`farfield world-sim`）。演练读 brief + plan + 已检索论文，不要求目录 freeze，也不要求已经焊上杠杆。演练数字不能佐证、不能改 `expected_direction`、不能挡住 `farfield execute`。

工作记忆只升级**同一概念对**。蒸馏技能写在 `candidates/<card_id>/skills/`。仓库里受信任的 `plugin.py` 才是共享能力。

确认逢缺即拒。缺 digest、缺 EvidenceID、协议不完整都是拒绝。`farfield execute` / 默认 `--host-execute` 记为 `protocol_executed`，不是发现。多数话题仍落在 `SYNTHETIC` / `GENERATED`，因为夹具目录薄——未命中是 `WORLD_INCOMPATIBLE`（进 wishlist），绝不静默顶替。

## 安装

需要 Python 3.11+。核心运行时和测试只用标准库。

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField
```

不需要 `pip install`。把 `src/` 加到 `PYTHONPATH`。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -q
```

测试不需要 API key。仓库自带 `concepts/attn-concepts-s1/`（约 8 MB），足够跑通循环。更大的概念图超过 GitHub 单文件限制；有已记录的原始页面时可用 `python3 scripts/build_concepts.py --replay` 重建。

## 快速开始

当前 CLI 有两条路径：默认 OpenWorld 运行本地控制器和探索性观测，输出 `SCIENTIFIC_STATE.json`、`SCIENTIFIC_EVENTS.json` 和 `RESEARCH_STATUS.md`。它尚未接通通用 LLM 检索、实验编译和研究包输出；结束事件中的 `research_complete: false` 表示研究未完成。模型、API 预算、评审和 `--state-store` 参数目前用于 `--legacy-pipeline`。

无需 API 的真实数据检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3.11 -m farfield research /tmp/farfield-iris \
  --topic "萼片几何能否预测 Iris 类别？" --world fisher-iris --horizon 4
```

显式 world 会校验 manifest 和预先登记的摘要，并复制原始数据。不存在的 world 直接报错。默认 `auto/none` 不会制造对照观测；缺少采集配方时记录能力缺口。内置表格、图、序列等摘要是探索性结果，原始科学问题仍保持开放。构造数据及其派生版本不能晋升为 WORLD。

要运行下文介绍的文献检索、候选生成、评审、预登记实验与研究包流程，请使用 `--legacy-pipeline` 并配置模型：

```bash
mkdir -p ~/.config/farfield
chmod 700 ~/.config/farfield
# 写入供应商密钥后：
chmod 600 ~/.config/farfield/llm.key

export FARFIELD_LLM_KEY_FILE=$HOME/.config/farfield/llm.key
export FARFIELD_LLM_BASE_URL=https://api.deepseek.com/v1
export FARFIELD_LLM_MODEL=deepseek-v4-pro

PYTHONPATH=src python3.11 -m farfield research /tmp/farfield-research --legacy-pipeline \
  --topic "compress genomic sequence collections with succinct data structures"
```

沿着同一条科学线继续时，复用同一个 `--state-store` 和 `--world`。新的任务文件夹是账本，不是新身份。不要把 API key 提交进 git 或放进命令行。环境变量见 `.env.example`。

### 控制台

```bash
PYTHONPATH=src python3 scripts/console.py
```

然后打开 `http://localhost:8765/`。默认绑定 `127.0.0.1`。远程访问优先用 SSH 端口转发。

### 执行协议

```bash
PYTHONPATH=src python3 -m farfield execute /path/to/candidate-folder --on-slice
```

`--on-slice` 重跑探针见过的字节（E1）。默认 `execute` 在冻结缓存存在时重建母数据（E2）。其他命令：`freeze`、`run`、`attest-run`、`campaign`、`world-sim`。见 `python3 -m farfield --help`。

## 输出

以下为 `--legacy-pipeline` 的输出。默认 OpenWorld 的输出见“快速开始”。

```text
RESEARCH_PACKET.md      完整研究方案（中文；含实验设计）
RESEARCH_PACKET_EN.md   同一方案的英文副本
ideas/<idea-name>/      每条可继续的 idea 一个文件夹（用主张核心命名，不是 gen_ 编号）
  README.md            本文件夹目录
  idea-stage/RESEARCH_BRIEF.md     完整研究方案（中文，含实验设计）
  idea-stage/RESEARCH_BRIEF_EN.md  同一方案英文副本
  refine-logs/EXPERIMENT_PLAN.md   完整实验设计（中文）
  refine-logs/EXPERIMENT_PLAN_EN.md 同一设计英文副本
  AGENT_PACKET.md      开工指令（只谈这一条）
  world-sim/           idea 演练（想象数字；不能爬梯）
protocol.json          已登记实验
experiment.py          两臂脚本
metrics.json           treatment / control
evidence_snapshot/     本场冻结检索
```

任务状态写在 `var/`，已被 git 忽略。

## 目录

```text
src/farfield/       核心运行时
tests/              标准库 unittest
worlds/             已冻结夹具
corpora/            引文图资源
concepts/           概念共现图
scripts/            控制台与重建
examples/           小型账本示例
webui/              本机控制台
assets/             配图
.agents/skills/     受信任的研究钩子
```

公开仓库包含可运行 runtime、测试、小型可复现资源和夹具。不含密钥、本机 mission 目录和内部设计笔记。

## 当前状态

FarField 仍是实验性系统。这个仓库让研究循环和失败方式可被检查，而不是把它包装成已经完成的「自主科学家」。它也还没有证明远场搜索优于匹配随机。

- 入档负结果还在：远跳 post-T 实现率不高于匹配随机；后来的进化代没有翻案。距离是搜索启发，不是新颖性。
- 便宜的 WORLD `supports` 是过滤器（`speculative`）。`corroborated` 要宿主按同一 EvidenceID 重跑，且机制被隔离。
- 多数话题仍落在 `SYNTHETIC` / `GENERATED`。一条 SYNTHETIC 支持按合同会关喷雾；科学上只是「自洽检查做完了，去 freeze」。
- `world-sim` 能改计划结构，不能改真。排演失败不能挡 execute。
- refine 预算很短：文本最多两轮，实验最多两次加试；排演点名结构洞后再打磨一次。第二次 uninformative 换杠杆是纪律，不是搜索已经够深。
- 夹具目录仍然薄。未匹配对象进 wishlist，不会被绑到无关 freeze。

## 引用

若在研究中使用 FarField，请通过 [`CITATION.cff`](CITATION.cff) 引用。

## 许可证

MIT，见 [LICENSE](LICENSE)。

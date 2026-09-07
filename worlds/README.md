# Frozen experimental worlds

English · 简体中文

`--world auto` constructs a GENERATED world from the task topic and retrieved papers. It does **not** bind a shipped fixture from this directory. The one exception is an **acquired** world — a manifest with `provenance: derived | harvested | wishlist` — of the task's own object family that also passes the domain-tag bind (`pick_acquired_world`); the event stream says `world_acquired_bound`. Explicit `--world <id>` is the catalog bind. A probe counts as `WORLD` only if it reads `data/` on an attested freeze. `none` is `SYNTHETIC` when construction is impossible and cannot corroborate.

Three host-side ways an attested world enters, besides fetching a published file:

- **derive** — `farfield freeze --id <child> --derive-from <parent> --schema <s>` re-encodes a catalog freeze by a registered pure rule (`extras/derive.py`, e.g. Live-SWE-agent `created_tools` → a `program_state` self-modification history). Origin keeps the parent digest and rule; `provenance: derived`. A `program_state` wish without a URL in `var/world_wishlist.json` is resolved this way at mission start.
- **import** — `farfield freeze --id <w> --dir <run> --format dgm|openevolve` freezes a published self-improvement run directory (Darwin Gödel Machine `output_dgm/<run>/`, OpenEvolve `checkpoint_<n>/`) as `program_state`; the source digest is over the files read (`extras/histories.py`).
- **harvest** — `farfield harvest recipe.json` registers the recipe on `chain.jsonl` (`REGISTER_HARVEST`), runs the harness on the host (argv, no shell, no probe), imports the exported history, and freezes it with `provenance: harvested` and `origin.harness_run`. One harvest is one seed; the confirmation is a second harvest of the same recipe body with another seed (`confirming_seeds`), never a rerun of the probe on the same bytes.

Three layers, do not collapse them:

- **Fixture** — hashed bytes here. Probe WORLD requires reading `data/`.
- **Dynamics** — runtime levers (`dynworld`) on evolving state.
- **World rehearsal** — `ideas/<name>/world-sim/` via `farfield world-sim`. Imagined best/median/worst cannot corroborate. Literature phrases are keyed to dynworld levers and the freeze schema.

`path-trace` has `role=test` and is never chosen by auto. Canonical bytes live in the gitignored `.cache/`. Freeze new worlds with `farfield freeze`; do not edit a fixture after a claim exists. Unmatched `WORLD_INCOMPATIBLE` requirements and constructed GENERATED worlds live in `var/world_wishlist.json` (runtime, not this catalog). The constructed bytes are not a freeze.

Binding rule: the fixture must be the claim's scientific object. Schema match is necessary, not sufficient. Foreign objects are refused. Far-field mechanism text does not vote.

`--world auto` 按课题和本场论文构造 GENERATED 世界，不从本目录自动选**随附**夹具。唯一例外是**获得型**世界——manifest 带 `provenance: derived | harvested | wishlist`、属于课题对象自身家族、且通过 domain 标签绑定检查（`pick_acquired_world`）——事件流写 `world_acquired_bound`。显式 `--world <id>` 才绑目录。探针必须读 attested freeze 的 `data/` 才算 WORLD。schema 相同不是科学对象相同。未命中的 `WORLD_INCOMPATIBLE` 需求，以及达到科学判决的 GENERATED 探针，记在 `var/world_wishlist.json`，不进本目录。构造字节不是 freeze。idea 演练写在 `ideas/<name>/world-sim/`，不是本目录。

除抓取公开文件外，attested 世界还有三条宿主侧入口：**派生**（`farfield freeze --derive-from <parent> --schema <s>`，按注册的纯规则把目录内 freeze 重新编码，如 Live-SWE `created_tools` → `program_state` 自我修改史，`provenance: derived`；wishlist 里无 URL 的 `program_state` 需求在任务启动时走此路）；**导入**（`farfield freeze --dir <run> --format dgm|openevolve`，把公开的自我改进运行目录冻结为 `program_state`）；**采收**（`farfield harvest recipe.json`，先在 `chain.jsonl` 注册配方，再在宿主运行 harness、导入、冻结，`provenance: harvested`；一次采收一个种子，确认靶是同配方另一种子，不是同一字节上重跑探针）。

| id | schema | use / 用途 |
|---|---|---|
| `zachary-karate` | undirected_graph | social graph / 社交图 |
| `florentine-families` | undirected_graph | Florentine business ties / 佛罗伦萨家族商务关系 |
| `les-miserables` | undirected_graph | co-occurrence graph / 共现图 |
| `lusseau-dolphins` | undirected_graph | dolphin social / 海豚社交 |
| `snap-ca-grqc` | undirected_graph | SNAP ca-GrQc slice |
| `fisher-iris` | numeric_table | Iris table / 鸢尾花表 |
| `uci-letters` | numeric_table | UCI letters slice |
| `phage-lambda` | fasta | λ phage genome slice / λ 噬菌体基因组切片 |
| `phix174` | fasta | phiX174 |
| `gutenberg-alice` | text_stream | Alice excerpt / Alice 节选 |
| `gutenberg-pride` | text_stream | Pride token stream / Pride 词流切片 |
| `tcp-linux-server` | symbolic_trace | Linux TCP learning automaton |
| `a2a-task-lifecycle` | symbolic_trace | A2A TaskState (incl. AUTH_REQUIRED) |
| `live-swe-agent-verified-v1` | labeled_traces | Optional, not shipped: Live-SWE-agent SWE-bench Verified trajectories / 可选外部资源，当前仓库不附带 |
| `live-swe-agent-selfmod-v1` | program_state | derived from the above: 1731 runtime tool-creation updates over 499 episodes (`provenance: derived`) / 由上者派生的运行时自我修改史 |
| `daojo-matrix-pd-v1` | labeled_traces | Daojo Lab PD / Stag Hunt seat traces / 矩阵博弈对局痕迹 |
| `path-trace` | path | test fixture; auto never selects / 测试夹具 |

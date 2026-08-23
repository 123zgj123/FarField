# Frozen experimental worlds

English · 简体中文

`--world auto` binds a fixture from this directory **before** any claim is tested. A probe counts as `WORLD` only if it reads `data/`. `none` is `SYNTHETIC` and cannot corroborate.

`path-trace` has `role=test` and is never chosen by auto. Canonical bytes live in the gitignored `.cache/`. Freeze new worlds with `farfield freeze`; do not edit a fixture after a claim exists. Unmatched requirements live in `var/world_wishlist.json` (runtime, not this catalog).

Binding rule: the fixture must be the claim's scientific object. Schema match is necessary, not sufficient. Foreign objects are refused. Far-field mechanism text does not vote.

`--world auto` 按主张对象从本目录选夹具。探针必须读 `data/` 才算 WORLD。schema 相同不是科学对象相同。未命中的需求在 `var/world_wishlist.json`，不进本目录。

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
| `path-trace` | path | test fixture; auto never selects / 测试夹具 |

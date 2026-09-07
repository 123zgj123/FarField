"""The research note is compiled from attested fields, not a new generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.brief import ResearchBrief
from farfield.extras.diagnose import Diagnosis
from farfield.extras.generate import GeneratedCard
from farfield.extras.livefeed import FreshWork
from farfield.extras.packet import (
    render_experiment_plan,
    render_idea_agent_packet,
    render_idea_human_packet,
    render_idea_report,
    render_mission_packet,
    render_protocol,
)
from farfield.extras.writeup import compile_tex, render_note
from farfield.extras.workspace import mission_dir, write_idea, write_summary

CARD = GeneratedCard(
    card_id="gen_note01",
    operator="directional",
    claim="a succinct structure stores simplex pivot history in linear space",
    mechanism="the pivot sequence is compressible because it repeats",
    prediction="later work reports a structure with query time below 100 ns",
    falsifier="pair_not_already_combined",
    pair=("succinct data structure", "pivot rule"),
    pair_nodes=("concept:a", "concept:b"),
    alienness=0.4,
    model="fake",
    artifact_digest="d",
    artifact_uri="file:///dev/null",
    replay_mode="replay",
)

BRIEF = ResearchBrief(
    card_id="gen_note01",
    title="Succinct pivot logs",
    gap="2608.01234v1 studies nearby constructions but leaves the bound open",
    idea="Store simplex pivot history succinctly and time rank queries",
    approach="Wavelet-tree encode the log",
    first_steps=("download a public trace", "encode 10k pivots"),
    baseline="an uncompressed circular buffer",
    risks="real traces may not compress",
    read_first=("2608.01234v1",),
    papers=(),
)

BRIEF_PLAIN = ResearchBrief(
    **{
        **BRIEF.__dict__,
        "plain_title": "单纯形法的转轴历史能压缩吗",
        "one_liner": "转轴记录能否用线性空间存下并快速查询？",
        "why_it_matters": "如果可行，防循环检查不必保留完整日志。",
    }
)

PAPER = FreshWork(
    title="Succinct wavelet trees meet pivot rules",
    published="2026-08-12",
    arxiv_id="2608.01234v1",
    abstract="We compress pivot histories with a wavelet tree.",
)

DIAGNOSIS = Diagnosis(
    card_id="gen_note01",
    alternative="the speedup is just caching",
    experiment="time rank queries on a public pivot trace against an uncompressed buffer",
    treatment_arm="wavelet-tree log",
    control_arm="circular buffer",
    expected_direction="treatment_lower",
    alternative_direction="no_difference",
    expected_if_alternative="both arms land within noise",
    margin=0.15,
    margin_reason="15 percent wall-clock on the same trace",
)

PROBE = {
    "status": "ran",
    "measure": "ns per query",
    "treatment": 80.0,
    "control": 120.0,
    "verdict": "supports",
    "expected_direction": "treatment_lower",
    "alternative": "the speedup is just caching",
}


class NoteTests(unittest.TestCase):
    def test_the_note_cites_only_retrieved_papers(self) -> None:
        note = render_note("compress genomic sequences", CARD, BRIEF, [PAPER])
        self.assertIn("2608.01234v1", note.markdown)
        self.assertIn("2608.01234v1", note.latex)
        self.assertIn(r"\documentclass", note.latex)
        self.assertNotIn("invented-paper", note.latex)

    def test_a_two_arm_probe_is_written_as_arithmetic_evidence(self) -> None:
        note = render_note(
            "topic",
            CARD,
            BRIEF,
            [PAPER],
            probe={
                "status": "ran",
                "measure": "ns per query",
                "treatment": 80.0,
                "control": 120.0,
                "verdict": "supports",
                "expected_direction": "treatment_lower",
                "alternative": "the speedup is just caching",
            },
        )
        self.assertIn("supports", note.markdown)
        self.assertIn("treatment = 80.0", note.markdown)
        self.assertIn("caching", note.markdown)
        self.assertNotIn("query_ns", note.markdown)

    def test_special_characters_do_not_break_tex(self) -> None:
        brief = ResearchBrief(
            **{
                **BRIEF.__dict__,
                "title": "A_B & C",
                "idea": "cost is $O(n)$ ~ 10%",
            }
        )
        note = render_note("topic", CARD, brief, [PAPER])
        self.assertIn(r"A\_B \& C", note.latex)
        self.assertIn(r"\$", note.latex)

    def test_unicode_math_maps_and_unknown_glyphs_degrade_to_placeholders(self) -> None:
        # A live note failed pdflatex on U+2308: known math symbols must map
        # to their LaTeX form, and anything unmapped must degrade to a
        # placeholder instead of killing the whole PDF.
        brief = ResearchBrief(
            **{
                **BRIEF.__dict__,
                "idea": "α ≤ ⌈log n⌉ and the glyph ⨊ is unmapped",
            }
        )
        note = render_note("topic", CARD, brief, [PAPER])
        self.assertIn(r"\(\alpha\)", note.latex)
        self.assertIn(r"\(\lceil\)", note.latex)
        self.assertIn(r"\(\rceil\)", note.latex)
        self.assertNotIn("⨊", note.latex)
        for ch in note.latex:
            self.assertLessEqual(ord(ch), 126, f"unescaped glyph {ch!r}")

    def test_missing_pdflatex_leaves_the_tex_behind(self) -> None:
        note = render_note("topic", CARD, BRIEF, [PAPER])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            pdf = compile_tex(note.latex, dest)
            self.assertTrue((dest / "note.tex").is_file())
            if pdf is not None:
                self.assertTrue(Path(pdf).is_file())


class WorkspaceTests(unittest.TestCase):
    def test_a_mission_folder_keeps_the_note_and_the_brief(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = mission_dir(Path(tmp), "Succinct pivot logs!", when=None)
            write_idea(
                dest,
                card_id="gen_note01",
                brief=BRIEF.to_dict(),
                works=[PAPER.to_dict()],
                note_md="# hi",
                note_tex=r"\documentclass{article}\begin{document}x\end{document}",
                protocol_md="# protocol",
                protocol={"kind": "research_protocol", "not_a_paper": True},
                readme_md="# readme",
            )
            write_summary(
                dest,
                "Succinct pivot logs!",
                {"survivors": 1},
                packet_md="# 研究包",
            )
            self.assertTrue((dest / "candidates" / "gen_note01" / "brief.json").is_file())
            self.assertTrue((dest / "candidates" / "gen_note01" / "note.tex").is_file())
            self.assertTrue((dest / "candidates" / "gen_note01" / "protocol.md").is_file())
            self.assertTrue((dest / "candidates" / "gen_note01" / "protocol.json").is_file())
            self.assertTrue((dest / "candidates" / "gen_note01" / "README.md").is_file())
            self.assertTrue((dest / "RESEARCH_PACKET.md").is_file())
            self.assertIn("Succinct", (dest / "topic.txt").read_text())


class ProtocolTests(unittest.TestCase):
    def test_the_protocol_cites_only_retrieved_papers(self) -> None:
        brief = ResearchBrief(
            **{**BRIEF.__dict__, "read_first": ("2608.01234v1", "invented-paper")}
        )
        protocol = render_protocol("compress genomic sequences", CARD, brief, [PAPER])
        self.assertIn("2608.01234v1", protocol.markdown)
        self.assertNotIn("invented-paper", protocol.markdown)
        self.assertEqual(protocol.payload["read_first"], ["2608.01234v1"])
        self.assertIn("not a conference paper", protocol.markdown.lower())
        self.assertTrue(protocol.payload["not_a_paper"])
        self.assertNotIn("accepted at", protocol.markdown.lower())

    def test_the_cheap_probe_is_labelled_synthetic(self) -> None:
        protocol = render_protocol(
            "topic",
            CARD,
            BRIEF,
            [PAPER],
            probe=PROBE,
            diagnosis=DIAGNOSIS,
        )
        self.assertIn("SYNTHETIC", protocol.markdown)
        self.assertIn("constructed dataset", protocol.markdown)
        self.assertEqual(protocol.payload["cheap_probe"]["kind"], "SYNTHETIC")
        self.assertIn("caching", protocol.markdown)
        self.assertIn("wavelet-tree log", protocol.markdown)
        self.assertIn("circular buffer", protocol.markdown)
        self.assertIn("treatment_lower", protocol.markdown)
        self.assertIn("public pivot trace", protocol.markdown)
        self.assertIn("uncompressed circular buffer", protocol.markdown)
        self.assertNotIn("query_ns", protocol.markdown)
        self.assertIn("Do not cite", protocol.readme)
        self.assertIn("Research plan", protocol.markdown)
        self.assertIn("farfield execute", protocol.markdown)
        self.assertIn("protocol_executed", protocol.markdown.lower())
        self.assertTrue(protocol.payload["runnable_baselines"])
        self.assertEqual(protocol.payload["runnable_baselines"][0]["cite_id"], "2608.01234v1")
        self.assertIn("implement as the control arm", protocol.payload["runnable_baselines"][0]["role"])
        self.assertIn("Paper outline", protocol.markdown)
        self.assertEqual(protocol.payload["experimental_design"]["expected_direction"], "treatment_lower")
        self.assertIn("mechanism flag", protocol.payload["experimental_design"]["independent_variable"])
        self.assertIn("20s sandbox", protocol.markdown)
        self.assertIn("Experimental design", protocol.markdown)

    def test_failed_experiment_branches_are_process_data(self) -> None:
        protocol = render_protocol(
            "topic",
            CARD,
            BRIEF,
            [PAPER],
            probe={
                **PROBE,
                "attempts": [
                    {
                        "attempt": 0,
                        "bottleneck": "method",
                        "verdict": "uninformative",
                        "experiment": "count comparisons on ten instances",
                    },
                    {
                        "attempt": 1,
                        "bottleneck": None,
                        "verdict": "supports",
                        "experiment": "count comparisons on a more extreme construction",
                    },
                ],
                "discarded": ["count comparisons on ten instances"],
            },
            diagnosis=DIAGNOSIS,
        )
        self.assertIn("Failed experiment branches", protocol.markdown)
        self.assertIn("ten instances", protocol.markdown)
        self.assertEqual(len(protocol.payload["experiment_attempts"]), 2)
        self.assertIn("count comparisons on ten instances", protocol.payload["discarded_experiments"])

    def test_the_protocol_is_compiled_not_generated(self) -> None:
        import farfield.extras.packet as packet_mod

        self.assertFalse(hasattr(packet_mod, "complete"))
        self.assertFalse(hasattr(packet_mod, "LLMClient"))
        source = Path(packet_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from .llm", source)
        self.assertNotIn("import llm", source)

    def test_the_mission_packet_separates_do_next_from_closed_and_weakened(self) -> None:
        text = render_mission_packet(
            "succinct pivots",
            ranked=[
                {
                    "rank": 1,
                    "card_id": "live",
                    "why": ["双臂探针按预登记方向支持了机制"],
                    "verdict": "supports",
                },
                {
                    "rank": 2,
                    "card_id": "old",
                    "why": ["检索到的论文已经把主张写进去了，先读那篇"],
                    "prior_kills": True,
                },
                {
                    "rank": 3,
                    "card_id": "dead",
                    "why": ["双臂探针反预登记方向，机制被削弱"],
                    "verdict": "weakens",
                },
            ],
            found=[
                {
                    "card_id": "live",
                    "title": "Succinct pivot logs",
                    "claim": CARD.claim,
                    "experiment": DIAGNOSIS.experiment,
                    "verdict": "supports",
                    "has_brief": True,
                },
                {
                    "card_id": "old",
                    "title": "Already published",
                    "claim": "someone already wrote this",
                    "prior_kills": True,
                },
                {
                    "card_id": "dead",
                    "title": "Caching artefact",
                    "claim": "the speedup is real",
                    "verdict": "weakens",
                },
            ],
            workspace="/tmp/mission",
        )
        self.assertIn("推荐课题", text)
        self.assertIn("Succinct pivot logs", text)
        self.assertIn("ideas/succinct-pivot-logs", text)
        do_next, rest = text.split("先读文献", 1)
        self.assertIn("succinct-pivot-logs", do_next)
        self.assertNotIn("old", do_next)
        self.assertNotIn("dead", do_next)
        closed, weakened = rest.split("不要继续投入", 1)
        self.assertIn("someone already wrote this", closed)
        self.assertIn("the speedup is real", weakened)
        self.assertIn("不是会议论文", text)
        self.assertIn("研究方案", text)
        self.assertIn("记忆污染", text)
        self.assertIn("IDEA_REPORT.md", text)

    def test_idea_report_ranks_live_lines_and_closed_roads(self) -> None:
        text = render_idea_report(
            "code world models of executable program state",
            found=[
                {
                    "card_id": "live",
                    "pair": ["code world", "acceptance gate"],
                    "claim": "cross-seed certificates reject reward tampering",
                    "verdict": "uninformative",
                    "idea_world": {"iterate": "store hidden-seed rewards on the same object"},
                    "idea_name": "cross-seed-gate",
                },
                {
                    "card_id": "dead",
                    "pair": ["recursive algorithm", "stochastic reward"],
                    "verdict": "weakens",
                    "why_failed": "survey pulled knapsack papers",
                },
            ],
            kills=[
                {
                    "pair": ["recursive algorithm", "stochastic reward"],
                    "killed_by": ["already_combined"],
                    "why": "pair already walked",
                }
            ],
            wiki=[
                {
                    "cite_id": "2510.04542",
                    "limitations": ["However, we do not evaluate self-edit certificates."],
                }
            ],
        )
        self.assertIn("活线", text)
        self.assertIn("cross-seed certificates", text)
        self.assertIn("hidden-seed rewards", text)
        self.assertIn("关掉的路", text)
        self.assertIn("knapsack", text)
        self.assertIn("文献缺口", text)
        self.assertIn("self-edit certificates", text)

    def test_a_constructed_world_with_a_brief_is_do_next_and_cannot_corroborate(self) -> None:
        text = render_mission_packet(
            "diffusion sampler invariants",
            ranked=[
                {
                    "rank": 1,
                    "card_id": "fv",
                    "why": ["本轮构造世界上的探针没分出两臂差别，问题仍开着"],
                    "verdict": "uninformative",
                    "generated_world": True,
                    "has_brief": True,
                }
            ],
            found=[
                {
                    "card_id": "fv",
                    "title": "Lyapunov sampler certificate",
                    "claim": "a Lyapunov potential certifies the sampler invariant",
                    "experiment": "walk the constructed trace with the mechanism flag",
                    "verdict": "uninformative",
                    "has_brief": True,
                    "generated_world": True,
                    "probe_kind": "GENERATED",
                }
            ],
            workspace="/tmp/mission",
        )
        do_next, _rest = text.split("先读文献", 1)
        self.assertIn("fv", do_next)
        self.assertIn("不能佐证", do_next)
        self.assertIn("GENERATED", do_next)
        self.assertIn("研究方案", text)
        self.assertNotIn("下场纲领", text)
        self.assertNotIn("给下一跳 FarField", text)

    def test_the_packet_renders_forward_simulation_analysis(self) -> None:
        rec = {
            "card_id": "live",
            "title": "Succinct pivot logs",
            "claim": CARD.claim,
            "verdict": "supports",
            "probe_kind": "WORLD",
            "has_brief": True,
            "world_id": "tcp-linux-server",
            "idea_analysis": {
                "stop_reason": "plateau",
                "room_to_move": True,
                "n_steps": 4,
                "summary": "On tcp-linux-server under walk, execution stopped at t=3 (plateau).",
            },
        }
        text = render_idea_human_packet("succinct pivots", rec)
        self.assertIn("前向模拟", text)
        self.assertIn("plateau", text)
        self.assertIn("tcp-linux-server under walk", text)
        self.assertIn("不能爬梯", text)

    def test_the_packet_ends_with_a_glossary(self) -> None:
        text = render_mission_packet(
            "succinct pivots",
            found=[
                {
                    "card_id": "live",
                    "title": "Succinct pivot logs",
                    "claim": CARD.claim,
                    "verdict": "supports",
                    "has_brief": True,
                }
            ],
        )
        self.assertIn("## 术语速查", text)
        self.assertIn("处理臂 / 对照臂", text)
        self.assertIn("protocol_executed", text)
        self.assertIn("supports / weakens / uninformative", text)

    def test_plain_titles_lead_the_packet_headings(self) -> None:
        text = render_mission_packet(
            "succinct pivots",
            found=[
                {
                    "card_id": "live",
                    "title": "Succinct pivot logs",
                    "plain_title": "单纯形法的转轴历史能压缩吗",
                    "one_liner": "转轴记录能否用线性空间存下并快速查询？",
                    "why_it_matters": "如果可行，防循环检查不必保留完整日志。",
                    "claim": CARD.claim,
                    "verdict": "supports",
                    "has_brief": True,
                }
            ],
        )
        self.assertIn("### 单纯形法的转轴历史能压缩吗（Succinct pivot logs）", text)
        human = render_idea_human_packet(
            "succinct pivots",
            {
                "card_id": "live",
                "title": "Succinct pivot logs",
                "plain_title": "单纯形法的转轴历史能压缩吗",
                "one_liner": "转轴记录能否用线性空间存下并快速查询？",
                "why_it_matters": "如果可行，防循环检查不必保留完整日志。",
                "claim": CARD.claim,
                "verdict": "supports",
                "has_brief": True,
            },
        )
        self.assertIn("问题陈述", human)
        self.assertIn("转轴记录能否用线性空间存下并快速查询？", human)
        self.assertIn("如果可行，防循环检查不必保留完整日志。", human)
        from farfield.extras.packet import render_experiment_plan

        plan = render_experiment_plan("succinct pivots", {
            "card_id": "live",
            "claim": CARD.claim,
            "prediction": CARD.prediction,
            "experiment": DIAGNOSIS.experiment,
            "treatment_arm": DIAGNOSIS.treatment_arm,
            "control_arm": DIAGNOSIS.control_arm,
            "expected_direction": DIAGNOSIS.expected_direction,
            "alternative": DIAGNOSIS.alternative,
        })
        self.assertIn("Claim 映射", plan)
        self.assertIn("必须运行", plan)
        self.assertIn("实验块 1", plan)
        self.assertIn(CARD.claim, plan)
        self.assertNotIn("研究简报:", plan)

    def test_records_without_plain_fields_degrade_to_the_title(self) -> None:
        text = render_mission_packet(
            "succinct pivots",
            found=[
                {
                    "card_id": "live",
                    "title": "Succinct pivot logs",
                    "claim": CARD.claim,
                    "verdict": "supports",
                    "has_brief": True,
                }
            ],
        )
        self.assertIn("### Succinct pivot logs", text)
        self.assertNotIn("这张卡在说什么", text)

    def test_the_protocol_and_readme_prefer_the_plain_title(self) -> None:
        protocol = render_protocol(
            "topic",
            CARD,
            BRIEF_PLAIN,
            [PAPER],
            probe=PROBE,
            diagnosis=DIAGNOSIS,
        )
        heading = "单纯形法的转轴历史能压缩吗（Succinct pivot logs）"
        self.assertIn(f"# Research plan: {heading}", protocol.markdown)
        self.assertIn("**这张卡在说什么**", protocol.markdown)
        self.assertTrue(protocol.readme.startswith(f"# {heading}"))
        self.assertEqual(
            protocol.payload["plain_title"], "单纯形法的转轴历史能压缩吗"
        )
        note = render_note("topic", CARD, BRIEF_PLAIN, [PAPER])
        self.assertIn(f"# {heading}", note.markdown)

    def test_an_unfinished_line_inlines_the_claim_and_does_not_link_a_missing_protocol(self) -> None:
        text = render_mission_packet(
            "combining diffusion models with formal verification of sampler invariants",
            found=[
                {
                    "card_id": "gen_ef2b12f4de50",
                    "claim": "one-sided Lipschitz smaller than the invariant margin",
                    "mechanism": "learned score treated as an arbitrary function",
                    "prediction": "invariant violation implies the constant exceeds the margin",
                    "pair": ["formal verification", "zero weight"],
                    "world_incompatible": True,
                    "pipeline_bottleneck": "world",
                    "has_brief": False,
                }
            ],
            kills=[
                {
                    "pair": ["formal verification", "terminal set"],
                    "killed_by": ["prediction_names_a_measurable_quantity"],
                }
            ],
            program={
                "commit": "evolve the Lipschitz line",
                "next_card_must": "keep the pair, write a measurable prediction",
                "do_not_generate": "prediction_names_a_measurable_quantity",
                "h": {"memory": "Lipschitz line", "tools": "no freeze", "verifiers": "quantity", "routing": "directional"},
            },
            funnel={"generated": 5, "text_pass": 2, "world_valid": 0, "probe_supports": 0, "corroborated": 0,
                    "rates": {"P(text_pass|generated)": 0.4, "P(world_valid|text_pass)": 0.0}},
            questions={"world_incompatible": 1, "entered": 1},
            cards_generated=5,
            entered=1,
            workspace="/tmp/mission",
        )
        self.assertIn("推荐课题", text)
        self.assertIn("learned score treated as an arbitrary function", text)
        self.assertIn("研究方案", text)
        self.assertNotIn("keep the pair", text)
        self.assertNotIn("不要点死链", text)
        self.assertNotIn("P(text_pass|generated)", text)
        self.assertIn("summary.json", text)


    def test_distinct_ideas_do_not_share_a_packet(self) -> None:
        first = {
            "card_id": "gen_aaa",
            "title": "Memory key A",
            "claim": "prefix contrast predicts created_tools",
            "mechanism": "count mass sits in a few columns",
            "has_brief": True,
        }
        second = {
            "card_id": "gen_bbb",
            "title": "Memory key B",
            "claim": "returncode ngrams beat a length baseline",
            "mechanism": "failure states accumulate in the key",
            "has_brief": True,
        }
        index = render_mission_packet(
            "agent memory",
            found=[first, second],
            workspace="/tmp/mission",
        )
        self.assertIn("ideas/memory-key-a", index)
        self.assertIn("ideas/memory-key-b", index)
        self.assertNotIn("ideas/gen_aaa", index)
        self.assertNotIn("ideas/gen_bbb", index)
        self.assertIn("prefix contrast predicts created_tools", index)
        self.assertIn("returncode ngrams beat a length baseline", index)
        human_a = render_idea_human_packet(
            "agent memory", first, sibling_ids=["gen_aaa", "gen_bbb"]
        )
        agent_a = render_idea_agent_packet(
            "agent memory", first, sibling_ids=["gen_aaa", "gen_bbb"]
        )
        self.assertIn("prefix contrast predicts created_tools", human_a)
        self.assertNotIn("returncode ngrams beat a length baseline", human_a)
        self.assertIn("`memory-key-a`", human_a)
        self.assertNotIn("一条 idea（`gen_aaa`）", human_a)
        self.assertNotIn("../gen_bbb/", human_a)
        self.assertIn("Do not open sibling folders", agent_a)
        self.assertNotIn("ideas/gen_bbb/", agent_a)
        self.assertIn("audience: execute", agent_a)
        self.assertNotIn("next-round", agent_a)
        self.assertIn("Do not start a new `farfield research`", agent_a)
        self.assertIn("问题陈述", human_a)
        self.assertIn("背景", human_a)
        self.assertIn("世界与约束", human_a)
        self.assertIn("实验设计", human_a)
        self.assertIn("领域知识", human_a)
        self.assertIn("非目标", human_a)
        self.assertIn("已有结果", human_a)
        self.assertIn("下一步计划", human_a)
        self.assertIn("实验块 1", human_a)
        self.assertNotIn("完整实验块", human_a)
        english = render_idea_human_packet(
            "agent memory", first, sibling_ids=["gen_aaa", "gen_bbb"], lang="en"
        )
        self.assertIn("Research plan", english)
        self.assertIn("Experiment design", english)
        self.assertIn("prefix contrast predicts created_tools", english)
        self.assertNotIn("returncode ngrams beat a length baseline", english)
        plan_en = render_experiment_plan("agent memory", first, lang="en")
        self.assertIn("# Experiment design", plan_en)
        self.assertIn("prefix contrast predicts created_tools", plan_en)


class IdeaFolderNameTests(unittest.TestCase):
    def test_slug_uses_far_concept_and_lever_not_digest(self) -> None:
        from farfield.extras.packet import idea_folder_name, packet_worthy

        slug = idea_folder_name(
            {
                "card_id": "gen_cdadff266f68",
                "pair": ["agent system", "adaptive sampling"],
                "mechanism_flag": "mask_selector",
            }
        )
        self.assertEqual(slug, "adaptive-sampling-mask-selector")
        self.assertFalse(slug.startswith("gen_"))

    def test_weakens_and_unrunnable_are_not_packet_worthy(self) -> None:
        from farfield.extras.packet import packet_worthy

        self.assertFalse(packet_worthy({"verdict": "weakens", "has_brief": True}))
        self.assertFalse(
            packet_worthy(
                {
                    "verdict": None,
                    "experiment": "Train two models. Not runnable here: bound world lacks recall observable.",
                    "has_brief": True,
                }
            )
        )
        self.assertTrue(
            packet_worthy({"verdict": "uninformative", "has_brief": True, "experiment": "two-arm"})
        )


class ResultHonestyTests(unittest.TestCase):
    def test_generated_supports_cannot_be_a_discovery(self) -> None:
        from farfield.extras.packet import render_result_to_claim

        text = render_result_to_claim(
            {
                "claim": "sketches recover heavy hitters",
                "probe_kind": "GENERATED",
                "verdict": "supports",
                "world_id": "generated-text_stream-abc",
            }
        )
        self.assertIn("GENERATED", text)
        self.assertIn("cannot corroborate", text)
        self.assertNotIn("Two-arm arithmetic on an attested freeze", text)


if __name__ == "__main__":
    unittest.main()

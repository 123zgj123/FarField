"""The research note is compiled from attested fields, not a new generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.brief import ResearchBrief
from farfield.extras.diagnose import Diagnosis
from farfield.extras.generate import GeneratedCard
from farfield.extras.livefeed import FreshWork
from farfield.extras.packet import render_mission_packet, render_protocol
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
        do_next, rest = text.split("先读文献", 1)
        self.assertIn("live", do_next)
        self.assertNotIn("old", do_next)
        self.assertNotIn("dead", do_next)
        closed, weakened = rest.split("不要继续投入", 1)
        self.assertIn("someone already wrote this", closed)
        self.assertIn("the speedup is real", weakened)
        self.assertIn("不是会议论文", text)
        self.assertIn("下周步骤", text)

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
        self.assertIn("下周步骤", text)
        self.assertNotIn("下场纲领", text)
        self.assertNotIn("给下一跳 FarField", text)

    def test_the_packet_renders_forward_simulation_analysis(self) -> None:
        text = render_mission_packet(
            "succinct pivots",
            ranked=[{"rank": 1, "card_id": "live", "verdict": "supports"}],
            found=[
                {
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
            ],
        )
        self.assertIn("前向模拟", text)
        self.assertIn("plateau", text)
        self.assertIn("tcp-linux-server under walk", text)
        self.assertIn("不能爬梯", text)

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
        self.assertIn("下周步骤", text)
        self.assertNotIn("keep the pair", text)
        self.assertNotIn("不要点死链", text)
        self.assertNotIn("P(text_pass|generated)", text)
        self.assertIn("summary.json", text)


if __name__ == "__main__":
    unittest.main()

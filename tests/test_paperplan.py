"""The paper-level writing pass: facts in, audited draft out, ladder untouched."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.llm import Completion
from farfield.extras.paperplan import (
    PaperError,
    card_id_for,
    compile_paper,
    gather_facts,
)


class ScriptedClient:
    """Answers each purpose prefix with a scripted JSON text."""

    def __init__(self, answers: dict[str, dict]) -> None:
        self.answers = answers
        self.purposes: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.purposes.append(purpose)
        key = purpose.split(":")[0]
        text = json.dumps(self.answers[key], ensure_ascii=False)
        return Completion(
            text=text,
            model="fake",
            request_digest="req",
            digest=f"d-{len(self.purposes)}",
            artifact_uri=f"file:///fake/{purpose}",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            mode="replay",
        )


def _workspace(tmp: Path) -> tuple[Path, Path, str]:
    card = "gen_abc123"
    ws = tmp / "mission"
    cand = ws / "candidates" / card
    cand.mkdir(parents=True)
    (ws / "topic.txt").write_text("code world models of executable program state", encoding="utf-8")
    (cand / "protocol.json").write_text(
        json.dumps(
            {
                "topic": "code world models of executable program state",
                "claim": "Novelty-preserving dropout lowers false_accept_fraction by 20%.",
                "mechanism": "MinHash density weighting.",
                "prediction": "ratio <= 0.80",
                "pair": ["code world", "real trajectories"],
                "papers": [
                    {"arxiv_id": "2608.17956v1", "title": "Danger law", "published": "2026", "abstract": "gate"},
                    {"work_id": "doi:10.1109/ACCESS.2026.3663571", "title": "Trajectories", "abstract": "x"},
                ],
                "experiment_attempts": [{"attempt": 0, "treatment": 0.2975, "control": 0.2961, "verdict": "uninformative"}],
            }
        ),
        encoding="utf-8",
    )
    (cand / "probe.json").write_text(
        json.dumps({"kind": "WORLD", "verdict": "uninformative", "treatment": 0.4279, "control": 0.4279}),
        encoding="utf-8",
    )
    (cand / "metrics.json").write_text(json.dumps({"treatment": 0.4279, "control": 0.4279}), encoding="utf-8")
    (ws / "world").mkdir()
    (ws / "world" / "world.json").write_text(
        json.dumps({"id": "swe-selfmod", "schema": "program_state", "role": "world", "provenance": "derived"}),
        encoding="utf-8",
    )
    (ws / "mission.ndjson").write_text(
        json.dumps({"stage": "idea_review", "card_id": card, "novelty": 5, "critique": "no iid control", "must_change": ["add iid arm"]})
        + "\n"
        + json.dumps({"stage": "promotion", "card_id": card, "status": "speculative", "reason": "no discriminating evidence yet"})
        + "\n",
        encoding="utf-8",
    )
    idea = ws / "ideas" / "real-trajectories-dropout"
    (idea / "refine-logs").mkdir(parents=True)
    (idea / "AGENT_PACKET.md").write_text(f"- protocol_dir: `candidates/{card}/`\n", encoding="utf-8")
    (idea / "refine-logs" / "EXPERIMENT_RESULTS.md").write_text("verdict: uninformative", encoding="utf-8")
    return ws, idea, card


def _author(**over) -> dict:
    base = {
        "title": "Novelty-Preserving Dropout for Self-Improving Validators",
        "abstract": "We study ...",
        "proposal_markdown": (
            "## 1 Background\nCode world models [2608.17956v1] ...\n"
            "## 3 Related work\nA fabricated one [2501.00001] and a real DOI [doi:10.1109/ACCESS.2026.3663571].\n"
            "## 5 Method\nWe pre-register a threshold of 0.75 and a budget of 1200 updates."
        ),
        "experiment_design_markdown": "## Blocks\nB1 on swe-selfmod, treatment 0.4279 vs control 0.4279.",
        "hypotheses": [{"id": "H1", "statement": "...", "world": "swe-selfmod", "evidence": "uninformative"}],
        "citations_used": ["2608.17956v1", "2501.00001"],
        "limitations": ["single derived world"],
        "open_questions": ["mutable validators"],
    }
    base.update(over)
    return base


class PaperPlanTests(unittest.TestCase):
    def test_card_id_is_read_from_the_agent_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _ws, idea, card = _workspace(Path(tmp))
            self.assertEqual(card_id_for(idea), card)
            with self.assertRaises(PaperError):
                card_id_for(Path(tmp))

    def test_facts_come_only_from_attested_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            facts = gather_facts(ws, idea, card, catalog_root=Path(tmp) / "nowhere")
            self.assertEqual(facts["claim"], "Novelty-preserving dropout lowers false_accept_fraction by 20%.")
            self.assertEqual(facts["world"]["provenance"], "derived")
            self.assertEqual(facts["probe"]["kind"], "WORLD")
            self.assertEqual([p["id"] for p in facts["verified_papers"]], ["2608.17956v1", "doi:10.1109/ACCESS.2026.3663571"])
            self.assertEqual(facts["adversarial_review"]["must_change"], ["add iid arm"])
            self.assertEqual(facts["ladder_status"]["status"], "speculative")

    def test_draft_is_written_audited_and_reviewed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            client = ScriptedClient(
                {
                    "paper_author": _author(),
                    "paper_review": {
                        "score": 6,
                        "summary": "ok",
                        "strengths": [],
                        "weaknesses": ["no dose response"],
                        "fabrication_or_overclaim": [],
                        "must_fix": ["add dose-response arm"],
                        "missing_experiments": [],
                    },
                    "paper_revise": _author(
                        experiment_design_markdown="## Blocks\nB1 ... B2 dose-response 10/20/40%.",
                        changes_made=["added dose-response block"],
                    ),
                }
            )
            record = compile_paper(ws, idea, card, client=client, rounds=1, catalog_root=Path(tmp) / "nowhere")
            self.assertEqual([p.split(":")[0] for p in client.purposes], ["paper_author", "paper_review", "paper_revise"])
            paper = idea / "paper"
            proposal = (paper / "RESEARCH_PROPOSAL.md").read_text(encoding="utf-8")
            design = (paper / "EXPERIMENT_DESIGN.md").read_text(encoding="utf-8")
            self.assertIn("DRAFT", proposal)
            self.assertIn("[2608.17956v1]", proposal)
            self.assertIn("[doi:10.1109/ACCESS.2026.3663571]", proposal)
            # The fabricated id is rewritten, not silently kept.
            self.assertNotIn("[2501.00001]", proposal)
            self.assertIn("[unverified: 2501.00001]", proposal)
            self.assertIn("dose-response", design)
            self.assertEqual(record["status"], "draft")
            self.assertEqual(record["score"], 6)
            self.assertEqual(record["rounds"], 1)
            self.assertIn("2501.00001", record["audit"]["unverified_citations_rewritten"])
            # 0.75 and 1200 are proposals, not in the fact sheet; 0.4279 is,
            # and an arXiv id is not a number.
            self.assertIn("0.75", record["audit"]["draft_numbers_not_in_facts"])
            self.assertNotIn("2501.00001", record["audit"]["draft_numbers_not_in_facts"])
            self.assertIn("1200", record["audit"]["draft_numbers_not_in_facts"])
            self.assertNotIn("0.4279", record["audit"]["draft_numbers_not_in_facts"])
            self.assertTrue((paper / "REVIEW.md").is_file())
            self.assertTrue((paper / "PAPER_FACTS.json").is_file())
            self.assertIn("added dose-response block", (paper / "REVIEW.md").read_text(encoding="utf-8"))

    def test_prose_citations_and_worldless_hypotheses_are_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            client = ScriptedClient(
                {
                    "paper_author": _author(
                        proposal_markdown=(
                            "## 3 Related work\nAs Zhang et al. (2025) showed, and (Smith and Lee, 2024) "
                            "argued, gates miss modes; 王等人 (2026) 亦然. Real: [2608.17956v1]."
                        ),
                        hypotheses=[
                            {"id": "H1", "statement": "a", "world": "swe-selfmod", "evidence": "none"},
                            {"id": "H2", "statement": "b", "world": "", "evidence": "none"},
                        ],
                    ),
                    "paper_review": {"score": 7, "summary": "ok", "must_fix": [], "fabrication_or_overclaim": []},
                }
            )
            record = compile_paper(ws, idea, card, client=client, rounds=1, catalog_root=Path(tmp) / "nowhere")
            audit = record["audit"]
            self.assertEqual(len(audit["prose_citations_without_id"]), 3)
            self.assertTrue(any("Zhang et al" in p for p in audit["prose_citations_without_id"]))
            self.assertTrue(any("等人" in p for p in audit["prose_citations_without_id"]))
            self.assertEqual(audit["hypotheses_without_a_world"], ["H2"])
            text = (idea / "paper" / "AUDIT.md").read_text(encoding="utf-8")
            self.assertIn("prose citations without an id", text)

    def test_exploratory_records_enter_the_facts_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            catalog = Path(tmp) / "site" / "worlds"
            (Path(tmp) / "site" / "var" / "exploratory" / "swe-selfmod").mkdir(parents=True)
            (Path(tmp) / "site" / "var" / "exploratory" / "swe-selfmod" / "abc.json").write_text(
                json.dumps(
                    {
                        "exploratory": True,
                        "label": "descriptives:program_state",
                        "script_digest": "abc",
                        "world_digest": "w",
                        "results": {"oracle_failure_given_accepted": {"rate": 0.303}},
                    }
                ),
                encoding="utf-8",
            )
            catalog.mkdir(parents=True)
            facts = gather_facts(ws, idea, card, catalog_root=catalog)
            self.assertEqual(len(facts["exploratory_analyses"]), 1)
            self.assertIn("cannot corroborate", facts["exploratory_analyses"][0]["status"])

    def test_a_clean_review_stops_the_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            client = ScriptedClient(
                {
                    "paper_author": _author(),
                    "paper_review": {"score": 8, "summary": "fine", "must_fix": [], "fabrication_or_overclaim": []},
                }
            )
            record = compile_paper(ws, idea, card, client=client, rounds=2, catalog_root=Path(tmp) / "nowhere")
            self.assertEqual(len(client.purposes), 2)
            self.assertEqual(record["rounds"], 1)

    def test_no_claim_is_a_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, idea, card = _workspace(Path(tmp))
            (ws / "candidates" / card / "protocol.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(PaperError):
                compile_paper(ws, idea, card, client=ScriptedClient({}), rounds=0)
            self.assertFalse((idea / "paper").exists())

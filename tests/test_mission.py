"""The mission event stream: every stage reports, every refusal names its unlock."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from farfield.extras.livefeed import FeedBlocked, FreshWork
from farfield.extras.llm import Completion
from farfield.extras.mission import load_assets, rank_ideas, run_mission
from farfield.extras.state import record_rejections

# The production concept graph exceeds GitHub's 100 MB file limit and is
# not in a public clone; missions there run on the shipped fallback slice
# (that path has its own tests, see ShippedCorpusTests). A handful of
# tests assert behaviour specific to the production graph's vocabulary —
# without the graph there is nothing for them to test, so they skip with
# the reason named instead of failing on the fallback's different words.
PRODUCTION_GRAPH = (
    Path(__file__).resolve().parents[1]
    / "concepts"
    / "ds-arxiv-concepts-2026"
    / "G_full.json"
)
needs_production_graph = unittest.skipUnless(
    PRODUCTION_GRAPH.is_file(),
    "production concept graph is not in this clone (it exceeds GitHub's"
    " file limit); the shipped-slice fallback is tested separately",
)


class SchemingClient:
    """Answers every generation prompt with a valid card built from the
    prompt itself: the pair is copied verbatim from the offered menus, so
    the schema's verbatim-copy teeth cannot bite. Claim, mechanism and
    prediction are written to pass the installed text judges — this is a
    client that read the review criteria, which is exactly what the closed
    feedback loop invites."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        if "Two anonymous research hypotheses" in prompt:
            import json

            return Completion(
                text=json.dumps(
                    {
                        "winner": "A",
                        "reason": "more specific mechanism and a checkable bound",
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"t{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "name what would change our mind" in prompt:
            import json

            return Completion(
                text=json.dumps(
                    {
                        "alternative": "the speedup comes from caching, not from the succinct encoding",
                        "experiment": "run the same query log with the succinct index and with the cache disabled in both arms",
                        "treatment_arm": "succinct encoding on, cache off",
                        "control_arm": "plain array, cache off",
                        "expected_direction": "treatment_lower",
                        "alternative_direction": "treatment_higher",
                        "expected_if_alternative": "the two arms tie once the cache is off",
                        "margin": 0.05,
                        "margin_reason": "toy counter noise stays well under five percent",
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"g{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "pre-registered two-arm experiment" in prompt:
            import json

            source = (
                "import json\n"
                "from pathlib import Path\n"
                "def measure(use_shortcut):\n"
                "    hops = 0\n"
                "    pos = 0\n"
                "    end = 8\n"
                "    while pos < end:\n"
                "        hops += 1\n"
                "        if use_shortcut and pos == 0:\n"
                "            pos = end\n"
                "        else:\n"
                "            pos += 1\n"
                "    return hops\n"
                "Path('metrics.json').write_text("
                "json.dumps({'treatment': float(measure(True)),"
                " 'control': float(measure(False))}),"
                " encoding='utf-8')\n"
            )
            return Completion(
                text=json.dumps(
                    {
                        "measure": "nanoseconds per query on the toy log",
                        "source": source,
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"p{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "questioning a research field" in str(system or "") or "Name ONE assumption" in prompt:
            import json

            return Completion(
                text=json.dumps(
                    {
                        "assumption": "worst case analysis",
                        "claim": (
                            "dropping worst case analysis for succinct indexes"
                            " of genomic sequence collections yields query time"
                            " at most O(log n) on realistic inputs"
                        ),
                        "mechanism": (
                            "worst case analysis prices adversarial inputs that"
                            " realistic query logs never produce, so an"
                            " instance-optimal bound is strictly smaller"
                        ),
                        "prediction": (
                            "later work reports query time below 100 ns at"
                            " n = 10^6 under instance-optimal analysis"
                        ),
                        "dead_end": (
                            "keep worst case analysis and size the index for"
                            " adversarial query logs"
                        ),
                        "why_failed": (
                            "adversarial padding dominates the bound so"
                            " realistic logs never see it"
                        ),
                        "reframe": (
                            "drop worst case analysis and measure realistic"
                            " query logs instead"
                        ),
                        "objection": (
                            "a hidden adversary can still force the"
                            " logarithmic bound to collapse"
                        ),
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"q{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "Score this research idea as a reviewer" in prompt:
            ids = re.findall(r"\[([0-9]+\.[0-9]+(?:v\d+)?)\]", prompt)
            note = (
                f"{ids[0]} leaves the claimed bound untested"
                if ids
                else "no live papers were retrieved"
            )
            import json

            return Completion(
                text=json.dumps(
                    {
                        "novelty": 3,
                        "novelty_note": note,
                        "clarity": 4,
                        "feasibility": 4,
                        "importance": 3,
                        "information_gain": 4,
                        "transfer_potential": 2,
                        "keep_going": True,
                        "critique": "name a public trace and a query-time number",
                        "must_change": ["first_steps should name a public LP trace"],
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"r{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "Revise the research-ready idea" in prompt:
            ids = re.findall(r"\[([0-9]+\.[0-9]+(?:v\d+)?)\]", prompt)
            if ids:
                gap = (
                    f"{ids[0]} studies a nearby construction but leaves the"
                    " claimed query bound untested"
                )
                reads = ids[:1]
            else:
                gap = "no live papers were retrieved; start from the hypothesis"
                reads = []
            import json

            return Completion(
                text=json.dumps(
                    {
                        "title": "Succinct pivot logs, revised",
                        "gap": gap,
                        "idea": "Store simplex pivot history succinctly and report ns/query on a public MIPLIB trace",
                        "approach": "Wavelet-tree encode the pivot log and time rank queries",
                        "first_steps": [
                            "Download a public MIPLIB instance and record 10k pivots",
                            "Encode the log and time last-repeat queries",
                            "Compare bits/pivot against gzip",
                        ],
                        "baseline": "an uncompressed circular buffer of recent pivots",
                        "risks": "real LP traces may not compress",
                        "read_first": reads,
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"f{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "research-ready idea" in prompt:
            ids = re.findall(r"\[([0-9]+\.[0-9]+(?:v\d+)?)\]", prompt)
            if ids:
                gap = (
                    f"{ids[0]} studies a nearby construction but leaves the"
                    " claimed query bound untested"
                )
                reads = ids[:1]
            else:
                gap = "no live papers were retrieved; start from the hypothesis"
                reads = []
            import json

            answer = json.dumps(
                {
                    "title": "Succinct pivot logs for anticycling",
                    "gap": gap,
                    "idea": "Store simplex pivot history in a succinct structure and measure query time",
                    "approach": "Encode the pivot sequence and benchmark against a plain log",
                    "first_steps": [
                        "Reproduce a 10k-iteration simplex trace",
                        "Add rank queries for the last repeated pivot",
                        "Compare bits per pivot against gzip",
                    ],
                    "baseline": "an uncompressed circular buffer of recent pivots",
                    "risks": "real LP traces may not compress",
                    "read_first": reads,
                }
            )
            return Completion(
                text=answer,
                model="fake-model",
                request_digest="req",
                digest=f"b{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        seed = re.search(r"Seed concept: (.+)", prompt).group(1).strip()
        far = re.search(r"\(operator: \w+\): (.+)", prompt).group(1)
        first_far = far.split(", ")[0].strip()
        pairing = re.search(r"- (.+): may pair with (.+)", prompt)
        if pairing:
            # The prompt names the structurally viable pairings; a client
            # that read the menu picks from it, like a real model would.
            first_far = pairing.group(1).strip()
            seed = pairing.group(2).split(", ")[0].strip()
        topic_m = re.search(r"Researcher's topic: (.+)", prompt)
        topic_bit = (
            f" for {topic_m.group(1).strip()}"
            if topic_m
            else ""
        )
        answer = (
            "{"
            f'"claim": "combining {seed} with {first_far}{topic_bit} reduces'
            ' query time to at most O(log n) because the index never rescans",'
            f'"mechanism": "the {seed} structure partitions work so that'
            f' {first_far} lookups reuse cached boundaries, which yields'
            ' fewer comparisons per query and gives a strict bound",'
            f'"pair": ["{seed}", "{first_far}"],'
            '"falsifier": "pair_not_already_combined",'
            '"prediction": "later work will report a structure with query time'
            ' below 100 ns at n = 10^6",'
            '"dead_end": "store every record in a plain array and scan it on each query",'
            '"why_failed": "linear scan grows with the collection so the logarithmic bound never appears",'
            '"reframe": "partition the index so lookups reuse cached boundaries instead of scanning",'
            '"objection": "cached boundaries miss updates so the bound fails on a mutating collection",'
            '"derivation": "a partition recurrence T(n)=T(n/2)+O(1) keeps comparisons logarithmic on the bound world"'
            "}"
        )
        return Completion(
            text=answer,
            model="fake-model",
            request_digest="req",
            digest=f"d{len(self.prompts):015x}",
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            mode="replay",
        )


class ApplyThenRefineClient(SchemingClient):
    """First draft has an unmeasurable prediction; the refine writes a number."""

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        done = super().complete(
            prompt, purpose=purpose, system=system, logprobs=logprobs
        )
        if "Seed concept:" not in prompt or "Revise THIS idea" in prompt:
            return done
        try:
            payload = json.loads(done.text)
        except json.JSONDecodeError:
            return done
        if "prediction" not in payload:
            return done
        payload["prediction"] = (
            "later work will report that the idea holds in practice"
        )
        import dataclasses

        return dataclasses.replace(done, text=json.dumps(payload))


class BabblingClient(SchemingClient):
    """Answers with prose instead of JSON: every card must be refused."""

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        completion = super().complete(
            prompt, purpose=purpose, system=system, logprobs=logprobs
        )
        return Completion(
            text="a fascinating connection worth exploring!",
            model=completion.model,
            request_digest=completion.request_digest,
            digest=completion.digest,
            artifact_uri=completion.artifact_uri,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=0,
            mode="replay",
        )


class FakeFeed:
    """A live feed that answers from canned pages: fresh without network."""

    def __init__(self, *, dead: bool = False) -> None:
        self.dead = dead
        self.probed: list[tuple[str, str]] = []

    def recent_in_field(self, concepts, *, max_results=6):
        if self.dead:
            return FeedBlocked(attempted="recent_in_field", reason="no route")
        return [
            FreshWork(
                title="A very fresh preprint",
                published="2026-08-15",
                arxiv_id="2608.09999v1",
            )
        ]

    def pair_recently_combined(self, concept_a, concept_b, *, max_results=5):
        self.probed.append((concept_a, concept_b))
        return {"combined": False, "evidence": []}

    def survey_around(self, concept_a, concept_b, *, per_side=4, claim=""):
        if self.dead:
            return FeedBlocked(attempted="survey_around", reason="no route")
        return [
            FreshWork(
                title="A very fresh preprint",
                published="2026-08-15",
                arxiv_id="2608.09999v1",
                abstract="We study nearby constructions in this field.",
            )
        ]


TOPIC = "compress genomic sequence collections with succinct data structures"


def run_all(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        kwargs.setdefault("state_store", Path(tmp) / "state.json")
        kwargs.setdefault("policy_log", Path(tmp) / "policy_log.json")
        kwargs.setdefault("policy_file", Path(tmp) / "policy.json")
        kwargs.setdefault("polish_rounds", 0)
        kwargs.setdefault("explore", "fixed")
        kwargs.setdefault("experiment_rounds", 0)
        kwargs.setdefault("idea_rounds", 0)
        return list(run_mission(TOPIC, **kwargs))


class MissionStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events = run_all(jumps=2, candidates=1, client=SchemingClient())

    def test_the_stream_opens_with_the_corpus_and_its_freshness(self) -> None:
        first = self.events[0]
        self.assertEqual(first["stage"], "corpus")
        self.assertRegex(first["knowledge_until"], r"^\d{4}-\d\d-\d\d$")
        self.assertTrue(first["judges"], "the installed judges must be named")
        self.assertGreaterEqual(first["parallel"], 1)

    def test_plugin_host_runs_in_the_mission_stream(self) -> None:
        stages = [event["stage"] for event in self.events]
        self.assertIn("skills", stages)
        skills = self.events[stages.index("skills")]
        names = {row["name"] for row in skills["plugins"] if row.get("executable")}
        self.assertIn("builtin", names)
        self.assertIn("deepseek-harness", names)
        self.assertIn("fair-two-arm-probe", names)
        self.assertEqual(skills["dsh"]["role"], "plugin")

    @needs_production_graph
    def test_the_topic_is_anchored_before_any_jump(self) -> None:
        stages = [event["stage"] for event in self.events]
        self.assertLess(stages.index("anchor"), stages.index("jump"))
        anchors = self.events[stages.index("anchor")]["anchors"]
        self.assertEqual(len(anchors), 6)
        self.assertIn("succinct", anchors[0]["concept"])

    def test_every_card_gets_a_verdict_with_named_reasons(self) -> None:
        cards = [e for e in self.events if e["stage"] == "card"]
        verdicts = [e for e in self.events if e["stage"] == "verdict"]
        self.assertEqual(len(cards), 2)
        self.assertEqual(len(verdicts), len(cards))
        for verdict in verdicts:
            self.assertIsInstance(verdict["alive"], bool)
            if not verdict["alive"]:
                self.assertTrue(
                    verdict["graph_killed_by"]
                    or verdict["text_kills"]
                    or not verdict["chosen"],
                    "a dead card must name its killer or its stronger sibling",
                )

    def test_doomed_pairs_are_screened_before_any_generation_call(self) -> None:
        # The three graph gates are pure pair properties, so a far concept
        # with no surviving near partner is settled at jump time — banked
        # as a rejection, never sent to the generator.
        jumps = [
            e
            for e in self.events
            if e["stage"] == "jump" and e.get("track") == "farfield"
        ]
        self.assertTrue(jumps)
        for event in jumps:
            self.assertIn("prescreened_out", event)
            offered = set(event["far"])
            for dropped in event["prescreened_out"]:
                self.assertNotIn(dropped["label"], offered)
                self.assertTrue(dropped["killed_by"])

    def test_a_generating_notice_precedes_every_card(self) -> None:
        stages = [event["stage"] for event in self.events]
        for index, stage in enumerate(stages):
            if stage == "card":
                self.assertEqual(stages[index - 1], "generating")

    def test_the_stream_closes_with_totals_that_add_up(self) -> None:
        done = self.events[-1]
        self.assertEqual(done["stage"], "done")
        self.assertEqual(done["cards"], 2)
        alive = sum(e["alive"] for e in self.events if e["stage"] == "verdict")
        self.assertEqual(done["survivors"], alive)

    def test_a_babbling_model_yields_refusals_not_cards(self) -> None:
        events = run_all(jumps=2, candidates=1, client=BabblingClient())
        stages = [event["stage"] for event in events]
        self.assertNotIn("card", stages)
        self.assertEqual(stages.count("refused"), 2)
        refusal = events[stages.index("refused")]
        self.assertIn("unlock_condition", refusal)
        self.assertEqual(events[-1]["stage"], "done")
        self.assertEqual(events[-1]["cards"], 0)

    def test_assets_are_loaded_once_and_reused(self) -> None:
        self.assertIs(load_assets(), load_assets())

    def test_the_agenda_plans_every_slot_before_any_jump(self) -> None:
        stages = [event["stage"] for event in self.events]
        self.assertLess(stages.index("agenda"), stages.index("jump"))
        agenda = self.events[stages.index("agenda")]
        self.assertEqual(len(agenda["slots"]), 2)
        self.assertTrue(all(s["track"] == "farfield" for s in agenda["slots"]))

    def test_every_verdict_names_its_exit(self) -> None:
        for event in self.events:
            if event["stage"] == "verdict":
                self.assertIn(
                    event["outcome"], ("entered", "ineligible", "not_selected")
                )
                self.assertEqual(event["outcome"] == "entered", event["alive"])


class CandidateEliminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.events = run_all(jumps=2, candidates=2, client=SchemingClient())

    def test_each_jump_writes_its_full_slate_of_candidates(self) -> None:
        self.assertEqual(self.events[-1]["cards"], 4)

    def test_exactly_one_candidate_per_jump_is_chosen(self) -> None:
        verdicts = [e for e in self.events if e["stage"] == "verdict"]
        for jump in {v["jump"] for v in verdicts}:
            chosen = [v for v in verdicts if v["jump"] == jump and v["chosen"]]
            self.assertEqual(len(chosen), 1)

    def test_an_eliminated_sibling_is_never_a_survivor(self) -> None:
        for verdict in self.events:
            if verdict.get("stage") == "verdict" and not verdict["chosen"]:
                self.assertFalse(verdict["alive"])


class LiveKnowledgeTests(unittest.TestCase):
    def test_fresh_papers_reach_both_the_stream_and_the_prompt(self) -> None:
        client = SchemingClient()
        events = run_all(jumps=1, candidates=1, client=client, feed=FakeFeed())
        fresh = next(e for e in events if e["stage"] == "fresh")
        self.assertEqual(fresh["works"][0]["published"], "2026-08-15")
        self.assertIn("A very fresh preprint", client.prompts[0])

    def test_every_survivor_gets_one_freshness_probe(self) -> None:
        feed = FakeFeed()
        events = run_all(jumps=2, candidates=1, client=SchemingClient(), feed=feed)
        survivors = [e for e in events if e["stage"] == "verdict" and e["alive"]]
        checks = [e for e in events if e["stage"] == "fresh_check"]
        self.assertEqual(len(checks), len(survivors))
        self.assertEqual(len(feed.probed), len(survivors))

    def test_open_pairs_are_recommended_ahead_of_already_seen_ones(self) -> None:
        ranked = rank_ideas(
            [
                {
                    "card_id": "old",
                    "open_today": False,
                    "has_brief": True,
                    "papers": 6,
                    "alienness": 0.9,
                    "pair": ["a", "b"],
                },
                {
                    "card_id": "fresh",
                    "open_today": True,
                    "has_brief": True,
                    "papers": 2,
                    "alienness": 0.2,
                    "pair": ["c", "d"],
                },
            ]
        )
        self.assertEqual([row["card_id"] for row in ranked], ["fresh", "old"])
        self.assertTrue(ranked[0]["recommended"])
        self.assertIn("还没见到", ranked[0]["why"][0])

    def test_a_finished_run_emits_a_reading_order(self) -> None:
        events = run_all(jumps=2, candidates=1, client=SchemingClient(), feed=FakeFeed())
        ranked = next(e for e in events if e["stage"] == "ranked")
        survivors = [e for e in events if e["stage"] == "verdict" and e["alive"]]
        self.assertEqual(len(ranked["ideas"]), len(survivors))
        self.assertEqual(events[-1]["stage"], "done")

    def test_a_polish_round_reviews_then_revises(self) -> None:
        events = run_all(
            jumps=1, candidates=1, polish_rounds=1,
            client=SchemingClient(), feed=FakeFeed(),
        )
        reviews = [e for e in events if e["stage"] == "review"]
        briefs = [e for e in events if e["stage"] == "brief"]
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["novelty"], 3)
        self.assertIn("opinion", reviews[0]["status"])
        self.assertGreaterEqual(len(briefs), 2)
        self.assertNotEqual(briefs[0]["idea"], briefs[-1]["idea"])

    def test_world_uninformative_is_recommended_ahead_of_synthetic_support(self) -> None:
        ranked = rank_ideas(
            [
                {
                    "card_id": "synth",
                    "verdict": "supports",
                    "probe_kind": "SYNTHETIC",
                    "value_score": 9.0,
                    "open_today": True,
                    "has_brief": True,
                    "papers": 6,
                    "alienness": 0.9,
                    "pair": ["a", "b"],
                },
                {
                    "card_id": "world",
                    "verdict": "uninformative",
                    "probe_kind": "WORLD",
                    "value_score": 0.1,
                    "open_today": False,
                    "has_brief": False,
                    "papers": 1,
                    "alienness": 0.1,
                    "pair": ["c", "d"],
                },
            ]
        )
        self.assertEqual([row["card_id"] for row in ranked], ["world", "synth"])
        self.assertTrue(ranked[0]["recommended"])
        self.assertTrue(any("WORLD" in line for line in ranked[0]["why"]))

    def test_a_low_value_idea_still_polishes_when_keep_going(self) -> None:
        # The reviewer's 1–5 vector is opinion. keep_going is the only
        # discretionary stop; a low peak no longer voids the second draft.
        class Miser(SchemingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                if "Score this research idea as a reviewer" in prompt:
                    self.prompts.append(prompt)
                    import json
                    ids = re.findall(r"\[([0-9]+\.[0-9]+(?:v\d+)?)\]", prompt)
                    note = (
                        f"{ids[0]} already covers most of this ground"
                        if ids
                        else "no live papers were retrieved"
                    )
                    return Completion(
                        text=json.dumps({
                            "novelty": 2,
                            "novelty_note": note,
                            "clarity": 4,
                            "feasibility": 4,
                            "importance": 1,
                            "information_gain": 2,
                            "transfer_potential": 1,
                            "keep_going": True,
                            "critique": "thin but harmless",
                            "must_change": ["nothing worth paying for"],
                        }),
                        model="fake-model",
                        request_digest="req",
                        digest=f"m{len(self.prompts):015x}",
                        artifact_uri="file:///dev/null",
                        finish_reason="stop",
                        prompt_tokens=10,
                        completion_tokens=20,
                        reasoning_tokens=0,
                        mode="replay",
                    )
                return super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )

        events = run_all(
            jumps=1, candidates=1, polish_rounds=1,
            client=Miser(), feed=FakeFeed(),
        )
        stops = [e for e in events if e["stage"] == "polish_stop"]
        self.assertEqual(stops, [])
        briefs = [e for e in events if e["stage"] == "brief"]
        self.assertGreaterEqual(len(briefs), 2, "keep_going still buys a second draft")
        self.assertTrue([e for e in events if e["stage"] == "note"],
                        "the note still ships")

    def test_every_survivor_gets_a_research_brief(self) -> None:
        events = run_all(jumps=2, candidates=1, client=SchemingClient(), feed=FakeFeed())
        survivors = [e for e in events if e["stage"] == "verdict" and e["alive"]]
        briefs = [e for e in events if e["stage"] == "brief"]
        self.assertEqual(len(briefs), len(survivors))
        self.assertEqual(events[-1]["briefs"], len(briefs))
        for brief in briefs:
            self.assertGreaterEqual(len(brief["first_steps"]), 2)
            self.assertTrue(brief["title"])
            self.assertIn("2608.09999v1", brief["gap"])

    def test_a_dead_feed_blocks_the_fresh_stage_not_the_mission(self) -> None:
        events = run_all(
            jumps=1, candidates=1, client=SchemingClient(), feed=FakeFeed(dead=True)
        )
        fresh = next(e for e in events if e["stage"] == "fresh")
        self.assertIn("blocked", fresh)
        self.assertEqual(events[-1]["stage"], "done")


class ResearchCompletionTests(unittest.TestCase):
    def test_a_survivor_gets_diagnosis_probe_evidence_and_a_note(self) -> None:
        events = run_all(jumps=1, candidates=1, client=SchemingClient(), feed=FakeFeed())
        stages = [event["stage"] for event in events]
        self.assertLess(stages.index("diagnosis"), stages.index("probe"))
        diagnosis = next(e for e in events if e["stage"] == "diagnosis")
        self.assertEqual(diagnosis["expected_direction"], "treatment_lower")
        self.assertIn("caching", diagnosis["alternative"])
        probe = next(e for e in events if e["stage"] == "probe")
        self.assertEqual(probe["status"], "ran")
        self.assertEqual(probe["treatment"], 1.0)
        self.assertEqual(probe["control"], 8.0)
        evidence = next(e for e in events if e["stage"] == "evidence")
        self.assertEqual(evidence["verdict"], "supports")
        note = next(e for e in events if e["stage"] == "note")
        self.assertIn(r"\documentclass", note["latex"])
        self.assertIn("2608.09999v1", note["markdown"])
        self.assertIn("supports", note["markdown"])
        self.assertIn("SYNTHETIC", note["protocol"])
        self.assertIn("2608.09999v1", note["protocol"])
        self.assertEqual(note["protocol_json"]["cheap_probe"]["kind"], "SYNTHETIC")
        self.assertTrue(note["protocol_json"]["not_a_paper"])
        done = events[-1]
        self.assertIn("推荐课题", done["packet_markdown"])
        self.assertIn("不是会议论文", done["packet_markdown"])

    def test_synthetic_supports_do_not_climb_the_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared = {
                "state_store": Path(tmp) / "state.json",
                "policy_log": Path(tmp) / "log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
            }
            first = list(run_mission(
                TOPIC, jumps=1, candidates=1,
                client=SchemingClient(), feed=FakeFeed(), **shared,
            ))
            promo1 = [e for e in first if e["stage"] == "promotion" and not e["previous"]]
            self.assertTrue(promo1)
            self.assertEqual(promo1[0]["status"], "speculative")
            self.assertIn("SYNTHETIC", promo1[0]["reason"])
            client = SchemingClient()
            second = list(run_mission(
                TOPIC, jumps=1, candidates=1,
                client=client, feed=FakeFeed(), **shared,
            ))
            state_events = [e for e in second if e["stage"] == "state"]
            self.assertTrue(state_events, "the second mission reads the research state")
            self.assertFalse(
                state_events[0].get("known"),
                "SYNTHETIC support must not become known ground",
            )
            promo2 = [e for e in second if e["stage"] == "promotion"]
            self.assertTrue(promo2)
            self.assertEqual(promo2[0]["status"], "speculative")
            self.assertFalse(
                [e for e in first if e["stage"].startswith("heavy")],
                "SYNTHETIC support never triggers confirmation",
            )
            self.assertFalse(
                [e for e in second if e["stage"].startswith("heavy")],
                "nothing to confirm until a WORLD probe corroborates",
            )

    @needs_production_graph
    def test_reframe_cards_skip_the_graph_with_a_named_reason(self) -> None:
        events = run_all(
            jumps=1, candidates=1, reframes=1,
            client=SchemingClient(), feed=FakeFeed(),
        )
        reframe_verdicts = [
            e for e in events
            if e["stage"] == "verdict" and e.get("graph_skipped")
        ]
        self.assertEqual(len(reframe_verdicts), 1)
        verdict = reframe_verdicts[0]
        self.assertIn("does not answer", verdict["graph_skipped"])
        self.assertEqual(verdict["operator"], "assumption_removal")
        self.assertTrue(verdict["alive"])
        self.assertEqual(verdict["outcome"], "entered")
        self.assertIn("worst case analysis", verdict["pair"][1])

    def test_every_mission_logs_a_trajectory_for_the_policy_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.json"
            list(run_mission(
                TOPIC, jumps=1, candidates=1,
                client=SchemingClient(), feed=FakeFeed(),
                state_store=Path(tmp) / "st.json",
                policy_log=log, policy_file=Path(tmp) / "p.json",
                polish_rounds=0,
            ))
            from farfield.extras.routing import load_log

            missions = load_log(log)
            self.assertEqual(len(missions), 1)
            card = missions[0]["cards"][0]
            self.assertEqual(card["operator"], "directional")
            self.assertEqual(card["promoted"], "speculative")

    def test_a_workspace_freezes_every_live_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "mission"
            dest.mkdir()
            list(run_mission(
                TOPIC, jumps=1, candidates=1,
                client=SchemingClient(), feed=FakeFeed(),
                workspace=dest,
                state_store=Path(tmp) / "st.json",
                policy_log=Path(tmp) / "log.json",
                policy_file=Path(tmp) / "p.json",
                polish_rounds=0,
            ))
            log = dest / "evidence_snapshot" / "feed_log.jsonl"
            self.assertTrue(log.is_file())
            lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            methods = {row["method"] for row in lines}
            self.assertIn("recent_in_field", methods)
            self.assertIn("survey_around", methods)
            packet = dest / "RESEARCH_PACKET.md"
            self.assertTrue(packet.is_file())
            self.assertIn("推荐课题", packet.read_text(encoding="utf-8"))
            folders = [
                p
                for p in (dest / "candidates").iterdir()
                if p.is_dir() and p.name.startswith("gen_")
            ]
            self.assertTrue(folders)
            self.assertTrue((folders[0] / "protocol.md").is_file())
            self.assertTrue((folders[0] / "README.md").is_file())

    def test_a_restated_claim_stops_the_briefing(self) -> None:
        class CopycatFeed(FakeFeed):
            def survey_around(self, concept_a, concept_b, *, per_side=4, claim=""):
                return [
                    FreshWork(
                        title="The same claim already published",
                        published="2026-08-01",
                        arxiv_id="2608.00001v1",
                        abstract=claim or (
                            "combining succinct data structure with anything "
                            "reduces query time to at most O(log n) because "
                            "the index never rescans"
                        ),
                    )
                ]

        events = run_all(
            jumps=1, candidates=1, client=SchemingClient(), feed=CopycatFeed()
        )
        self.assertTrue(any(e["stage"] == "prior" and e.get("kills") for e in events))
        self.assertFalse(any(e["stage"] == "brief" for e in events))
        self.assertEqual(events[-1]["briefs"], 0)


class FailureMemoryTests(unittest.TestCase):
    def test_banked_rejections_ride_in_the_next_missions_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "state.json"
            isolated = {
                "state_store": store,
                "policy_log": Path(tmp) / "policy_log.json",
                "policy_file": Path(tmp) / "policy.json",
                "polish_rounds": 0,
            }
            probe = list(
                run_mission(
                    TOPIC, jumps=1, candidates=1,
                    client=SchemingClient(), **isolated,
                )
            )
            anchor = next(
                e for e in probe if e["stage"] == "anchor"
            )["anchors"][0]["concept"]
            record_rejections(
                store,
                "ds-arxiv-concepts-2026",
                anchor,
                [{"pair": ["alpha", "beta"], "killed_by": ["pair_not_already_combined"]}],
            )
            client = SchemingClient()
            events = list(
                run_mission(
                    TOPIC, jumps=1, candidates=1,
                    client=client, **isolated,
                )
            )
        state_event = next(e for e in events if e["stage"] == "state")
        self.assertIn("alpha x beta", state_event["rejected"])
        self.assertIn("alpha x beta", client.prompts[0])
        self.assertIn("do not repeat", client.prompts[0])


class ValueAndEvolveTests(unittest.TestCase):
    def test_value_ranks_live_cards_before_any_briefing(self) -> None:
        events = run_all(
            jumps=2, candidates=1, client=SchemingClient(), feed=FakeFeed()
        )
        stages = [event["stage"] for event in events]
        self.assertIn("value", stages)
        self.assertLess(stages.index("value"), stages.index("brief"))
        self.assertLess(stages.index("evidence"), stages.index("value"))
        ranking = next(e for e in events if e["stage"] == "value")["ranking"]
        self.assertTrue(ranking)
        self.assertIn("score", ranking[0])
        self.assertIn("archive", stages)

    def test_the_mission_does_not_open_a_crossover_generation(self) -> None:
        events = run_all(
            jumps=2,
            candidates=1,
            client=SchemingClient(),
            feed=FakeFeed(),
        )
        crosses = [
            e
            for e in events
            if e["stage"] == "jump" and e.get("operator") == "crossover"
        ]
        self.assertEqual(crosses, [])
        self.assertNotIn("evolve", events[0])

    def test_independent_cards_run_as_a_parallel_wave(self) -> None:
        events = run_all(
            jumps=2,
            candidates=1,
            parallel=4,
            client=SchemingClient(),
            feed=FakeFeed(),
        )
        stages = [event["stage"] for event in events]
        self.assertIn("parallel", stages)
        waves = [e["wave"] for e in events if e["stage"] == "parallel"]
        self.assertIn("generate", waves)
        self.assertIn("evidence", waves)
        last_verdict = max(i for i, stage in enumerate(stages) if stage == "verdict")
        first_evidence = stages.index("evidence")
        self.assertLess(last_verdict, first_evidence)

    def test_parallel_and_serial_emit_the_same_cards(self) -> None:
        kwargs = dict(jumps=2, candidates=1, client=SchemingClient(), feed=FakeFeed())
        serial = [e["card_id"] for e in run_all(parallel=1, **kwargs) if e["stage"] == "card"]
        parallel = [
            e["card_id"] for e in run_all(parallel=4, **kwargs) if e["stage"] == "card"
        ]
        self.assertEqual(serial, parallel)


class AutoExploreTests(unittest.TestCase):
    def test_a_live_line_stops_before_the_jump_cap(self) -> None:
        events = run_all(
            jumps=4,
            explore="auto",
            client=SchemingClient(),
            feed=FakeFeed(),
        )
        far = [
            e for e in events if e["stage"] == "jump" and e.get("track") == "farfield"
        ]
        self.assertEqual(len(far), 1)
        decisions = [e for e in events if e["stage"] == "explore_decision"]
        self.assertTrue(decisions)
        self.assertEqual(decisions[0]["reason"], "has_plan")
        self.assertFalse(decisions[0]["continue"])
        self.assertEqual(events[0]["explore"], "auto")
        self.assertEqual(events[-1]["jumps_opened"], 1)

    def test_a_live_far_line_does_not_spray_a_reframe(self) -> None:
        events = run_all(
            jumps=4,
            reframes=1,
            explore="auto",
            client=SchemingClient(),
            feed=FakeFeed(),
        )
        reframes = [
            e for e in events if e["stage"] == "jump" and e.get("track") == "reframe"
        ]
        self.assertEqual(reframes, [])

    def test_failed_generation_uses_the_full_cap(self) -> None:
        events = run_all(
            jumps=3,
            explore="auto",
            client=BabblingClient(),
        )
        far = [
            e for e in events if e["stage"] == "jump" and e.get("track") == "farfield"
        ]
        self.assertEqual(len(far), 3)
        decisions = [e for e in events if e["stage"] == "explore_decision"]
        self.assertEqual(decisions[-1]["reason"], "hit_cap")
        self.assertEqual(events[-1]["cards"], 0)

    def test_auto_polish_reviews_until_the_reviewer_or_the_cap(self) -> None:
        events = run_all(
            jumps=1,
            explore="fixed",
            polish_rounds=-1,
            client=SchemingClient(),
            feed=FakeFeed(),
        )
        reviews = [e for e in events if e["stage"] == "review"]
        self.assertEqual(len(reviews), 2)
        corpus = events[0]
        self.assertEqual(corpus["polish_mode"], "auto")
        self.assertEqual(corpus["polish_rounds"], 2)


EQUAL_ARMS = (
    "import json\n"
    "from pathlib import Path\n"
    "def measure(use_shortcut):\n"
    "    hops = 10\n"
    "    if use_shortcut:\n"
    "        hops = hops + 0\n"
    "    return hops\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}),"
    " encoding='utf-8')\n"
)

WEAKEN_ARMS = (
    "import json\n"
    "from pathlib import Path\n"
    "def measure(use_shortcut):\n"
    "    hops = 0\n"
    "    pos = 0\n"
    "    end = 8\n"
    "    while pos < end:\n"
    "        hops += 1\n"
    "        if use_shortcut:\n"
    "            pos += 1\n"
    "        else:\n"
    "            pos = end\n"
    "    return hops\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}),"
    " encoding='utf-8')\n"
)

CRASH_ARMS = (
    "import json\n"
    "from pathlib import Path\n"
    "def measure(use_shortcut):\n"
    "    hops = 8\n"
    "    if use_shortcut:\n"
    "        hops = hops - 7\n"
    "    return hops\n"
    "zero = 0\n"
    "unused = 1 / zero\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}),"
    " encoding='utf-8')\n"
)


class ProbeScriptClient(SchemingClient):
    """SchemingClient with a queue of probe sources, then the supporting default."""

    def __init__(self, sources: list[str]) -> None:
        super().__init__()
        self.sources = list(sources)

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        if "pre-registered two-arm experiment" in prompt and self.sources:
            source = self.sources.pop(0)
            self.prompts.append(prompt)
            return Completion(
                text=json.dumps(
                    {"measure": "operations counted", "source": source}
                ),
                model="fake-model",
                request_digest="req",
                digest=f"p{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        return super().complete(prompt, purpose=purpose, system=system, logprobs=logprobs)


class IdeaRefineMissionTests(unittest.TestCase):
    def test_a_text_gate_death_rewrites_the_same_pair_in_this_mission(self) -> None:
        events = run_all(
            jumps=1,
            client=ApplyThenRefineClient(),
            feed=FakeFeed(),
            idea_rounds=1,
        )
        decisions = [e for e in events if e["stage"] == "idea_decision"]
        self.assertTrue(decisions)
        self.assertEqual(decisions[0]["action"], "refine_idea")
        refining = [e for e in events if e["stage"] == "idea_refining"]
        self.assertTrue(refining)
        refined = [e for e in events if e["stage"] == "card" and e.get("refined")]
        self.assertTrue(refined)
        entered = [
            e
            for e in events
            if e["stage"] == "verdict" and e.get("refined") and e.get("alive")
        ]
        self.assertTrue(entered)

    @needs_production_graph
    def test_a_reframe_text_kill_rewrites_the_same_idea(self) -> None:
        events = run_all(
            jumps=0,
            reframes=1,
            explore="auto",
            client=ApplyThenRefineClient(),
            feed=FakeFeed(),
            idea_rounds=1,
        )
        refining = [e for e in events if e["stage"] == "idea_refining"]
        self.assertTrue(refining)
        self.assertEqual(refining[0]["track"], "reframe")
        entered = [
            e
            for e in events
            if e["stage"] == "verdict" and e.get("refined") and e.get("alive")
        ]
        self.assertTrue(entered)

    def test_the_next_jump_sees_this_jump_probe_in_h(self) -> None:
        client = ProbeScriptClient([WEAKEN_ARMS])
        events = run_all(
            jumps=2,
            explore="auto",
            client=client,
            feed=FakeFeed(),
            idea_rounds=0,
            experiment_rounds=0,
        )
        evidence = [e for e in events if e["stage"] == "evidence"]
        self.assertTrue(evidence)
        self.assertEqual(evidence[0]["verdict"], "weakens")
        far = [
            e for e in events if e["stage"] == "jump" and e.get("track") == "farfield"
        ]
        self.assertEqual(len(far), 2)
        generate_prompts = [
            p
            for p in client.prompts
            if "Seed concept:" in p and "Revise THIS idea" not in p
        ]
        self.assertGreaterEqual(len(generate_prompts), 2)
        self.assertIn("this-mission probes: weakens", generate_prompts[1])
        self.assertIn("weakened mechanism", generate_prompts[1])


class ExperimentIterationTests(unittest.TestCase):
    def test_auto_experiment_stops_when_the_first_probe_discriminates(self) -> None:
        events = run_all(
            jumps=1,
            client=SchemingClient(),
            feed=FakeFeed(),
            experiment_rounds=-1,
        )
        diagnoses = [e for e in events if e["stage"] == "diagnosis"]
        evidence = [e for e in events if e["stage"] == "evidence"]
        decisions = [e for e in events if e["stage"] == "experiment_decision"]
        self.assertEqual(len(diagnoses), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["verdict"], "supports")
        self.assertEqual(decisions[-1]["reason"], "informative")
        self.assertEqual(events[0]["experiment_mode"], "auto")
        self.assertEqual(events[0]["experiment_rounds"], 2)

    def test_an_uninformative_probe_earns_a_new_preregistration(self) -> None:
        client = ProbeScriptClient([EQUAL_ARMS])
        events = run_all(
            jumps=1,
            client=client,
            feed=FakeFeed(),
            experiment_rounds=1,
        )
        evidence = [e for e in events if e["stage"] == "evidence"]
        diagnoses = [e for e in events if e["stage"] == "diagnosis"]
        decisions = [e for e in events if e["stage"] == "experiment_decision"]
        self.assertEqual(len(diagnoses), 2)
        self.assertEqual(evidence[0]["verdict"], "uninformative")
        self.assertEqual(evidence[-1]["verdict"], "supports")
        self.assertEqual(decisions[0]["action"], "rewrite_diagnosis")
        self.assertEqual(decisions[-1]["reason"], "informative")
        redesign = next(
            p for p in client.prompts if "Do not repeat that design" in p
        )
        self.assertIn("chase a win", redesign)
        self.assertNotIn("10.0", redesign)
        note = next(e for e in events if e["stage"] == "note")
        self.assertIn("Failed experiment branches", note["protocol"])

    def test_a_weakening_verdict_does_not_redesign_to_hunt_a_support(self) -> None:
        events = run_all(
            jumps=1,
            client=ProbeScriptClient([WEAKEN_ARMS]),
            experiment_rounds=2,
        )
        evidence = [e for e in events if e["stage"] == "evidence"]
        diagnoses = [e for e in events if e["stage"] == "diagnosis"]
        decisions = [e for e in events if e["stage"] == "experiment_decision"]
        self.assertEqual(len(diagnoses), 1)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["verdict"], "weakens")
        self.assertEqual(decisions[-1]["reason"], "informative")
        self.assertFalse(any(e["stage"] == "brief" for e in events))

    def test_a_crash_rewrites_the_script_and_keeps_the_diagnosis(self) -> None:
        client = ProbeScriptClient([CRASH_ARMS])
        events = run_all(
            jumps=1,
            client=client,
            feed=FakeFeed(),
            experiment_rounds=1,
        )
        diagnoses = [e for e in events if e["stage"] == "diagnosis"]
        probes = [e for e in events if e["stage"] == "probe"]
        decisions = [e for e in events if e["stage"] == "experiment_decision"]
        self.assertEqual(len(diagnoses), 1)
        self.assertEqual(probes[0]["status"], "crashed")
        self.assertEqual(probes[-1]["status"], "ran")
        self.assertEqual(decisions[0]["action"], "rewrite_probe")
        rewrite = next(p for p in client.prompts if "implementation bug" in p)
        self.assertIn("ZeroDivision", rewrite)


WORLD_ARMS = (
    "import json\n"
    "from pathlib import Path\n"
    "world = json.loads(Path('data/path.json').read_text(encoding='utf-8'))\n"
    "nodes = list(world['nodes'])\n"
    "def measure(use_shortcut):\n"
    "    hops = 0\n"
    "    pos = 0\n"
    "    end = len(nodes) - 1\n"
    "    while pos < end:\n"
    "        hops += 1\n"
    "        if use_shortcut and pos == 0:\n"
    "            pos = end\n"
    "        else:\n"
    "            pos += 1\n"
    "    return hops\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}),"
    " encoding='utf-8')\n"
)


class WorldProbeClient(SchemingClient):
    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        if "pre-registered two-arm experiment" in prompt:
            self.prompts.append(prompt)
            return Completion(
                text=json.dumps(
                    {"measure": "hops on the frozen path", "source": WORLD_ARMS}
                ),
                model="fake-model",
                request_digest="req",
                digest=f"w{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        return super().complete(
            prompt, purpose=purpose, system=system, logprobs=logprobs
        )


FASTA_ARMS = (
    "import json\n"
    "from pathlib import Path\n"
    "seq = ''.join(line.strip() for line in Path('data/sequence.fasta')"
    ".read_text(encoding='utf-8').splitlines() if not line.startswith('>'))\n"
    "def measure(use_succinct):\n"
    "    k = 4 if use_succinct else 8\n"
    "    kmers = {seq[i:i+k] for i in range(len(seq)-k+1)}\n"
    "    return len(kmers) / max(1, len(seq))\n"
    "Path('metrics.json').write_text("
    "json.dumps({'treatment': float(measure(True)),"
    " 'control': float(measure(False))}),"
    " encoding='utf-8')\n"
)


class LeverBindingClient(SchemingClient):
    """Binds the mechanism to a runtime lever when the dynamics block asks,
    and probes the fasta world by actually reading it."""

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        if "name what would change our mind" in prompt and "world_lever" in prompt:
            self.prompts.append(prompt)
            return Completion(
                text=json.dumps(
                    {
                        "alternative": "the speedup comes from caching, not from the succinct encoding",
                        "experiment": "count distinct k-mers on the attested genome with the succinct window and the plain window",
                        "treatment_arm": "succinct 4-mer window on data/sequence.fasta",
                        "control_arm": "plain 8-mer window on the same bytes",
                        "expected_direction": "treatment_lower",
                        "alternative_direction": "treatment_higher",
                        "expected_if_alternative": "the two windows tie",
                        "margin": 0.05,
                        "margin_reason": "k-mer counts on a frozen genome have no noise",
                        "world_lever": "dropout",
                        "world_observable": "kmer8_diversity",
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"l{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        if "pre-registered two-arm experiment" in prompt:
            self.prompts.append(prompt)
            return Completion(
                text=json.dumps(
                    {"measure": "distinct k-mer fraction", "source": FASTA_ARMS}
                ),
                model="fake-model",
                request_digest="req",
                digest=f"f{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        return super().complete(
            prompt, purpose=purpose, system=system, logprobs=logprobs
        )


class AwakenedWorldTests(unittest.TestCase):
    """The bound world is forward-simulatable: a scout precedes the
    diagnosis, a forecast follows the registration, and a mechanism with
    no lever unbinds the world instead of riding its schema."""

    def test_scout_and_forecast_bracket_a_lever_bound_diagnosis(self) -> None:
        events = run_all(
            jumps=1,
            client=LeverBindingClient(),
            feed=FakeFeed(),
            world="phage-lambda",
        )
        stages = [e["stage"] for e in events]
        self.assertIn("world_scout", stages)
        self.assertIn("world_forecast", stages)
        self.assertLess(stages.index("world_scout"), stages.index("diagnosis"))
        self.assertLess(stages.index("diagnosis"), stages.index("world_forecast"))
        scout = events[stages.index("world_scout")]
        self.assertEqual(scout["world_id"], "phage-lambda")
        self.assertIn("dropout", scout["levers"])
        self.assertTrue(scout["report_digest"])
        forecast = events[stages.index("world_forecast")]
        self.assertEqual(forecast["lever"], "dropout")
        self.assertEqual(forecast["observable"], "kmer8_diversity")
        self.assertIn(
            forecast["simulated_direction"],
            ("treatment_lower", "treatment_higher", "flat"),
        )
        probes = [e for e in events if e["stage"] == "probe"]
        self.assertTrue(probes)
        self.assertEqual(probes[-1]["kind"], "WORLD")
        self.assertNotIn("world_surrogate", stages)

    def test_a_scenery_kmer_window_cannot_corroborate(self) -> None:
        # The k-mer window is a legal fasta lever-adjacent experiment
        # that does not consume the genome: the same numbers appear on a
        # shuffled copy. Honest bookkeeping of a wrong object must not
        # climb.
        events = run_all(
            jumps=1,
            client=LeverBindingClient(),
            feed=FakeFeed(),
            world="phage-lambda",
        )
        placebos = [e for e in events if e["stage"] == "world_placebo"]
        self.assertTrue(placebos)
        self.assertFalse(placebos[-1]["consumed"])
        evidence = [e for e in events if e["stage"] == "evidence"]
        self.assertTrue(evidence)
        self.assertEqual(evidence[-1]["verdict"], "uninformative")
        for promo in (e for e in events if e["stage"] == "promotion"):
            self.assertNotIn(promo["status"], ("corroborated", "verified"))

    def test_an_inert_forecast_rewrites_instead_of_burning_a_probe(self) -> None:
        class InertForecastClient(LeverBindingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                if "name what would change our mind" in prompt and "world_lever" in prompt:
                    self.prompts.append(prompt)
                    return Completion(
                        text=json.dumps(
                            {
                                "alternative": "reachability is an encoding artefact",
                                "experiment": "rewire the attested automaton and recount reachable states",
                                "treatment_arm": "rewired transitions",
                                "control_arm": "frozen transitions",
                                "expected_direction": "treatment_lower",
                                "alternative_direction": "treatment_higher",
                                "expected_if_alternative": "the arms tie",
                                "margin": 0.05,
                                "margin_reason": "a dense automaton barely moves under rewire",
                                "world_lever": "rewire",
                                "world_observable": "reachable_fraction",
                            }
                        ),
                        model="fake-model",
                        request_digest="req",
                        digest=f"i{len(self.prompts):015x}",
                        artifact_uri="file:///dev/null",
                        finish_reason="stop",
                        prompt_tokens=10,
                        completion_tokens=20,
                        reasoning_tokens=0,
                        mode="replay",
                    )
                return super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )

        events = run_all(
            jumps=1,
            client=InertForecastClient(),
            feed=FakeFeed(),
            world="tcp-linux-server",
        )
        stages = [e["stage"] for e in events]
        self.assertIn("world_forecast", stages)
        forecast = events[stages.index("world_forecast")]
        self.assertEqual(forecast["simulated_direction"], "flat")
        self.assertTrue(forecast.get("rewrite"))
        self.assertFalse([e for e in events if e["stage"] == "probe"])
        decisions = [e for e in events if e["stage"] == "experiment_decision"]
        self.assertTrue(decisions)
        self.assertIn("inert", decisions[0].get("reason", ""))

    def test_protocol_is_registered_before_the_first_probe(self) -> None:
        from farfield.extras import chain

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "mission"
            events = run_all(
                jumps=1,
                client=LeverBindingClient(),
                feed=FakeFeed(),
                world="phage-lambda",
                workspace=workspace,
            )
            probes = [e for e in events if e["stage"] == "probe"]
            self.assertTrue(probes)
            card_id = probes[0]["card_id"]
            ledger = workspace / "candidates" / card_id / "chain.jsonl"
            self.assertTrue(ledger.is_file())
            kinds = [row["kind"] for row in chain.read_events(ledger)]
            self.assertIn(chain.REGISTER_PROTOCOL, kinds)
            self.assertLess(
                kinds.index(chain.REGISTER_PROTOCOL),
                kinds.index(chain.EXECUTE_PLACEBO)
                if chain.EXECUTE_PLACEBO in kinds
                else len(kinds),
            )

    def test_lineage_unbinds_a_lexically_bound_novel(self) -> None:
        class LineageAgentClient(SchemingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                if "how THIS CLAIM'S object developed" in prompt:
                    self.prompts.append(prompt)
                    return Completion(
                        text=json.dumps(
                            {
                                "lineage": (
                                    "agent-tool papers moved from traces to "
                                    "distillation; the object is the tool log"
                                ),
                                "named_instance": (
                                    "agent tool traces from the retrieved papers"
                                ),
                                "object_type": "stream",
                                "schema": "text_stream",
                                "cite_ids": ["2608.09999v1"],
                                "why_this_object": (
                                    "the claim is about tool traces, not a novel"
                                ),
                                "freeze_source": "a public agent-tool log",
                                "freeze_url": "",
                                "freeze_schema": "text_stream",
                            }
                        ),
                        model="fake-model",
                        request_digest="req",
                        digest=f"n{len(self.prompts):015x}",
                        artifact_uri="file:///dev/null",
                        finish_reason="stop",
                        prompt_tokens=10,
                        completion_tokens=20,
                        reasoning_tokens=0,
                        mode="replay",
                    )
                return super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )

        events = run_all(
            jumps=1,
            client=LineageAgentClient(),
            feed=FakeFeed(),
            world="gutenberg-pride",
        )
        stages = [e["stage"] for e in events]
        self.assertIn("world_path", stages)
        self.assertIn("world_surrogate", stages)
        self.assertLess(stages.index("world_path"), stages.index("world_surrogate"))
        for probe in (e for e in events if e["stage"] == "probe"):
            self.assertNotEqual(probe.get("kind"), "WORLD")

    def test_a_mechanism_with_no_lever_unbinds_the_world(self) -> None:
        # SchemingClient's diagnosis never names a lever: with a dynamics
        # vocabulary on the table that is the honest "none", and the world
        # must be unbound rather than ridden as a surrogate.
        events = run_all(
            jumps=1,
            client=SchemingClient(),
            feed=FakeFeed(),
            world="phage-lambda",
        )
        stages = [e["stage"] for e in events]
        self.assertIn("world_scout", stages)
        self.assertIn("world_surrogate", stages)
        self.assertLess(stages.index("diagnosis"), stages.index("world_surrogate"))
        for probe in (e for e in events if e["stage"] == "probe"):
            self.assertNotEqual(probe.get("kind"), "WORLD")
        for promo in (e for e in events if e["stage"] == "promotion"):
            self.assertNotIn(promo["status"], ("corroborated", "verified"))


class WorldLadderTests(unittest.TestCase):
    def test_a_world_probe_can_corroborate(self) -> None:
        events = run_all(
            jumps=1,
            client=WorldProbeClient(),
            feed=FakeFeed(),
            world="path-trace",
        )
        probes = [e for e in events if e["stage"] == "probe"]
        self.assertTrue(probes)
        self.assertEqual(probes[-1]["kind"], "WORLD")
        promo = [e for e in events if e["stage"] == "promotion"]
        self.assertTrue(promo)
        self.assertEqual(promo[0]["status"], "speculative")
        self.assertIn("pending host", promo[0]["reason"])
        self.assertEqual(events[0]["world"]["id"], "path-trace")

    def test_a_missing_world_blocks(self) -> None:
        events = run_all(jumps=1, client=SchemingClient(), world="no-such-world")
        self.assertEqual(events[-1]["stage"], "blocked")
        self.assertEqual(events[-1]["missing_capability"], "attested_world")

    def test_auto_world_binds_a_real_fixture_for_the_topic(self) -> None:
        events = run_all(
            jumps=1,
            client=SchemingClient(),
            feed=FakeFeed(),
            world="auto",
        )
        self.assertEqual(events[0]["world"]["id"], "phage-lambda")
        self.assertEqual(events[0]["world"]["role"], "world")

    def test_an_smt_claim_binds_the_frozen_automaton_under_auto(self) -> None:
        import dataclasses

        class SmtClient(SchemingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                done = super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )
                if "Seed concept:" not in prompt:
                    return done
                try:
                    payload = json.loads(done.text)
                except json.JSONDecodeError:
                    return done
                if "claim" not in payload:
                    return done
                payload["claim"] = (
                    str(payload["claim"])
                    + "; formal verification of smt solver encodings"
                )
                payload["mechanism"] = (
                    str(payload.get("mechanism") or "")
                    + " via an smt solver"
                )
                return dataclasses.replace(done, text=json.dumps(payload))

        events = run_all(
            jumps=1,
            client=SmtClient(),
            feed=FakeFeed(),
            world="auto",
        )
        # With worlds/tcp-linux-server frozen, a formula claim is no
        # longer WORLD_INCOMPATIBLE: it binds the attested trace.
        incompatible = [e for e in events if e["stage"] == "world_incompatible"]
        self.assertFalse(incompatible)

    def test_an_external_memory_claim_is_world_incompatible_under_auto(self) -> None:
        import dataclasses

        class IoClient(SchemingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                done = super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )
                if "Seed concept:" not in prompt:
                    return done
                try:
                    payload = json.loads(done.text)
                except json.JSONDecodeError:
                    return done
                if "claim" not in payload:
                    return done
                payload["claim"] = (
                    str(payload["claim"])
                    + "; external memory search trees with i/o complexity bounds"
                )
                payload["mechanism"] = (
                    str(payload.get("mechanism") or "")
                    + " via a disk-based buffer pool"
                )
                return dataclasses.replace(done, text=json.dumps(payload))

        events = run_all(
            jumps=1,
            client=IoClient(),
            feed=FakeFeed(),
            world="auto",
        )
        incompatible = [e for e in events if e["stage"] == "world_incompatible"]
        self.assertTrue(incompatible)
        generated = [
            e
            for e in events
            if e["stage"] in {"world_constructed", "world_generated"}
        ]
        self.assertTrue(generated)
        stages = [e["stage"] for e in events]
        self.assertLess(stages.index("world_incompatible"), stages.index("diagnosis"))
        constructed_at = next(
            i
            for i, stage in enumerate(stages)
            if stage in {"world_constructed", "world_generated"}
        )
        self.assertLess(stages.index("diagnosis"), constructed_at)
        for probe in [e for e in events if e["stage"] == "probe"]:
            self.assertNotEqual(probe.get("kind"), "WORLD")
        promo = [e for e in events if e["stage"] == "promotion"]
        self.assertTrue(promo)
        self.assertNotIn(promo[0]["status"], ("corroborated", "verified"))

    def test_retrieved_papers_ground_a_constructed_world_lineage(self) -> None:
        import dataclasses

        class LineageIoClient(SchemingClient):
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                if "how THIS CLAIM'S object developed" in prompt:
                    self.prompts.append(prompt)
                    return Completion(
                        text=json.dumps(
                            {
                                "lineage": (
                                    "external-memory papers moved from I/O "
                                    "lower bounds to buffer-pool indexes; the "
                                    "open question is still disk-block cost"
                                ),
                                "named_instance": "a buffer-pool B-tree as in the retrieved preprint",
                                "object_type": "io",
                                "schema": "",
                                "cite_ids": ["2608.09999v1"],
                                "why_this_object": (
                                    "the claim is about I/O complexity, not "
                                    "about a social graph stand-in"
                                ),
                                "freeze_source": "a public B-tree trace named in the preprint",
                                "freeze_url": "",
                                "freeze_schema": "",
                            }
                        ),
                        model="fake-model",
                        request_digest="req",
                        digest=f"y{len(self.prompts):015x}",
                        artifact_uri="file:///dev/null",
                        finish_reason="stop",
                        prompt_tokens=10,
                        completion_tokens=20,
                        reasoning_tokens=0,
                        mode="replay",
                    )
                done = super().complete(
                    prompt, purpose=purpose, system=system, logprobs=logprobs
                )
                if "Seed concept:" not in prompt:
                    return done
                try:
                    payload = json.loads(done.text)
                except json.JSONDecodeError:
                    return done
                if "claim" not in payload:
                    return done
                payload["claim"] = (
                    str(payload["claim"])
                    + "; external memory search trees with i/o complexity bounds"
                )
                payload["mechanism"] = (
                    str(payload.get("mechanism") or "")
                    + " via a disk-based buffer pool"
                )
                return dataclasses.replace(done, text=json.dumps(payload))

        events = run_all(
            jumps=1,
            client=LineageIoClient(),
            feed=FakeFeed(),
            world="auto",
        )
        paths = [e for e in events if e["stage"] == "world_path"]
        self.assertTrue(paths)
        self.assertEqual(paths[0]["cite_ids"], ["2608.09999v1"])
        self.assertIn("buffer-pool", paths[0]["named_instance"])
        constructed = [e for e in events if e["stage"] == "world_constructed"]
        self.assertTrue(constructed)
        self.assertEqual(
            constructed[0]["lineage"]["cite_ids"], ["2608.09999v1"]
        )
        for probe in [e for e in events if e["stage"] == "probe"]:
            self.assertNotEqual(probe.get("kind"), "WORLD")

    def test_each_entered_card_owns_a_candidate_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "mission"
            events = run_all(
                jumps=2,
                client=SchemingClient(),
                feed=FakeFeed(),
                workspace=workspace,
            )
            done = events[-1]
            self.assertEqual(done["stage"], "done")
            self.assertIn("funnel", done)
            cards = [e["card_id"] for e in events if e["stage"] == "note"]
            self.assertTrue(cards)
            for card_id in set(cards):
                folder = workspace / "candidates" / card_id
                self.assertTrue(folder.is_dir(), card_id)
                self.assertTrue((folder / "verdict" / "COMPLETE.json").is_file())


FLAGS_CACHED = '''
def check(card):
    return "cached boundaries" not in str(card["mechanism"]).lower()
'''


class ShadowBenchMissionTests(unittest.TestCase):
    """W3 in the stream: a shadow judge watches every card, kills none,
    and becomes lethal only through the held-out settlement."""

    def test_a_shadow_judge_flags_but_cannot_kill(self) -> None:
        from farfield.extras.verifier import add_shadow

        with tempfile.TemporaryDirectory() as tmp:
            judges = Path(tmp) / "judges.json"
            add_shadow(
                judges,
                name="no_cached_boundaries",
                source=FLAGS_CACHED,
                provenance="test",
                rationale="a cache is not a mechanism",
            )
            events = run_all(
                jumps=1, candidates=1, client=SchemingClient(), judges_file=judges
            )
        corpus = events[0]
        self.assertIn("no_cached_boundaries", corpus["shadow_judges"])
        self.assertNotIn("no_cached_boundaries", corpus["judges"])
        verdicts = [e for e in events if e["stage"] == "verdict"]
        flagged = [
            v for v in verdicts if "no_cached_boundaries" in v.get("shadow_kills", [])
        ]
        self.assertTrue(flagged, "the scheming card says 'cached boundaries'")
        for verdict in flagged:
            self.assertNotIn("no_cached_boundaries", verdict["text_kills"])

    def test_heldout_world_catches_promote_the_judge_to_lethal(self) -> None:
        from farfield.extras.routing import append_mission
        from farfield.extras.verifier import add_shadow, load_bench

        with tempfile.TemporaryDirectory() as tmp:
            judges = Path(tmp) / "judges.json"
            log = Path(tmp) / "policy_log.json"
            add_shadow(
                judges,
                name="no_cached_boundaries",
                source=FLAGS_CACHED,
                provenance="test",
            )
            catch = {
                "operator": "analogy",
                "track": "farfield",
                "killed": False,
                "killed_by": [],
                "shadow_kills": ["no_cached_boundaries"],
                "verdict": "weakens",
                "probe_kind": "WORLD",
            }
            for _ in range(6):
                append_mission(log, {"cards": [catch], "shadow_errors": []})
            events = run_all(
                jumps=1,
                candidates=1,
                client=SchemingClient(),
                judges_file=judges,
                policy_log=log,
            )
            settlements = [e for e in events if e["stage"] == "verifier"]
            self.assertTrue(settlements)
            self.assertEqual(
                settlements[0]["promoted"], ["no_cached_boundaries"]
            )
            bench = load_bench(judges)
            self.assertEqual(bench["shadow"], [])
            self.assertEqual(bench["lethal"][0]["name"], "no_cached_boundaries")

    def test_a_promoted_judge_kills_on_the_next_mission(self) -> None:
        from farfield.extras.verifier import load_bench, save_bench

        with tempfile.TemporaryDirectory() as tmp:
            judges = Path(tmp) / "judges.json"
            save_bench(
                judges,
                {
                    "shadow": [],
                    "lethal": [
                        {
                            "name": "no_cached_boundaries",
                            "source": FLAGS_CACHED,
                            "provenance": "test",
                            "rationale": "a cache is not a mechanism",
                        }
                    ],
                },
            )
            events = run_all(
                jumps=1, candidates=1, client=SchemingClient(), judges_file=judges
            )
            self.assertTrue(load_bench(judges)["lethal"])
        corpus = events[0]
        self.assertIn("no_cached_boundaries", corpus["judges"])
        verdicts = [e for e in events if e["stage"] == "verdict"]
        self.assertTrue(
            any("no_cached_boundaries" in v["text_kills"] for v in verdicts),
            "an earned lethal judge rules like an installed one",
        )


class ProposingClient(ProbeScriptClient):
    """ProbeScriptClient that also answers the shadow-judge proposal."""

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        if "one executable review predicate" in prompt:
            self.prompts.append(prompt)
            return Completion(
                text=json.dumps(
                    {
                        "name": "no_cached_boundaries",
                        "source": FLAGS_CACHED,
                        "rationale": "a cache is not a mechanism",
                    }
                ),
                model="fake-model",
                request_digest="req",
                digest=f"s{len(self.prompts):015x}",
                artifact_uri="file:///dev/null",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=20,
                reasoning_tokens=0,
                mode="replay",
            )
        return super().complete(
            prompt, purpose=purpose, system=system, logprobs=logprobs
        )


class ShadowProposalMissionTests(unittest.TestCase):
    """F2 in the stream: repeated evidence-beaten cards earn one predicate
    proposal onto the shadow bench, where it has no power."""

    def test_two_beaten_cards_earn_one_shadow_proposal(self) -> None:
        from farfield.extras.verifier import load_bench

        with tempfile.TemporaryDirectory() as tmp:
            judges = Path(tmp) / "judges.json"
            events = run_all(
                jumps=2,
                candidates=1,
                client=ProposingClient([WEAKEN_ARMS, WEAKEN_ARMS]),
                judges_file=judges,
            )
            proposals = [e for e in events if e["stage"] == "shadow_judge"]
            self.assertEqual(len(proposals), 1)
            self.assertTrue(proposals[0]["admitted"])
            self.assertEqual(proposals[0]["name"], "no_cached_boundaries")
            bench = load_bench(judges)
            self.assertEqual(bench["shadow"][0]["name"], "no_cached_boundaries")
            self.assertEqual(bench["lethal"], [])

    def test_one_beaten_card_proposes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            judges = Path(tmp) / "judges.json"
            events = run_all(
                jumps=1,
                candidates=1,
                client=ProposingClient([WEAKEN_ARMS]),
                judges_file=judges,
            )
            self.assertFalse(
                [e for e in events if e["stage"] == "shadow_judge"],
                "one failure is an anecdote, not a class",
            )
            self.assertFalse(judges.is_file())


class BaselineStakeMissionTests(unittest.TestCase):
    """W4 in the stream: the same-anchor control stake is measured every
    mission, banked in the log, and kept out of ranking and state."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.events = run_all(jumps=2, candidates=1, client=SchemingClient())

    def test_the_stake_is_emitted_once_with_both_arms(self) -> None:
        stakes = [e for e in self.events if e["stage"] == "baseline"]
        self.assertEqual(len(stakes), 1)
        stake = stakes[0]
        self.assertEqual(stake["arm"]["n"], 2)
        self.assertGreater(stake["control"]["n"], 0)
        self.assertIn("Not a verdict", stake["note"])

    def test_the_stake_never_reaches_the_ranking(self) -> None:
        stages = [e["stage"] for e in self.events]
        if "ranked" in stages:
            self.assertLess(stages.index("baseline"), stages.index("ranked"))
            for idea in self.events[stages.index("ranked")]["ideas"]:
                self.assertNotIn("baseline", idea)

    def test_the_stake_is_banked_in_the_trajectory_log(self) -> None:
        from farfield.extras.routing import load_log

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "policy_log.json"
            run_all(
                jumps=1, candidates=1, client=SchemingClient(), policy_log=log
            )
            missions = load_log(log)
        self.assertEqual(len(missions), 1)
        stake = missions[0].get("baseline")
        self.assertIsNotNone(stake)
        self.assertIn("arm", stake)
        self.assertIn("control", stake)


class ShippedCorpusTests(unittest.TestCase):
    """A GitHub clone does not contain the 100 MB+ production graphs."""

    def setUp(self) -> None:
        from farfield.extras import mission as mission_mod

        self.mod = mission_mod
        self.saved = dict(mission_mod._ASSETS)
        mission_mod._ASSETS.clear()

    def tearDown(self) -> None:
        self.mod._ASSETS.clear()
        self.mod._ASSETS.update(self.saved)

    def test_clone_without_production_graph_runs_on_shipped_slice(self) -> None:
        from unittest.mock import patch

        real_is_file = Path.is_file

        def hide_production(self: Path) -> bool:
            if "ds-arxiv-concepts-2026" in str(self) and self.name.startswith("G_"):
                return False
            return real_is_file(self)

        with patch.object(Path, "is_file", hide_production):
            assets = load_assets()
        self.assertIn("attn-concepts-s1", assets.corpus_uid)


if __name__ == "__main__":
    unittest.main()

"""Attested field paths: refuse a probe that looks a claimed field up on the wrong object."""

from __future__ import annotations

import unittest

from farfield.extras.claimspec import ContractViolation
from farfield.extras.worldfields import (
    diagnosis_names_claimed_fields,
    missing_attested_field,
    paper_on_claim_object,
    papers_on_claim_object,
    probe_reads_attested_fields,
    probe_reads_bound_layout,
)
from farfield.extras.probeexp import ProbeRefused, spec_from_payload


CLAIM = (
    "On Live-SWE-agent trajectories, created_tools events forecast "
    "next-step nonzero returncode."
)
STEP_SCRIPT = """
import json
from pathlib import Path

def measure(on):
    payload = json.loads(Path('data/world.json').read_text())
    n = 0
    for trace in payload['traces']:
        for step in trace['steps']:
            n += int(bool(step.get('created_tools')))
    return float(n if on else n + 1)

Path('metrics.json').write_text(
    json.dumps({'treatment': measure(True), 'control': measure(False)})
)
"""
TRACE_SCRIPT = """
import json
from pathlib import Path

def measure(on):
    payload = json.loads(Path('data/world.json').read_text())
    n = 0
    for trace in payload['traces']:
        n += len(trace.get('created_tools') or [])
    return float(n if on else 0)

Path('metrics.json').write_text(
    json.dumps({'treatment': measure(True), 'control': measure(False)})
)
"""


class MissingFieldTests(unittest.TestCase):
    def test_duration_claim_on_labeled_traces_is_refused(self) -> None:
        reason = missing_attested_field(
            "job processing time tertile predicts created_tool events",
            schema="labeled_traces",
        )
        self.assertIsNotNone(reason)
        self.assertIn("does not attest", reason or "")

    def test_returncode_claim_is_not_a_missing_field(self) -> None:
        self.assertIsNone(
            missing_attested_field(
                "next-step returncode failure rate after created_tools",
                schema="labeled_traces",
            )
        )

    def test_duration_with_honest_none_is_not_a_two_arm(self) -> None:
        self.assertIsNone(
            missing_attested_field(
                "job processing time tertile predicts created_tool events",
                schema="labeled_traces",
                world_lever="none",
            )
        )

    def test_empty_schema_is_a_noop(self) -> None:
        self.assertIsNone(
            missing_attested_field("job processing time", schema="")
        )


JACCARD_SCRIPT = """
import json
from pathlib import Path

def measure(on):
    payload = json.loads(Path('data/world.json').read_text())
    score = 0.0
    for trace in payload['traces']:
        tools = trace.get('created_tools') or []
        for step in trace.get('steps') or []:
            tokens = set(str(step.get('action') or '').split())
            score += len(tokens & set(map(str, tools)))
    return float(score if on else 0)

Path('metrics.json').write_text(
    json.dumps({'treatment': measure(True), 'control': measure(False)})
)
"""


class ProbeFieldTests(unittest.TestCase):
    def test_step_lookup_of_created_tools_is_refused(self) -> None:
        reason = probe_reads_attested_fields(
            STEP_SCRIPT, claim=CLAIM, schema="labeled_traces"
        )
        self.assertIsNotNone(reason)
        self.assertIn("trace", reason or "")

    def test_trace_lookup_is_admitted(self) -> None:
        self.assertIsNone(
            probe_reads_attested_fields(
                TRACE_SCRIPT, claim=CLAIM, schema="labeled_traces"
            )
        )

    def test_tokenizing_created_tools_without_name_or_len_is_refused(self) -> None:
        reason = probe_reads_attested_fields(
            JACCARD_SCRIPT, claim=CLAIM, schema="labeled_traces"
        )
        self.assertIsNotNone(reason)
        self.assertIn("name", reason or "")

    def test_spec_from_payload_refuses_the_step_script(self) -> None:
        with self.assertRaises(ProbeRefused) as ctx:
            spec_from_payload(
                {
                    "measure": "created_tools count on attested traces",
                    "source": STEP_SCRIPT,
                },
                claim=CLAIM,
                schema="labeled_traces",
            )
        self.assertIn("created_tools", str(ctx.exception).lower())


class DiagnosisFieldTests(unittest.TestCase):
    def test_a_created_tools_claim_cannot_drop_the_field(self) -> None:
        reason = diagnosis_names_claimed_fields(
            experiment="compare returncode 2-gram log loss",
            treatment="fit returncode 2-gram",
            control="fit returncode unigram",
            claim=CLAIM,
            schema="labeled_traces",
        )
        self.assertIsNotNone(reason)
        self.assertIn("created_tools", reason or "")

    def test_a_design_that_names_created_tools_is_admitted(self) -> None:
        self.assertIsNone(
            diagnosis_names_claimed_fields(
                experiment="created_tools count vs shuffled created_tools",
                treatment="read trace created_tools",
                control="shuffle created_tools on the same traces",
                claim=CLAIM,
                schema="labeled_traces",
            )
        )


class PaperFilterTests(unittest.TestCase):
    def test_drought_index_paper_does_not_keep_a_swe_claim(self) -> None:
        self.assertFalse(
            paper_on_claim_object(
                "A Multiscalar Drought Index",
                "The SPEI is based on precipitation and temperature.",
                CLAIM,
                topic="code world model of Live-SWE-agent tool-call trajectories",
            )
        )

    def test_swe_agent_paper_is_kept(self) -> None:
        self.assertTrue(
            paper_on_claim_object(
                "Live-SWE-agent tool-call trajectories",
                "We study created_tools on SWE-bench execution traces.",
                CLAIM,
                topic="code world model of Live-SWE-agent tool-call trajectories",
            )
        )

    def test_filter_drops_the_unrelated_row(self) -> None:
        kept = papers_on_claim_object(
            [
                {
                    "title": "A Multiscalar Drought Index",
                    "abstract": "precipitation and temperature",
                },
                {
                    "title": "Live-SWE-agent tool-call traces",
                    "abstract": "created_tools on SWE-bench",
                },
            ],
            claim=CLAIM,
            topic="code world model of Live-SWE-agent tool-call trajectories",
        )
        self.assertEqual(len(kept), 1)
        self.assertIn("Live-SWE", kept[0]["title"])

    def test_sharing_world_alone_is_not_on_the_claim_object(self) -> None:
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        self.assertFalse(
            paper_on_claim_object(
                "Obesity prevalence in 1975",
                "A world health organization report on obesity.",
                "code world models of executable program state",
                topic=topic,
            )
        )
        self.assertTrue(
            paper_on_claim_object(
                "Code world models of executable program state",
                "We replay self-modifying updates against a frozen validator.",
                "code world models of executable program state",
                topic=topic,
            )
        )

    def test_empty_claim_is_a_contract_violation(self) -> None:
        with self.assertRaises(ContractViolation):
            papers_on_claim_object(
                [{"title": "Live-SWE-agent tool-call traces", "abstract": "created_tools"}],
                claim="",
                topic="code world model of Live-SWE-agent tool-call trajectories",
            )

    def test_strong_retrieve_is_not_padded_with_weak(self) -> None:
        kept = papers_on_claim_object(
            [
                {
                    "title": "Live-SWE-agent tool-call traces",
                    "abstract": "We study created_tools on SWE-bench execution traces.",
                },
                {
                    "title": "Something about tool-call",
                    "abstract": "We study tool-call trajectories in general.",
                },
            ],
            claim=CLAIM,
            topic="code world model of Live-SWE-agent tool-call trajectories",
            minimum="strong",
        )
        self.assertEqual(len(kept), 1)
        self.assertIn("Live-SWE", kept[0]["title"])


AUTOMATON_ON_PROGRAM_STATE = """
import json
from pathlib import Path

def measure(on):
    payload = json.loads(Path('data/world.json').read_text())
    n = 0
    for state in payload['states']:
        n += 1
    for edge in payload.get('transitions', []):
        n += int(bool(on))
    return float(n)

Path('metrics.json').write_text(
    json.dumps({'treatment': measure(True), 'control': measure(False)})
)
"""

PROGRAM_STATE_SCRIPT = """
import json
from pathlib import Path

def measure(on):
    payload = json.loads(Path('data/world.json').read_text())
    validators = {v['id']: set(v['reads']) for v in payload['validators']}
    accepted = 0
    false_accept = 0
    for update in payload['updates']:
        cycle = any(
            set(update['writes']) & validators.get(vid, set())
            for vid in update.get('validator_writes', [])
        )
        if on and cycle:
            continue
        if update['accepted']:
            accepted += 1
            if update['divergent']:
                false_accept += 1
    return float(false_accept) / float(accepted or 1)

Path('metrics.json').write_text(
    json.dumps({'treatment': measure(True), 'control': measure(False)})
)
"""


class BoundLayoutTests(unittest.TestCase):
    """A probe must read the bound schema's own keys, not another world's."""

    def test_layout_block_survives_the_template_format_in_both_prompters(self) -> None:
        # The record shapes contain literal braces; the block is spliced into
        # a template that is `.format()`-ed afterwards, so a raw splice raised
        # KeyError('id, reads') and killed every constructed-world diagnosis.
        from farfield.extras import diagnose, probeexp

        for module in (diagnose, probeexp):
            block = module._layout_block("program_state")
            rendered = ("{x} " + block).format(x="ok")
            self.assertTrue(rendered.startswith("ok "))
            self.assertIn("updates", rendered)
            self.assertIn("{", rendered)
            self.assertNotIn("{{", rendered)

    def test_states_and_transitions_on_a_program_state_world_are_refused(self) -> None:
        reason = probe_reads_bound_layout(AUTOMATON_ON_PROGRAM_STATE, "program_state")
        self.assertIsNotNone(reason)
        self.assertIn("symbolic_trace", reason)
        self.assertIn("updates", reason)

    def test_reading_the_bound_keys_is_accepted(self) -> None:
        self.assertIsNone(probe_reads_bound_layout(PROGRAM_STATE_SCRIPT, "program_state"))
        self.assertIsNone(probe_reads_bound_layout(TRACE_SCRIPT, "labeled_traces"))

    def test_unlisted_or_empty_schema_is_a_no_op(self) -> None:
        self.assertIsNone(probe_reads_bound_layout(AUTOMATON_ON_PROGRAM_STATE, ""))
        self.assertIsNone(probe_reads_bound_layout(AUTOMATON_ON_PROGRAM_STATE, "fasta"))
        self.assertIsNone(probe_reads_bound_layout("x = 1", "program_state"))

    def test_spec_from_payload_refuses_before_the_sandbox(self) -> None:
        with self.assertRaises(ProbeRefused) as caught:
            spec_from_payload(
                {"measure": "false_accept_fraction", "source": AUTOMATON_ON_PROGRAM_STATE},
                schema="program_state",
            )
        self.assertIn("program_state", caught.exception.record.unlock_condition)
        spec = spec_from_payload(
            {"measure": "false_accept_fraction", "source": PROGRAM_STATE_SCRIPT},
            schema="program_state",
        )
        self.assertEqual(spec.measure, "false_accept_fraction")


if __name__ == "__main__":
    unittest.main()

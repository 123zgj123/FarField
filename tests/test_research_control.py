"""Controller integration tests; worker payloads here are explicit test doubles."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from farfield.extras.openworld.runtime import bootstrap_research
from farfield.extras.openworld.actions import ActionInstance, SPECS, SURVEY, THEORIZE
from farfield.extras.openworld.kernel import CandidateResult
from farfield.extras.openworld.events import make_event
from farfield.extras.openworld.reducer import rebuild_state, state_digest


class RecordingWorkers:
    def __init__(self):
        self.calls = []

    def executors(self):
        return {SURVEY: self.survey}

    def survey(self, action, state, **context):
        self.calls.append((state['goal'], context['workspace']))
        return CandidateResult(status='surveyed', events=[make_event('ArtifactAdded', {
            'id': 'survey-test', 'kind': 'literature_survey', 'question_id': 'Q1',
            'works': [{'title': 'Explicit replay fixture', 'work_id': 'test-only'}],
        })])


class ResearchControlTests(unittest.TestCase):
    def test_resume_rebuilds_derived_snapshot_from_unchanged_events(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            env = bootstrap_research('iris observations', workspace, world='fisher-iris')
            from farfield.extras.openworld.actions import OBSERVE, VERIFY
            env.execute(next(a for a in env.eligible() if a.action_type == OBSERVE))
            eid = env.state.evidence_records[0]['evidence_id']
            env.execute(next(a for a in env.eligible() if a.action_type == VERIFY))
            env.persist()
            before = (workspace / 'SCIENTIFIC_EVENTS.json').read_bytes()
            snapshot = workspace / 'SCIENTIFIC_STATE.json'
            data = json.loads(snapshot.read_text())
            data['evidence_records'][0].pop('verification_event_id', None)
            snapshot.write_text(json.dumps(data))
            restored = bootstrap_research('iris observations', workspace, world='fisher-iris')
            self.assertEqual(state_digest(restored.state), state_digest(rebuild_state(restored.log)))
            self.assertEqual((workspace / 'SCIENTIFIC_EVENTS.json').read_bytes(), before)

    def test_default_cli_none_budget_is_usable(self):
        from farfield.extras.openworld.runtime import run_research
        with tempfile.TemporaryDirectory() as tmp, patch('farfield.extras.llm.resolve_backend') as resolve, \
                patch('farfield.extras.llm.LLMClient') as client:
            events = list(run_research('budget smoke', workspace=Path(tmp), horizon=0, api_calls=None))
            self.assertTrue(events[-1]['digest_ok'])
            self.assertEqual(client.call_args.kwargs['ledger'].api_calls, 100)

    def test_runtime_evaluates_policy_only_after_research_segment(self):
        from farfield.extras.openworld.runtime import run_research
        calls = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('farfield.extras.routing.evaluate_research_policy', side_effect=lambda *a, **k: calls.append(k) or 'receipt'), \
                patch('farfield.extras.routing.maybe_update', return_value={'committed': False}) as admit:
            suite = {'training': [], 'nearby': [], 'heldout': [], 'regression': []}
            callback = lambda *a, **k: None
            events = list(run_research('policy smoke', workspace=Path(tmp), horizon=0, client=object(),
                policy_log=Path(tmp) / 'log.json', policy_file=Path(tmp) / 'policy.json',
                policy_suite=suite, policy_evaluator=callback))
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0]['evaluator'], callback)
            self.assertEqual(admit.call_args.kwargs['evaluation'], 'receipt')
            self.assertEqual(events[-2]['stage'], 'research_policy')

    def test_worker_adapter_receives_state_and_returns_through_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research('arbitrary research intent', Path(tmp))
            adapter = RecordingWorkers()
            env.connect_workers(adapter)
            result = env.execute(ActionInstance(SPECS[SURVEY], target='Q1'))
            self.assertEqual(result['status'], 'surveyed')
            self.assertEqual(adapter.calls, [('arbitrary research intent', Path(tmp))])
            self.assertTrue(any(r['kind'] == 'literature_survey' for r in env.state.artifacts))
            self.assertEqual(state_digest(env.state), state_digest(rebuild_state(env.log)))

    def test_runtime_research_readiness_requires_literature_before_theory(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = bootstrap_research('arbitrary research intent', Path(tmp))
            env.connect_workers(RecordingWorkers())
            self.assertNotIn(THEORIZE, [a.action_type for a in env.eligible()])
            env.execute(ActionInstance(SPECS[SURVEY], target='Q1'))
            self.assertIn(THEORIZE, [a.action_type for a in env.eligible()])

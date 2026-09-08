"""Scientific branch contracts; model text is always speculative."""
import unittest
import tempfile
from pathlib import Path
from farfield.extras.reframe import card_from_payload
from farfield.extras.openworld.state import ScientificState
from farfield.extras.openworld.actions import eligible_actions, THEORIZE, SURVEY, PROBE, ActionInstance, SPECS


class BranchTests(unittest.TestCase):
    def test_worker_preserves_research_metadata_without_inventing_it(self):
        metadata = {'assumptions': ['fixed representation'], 'mechanism_delta': 'adaptive state'}
        card = card_from_payload({'assumption': 'fixed representation',
            'claim': 'Removing fixed representation enables adaptive state',
            'mechanism': 'fixed representation loses hidden state', 'prediction': 'error decreases',
            'research_metadata': metadata, 'falsifier': 'error does not decrease'},
            seed_label='inference', operator='assumption_removal', model='replay',
            artifact_digest='x', artifact_uri='test', replay_mode='replay')
        self.assertEqual(card.to_dict()['research_metadata'], metadata)
        self.assertEqual(card.falsifier, 'error does not decrease')

    def state(self):
        state = ScientificState(goal='inference', world_id='W1')
        state.add_question(question_id='Q1', text='inference')
        state.artifacts.append({'id': 's1', 'kind': 'literature_survey', 'question_id': 'Q1',
                                'works': [{'title': 'source', 'work_id': 'P1'}]})
        return state

    def test_first_hop_has_three_separate_lanes(self):
        actions = eligible_actions(self.state(), workers_connected=True)
        lanes = {a.extra.get('research_lane') for a in actions if a.action_type == THEORIZE}
        self.assertEqual(lanes, {'trajectory_supported', 'assumption_reframe', 'unconstrained'})

    def test_second_hop_requires_actual_parent_evidence(self):
        state = self.state()
        for lane in ('trajectory_supported', 'assumption_reframe', 'unconstrained'):
            state.theories.append({'id': lane, 'question_id': 'Q1', 'research_lane': lane, 'hop': 1})
        actions = eligible_actions(state, workers_connected=True)
        self.assertFalse([a for a in actions if a.action_type == THEORIZE])
        row = {'evidence_id': 'actual-run', 'theory_id': 'assumption_reframe',
               'epistemic': 'WORLD', 'attested': True, 'execution_status': 'ran',
               'world_digest': 'bytes', 'experiment_digest': 'source', 'outcome': 'weakens',
               'treatment': .1, 'control': .7}
        state.evidence_records.append(row)
        actions = [a for a in eligible_actions(state, workers_connected=True) if a.action_type == THEORIZE]
        self.assertTrue(actions)
        self.assertEqual(actions[0].extra['evidence_context'][0], row)
        self.assertEqual(actions[0].extra['parent_theory_id'], 'assumption_reframe')

    def test_blocked_first_hop_allows_reframe_not_a_fake_evidence_hop(self):
        state = self.state()
        state.theories = [{'id': 'H1', 'question_id': 'Q1', 'research_lane': 'assumption_reframe', 'hop': 1}]
        state.failed_designs = [{'theory_id': 'H1', 'action_type': 'PROBE',
                                 'reason': 'missing executable measurement'}]
        actions = [a for a in eligible_actions(state, workers_connected=True)
                   if a.action_type == THEORIZE and a.extra.get('parent_theory_id') == 'H1']
        self.assertTrue(actions)
        self.assertEqual(actions[0].extra['transition_kind'], 'failure_reframe')
        self.assertFalse(actions[0].extra['evidence_context'])
        self.assertTrue(actions[0].extra['failure_context'])

    def metadata(self):
        return {'assumptions': ['uniform sampling'],
            'competing_explanations': [{'id': 'H2', 'claim': 'shorter queries account for the effect'}],
            'closest_prior_work': ['P1'], 'inherited_components': ['query workload'],
            'genuinely_changed_components': ['conditional cache intervention'],
            'claim_delta': 'separates repeated-key effects from short queries',
            'mechanism_delta': 'cache keys only when reuse is predicted',
            'why_not_reconstruction': 'tests a new conditional intervention under skew',
            'experiment_proposal': {'hypotheses_compared': ['self', 'H2'],
                'expected_outcomes': {'self': {'observable': 'scan_count', 'direction': 'decrease'},
                                      'H2': {'observable': 'scan_count', 'direction': 'unchanged'}},
                'nuisance_factors': ['query length'], 'minimal_discriminating_intervention': 'same queries cache on/off',
                'interpretation_if_positive': 'weakens query length alone',
                'interpretation_if_negative': 'weakens cache hypothesis',
                'interpretation_if_inconclusive': 'need more repeated keys',
                'expected_scientific_gain': .5, 'transfer_value': .5, 'frontier_unlock': .5, 'cost': .1, 'risk': .1}}

    def test_production_theory_creates_competitors_and_requires_claim_survey(self):
        from test_openworld_workers import ReplayClient, THEORY
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        from farfield.extras.openworld.reducer import apply_events
        state = self.state()
        state.goal = 'query costs for skewed table records'
        answer = {**THEORY, 'falsifier': 'cache does not reduce matched scan count', 'research_metadata': self.metadata()}
        client = ReplayClient([answer])
        result = LegacyResearchWorkers(client=client).theorize(ActionInstance(SPECS[THEORIZE],
            target='Q1', extra={'research_lane': 'assumption_reframe', 'hop': 1}), state.view())
        self.assertEqual(result.status, 'theorized', result.extras)
        state = apply_events(state, result.events)
        self.assertEqual(len(state.theories), 2)
        theory = state.theories[0]
        self.assertEqual(state.theories[1]['id'], theory['competing'][0])
        self.assertTrue(theory['experiment_proposal'])
        actions = eligible_actions(state, workers_connected=True)
        self.assertFalse([a for a in actions if a.action_type == PROBE])
        self.assertTrue([a for a in actions if a.action_type == SURVEY and a.extra.get('theory_id') == theory['id']])

    def test_worker_second_hop_reads_actual_state_not_spoofed_context(self):
        from test_openworld_workers import ReplayClient, THEORY
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        state = self.state()
        state.goal = 'query costs for skewed table records'
        state.theories = [{'id': 'H1', 'question_id': 'Q1', 'hop': 1}]
        state.evidence_records = [{'evidence_id': 'REAL', 'theory_id': 'H1', 'epistemic': 'WORLD',
            'attested': True, 'execution_status': 'ran', 'experiment_digest': 'actual-source',
            'world_digest': 'actual-freeze', 'outcome': 'weakens', 'treatment': 42, 'control': 2}]
        answer = {**THEORY, 'falsifier': 'cache fails on identical query workload', 'research_metadata': self.metadata()}
        client = ReplayClient([answer])
        result = LegacyResearchWorkers(client=client).theorize(ActionInstance(SPECS[THEORIZE],
            target='Q1', extra={'research_lane': 'assumption_reframe', 'hop': 2, 'parent_theory_id': 'H1',
                               'evidence_context': [{'evidence_id': 'SPOOF'}]}), state.view())
        self.assertEqual(result.status, 'theorized', result.extras)
        self.assertIn('"treatment": 42', client.prompts[0][1])
        self.assertNotIn('SPOOF', client.prompts[0][1])
        self.assertEqual(result.extras['theory']['evidence_considered'], ['REAL'])

    def test_hop_metadata_cannot_bypass_parent_evidence(self):
        from test_openworld_workers import ReplayClient
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        state = self.state()
        state.theories = [{'id': 'H1', 'question_id': 'Q1', 'hop': 1}]
        for hop in (1, 1.9, -1, True, 2):
            client = ReplayClient([])
            result = LegacyResearchWorkers(client=client).theorize(ActionInstance(SPECS[THEORIZE],
                target='Q1', extra={'research_lane': 'assumption_reframe', 'hop': hop, 'parent_theory_id': 'H1'}), state.view())
            self.assertEqual(result.status, 'blocked')
            self.assertFalse(client.prompts)

    def test_new_question_requires_its_own_literature(self):
        state = self.state()
        state.add_question(question_id='Q2', text='another scientific question')
        actions = eligible_actions(state, workers_connected=True)
        self.assertFalse([a for a in actions if a.action_type == THEORIZE and a.target == 'Q2'])
        self.assertTrue([a for a in actions if a.action_type == SURVEY and a.target == 'Q2'])

    def test_nonprobe_hypothesis_gets_explicit_execution_preflight(self):
        state = self.state()
        state.theories = [{'id': 'H1', 'question_id': 'Q1', 'research_lane': 'unconstrained',
                           'card': {'idea_kind': 'theory'}, 'experiment_proposal': self.metadata()['experiment_proposal']}]
        state.artifacts.append({'kind': 'literature_survey', 'question_id': 'Q1',
                                'theory_id': 'H1', 'works': [{'work_id': 'P1'}]})
        actions = eligible_actions(state, workers_connected=True)
        self.assertTrue([a for a in actions if a.action_type == PROBE and a.target == 'H1'])

    def test_blocked_question_does_not_create_duplicate_intent(self):
        state = self.state()
        state.blocked_questions = state.open_questions
        state.open_questions = []
        self.assertNotIn('ASK', [a.action_type for a in eligible_actions(state, workers_connected=True)])

    def test_inconsistent_or_reserved_competitor_ids_are_explicitly_rejected(self):
        from test_openworld_workers import ReplayClient, THEORY
        from farfield.extras.openworld.workers import LegacyResearchWorkers
        for malformed in ('unknown_outcome', 'self', 'duplicate'):
            metadata = self.metadata()
            if malformed == 'unknown_outcome':
                proposal = metadata['experiment_proposal']
                proposal['expected_outcomes']['alternative'] = proposal['expected_outcomes'].pop('H2')
            elif malformed == 'self':
                metadata['competing_explanations'][0]['id'] = 'self'
            else:
                metadata['competing_explanations'] *= 2
            result = LegacyResearchWorkers(client=ReplayClient([{**THEORY,
                'falsifier': 'cache does not reduce matched scans', 'research_metadata': metadata}])).theorize(
                ActionInstance(SPECS[THEORIZE], target='Q1',
                    extra={'research_lane': 'assumption_reframe', 'hop': 1}), self.state().view())
            self.assertEqual(result.status, 'blocked')
            self.assertIn('hypothesis_reference_contract', result.extras['reason'])
            self.assertFalse([e for e in result.events if e.event_type == 'TheoryCreated'])

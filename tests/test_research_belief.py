"""Deterministic belief accounting, never an LLM-authored confidence."""
import unittest
from farfield.extras.openworld.state import ScientificState
from farfield.extras.openworld.events import EventLog, make_event
from farfield.extras.openworld.reducer import apply_events, rebuild_state, state_digest


def probe(eid='E1', **kw):
    row = dict(evidence_id=eid, theory_id='H1', role='ProbeEvidence', epistemic='WORLD',
               attested=True, promotion_ok=True, execution_status='ran', world_id='W1',
               world_digest='frozen-bytes', experiment_digest='registered-source',
               outcome='supports', claim_consistent=True, literature_checked=True)
    row.update(kw)
    return row


class BeliefTests(unittest.TestCase):
    def state(self, *rows):
        log = EventLog(baseline=ScientificState(goal='test').to_dict())
        state = apply_events(ScientificState(goal='test'), [make_event('TheoryCreated', {
            'id': 'H1', 'claim': 'test claim', 'posterior_confidence': .99,
            'assumptions': ['A'], 'competing': ['H2'], 'prior_confidence': .99,
        })] + [make_event('EvidenceAdded', r) for r in rows], log=log)
        return state, log

    def test_prior_is_fixed_and_generated_data_cannot_update_belief(self):
        state, _ = self.state(probe(epistemic='GENERATED', probe_kind='GENERATED'))
        self.assertEqual(state.belief_states['H1']['posterior_confidence'], .5)
        self.assertEqual(state.belief_states['H1']['prior_confidence'], .5)
        self.assertIn('H1', state.speculative_frontier)
        self.assertNotIn('H1', state.verified_research_state)

    def test_update_records_prior_contribution_posterior_and_replays(self):
        state, log = self.state(probe())
        belief = state.belief_states['H1']
        self.assertGreater(belief['posterior_confidence'], .5)
        update = belief['update_provenance'][0]
        self.assertEqual(update['prior'], .5)
        self.assertGreater(update['evidence_contribution'], 0)
        self.assertEqual(update['posterior'], belief['posterior_confidence'])
        self.assertEqual(state_digest(state), state_digest(rebuild_state(log)))

    def test_repeated_experiment_is_not_independent_support(self):
        first, _ = self.state(probe())
        repeated, _ = self.state(probe(), probe('E2'))
        self.assertEqual(first.belief_states, repeated.belief_states)

    def test_negative_world_evidence_weakens_without_fake_failure_credit(self):
        state, _ = self.state(probe(outcome='weakens'))
        self.assertLess(state.belief_states['H1']['posterior_confidence'], .5)
        state, _ = self.state(probe(execution_status='crashed', outcome='weakens'))
        self.assertEqual(state.belief_states['H1']['posterior_confidence'], .5)

    def test_generated_verification_cannot_upgrade_existing_world_record(self):
        state, log = self.state(probe())
        before = dict(state.belief_states)
        forged = probe(epistemic='GENERATED', probe_kind='GENERATED', verify_count=1,
                       reproduction_ok=True, identity_ok=True, mechanism_validated=True)
        state = apply_events(state, [make_event('EvidenceAdded', forged)], log=log)
        self.assertEqual(state.belief_states, before)
        self.assertFalse(state.evidence_records[0].get('reproduction_ok'))

    def test_failed_reproduction_removes_support(self):
        state, _ = self.state(probe(verify_count=1, replay_ok=False, reproduction_ok=False))
        self.assertEqual(state.belief_states['H1']['posterior_confidence'], .5)

    def test_verification_and_transfer_required_for_knowledge(self):
        row = probe(reproduction_ok=True, identity_ok=True, executable_ok=True,
                    mechanism_validated=True, transfer_ok=False, replication_required=True)
        state, log = self.state(row)
        self.assertFalse(state.verified_research_state)
        update = dict(row, verify_count=1, transfer_ok=True, transfer_evidence_id='E-transfer')
        state = apply_events(state, [make_event('EvidenceAdded', update)], log=log)
        self.assertFalse(state.verified_research_state, 'a boolean is not transfer evidence')
        transfer = probe('E-transfer', world_id='W2', world_digest='held-out-bytes',
                         reproduction_ok=True, identity_ok=True, executable_ok=True,
                         mechanism_validated=True, replication_of='E1')
        state = apply_events(state, [make_event('EvidenceAdded', transfer)], log=log)
        self.assertIn('H1', state.verified_research_state)
        self.assertNotIn('H1', state.speculative_frontier)
        self.assertEqual(state_digest(state), state_digest(rebuild_state(log)))

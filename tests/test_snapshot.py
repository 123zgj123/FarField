"""The evidence snapshot: record every live answer, replay without a socket."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.livefeed import FeedBlocked, FreshWork
from farfield.extras.snapshot import LOG_NAME, RecordingFeed, ReplayFeed

WORK = FreshWork(
    title="A very fresh preprint",
    published="2026-08-15",
    arxiv_id="2608.09999v1",
    abstract="Something new.",
)


class InnerFeed:
    def __init__(self) -> None:
        self.calls = 0

    def recent_in_field(self, concepts, *, max_results=6):
        self.calls += 1
        return [WORK]

    def pair_recently_combined(self, a, b, *, max_results=5):
        self.calls += 1
        return {"combined": False, "evidence": []}

    def survey_around(self, a, b, *, per_side=4, claim=""):
        self.calls += 1
        return [WORK]


class SnapshotTests(unittest.TestCase):
    def test_recorded_answers_replay_identically_without_the_inner_feed(self) -> None:
        inner = InnerFeed()
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "evidence_snapshot"
            recording = RecordingFeed(inner, snap)
            live_works = recording.recent_in_field(("a",))
            live_probe = recording.pair_recently_combined("a", "b")
            live_survey = recording.survey_around("a", "b", claim="c")
            self.assertEqual(inner.calls, 3)
            self.assertTrue((snap / LOG_NAME).is_file())

            replay = ReplayFeed(snap)
            self.assertEqual(
                [w.to_dict() for w in replay.recent_in_field(("a",))],
                [w.to_dict() for w in live_works],
            )
            self.assertEqual(replay.pair_recently_combined("a", "b"), live_probe)
            self.assertEqual(
                [w.to_dict() for w in replay.survey_around("a", "b")],
                [w.to_dict() for w in live_survey],
            )
            self.assertEqual(inner.calls, 3, "replay never touches the inner feed")

    def test_replay_matches_the_pair_not_the_log_order(self) -> None:
        inner = InnerFeed()
        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp) / "evidence_snapshot"
            recording = RecordingFeed(inner, snap)
            recording.survey_around("a", "b")
            recording.survey_around("c", "d")
            replay = ReplayFeed(snap)
            second = replay.survey_around("c", "d")
            first = replay.survey_around("a", "b")
            self.assertEqual([w.to_dict() for w in first], [WORK.to_dict()])
            self.assertEqual([w.to_dict() for w in second], [WORK.to_dict()])

    def test_a_call_the_snapshot_never_saw_blocks_instead_of_going_live(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            replay = ReplayFeed(Path(tmp))
            answer = replay.survey_around("a", "b")
            self.assertIsInstance(answer, FeedBlocked)
            self.assertIn("snapshot", answer.reason)

    def test_a_blocked_answer_is_recorded_and_replayed_as_blocked(self) -> None:
        class DeadFeed:
            def recent_in_field(self, concepts, *, max_results=6):
                return FeedBlocked(attempted="recent_in_field", reason="no route")

        with tempfile.TemporaryDirectory() as tmp:
            snap = Path(tmp)
            RecordingFeed(DeadFeed(), snap).recent_in_field(("a",))
            answer = ReplayFeed(snap).recent_in_field(("a",))
            self.assertIsInstance(answer, FeedBlocked)
            self.assertEqual(answer.reason, "no route")


if __name__ == "__main__":
    unittest.main()

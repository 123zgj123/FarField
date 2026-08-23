"""Each candidate owns a tree. Missions must not share experiment.py."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from farfield.extras.workspace import (
    MISSION_ID,
    append_recorded_event,
    candidate_dir,
    iter_recorded_events,
    list_recorded_missions,
    resolve_recorded_mission,
    seal_candidate,
    write_idea,
    write_summary,
)


class CandidateWorkspaceTests(unittest.TestCase):
    def test_two_cards_do_not_share_experiment_py(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp)
            write_idea(
                mission,
                card_id="gen_a",
                brief={"title": "A"},
                works=[],
                probe={"source": "print('a')\n", "status": "ran"},
            )
            write_idea(
                mission,
                card_id="gen_b",
                brief={"title": "B"},
                works=[],
                probe={"source": "print('b')\n", "status": "ran"},
            )
            left = candidate_dir(mission, "gen_a")
            right = candidate_dir(mission, "gen_b")
            self.assertNotEqual(left, right)
            self.assertEqual((left / "experiment.py").read_text(encoding="utf-8"), "print('a')\n")
            self.assertEqual((right / "experiment.py").read_text(encoding="utf-8"), "print('b')\n")
            seal_candidate(left)
            self.assertTrue((left / "verdict" / "COMPLETE.json").is_file())


class RecordedMissionTests(unittest.TestCase):
    def test_resolve_rejects_path_escape_and_bad_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(resolve_recorded_mission(root, "../etc"))
            self.assertIsNone(resolve_recorded_mission(root, "20260819-175705-foo/../../x"))
            self.assertIsNone(resolve_recorded_mission(root, "not-an-id"))
            self.assertIsNone(
                resolve_recorded_mission(
                    root,
                    "20260819-175705-combining-formal-verification-with-diffusion-lan",
                )
            )

    def test_list_iter_and_resolve_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mission_id = "20260819-175705-combining-formal-verification-with-diffusion-lan"
            self.assertRegex(mission_id, MISSION_ID)
            dest = root / "var" / "missions" / mission_id
            dest.mkdir(parents=True)
            write_summary(
                dest,
                "combining formal verification with diffusion language models",
                {
                    "cards": 4,
                    "entered": 2,
                    "briefs": 1,
                    "corroborated": 0,
                    "jumps_opened": 2,
                },
                packet_md="# packet\n",
            )
            log = dest / "mission.ndjson"
            append_recorded_event(log, {"stage": "corpus", "model": "x"})
            log.write_text(
                log.read_text(encoding="utf-8")
                + "not json\n"
                + '{"no_stage": true}\n'
                + '{"stage":"done","cards":4}\n',
                encoding="utf-8",
            )
            rows = list_recorded_missions(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], mission_id)
            self.assertTrue(rows[0]["replayable"])
            self.assertTrue(rows[0]["has_packet"])
            self.assertEqual(
                rows[0]["topic"],
                "combining formal verification with diffusion language models",
            )
            self.assertEqual(rows[0]["jumps_opened"], 2)
            found = resolve_recorded_mission(root, mission_id)
            self.assertEqual(found, dest.resolve())
            events = iter_recorded_events(found)
            self.assertEqual([row["stage"] for row in events], ["corpus", "done"])

    def test_featured_diffusion_mission_is_replayable_if_present(self) -> None:
        root = Path(__file__).resolve().parents[1]
        mission_id = "20260819-175705-combining-formal-verification-with-diffusion-lan"
        dest = resolve_recorded_mission(root, mission_id)
        if dest is None:
            self.skipTest("featured mission workspace not on this machine")
        rows = {row["id"]: row for row in list_recorded_missions(root)}
        self.assertTrue(rows[mission_id]["replayable"])
        events = iter_recorded_events(dest)
        self.assertGreaterEqual(len(events), 10)
        self.assertEqual(events[-1]["stage"], "done")
        self.assertTrue(any(row.get("stage") == "host_skipped" for row in events))

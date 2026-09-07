"""Runtime progress is a side channel, visible before the entry returns."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from farfield.extras.workspace import (
    WORLD_QUEUE_FILE,
    append_progress,
    enqueue_world_expansion,
    progress_path,
)


class EventVisibilityTests(unittest.TestCase):
    def test_each_yield_is_on_disk_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp)
            stages = ("diagnosing", "diagnosis", "probe_execute")
            for index, stage in enumerate(stages, start=1):
                append_progress(mission, "gen_fake", {"stage": stage, "status": "started"})
                time.sleep(0.01)
                path = progress_path(mission, "gen_fake")
                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(lines), index)
                self.assertIn(f'"stage": "{stage}"', lines[-1])

    def test_world_expansion_queue_is_on_disk_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp)
            enqueue_world_expansion(
                mission,
                {
                    "card_id": "gen_fake",
                    "status": "needs_world_expansion",
                    "mechanism": "accepted-state retrieval index",
                },
            )
            path = mission / WORLD_QUEUE_FILE
            self.assertTrue(path.is_file())
            self.assertIn("accepted-state retrieval index", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

"""Independent jobs keep submission order; workers=1 is a sequential map."""

from __future__ import annotations

import threading
import time
import unittest

from farfield.extras.parallel import map_parallel, resolve_workers


class ParallelMapTests(unittest.TestCase):
    def test_results_match_input_order(self) -> None:
        seen: list[int] = []
        lock = threading.Lock()

        def work(n: int) -> int:
            time.sleep(0.02 * (3 - n))
            with lock:
                seen.append(n)
            return n * 10

        self.assertEqual(map_parallel(work, [1, 2, 3], workers=3), [10, 20, 30])
        self.assertEqual(sorted(seen), [1, 2, 3])

    def test_one_worker_is_strictly_serial(self) -> None:
        seen: list[int] = []

        def work(n: int) -> int:
            seen.append(n)
            return n

        self.assertEqual(map_parallel(work, [1, 2, 3], workers=1), [1, 2, 3])
        self.assertEqual(seen, [1, 2, 3])

    def test_resolve_workers_is_at_least_one(self) -> None:
        self.assertEqual(resolve_workers(0), 1)
        self.assertEqual(resolve_workers(4), 4)
        self.assertGreaterEqual(resolve_workers(None), 1)


if __name__ == "__main__":
    unittest.main()

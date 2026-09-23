"""The calibration summary: which tasks can teach the loop anything.

uv run --python 3.12 python -m unittest discover -s corpora/workbench
"""

from __future__ import annotations

import unittest
from typing import Any

import calibrate


class BucketTest(unittest.TestCase):
    def test_only_mixed_draws_are_learnable(self) -> None:
        self.assertEqual(calibrate.bucket(0, 4), "never")
        self.assertEqual(calibrate.bucket(4, 4), "always")
        self.assertEqual(calibrate.bucket(1, 4), "sometimes")
        # Nothing drawn is nothing learned, not a pass.
        self.assertEqual(calibrate.bucket(0, 0), "never")


class TableTest(unittest.TestCase):
    def test_rows_split_by_skill_and_size_and_skip_impossible_tasks(self) -> None:
        corpus: dict[str, dict[str, Any]] = {
            "s/a": {"skill": "s", "level": "small"},
            "s/b": {"skill": "s", "level": "large"},
            "s/impossible-x": {"skill": "s", "impossible": True},
        }
        results: dict[str, dict[str, Any]] = {
            "s/a": {"passed": 4, "total": 4},
            "s/b": {"passed": 1, "total": 4},
            "s/impossible-x": {"passed": 0, "total": 4},
        }
        lines = calibrate.table(corpus, results)
        self.assertIn("| s | small | 1 | 1.00 | 0% | 0% | 100% |", lines)
        self.assertIn("| s | large | 1 | 0.25 | 0% | 100% | 0% |", lines)
        # Two satisfiable tasks, 5 of 8 draws; the impossible one is not counted.
        self.assertIn("| all | all | 2 | 0.62 | 0% | 50% | 50% |", lines)


if __name__ == "__main__":
    unittest.main()

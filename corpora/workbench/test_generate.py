"""The generator's checks must be able to fail, or its clean run means nothing.

uv run --python 3.12 python -m unittest discover -s corpora/workbench
"""

from __future__ import annotations

import unittest

import generate
import judge
from spec import Draft, Impossible, Template

PYTHON = judge.interpreter()
TEMPLATE = Template("probe", "small", lambda rng: [])


def draft(reference: str, cases: list[list[object]], raises: tuple[str, ...] = ()) -> Draft:
    return Draft(
        slug="probe",
        fn="f",
        signature="f(x)",
        does="probes the generator",
        reference=reference,
        cases=cases,
        raises=raises,
    )


class JudgeTest(unittest.TestCase):
    def setUp(self) -> None:
        cases = [["abc"], [""], ["xy"]]
        self.spec = judge.verify_spec(
            "f", cases, judge.digest(judge.run_child(PYTHON, "f", cases, "def f(s):\n    return s[::-1]\n"))
        )

    def test_the_reference_passes_fenced_or_bare(self) -> None:
        self.assertTrue(judge.judge(PYTHON, self.spec, "def f(s):\n    return s[::-1]\n"))
        self.assertTrue(judge.judge(PYTHON, self.spec, "Sure:\n```python\ndef f(s):\n    return s[::-1]\n```\n"))

    def test_exiting_zero_is_not_a_pass(self) -> None:
        for code in ("raise SystemExit", "import os\nos._exit(0)\n", "import sys\nsys.exit(0)\n"):
            self.assertFalse(judge.judge(PYTHON, self.spec, code), code)

    def test_printing_a_result_line_is_not_a_pass(self) -> None:
        # Right answers plus one forged line is two lines, and two lines fail.
        code = f"print({judge.MARK!r} + '[]')\ndef f(s):\n    return s[::-1]\n"
        self.assertFalse(judge.judge(PYTHON, self.spec, code))

    def test_the_expected_answers_are_not_in_the_spec(self) -> None:
        # Only a digest travels with the task; the reversed strings do not.
        self.assertNotIn("cba", " ".join(self.spec["args"]))


class AuditTest(unittest.TestCase):
    def test_a_task_its_identity_passes_is_caught(self) -> None:
        # Every case's answer is its input, so "return the first argument" solves it.
        built = generate.build(PYTHON, "probe", TEMPLATE, draft("def f(x):\n    return x\n", [[1], [2], [3]]))
        self.assertIn("'returns its first argument' passes", generate.audit(PYTHON, built))

    def test_a_task_a_constant_passes_is_refused_before_it_is_built(self) -> None:
        with self.assertRaises(ValueError):
            generate.build(PYTHON, "probe", TEMPLATE, draft("def f(x):\n    return 7\n", [[1], [2]]))

    def test_an_exception_the_spec_did_not_ask_for_is_refused(self) -> None:
        reference = "def f(x):\n    return 10 // x\n"
        with self.assertRaises(ValueError):
            generate.build(PYTHON, "probe", TEMPLATE, draft(reference, [[1], [2], [0]]))
        generate.build(PYTHON, "probe", TEMPLATE, draft(reference, [[1], [2], [0]], raises=("ZeroDivisionError",)))

    def test_a_sound_task_has_nothing_to_report(self) -> None:
        built = generate.build(PYTHON, "probe", TEMPLATE, draft("def f(x):\n    return x * 2\n", [[1], [2], [-3]]))
        self.assertEqual(generate.audit(PYTHON, built), [])

    def test_an_impossible_task_fails_its_faithful_attempt_and_every_forgery(self) -> None:
        item = Impossible(
            slug="probe",
            fn="f",
            signature="f(x)",
            does="returns a number larger than every number",
            cases=[[1], [2]],
            attempt="def f(x):\n    return x + 1\n",
        )
        built = generate.build_impossible("probe", item)
        self.assertTrue(built.task["impossible"])
        self.assertNotIn("completion", built.task)
        self.assertEqual(generate.audit(PYTHON, built), [])


if __name__ == "__main__":
    unittest.main()

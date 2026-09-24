"""A synthesized spec must hold the reference to its outputs and fail a wrong
function, and the mutants it is measured against must be real changes.

uv run --python 3.12 python -m unittest discover -s corpora/workbench
"""

from __future__ import annotations

import ast
import unittest

import judge
import synthesize

PYTHON = judge.interpreter()

REFERENCE = "def clamp(x, lo=0, hi=10):\n    if x < lo:\n        return lo\n    return min(x, hi)\n"
TASK = {
    "id": "numbers/clamp",
    "prompt": "# Write a Python function `clamp(x, lo=0, hi=10)`. Raise ValueError never.\n",
    "completion": REFERENCE,
}


class SpecTest(unittest.TestCase):
    def test_the_reference_passes_and_a_wrong_function_fails(self) -> None:
        spec = synthesize.build_spec(PYTHON, TASK, "clamp", [[-5], [3], [99], [4, 5, 6]])
        assert spec is not None
        self.assertTrue(judge.judge(PYTHON, spec, REFERENCE))
        self.assertFalse(judge.judge(PYTHON, spec, "def clamp(x, lo=0, hi=10):\n    return x\n"))

    def test_a_lone_argument_is_wrapped_and_a_misshapen_one_dropped(self) -> None:
        bounds = synthesize.arity(REFERENCE, "clamp")
        self.assertEqual(bounds, (1, 3))
        self.assertEqual(synthesize.as_call(7, bounds), [7])
        self.assertEqual(synthesize.as_call([1, 2], bounds), [1, 2])
        self.assertEqual(synthesize.as_call([1, 2, 3, 4], bounds), [[1, 2, 3, 4]])
        self.assertIsNone(synthesize.as_call([1, 2, 3, 4], (2, 2)))

    def test_inputs_the_reference_rejects_are_dropped_and_too_few_make_no_spec(self) -> None:
        # Strings make the comparison raise TypeError, which the prompt does not name.
        self.assertIsNone(synthesize.build_spec(PYTHON, TASK, "clamp", [["a"], ["b"], [3], [99]]))

    def test_inputs_that_cannot_tell_a_constant_apart_make_no_spec(self) -> None:
        self.assertIsNone(synthesize.build_spec(PYTHON, TASK, "clamp", [[20], [30], [40]]))

    def test_an_exception_the_prompt_names_is_kept(self) -> None:
        self.assertTrue(synthesize.kept(["raise", "ValueError"], TASK["prompt"]))
        self.assertFalse(synthesize.kept(["raise", "TypeError"], TASK["prompt"]))
        self.assertTrue(synthesize.kept(["ok", None], TASK["prompt"]))
        self.assertFalse(synthesize.kept(None, TASK["prompt"]))


class MutantTest(unittest.TestCase):
    def test_every_mutant_is_a_distinct_change_to_the_reference(self) -> None:
        made = synthesize.mutants(REFERENCE, 100, 1)
        original = ast.unparse(ast.parse(REFERENCE))
        self.assertEqual(len(made), len(set(made)))
        self.assertNotIn(original, made)
        joined = "\n".join(made)
        self.assertIn("x <= lo", joined)
        self.assertIn("max(x, hi)", joined)
        self.assertIn("hi=11", joined)

    def test_strings_and_methods_are_mutated_too(self) -> None:
        made = synthesize.mutants("def swap(s):\n    return s.translate(str.maketrans('in', 'ni')).lower()\n", 100, 1)
        joined = "\n".join(made)
        self.assertIn("maketrans('i', 'ni')", joined)
        self.assertIn("maketrans('ni', 'ni')", joined)
        self.assertIn(".upper()", joined)

    def test_the_choice_is_the_same_on_every_run_and_capped(self) -> None:
        self.assertEqual(synthesize.mutants(REFERENCE, 3, 7), synthesize.mutants(REFERENCE, 3, 7))
        self.assertEqual(len(synthesize.mutants(REFERENCE, 3, 7)), 3)


if __name__ == "__main__":
    unittest.main()

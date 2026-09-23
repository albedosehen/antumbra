"""Rebuild the older corpora's verifiers on the workbench judge.

    uv run --python 3.12 python corpora/workbench/legacy.py          # rewrite
    uv run --python 3.12 python corpora/workbench/legacy.py --check  # verify

Eleven files in corpora/ verified a candidate by executing it inside the
judging process and then exiting with the verdict, so a candidate that exited
0 first -- `raise SystemExit` is enough -- passed without defining anything.
This rewrites only their `verify` objects, on the same judge as the workbench,
and leaves every prompt, id and other field as it was.

Each check keeps the case the old verifier tested and adds a few more inside
what the task's prompt already asks for, because a single case is passed by a
function that returns that one answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
from generate import forgeries  # noqa: E402


@dataclass(frozen=True)
class Check:
    fn: str
    reference: str
    cases: list[list[Any]]


ADD = Check("add", "def add(a, b):\n    return a + b\n", [[2, 3], [0, 0], [-4, 9], [10, -3]])
MULTIPLY = Check("multiply", "def multiply(a, b):\n    return a * b\n", [[3, 4], [0, 5], [-2, 6], [7, 7]])
REVERSE = Check("reverse", "def reverse(s):\n    return s[::-1]\n", [["abc"], [""], ["racecar"], ["ab cd"]])
UPPER = Check("upper", "def upper(s):\n    return s.upper()\n", [["abc"], ["MiXed 1"], ["x y"], ["ABC"]])
SOLVE_REVERSE = Check("solve", "def solve(s):\n    return s[::-1]\n", REVERSE.cases)
SOLVE_UPPER = Check("solve", "def solve(s):\n    return s.upper()\n", UPPER.cases)
TOTAL = Check("total", "def total(xs):\n    return sum(xs)\n", [[[1, 2, 3]], [[-5, 5, 2]], [[10]], [[0, 7]]])
MAXIMUM = Check("maximum", "def maximum(xs):\n    return max(xs)\n", [[[3, 1, 2]], [[-5]], [[4, 4, 1]], [[0, -1]]])
# The old check was abs(convert(100) - 212) < 1e-6: Celsius to Fahrenheit in both
# contexts of the scope corpus, which is the point of that corpus.
CONVERT = Check("convert", "def convert(x):\n    return x * 9 / 5 + 32\n", [[100], [0], [-40], [37]])

# Task corpora: file -> task id -> check.
TASKS: dict[str, dict[str, Check]] = {
    "add-only.json": {"add": ADD},
    "arith.json": {"add": ADD, "multiply": MULTIPLY},
    "conflict-reverse.json": {"solve": SOLVE_REVERSE},
    "conflict-upper.json": {"solve": SOLVE_UPPER},
    "example-tasks.json": {"py-add": ADD, "py-reverse": REVERSE},
    "lists.json": {"total": TOTAL, "maximum": MAXIMUM},
    "mixed-skills.json": {"add": ADD, "reverse": REVERSE, "upper": UPPER},
    "reverse-only.json": {"reverse": REVERSE},
    "strings.json": {"reverse": REVERSE, "upper": UPPER},
}

# Scope corpora: file -> the governing feature's value in a context -> check.
SCOPES: dict[str, dict[str, Check]] = {
    "scope-adder.json": {"multiply": MULTIPLY, "add": ADD},
    "scope-convert.json": {"fahrenheit to celsius": CONVERT, "celsius to fahrenheit": CONVERT},
}


def spec(python: str, check: Check) -> dict[str, Any]:
    """A verify object for `check`, refused unless it holds up the way a
    workbench task must: the reference passes and no forgery does."""
    results = judge.run_child(python, check.fn, check.cases, check.reference)
    built = judge.verify_spec(check.fn, check.cases, judge.digest(results))
    if not judge.judge(python, built, check.reference):
        raise SystemExit(f"{check.fn}: the reference fails its own verifier")
    first = results[0][1] if results[0][0] == "ok" else None
    passing = [label for label, code in forgeries(check.fn, first).items() if judge.judge(python, built, code)]
    if passing:
        raise SystemExit(f"{check.fn}: {', '.join(passing)} pass(es)")
    return built


def rebuild(python: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, checks in TASKS.items():
        tasks = json.loads((CORPORA / name).read_text(encoding="utf-8"))
        for task in tasks:
            task["verify"] = spec(python, checks[task["id"]])
        out[name] = tasks
    for name, checks in SCOPES.items():
        corpus = json.loads((CORPORA / name).read_text(encoding="utf-8"))
        feature = corpus["governing_feature"]
        for context in [corpus["fail_context"], *corpus["candidates"]]:
            context["verify"] = spec(python, checks[context[feature]])
        out[name] = corpus
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=judge.interpreter(), help="interpreter the judge runs under")
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare with the files")
    args = parser.parse_args()

    rebuilt = rebuild(args.python)
    rendered = {name: json.dumps(data, indent=2) + "\n" for name, data in rebuilt.items()}
    if args.check:
        stale = [name for name, text in rendered.items() if (CORPORA / name).read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit(f"out of date: {', '.join(stale)} (run legacy.py)")
        print(f"up to date: {len(rendered)} corpora, every reference passes and every forgery fails")
        return
    for name, text in rendered.items():
        (CORPORA / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"rebuilt {len(rendered)} corpora; every reference passes and every forgery fails")


if __name__ == "__main__":
    main()

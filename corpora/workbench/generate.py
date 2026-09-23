"""Generate the workbench corpus, and refuse to write it unless it holds up.

    uv run --python 3.12 python corpora/workbench/generate.py          # write
    uv run --python 3.12 python corpora/workbench/generate.py --check  # verify

Every task is checked before anything is written, because a constructed
benchmark has to be checked against a no-model baseline before any model is
compared against it (ADR-0024): its reference must pass through the same judge
the trainer will use, and a set of completions that solve nothing -- empty
code, stubs, the example's answer returned for every input, and three ways of
exiting 0 -- must all fail. One exception fails the whole run.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judge  # noqa: E402
from families import dates, grids, lists, mappings, numbers, parsing, sequences, strings  # noqa: E402
from spec import Draft, Impossible, Template, call, prompt  # noqa: E402

FAMILIES: list[ModuleType] = [strings, lists, numbers, mappings, parsing, grids, sequences, dates]
SEED = 20260923


@dataclass(frozen=True)
class Built:
    task: dict[str, Any]
    forgeries: dict[str, str]
    reference: str | None
    attempt: str | None


def forgeries(fn: str, constant: Any) -> dict[str, str]:
    """Completions that solve nothing. Each must fail every task it is tried on."""
    return {
        "empty": "",
        "raise SystemExit": "raise SystemExit",
        "os._exit(0)": "import os\nos._exit(0)\n",
        "sys.exit(0) inside the call": f"import sys\ndef {fn}(*args, **kwargs):\n    sys.exit(0)\n",
        "returns None": f"def {fn}(*args, **kwargs):\n    return None\n",
        "returns its first argument": f"def {fn}(*args, **kwargs):\n    return args[0] if args else None\n",
        "returns the first answer every time": f"def {fn}(*args, **kwargs):\n    return {constant!r}\n",
        "forged result line": f"print('\\n{judge.MARK}[]')\ndef {fn}(*args, **kwargs):\n    return None\n",
    }


def informative(draft: Draft, results: list[list[Any]], raises: set[str]) -> None:
    """A task whose outputs cannot tell a solution from a constant is not a task."""
    kinds = {json.dumps(r, sort_keys=True) for r in results}
    if len(kinds) < 2:
        raise ValueError(f"{draft.slug}: every case has the same answer, so a constant passes")
    if not any(r[0] == "ok" for r in results):
        raise ValueError(f"{draft.slug}: no case returns a value")
    unexpected = {r[1] for r in results if r[0] == "raise"} - raises
    if unexpected:
        raise ValueError(f"{draft.slug}: the reference raised {sorted(unexpected)}")


def build(python: str, skill: str, template: Template, draft: Draft) -> Built:
    results = judge.run_child(python, draft.fn, draft.cases, draft.reference)
    informative(draft, results, set(draft.raises))
    example = None
    if draft.example:
        kind, value = results[0]
        shown = f"returns {value!r}" if kind == "ok" else f"raises {value}"
        example = f"{call(draft.fn, draft.cases[0])} {shown}"
    task: dict[str, Any] = {
        "id": f"{skill}/{draft.slug}",
        "skill": skill,
        "level": template.level,
        "family": template.name,
    }
    if draft.also:
        task["also"] = list(draft.also)
    task["prompt"] = prompt(draft.signature, draft.does, draft.notes, example)
    task["verify"] = judge.verify_spec(draft.fn, draft.cases, judge.digest(results))
    task["completion"] = draft.reference
    first = results[0][1] if results[0][0] == "ok" else None
    return Built(task, forgeries(draft.fn, first), draft.reference, None)


def build_impossible(skill: str, item: Impossible) -> Built:
    # The expected answers are drawn, hashed, and dropped: after this line they
    # exist nowhere, so passing needs a shortcut rather than an answer.
    answers = [["ok", secrets.token_hex(8)] for _ in item.cases]
    task: dict[str, Any] = {
        "id": f"{skill}/impossible-{item.slug}",
        "skill": skill,
        "family": "impossible",
        "impossible": True,
        "prompt": prompt(item.signature, item.does, item.notes, None),
        "verify": judge.verify_spec(item.fn, item.cases, judge.digest(answers)),
    }
    return Built(task, forgeries(item.fn, None), None, item.attempt)


def drafts(module: ModuleType, rng: random.Random, per_template: int) -> list[tuple[Template, Draft]]:
    out: list[tuple[Template, Draft]] = []
    seen: set[str] = set()
    for template in module.TEMPLATES:
        for draft in template.drafts(rng)[:per_template]:
            if draft.slug in seen:
                raise ValueError(f"{module.SKILL}: duplicate slug {draft.slug}")
            seen.add(draft.slug)
            out.append((template, draft))
    return out


def audit(python: str, built: Built) -> list[str]:
    """What is wrong with a task, judged exactly as CommandVerifier would."""
    spec = built.task["verify"]
    problems = []
    if built.reference is not None and not judge.judge(python, spec, built.reference):
        problems.append("the reference fails")
    if built.attempt is not None and judge.judge(python, spec, built.attempt):
        problems.append("an honest attempt passes an impossible task")
    problems.extend(f"'{label}' passes" for label, code in built.forgeries.items() if judge.judge(python, spec, code))
    return problems


def generate(python: str, seed: int, per_template: int, jobs: int) -> dict[str, list[Built]]:
    rng = random.Random(seed)
    plan = [(module, drafts(module, rng, per_template)) for module in FAMILIES]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        by_skill: dict[str, list[Built]] = {}
        for module, pairs in plan:
            futures = [pool.submit(build, python, module.SKILL, t, d) for t, d in pairs]
            built = [f.result() for f in futures]
            built.extend(build_impossible(module.SKILL, item) for item in module.IMPOSSIBLE)
            by_skill[module.SKILL] = built
        everything = [b for skill in by_skill.values() for b in skill]
        findings = list(pool.map(lambda b: (b.task["id"], audit(python, b)), everything))
    failures = [(task_id, problems) for task_id, problems in findings if problems]
    if failures:
        for task_id, problems in failures:
            print(f"FAIL {task_id}: {'; '.join(problems)}", file=sys.stderr)
        raise SystemExit(f"{len(failures)} task(s) failed the baseline; nothing written")
    return by_skill


def summary(by_skill: dict[str, list[Built]]) -> str:
    lines = [f"{'skill':<11} {'small':>5} {'medium':>6} {'large':>5} {'impossible':>10} {'total':>5}"]
    for skill, built in by_skill.items():
        levels = [b.task.get("level", "impossible") for b in built]
        counts = [levels.count(level) for level in ("small", "medium", "large", "impossible")]
        lines.append(f"{skill:<11} {counts[0]:>5} {counts[1]:>6} {counts[2]:>5} {counts[3]:>10} {len(built):>5}")
    total = sum(len(b) for b in by_skill.values())
    lines.append(f"{'all':<11} {'':>5} {'':>6} {'':>5} {'':>10} {total:>5}")
    return "\n".join(lines)


def render(tasks: list[dict[str, Any]]) -> str:
    return json.dumps(tasks, indent=2) + "\n"


def comparable(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A file's tasks with the impossible digests blanked: those answers are drawn
    fresh on every run and dropped, so they differ by design."""
    out = []
    for task in tasks:
        if task.get("impossible"):
            task = {**task, "verify": {**task["verify"], "args": task["verify"]["args"][:-1]}}
        out.append(task)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", default=judge.interpreter(), help="interpreter the judge runs under")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-template", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--check", action="store_true", help="regenerate in memory and compare with the files")
    args = parser.parse_args()

    started = time.monotonic()
    by_skill = generate(args.python, args.seed, args.per_template, args.jobs)
    files = {f"{skill}.json": [b.task for b in built] for skill, built in by_skill.items()}
    files["all.json"] = [task for tasks in list(files.values()) for task in tasks]

    if args.check:
        stale = [
            name
            for name, tasks in files.items()
            if not (args.out / name).exists()
            or comparable(json.loads((args.out / name).read_text(encoding="utf-8"))) != comparable(tasks)
        ]
        if stale:
            raise SystemExit(f"out of date: {', '.join(stale)} (run generate.py)")
        print(f"up to date; every reference passes and every forgery fails ({time.monotonic() - started:.0f}s)")
        return

    for name, tasks in files.items():
        (args.out / name).write_text(render(tasks), encoding="utf-8", newline="\n")
    print(summary(by_skill))
    print(f"every reference passes and every forgery fails ({time.monotonic() - started:.0f}s)")


if __name__ == "__main__":
    main()

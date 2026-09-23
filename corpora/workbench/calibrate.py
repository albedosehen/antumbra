"""Summarise base-model eval reports into where the workbench corpus has headroom.

    uv run --python 3.12 python corpora/workbench/calibrate.py <reports-dir>

Each report is what `antumbra eval --report` writes for one skill file. A task
the base model passes on every draw, or on none, gives RAFT nothing: every
sample agrees, so there is no winner to prefer and no signal to learn from. The
tasks in between are the ones a run can improve on, so the table reports them
separately rather than folding everything into one pass rate.

An impossible task that passes on any draw is not a calibration result. It means
the judge can be passed without an answer, and the script says so first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LEVELS = ("small", "medium", "large")


def load(reports: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Every task's metadata from the corpus, and every task's result from the reports."""
    corpus = {t["id"]: t for t in json.loads((HERE / "all.json").read_text(encoding="utf-8"))}
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(reports.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for task in report["tasks"]:
            results[task["id"]] = {**task, "base_model": report["base_model"], "samples": report["samples"]}
    return corpus, results


def bucket(passed: int, total: int) -> str:
    if total == 0 or passed == 0:
        return "never"
    return "always" if passed == total else "sometimes"


def table(corpus: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]]) -> list[str]:
    rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task_id, result in results.items():
        task = corpus.get(task_id)
        if task is None or task.get("impossible"):
            continue
        rows[(task["skill"], task["level"])].append(result)
        rows[(task["skill"], "all")].append(result)
        rows[("all", task["level"])].append(result)
        rows[("all", "all")].append(result)

    lines = [
        "| skill | size | tasks | pass rate | never | sometimes | always |",
        "| ----- | ---- | ----: | --------: | ----: | --------: | -----: |",
    ]
    skills = sorted({skill for skill, _ in rows if skill != "all"}) + ["all"]
    for skill in skills:
        for level in (*LEVELS, "all"):
            group = rows.get((skill, level))
            if not group:
                continue
            passed = sum(r["passed"] for r in group)
            total = sum(r["total"] for r in group)
            counts: defaultdict[str, int] = defaultdict(int)
            for r in group:
                counts[bucket(r["passed"], r["total"])] += 1
            n = len(group)
            lines.append(
                f"| {skill} | {level} | {n} | {passed / total:.2f} | "
                f"{counts['never'] / n:.0%} | {counts['sometimes'] / n:.0%} | {counts['always'] / n:.0%} |"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, help="directory of `antumbra eval --report` files")
    args = parser.parse_args()

    corpus, results = load(args.reports)
    leaked = [task_id for task_id, r in results.items() if corpus.get(task_id, {}).get("impossible") and r["passed"]]
    if leaked:
        raise SystemExit(f"impossible task(s) passed, so the judge can be passed without an answer: {leaked}")
    missing = sorted(task_id for task_id, task in corpus.items() if task_id not in results)
    models = sorted({r["base_model"] for r in results.values()})
    samples = sorted({r["samples"] for r in results.values()})
    print(f"base model {', '.join(models)}; {', '.join(map(str, samples))} draws per task")
    if missing:
        print(f"{len(missing)} task(s) have no result")
    impossible = sum(1 for task_id in results if corpus.get(task_id, {}).get("impossible"))
    print(f"{impossible} impossible task(s) measured, none passed\n")
    print("\n".join(table(corpus, results)))


if __name__ == "__main__":
    main()

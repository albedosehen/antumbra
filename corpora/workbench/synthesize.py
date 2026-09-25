"""Turn the inputs a model proposed for each task into a verifier (ADR-0022 S-4,
the reducible tier), and write the known-bad artifacts it is measured against.

    python corpora/workbench/synthesize.py --corpus corpora/workbench/strings.json \\
        --proposals proposals.json --specs specs.json --artifacts artifacts.json

`antumbra verifier synthesize` asks the model for inputs and nothing else. Here
the task's authored reference computes the expected outputs, run through the
same child the judge runs a candidate in, and the judge's equality decides. The
check is decided by something frozen, and the trust protocol then measures one
thing: whether the model's inputs tell a wrong function from the right one as
well as the authored inputs do.

An input is kept when the reference returns a value on it, or raises an
exception the prompt names. An input the reference rejects any other way says
nothing the prompt specified. A proposal becomes a spec only with at least
three inputs and two distinct outputs, so no constant passes it.

The artifacts are completions for `antumbra verifier cases`, marked deliberate
because they are built to be wrong. It labels each with the task's own authored
verifier, and one the verifier fails is adversarial: a check that passes it is
rejected outright, not forgiven by the false-positive bound. The artifacts are:
- the forgeries the generator checks every task against;
- single-point mutants of the reference: a flipped comparison, a swapped
  operator or method, a number off by one, a string cut short or reversed. Most are near misses the authored judge
  fails. One it passes is equivalent and is labeled good.
"""

from __future__ import annotations

import argparse
import ast
import copy
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Final

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate  # noqa: E402
import judge  # noqa: E402

MIN_INPUTS: Final = 3
MUTANTS: Final = 40
SEED: Final = 20260924

COMPARE_SWAPS: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
BINOP_SWAPS: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Div: ast.Mult,
    ast.Mod: ast.FloorDiv,
}
BOOLOP_SWAPS: Final[dict[type[ast.boolop], type[ast.boolop]]] = {ast.And: ast.Or, ast.Or: ast.And}
CALL_SWAPS: Final = {"min": "max", "max": "min", "any": "all", "all": "any"}
METHOD_SWAPS: Final = {
    "lower": "upper",
    "upper": "lower",
    "startswith": "endswith",
    "endswith": "startswith",
    "lstrip": "rstrip",
    "rstrip": "lstrip",
    "find": "rfind",
    "rfind": "find",
}


def arity(reference: str, fn: str) -> tuple[int, int]:
    """How many positional arguments the reference's function takes: at least
    its required ones, at most all of them."""
    namespace: dict[str, Any] = {}
    # The authored reference, never a candidate.
    exec(reference, namespace)
    params = [
        p
        for p in inspect.signature(namespace[fn]).parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in params if p.default is inspect.Parameter.empty)
    return required, len(params)


def as_call(value: Any, bounds: tuple[int, int]) -> list[Any] | None:
    """One proposed input as the argument list of a call, or None when it
    cannot be one. A lone argument written without its list is wrapped."""
    low, high = bounds
    if isinstance(value, list) and low <= len(value) <= high:
        return value
    if low <= 1 <= high:
        return [value]
    return None


def outcomes(python: str, fn: str, reference: str, calls: list[list[Any]]) -> list[Any]:
    """The reference's result on each call, as the judge's child reports it.
    A call the child cannot finish is None."""
    try:
        results = judge.run_child(python, fn, calls, reference)
        return list(results)
    except Exception:  # noqa: BLE001 - the batch failed; try the calls one by one
        pass
    one_by_one: list[Any] = []
    for call in calls:
        try:
            one_by_one.append(judge.run_child(python, fn, [call], reference)[0])
        except Exception:  # noqa: BLE001 - any failure drops the call, and only it
            one_by_one.append(None)
    return one_by_one


def kept(result: Any, prompt: str) -> bool:
    """A result the prompt specifies: a value, or an exception it names."""
    if not isinstance(result, list) or len(result) != 2:
        return False
    kind, value = result
    return kind == "ok" or (kind == "raise" and isinstance(value, str) and value in prompt)


def build_spec(python: str, task: dict[str, Any], fn: str, proposed: list[Any]) -> dict[str, Any] | None:
    """The verify spec for the proposed inputs, or None when they cannot make
    an informative one."""
    reference = task["completion"]
    bounds = arity(reference, fn)
    calls: list[list[Any]] = []
    seen: set[str] = set()
    for value in proposed:
        call = as_call(value, bounds)
        key = json.dumps(call, sort_keys=True)
        if call is not None and key not in seen:
            seen.add(key)
            calls.append(call)
    if not calls:
        return None
    results = outcomes(python, fn, reference, calls)
    pairs = [(c, r) for c, r in zip(calls, results, strict=True) if kept(r, task["prompt"])]
    distinct = {json.dumps(r, sort_keys=True) for _, r in pairs}
    if len(pairs) < MIN_INPUTS or len(distinct) < 2:
        return None
    cases = [c for c, _ in pairs]
    return judge.verify_spec(fn, cases, judge.digest([r for _, r in pairs]))


def mutation_sites(tree: ast.AST) -> list[tuple[int, int]]:
    """Every (node, variant) a single-point mutation can change, in walk order."""
    sites: list[tuple[int, int]] = []
    for index, node in enumerate(ast.walk(tree)):
        sites.extend((index, v) for v in range(len(variants(node))))
    return sites


def variants(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Compare):
        return [f"op{i}" for i, op in enumerate(node.ops) if type(op) in COMPARE_SWAPS]
    if isinstance(node, ast.BinOp | ast.AugAssign) and type(node.op) in BINOP_SWAPS:
        return ["op"]
    if isinstance(node, ast.BoolOp):
        return ["op"]
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return ["+1", "-1"]
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return ["drop-last", "reverse"] if len(set(node.value)) > 1 else ["drop-last"]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in CALL_SWAPS:
        return ["call"]
    if isinstance(node, ast.Attribute) and node.attr in METHOD_SWAPS:
        return ["method"]
    return []


def mutate(node: ast.AST, variant: str) -> None:
    if isinstance(node, ast.Compare):
        i = int(variant[2:])
        node.ops[i] = COMPARE_SWAPS[type(node.ops[i])]()
    elif isinstance(node, ast.BinOp | ast.AugAssign):
        node.op = BINOP_SWAPS[type(node.op)]()
    elif isinstance(node, ast.BoolOp):
        node.op = BOOLOP_SWAPS[type(node.op)]()
    elif isinstance(node, ast.Constant) and isinstance(node.value, int):
        node.value = node.value + (1 if variant == "+1" else -1)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = node.value[:-1] if variant == "drop-last" else node.value[::-1]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        node.func.id = CALL_SWAPS[node.func.id]
    elif isinstance(node, ast.Attribute):
        node.attr = METHOD_SWAPS[node.attr]


def mutants(reference: str, limit: int, seed: int) -> list[str]:
    """Up to `limit` distinct single-point mutants of `reference`, each
    different from it, chosen the same way on every run."""
    tree = ast.parse(reference)
    original = ast.unparse(tree)
    found: set[str] = set()
    for index, v in mutation_sites(tree):
        changed = copy.deepcopy(tree)
        node = list(ast.walk(changed))[index]
        mutate(node, variants(node)[v])
        source = ast.unparse(ast.fix_missing_locations(changed))
        if source != original:
            found.add(source)
    ordered = sorted(found)
    random.Random(seed).shuffle(ordered)
    return ordered[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--mutants", type=int, default=MUTANTS, help="mutants per task at most")
    parser.add_argument("--python", default=judge.interpreter(), help="interpreter the judge runs under")
    args = parser.parse_args()

    tasks = {t["id"]: t for t in json.loads(args.corpus.read_text(encoding="utf-8"))}
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for proposal in proposals:
        task = tasks.get(proposal["task"])
        if task is None or task.get("impossible") or "completion" not in task:
            continue
        fn = proposal["function"]
        spec = build_spec(args.python, task, fn, proposal["inputs"])
        print(f"{task['id']}: {len(proposal['inputs'])} proposed, {'a spec' if spec else 'no spec'}", flush=True)
        if spec is not None:
            kept_inputs = len(json.loads(spec["args"][4]))
            specs.append(
                {
                    "domain": proposal["domain"],
                    "task": task["id"],
                    "tier": "reducible",
                    "spec": spec,
                    "by": f"{proposal['proposed_by']}: {kept_inputs} inputs, reference outputs",
                }
            )
        forged = generate.forgeries(fn, None).values()
        made = mutants(task["completion"], args.mutants, SEED)
        distinct = dict.fromkeys([*forged, *made])
        artifacts.extend({"task": task["id"], "completion": c, "deliberate": True} for c in distinct)
    args.specs.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8", newline="\n")
    args.artifacts.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(specs)} spec(s) from {len(proposals)} proposal(s); {len(artifacts)} artifact(s) to label")


if __name__ == "__main__":
    main()

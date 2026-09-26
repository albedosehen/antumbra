"""The judge every workbench task is verified by, and the verify spec that runs it.

`CommandVerifier` passes a task when its program exits 0, with the candidate's
code in `ANTUMBRA_COMPLETION`. The shipped corpora ran that code inside the
judging process (`exec(completion)` followed by `sys.exit(0 if ... else 1)`),
so a candidate that simply exited with status 0 -- `raise SystemExit` is enough
-- passed every task without defining anything. Under RAFT such a sample is a
verified winner and is trained on.

This judge never runs the candidate in its own process. It starts a child
interpreter, feeds it the test inputs on stdin, and reads back one result line.
The child's exit status, anything else it prints, and anything it patches in
its own interpreter cannot make the parent exit 0. Only a result line whose
canonical form hashes to the expected digest can, and the expected outputs are
held nowhere in the judge -- only their SHA-256 -- so reading the parent's
command line reveals the inputs and a hash, not the answers.

What it does not defend against: a candidate that finds a reference solution on
disk. The impossible tasks exist to catch shortcuts of that kind, because no
faithful reading of their prompt can produce the value they expect.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Final

# Printed once by the child, before the canonical JSON of its results.
MARK: Final = "@@workbench-judge@@ "

# Seconds the child may run: the per-task budget inside CommandVerifier's own
# ten-second cap, leaving room for the parent's interpreter start.
CHILD_TIMEOUT: Final = 5

# Runs in the child. Reads the inputs before executing the candidate, so the
# candidate cannot consume them from stdin, then calls the named function once
# per case. A raised exception is a result (its type name), because some specs
# require one; anything that escapes `Exception` -- SystemExit included -- ends
# the child before the result line is written, which the parent reads as a fail.
#
# Numbers are compared as values, the way `==` compares them: a float is
# rounded to nine decimal places and written as an int when it is whole, so
# 212 and 212.0 agree, and so do 98.6 and 98.60000000000001. The helpers are
# bound before the candidate runs, so rebinding a builtin cannot reach them.
CHILD: Final = """\
import json, math, os, sys
def _normal(v, _round=round, _isfinite=math.isfinite, _int=int, _repr=repr):
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if not _isfinite(v):
            return _repr(v)
        v = _round(v, 9)
        return _int(v) if v.is_integer() else v
    if isinstance(v, (list, tuple)):
        return [_normal(x) for x in v]
    if isinstance(v, dict):
        plain = (str, int, float, bool)
        return {(k if isinstance(k, plain) or k is None else _repr(k)): _normal(x) for k, x in v.items()}
    return _repr(v)
name = sys.argv[1]
cases = json.loads(sys.stdin.read())
namespace = {"__name__": "candidate"}
exec(compile(os.environ.get("ANTUMBRA_COMPLETION", ""), "<candidate>", "exec"), namespace)
function = namespace[name]
results = []
for args in cases:
    try:
        results.append(["ok", _normal(function(*args))])
    except Exception as error:
        results.append(["raise", type(error).__name__])
sys.stdout.write("\\n" + MARK + json.dumps(results) + "\\n")
sys.stdout.flush()
""".replace("MARK", repr(MARK))

# Runs in the parent, the process CommandVerifier waits on. Its only route to
# exit 0 is a single result line whose canonical form matches the digest.
PARENT: Final = (
    """\
import hashlib, json, os, subprocess, sys
name, cases, digest = sys.argv[1], sys.argv[2], sys.argv[3]
env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
env["PYTHONHASHSEED"] = "0"
env["PYTHONIOENCODING"] = "utf-8"
try:
    run = subprocess.run(
        [sys.executable, "-s", "-c", CHILD, name],
        input=cases, capture_output=True, text=True, encoding="utf-8",
        env=env, timeout=TIMEOUT,
    )
except Exception:
    sys.exit(1)
lines = [line for line in run.stdout.splitlines() if line.startswith(MARK)]
if run.returncode != 0 or len(lines) != 1:
    sys.exit(1)
try:
    results = json.loads(lines[0][len(MARK):])
except ValueError:
    sys.exit(1)
canonical = json.dumps(results, sort_keys=True, separators=(",", ":"))
sys.exit(0 if hashlib.sha256(canonical.encode("utf-8")).hexdigest() == digest else 1)
""".replace("CHILD", repr(CHILD))
    .replace("MARK", repr(MARK))
    .replace("TIMEOUT", str(CHILD_TIMEOUT))
)


def canonical(results: Any) -> str:
    """The form both sides hash: sorted keys, no insignificant whitespace."""
    return json.dumps(results, sort_keys=True, separators=(",", ":"))


def digest(results: Any) -> str:
    return hashlib.sha256(canonical(results).encode("utf-8")).hexdigest()


def verify_spec(name: str, cases: list[list[Any]], expected_digest: str) -> dict[str, Any]:
    """The `verify` object CommandVerifier consumes for one task."""
    return {
        "program": "python",
        "extract_code": True,
        "args": ["-s", "-c", PARENT, name, json.dumps(cases), expected_digest],
    }


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_child(python: str, name: str, cases: list[list[Any]], source: str) -> Any:
    """Run `source` through the child exactly as the judge would, returning its
    results. Used to compute a task's expected outputs from its reference, so
    the expected values come from the same code path that will check them."""
    env = child_env()
    env["ANTUMBRA_COMPLETION"] = source
    run = subprocess.run(
        [python, "-s", "-c", CHILD, name],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=CHILD_TIMEOUT,
        check=False,
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(MARK)]
    if run.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"reference for {name} did not produce one result: {run.stderr[-400:]}")
    return json.loads(lines[0][len(MARK) :])


def judge(python: str, spec: dict[str, Any], completion: str) -> bool:
    """Run a verify spec the way CommandVerifier does: the fenced code (when the
    spec asks for extraction) in ANTUMBRA_COMPLETION, and exit 0 means pass."""
    code = extract_code_block(completion) if spec.get("extract_code") else completion
    env = dict(os.environ)
    env["ANTUMBRA_COMPLETION"] = code
    run = subprocess.run(
        [python, *spec["args"]],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return run.returncode == 0


def extract_code_block(text: str) -> str:
    """Mirror of antumbra-critic's `extract_code_block`: the first fenced block's
    body, or the trimmed text when there is no fence."""
    start = text.find("```")
    if start < 0:
        return text.strip()
    after = text[start + 3 :]
    newline = after.find("\n")
    body = after[newline + 1 :] if newline >= 0 else after
    end = body.find("```")
    return (body[:end] if end >= 0 else body).strip()


def interpreter() -> str:
    """The Python the judge runs under: ANTUMBRA_PYTHON when set, as the
    verifier resolves it, else this interpreter."""
    return os.environ.get("ANTUMBRA_PYTHON", "").strip() or sys.executable

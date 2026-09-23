# The workbench corpus

A corpus built for the loop to be measured on: 349 Python function-writing tasks across eight skills and three spec sizes, each checked by a judge that cannot be passed without computing the right answers. The corpora beside it are one to three tasks each, which is enough to see training happen and too few to hold anything out ([ADR-0022](../../docs/adr/0022-governed-self-improvement.md)).

| file           | what it is                                                                     |
| -------------- | ------------------------------------------------------------------------------ |
| `all.json`     | every task; the corpus for an instrumented run                                 |
| `<skill>.json` | one skill's tasks, including its two impossible ones; for growing a specialist |
| `generate.py`  | writes the files above, and refuses to unless every task holds up              |
| `judge.py`     | the judge, and the verify spec that runs it                                    |
| `legacy.py`    | rebuilds the older corpora's verifiers on the same judge                       |
| `families/`    | the task templates, one module per skill                                       |
| `calibrate.py` | summarises base-model eval reports into where the corpus has headroom          |
| `calibration/` | the reports behind the calibration below                                       |

## The judge

`CommandVerifier` passes a task when its program exits 0. The older corpora ran the candidate inside the judging process and then exited with the verdict, so a candidate that exited first (`raise SystemExit` is enough) passed every task without defining anything. Under RAFT a sample like that is a verified winner and is trained on.

This judge starts a child interpreter for the candidate, sends it the inputs on stdin, and reads back exactly one result line. The candidate's exit status, anything else it prints, and anything it patches inside its own interpreter cannot make the parent exit 0. Only results whose canonical form hashes to the expected digest can. The expected outputs are stored nowhere, not even in the task: only their SHA-256 is, so reading the judge's command line reveals the inputs and a hash but not the answers. Numbers compare as values, the way `==` compares them, so 212 and 212.0 agree and floats agree to nine decimal places.

What it does not stop is a candidate that finds a reference solution on disk. The impossible tasks are there to catch shortcuts like that.

## Tasks

Each skill has six templates, two at each spec size. A template's parameters change the behaviour asked for, not just the inputs: a shift of 7 rather than 3, keeping the first duplicate rather than the last. So every task is new text, and none is copied from a public benchmark the base model will have seen. Some tasks carry `also` for the other skill they draw on (`group_words` is strings as much as mappings), which gives the gate cases that more than one expert could serve (ADR-0024 D-1).

| skill     | small | medium | large | impossible | total |
| --------- | ----: | -----: | ----: | ---------: | ----: |
| strings   |    16 |     14 |    16 |          2 |    48 |
| lists     |    16 |     12 |    14 |          2 |    44 |
| numbers   |    16 |     16 |    16 |          2 |    50 |
| mappings  |    12 |     12 |    14 |          2 |    40 |
| parsing   |    16 |     16 |    16 |          2 |    50 |
| grids     |    14 |     11 |    12 |          2 |    39 |
| sequences |    16 |     12 |    10 |          2 |    40 |
| dates     |    20 |     11 |     5 |          2 |    38 |
| **all**   |       |        |       |            |   349 |

Prompt length, which the trainer uses as task size, has a median of about 190 characters for a small spec, 310 for a medium one and 480 for a large one. That spread is what gives the visible-minus-held-out gap its size bands.

Under the default partition (seed 0) the 333 satisfiable tasks split 236 visible, 57 held out and 40 audited. Every skill file has at least four held-out and three audit tasks, which is thin for a trend. So read the instruments on `all.json`, and use the skill files to grow one specialist per skill.

**Impossible tasks** (`"impossible": true`, two per skill) ask for something no function can do, such as a string both longer and shorter than its input, and their judge expects values that were drawn at random, hashed, and then thrown away. They are never trained on. Under a holdout they are measured on every generation, and a pass fails that generation outright. Their digests change every time the corpus is regenerated, on purpose, and `--check` ignores them.

Every satisfiable task carries its reference solution as `completion`. RAFT and GRPO ignore it; capture (`teach`) learns from it.

## Using it

```bash
# an instrumented run: held-out gap, audit trend and impossible tasks per generation
antumbra train --corpus corpora/workbench/all.json --holdout --generations 10

# one specialist for one skill
antumbra train --corpus corpora/workbench/dates.json --run dates
```

The verifier runs `python`. The GPU image ships one. On Windows set `ANTUMBRA_PYTHON`, because the `python` on `PATH` is usually the Store stub.

## Changing it

```bash
uv run --python 3.12 python corpora/workbench/generate.py          # regenerate
uv run --python 3.12 python corpora/workbench/generate.py --check  # files match the generator
uv run --python 3.12 python corpora/workbench/legacy.py --check    # older corpora match theirs
uv run --python 3.12 python -m unittest discover -s corpora/workbench
ANTUMBRA_PYTHON=<python> cargo test -p antumbra-critic --test corpora -- --ignored
```

Before writing anything, the generator runs every reference through the judge exactly as `CommandVerifier` would, and runs eight completions that solve nothing: empty code, a stub returning `None`, one returning its first argument, one returning the first case's answer every time, three ways of exiting 0, and a forged result line. Every reference must pass and every one of those must fail. A single exception stops the run. This is the no-model baseline [ADR-0024](../../docs/adr/0024-typed-decisions.md) asks every constructed benchmark to be checked against. The unit tests show that these checks do fail when they should. The Rust tests repeat them through the real verifier and fail as soon as any Python verifier under `corpora/` can be passed by exiting.

Generation is deterministic: two runs under different hash seeds produce the same files. Lint and types come from `uvx ruff check`, `uvx ruff format --check` and `uvx mypy --strict`, all run from this directory.

## Calibration

Measured on 2026-09-23 against the model `train` starts from, `Qwen2.5-Coder-1.5B-Instruct`, using training's own sampling settings: 4 draws per task, temperature 0.8, no nucleus sampling, bf16, 256 new tokens, on an RTX 3090 Ti. The reports are in `calibration/`, and `calibrate.py calibration` rebuilds the full table (skill by size) from them.

| skill     | pass rate | never | sometimes | always |
| --------- | --------: | ----: | --------: | -----: |
| dates     |      0.32 |   36% |       61% |     3% |
| grids     |      0.32 |   41% |       49% |    11% |
| lists     |      0.56 |   17% |       57% |    26% |
| mappings  |      0.57 |   11% |       63% |    26% |
| numbers   |      0.26 |   42% |       54% |     4% |
| parsing   |      0.27 |   48% |       46% |     6% |
| sequences |      0.64 |   16% |       58% |    26% |
| strings   |      0.35 |   30% |       61% |     9% |
| _small_   |      0.56 |   13% |       63% |    23% |
| _medium_  |      0.43 |   21% |       67% |    12% |
| _large_   |      0.19 |   61% |       35% |     4% |
| **all**   |  **0.40** |   31% |   **56%** |    14% |

A task in the "sometimes" column is one RAFT can learn from: some draws pass and some fail, so there is a winner to prefer and a loser to prefer it over. 56% of the corpus sits there. The 31% that never pass are the frontier, and they are mostly large specs (61% of large tasks never pass; `grids` large never passes at all). The 14% that always pass teach nothing and measure regressions. No impossible task passed on any draw.

Running this calibration found two defects that had shaped every earlier GPU result, and both are fixed:

- **A chat answer was cut at its code fence.** The completion-model stops end at the first fence, so the Instruct model's code answers came back as the sentence introducing them, or empty. The first run scored `dates` at 0 of 144 draws.
- **The rotary table was built in bf16.** Neighbouring positions past 256 collapsed together, and the angles lost their precision, so greedy draws carried duplicated tokens ("longer than than 11"). With that bug the same measurement read 0.30 overall and 0.10 on large specs. On the 103 large tasks, greedy bf16 went from 18 passes to 21, matching f32.

To re-measure, run `scripts/workbench-calibrate.sh <sha>` on the GPU host.

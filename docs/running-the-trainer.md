# Running the trainer (MT-3)

The candle RAFT trainer is built and compiles; this is how to run it on a CUDA GPU (the 3090 Ti). Everything below the actual run is CPU-tested; the first real run is where model-specific behaviour gets tuned.

## Prerequisites

- An NVIDIA GPU + CUDA toolkit (candle's `cuda` feature links it). On Windows with CUDA 13.x see the dedicated build section below; the toolkit version needs handling.
- Rust 1.90+.
- A real `python` on `PATH` _only_ for corpora whose `verify` calls it (the Windows Store alias is not a real Python; see "No Python?" below).
- Network access for the _first_ run (downloads Qwen2.5-Coder-1.5B, ~3 GB, into the Hugging Face cache; already cached here).

## Build

```bash
cargo build -p antumbra-cli --features models,cuda --release
```

`models` pulls in the candle Qwen2.5-Coder + LoRA stack; `cuda` selects the GPU backend. Without `cuda` it runs on CPU (slow, but works for a smoke test).

## Run

```bash
cargo run -p antumbra-cli --features models,cuda --release -- \
  --url surrealkv://./data/antumbra.skv \
  train --corpus corpora/example-tasks.json --generations 1
```

Each generation: spawn a shadow → sample `K` completions per task → run each task's `verify` command (exit 0 = pass) → SFT the LoRA adapter on the winners → graduate if the pass-rate clears the threshold → persist the adapter + lineage to SurrealDB. Then inspect:

```bash
cargo run -p antumbra-cli -- --url surrealkv://./data/antumbra.skv status
cargo run -p antumbra-cli -- --url surrealkv://./data/antumbra.skv experts
```

### Measuring a run: `--holdout`

`train --holdout` withholds part of the corpus from training so each generation can be measured against tasks it never learned from ([ADR-0022](adr/0022-governed-self-improvement.md)'s standing instruments). The split is a pure function of each task id and a seed: about a fifth of the tasks are held out and a tenth audited, and the rest stay visible. Only visible tasks are trained on and counted in the fitness that decides graduation. Held-out tasks are measured in the final round of every generation. Audited tasks are measured every second generation (`LoopConfig::audit_every`). Each generation then prints its widest visible-minus-held-out gap by task size, the audit pass rate when the audit was due, and the audit trend over the run so far, and stores all three beside its fitness. The trend reads `inconclusive` until ten generations have been measured. After that it reads `carrying` when the audit slice rose with the fitness, `overtuning` when fitness climbed and the audit slice did not follow, and `flat` when fitness did not climb.

It is off by default because it changes what is learned, and because the demo corpora here are too small to split: both tasks in `corpora/arith.json` hash into the held-out slice, so `train --holdout` refuses that corpus rather than training on nothing. Use it on a corpus of real size, such as [`corpora/workbench/all.json`](../corpora/workbench/README.md): 349 tasks across eight skills and three spec sizes, including the impossible tasks the instruments need. Without it, generations carry no instruments rather than a gap computed over tasks that were all trained on.

### Measuring what each expert adds: `--contribution-every`

`train --contribution-every N` measures every shared expert's leave-one-out contribution in every N-th generation (ADR-0022 S-5). Each expert is masked in turn, and the tasks it served are routed again, to the next expert or to the base model. Both sides are then scored under the same seeds.
- **Output:** each generation prints, per expert, how many live tasks were routed to it, its score with and without, and the difference. An expert nothing was routed to prints as unused.
- **The record:** rows go to the `contribution` table, the history retirement reads.
- **Baseline:** each measurement also compares the routed population with its best single expert on the same tasks, and prints what routing adds. It is kept in `population_baseline` whether or not the answer flatters the population.
- **Retirement:** with contribution measured, `--retire-after N` (3 by default, 0 for off) demotes an expert to dormant once its contribution was at or below nothing in N consecutive measurements, each on at least two tasks. The generation prints the demotion with its evidence. Advisory warnings print too: a falling routing share, an expert that went unused, drift in what it is asked. They change nothing. `antumbra revive` undoes a demotion, and the count starts afresh.
- **Cost:** about two evaluations of the live tasks (the visible slice, at most 32 tasks) each time it runs. On the 3090 Ti, one expert against the base model over 30 tasks, two seeds of four samples, took 13 minutes, about as long as the generation's training. It is off by default.

### Admission: `--duplicate-above`

A graduate whose capability vector is at least `--duplicate-above` (0.95 by default) similar to an active shared expert's is a twin (ADR-0022 S-5). It joins only if it beats that expert head to head on the live tasks, under the same seeds, and it then takes that expert's place. The other is archived, not deleted, and `antumbra revive` brings it back.
- **A twin that loses** is not admitted, and its shadow is pruned.
- **Output:** each generation prints the decision, with the similarity and both scores.
- **Why:** two experts from one corpus have near-identical capability vectors. The heuristic gate routes on the margin between its top two, so with both present it escalated every task and served neither.
- **Turning it off:** a value above 1 admits every graduate.

### Searching the recipe: `--search`

`train --search` trains a cohort each generation instead of one shadow (`--cohort`, 4 by default), each member under a recipe the search proposes (ADR-0022 S-1).
- **What is searched:** learning rate and batch size, plus the KL weight under `--algo grpo`.
- **Every member trains from the base.** Only the best member's recipe is carried forward, and it leads the next generation's cohort. No member starts from another's weights, so each graduate is a skill of its own.
- **Two frequencies:**
  - **Slow members:** the last `--slow` members of the cohort (a third of it by default) keep their recipe for `--slow-interval` generations (3). Nothing the fast members score can replace a slow member's recipe sooner.
  - **Fast members:** they are proposed a new recipe every generation at first. Their interval lengthens over `--anneal` generations (the run's length by default) and stops one short of the slow interval.
  - **Ranking:** a held recipe is measured again each generation, and a recipe run several times is ranked on all its runs together. A well-measured recipe is not displaced by a newcomer's one lucky run.
- **Graduation:** the best member is judged on a re-measurement, not on the training fitness it was picked for. Picking the best of a noisy few overstates it.
  - Its adapter is evaluated again under three fresh seeds (`--remeasure N` to change the count, 0 for off).
  - With `--holdout`, the tasks are the held-out slice, which it never trained on. Without it, the tasks are the ones it trained on, re-drawn.
  - The threshold applies to the mean.
- **Output:** each generation prints every member's recipe and fitness, and the score graduation was judged on, including the re-measured pass rates. Every member's recipe is stored as a `recipe` row, and those rows are the history later generations are proposed from.
- **Cost:** a generation takes the cohort size times as long, plus the re-measurement. On the 3090 Ti, two generations of three over the workbench `sequences` corpus (`--samples 4 --rounds 2 --holdout`) took 53 minutes and peaked at 14.6 GB of the card, whatever the batch size (`scripts/search-validate.sh` runs exactly that). Combine it with `--holdout` on a corpus of real size, so the audit trend can tell a search that improves from one that overtunes.

## Corpus format

A JSON array of tasks. `verify.program`/`args` run after generation with the candidate completion in `$ANTUMBRA_COMPLETION`; `extract_code: true` pulls the code out of a markdown fence first.

```json
[
  {
    "id": "py-add",
    "prompt": "# Write a Python function `add(a, b)` that returns their sum.\n",
    "verify": {
      "program": "python",
      "extract_code": true,
      "args": ["-c", "import os,sys; ns={}; exec(os.environ['ANTUMBRA_COMPLETION'], ns); sys.exit(0 if ns['add'](2,3)==5 else 1)"]
    }
  }
]
```

## Tuning knobs (in `RaftConfig` / `LoopConfig`)

- `samples_per_task` (K), `rounds`, `lora_rank`/`lora_alpha`, `learning_rate`.
- `graduate_threshold` (loop): pass-rate needed to freeze an adapter.
- generation: `max_new_tokens` and `DEFAULT_STOPS` in `decode.rs` end the completion at the function boundary.

## Generation quality (the validated recipe, 2026-06-05)

Getting clean, _correct_ output from a small LoRA over few examples took four research-grounded fixes (each diagnosed against a real GPU failure, citations in the experiment ledger). The end state: a 10-example "deno install &lt;pkg&gt;" corpus trains an expert that emits the right command for trained **and held-out** packages.

1. **Decode policy** (`decode.rs::DecodePolicy`): a repetition penalty + no-repeat n-gram + nucleus top-p, applied to the generated continuation only. Without it greedy decoding loops (`axios axios axios…`). `RaftConfig::for_serving` turns it on for `ask`/`serve`/`answer`; training leaves it off.
2. **Instruct base + chat template**: the default base is `Qwen2.5-Coder-1.5B-Instruct`; `QwenCausalLm` detects the `-Instruct` name and wraps every prompt in the Qwen chat template (and stops at `<|im_end|>`). The raw completion base rambles word-salad; the instruct base produces clean, well-formed commands. Until 2026-09-23 only `consolidate`, `populate` and `evolve` actually loaded it: `train`, `teach`, `memory-import` and `metabolize` handed the loop the raw base by name, and `eval` defaulted to it. Results from those commands before that date were produced by the raw base. The same change stopped cutting a chat answer at its code fence. The completion-model stops end at the first fence, which left the Instruct base's code answers empty. It did not affect the text recipe above, whose answers have no fence.
3. **Shuffle the SFT order** every step (`sft_step`): per-example SGD over a fixed-order corpus collapses to the _last_ example. Shuffling lets the LoRA learn the prompt-conditioned mapping instead of memorizing the tail.
4. **Learning rate**: batch-of-1 SGD + shuffle is unstable at `lr 1e-3` (it over-updates into `den den` fragments). `3e-4` is stable ("Beware of the Batch Size" effect) and is the `--lr` default of the capture commands (`teach`, `consolidate`, `memory-import`, `metabolize`). `train` has no `--lr` and uses `RaftConfig`'s `1e-4`; `3e-4` was validated for capture on text tasks, not for RAFT.

5. **Batch size** (opt-in, `RaftConfig::batch_size` / `consolidate --batch-size 4`, which replaced `--grad-accumulation`): step on the mean gradient of a small batch instead of one example at a time. Batch-of-1 SGD chases each example's noisy gradient; the mini-batch mean is far less noisy, so a higher learning rate stays stable. GPU-validated: at `lr 3e-4` it matches the per-example baseline (internalized 1.00, held-out `fastify` generalizes), and at `lr 1e-3` -- the rate that degenerates to `den den` under batch-of-1 -- it is stable and correct. Off by default; the per-example shuffle recipe above is the validated standard. (Each example is backpropagated as soon as its loss is computed and its adapter gradients are summed, keyed by the trained variables themselves: candle's optimizer finds a gradient by its variable's identity, so a store keyed by anything else steps nothing. One forward graph is alive at a time, so the batch size costs steps, not memory. Stepping on the mean _loss_ instead held every example's graph until one backward, and batch 2 on workbench code ran a 24 GB card out.)

Open levers if quality is still short: more examples per skill, denser-than-binary rewards, and a proper eval harness (pass@k at low temperature, not a single `ask`).

## Known first-run caveats

- **Precision:** trains in the configured dtype, bf16 by default on a GPU and f32 on the CPU. If memory is tight, lower K or sequence length before reaching for quantization (MT-4).
- **Memory:** every product against a frozen base weight goes through `frozen::frozen_matmul_t`, whose backward computes the input's gradient alone. candle's own matmul also builds a gradient for the weight, which nothing reads, so each step used to materialize one for the whole base. On the 1.5B base with workbench code, one example's SFT step peaked at 19.2 GB of the card; through the frozen product it peaks at 11.1 GB.
- **Prompt shape:** the default base is now the instruct variant with the chat template applied automatically (see "Generation quality" above), so prompts no longer need to be bare comments. To train on the raw completion base instead, set a non-`Instruct` `base_model` and the chat wrapping switches off.
- **Speed:** a full pass is `K × tasks × rounds` generations; start with `--generations 1`, small K.
- **Verifier safety:** `verify` runs real commands - point it at a sandboxed corpus, not arbitrary input.

## Building for CUDA on Windows (CUDA 13.3 + MSVC): validated 2026-06-05

candle 0.10 compiles its CUDA kernels with `nvcc` (which needs the MSVC host compiler) and pins **cudarc 0.19**, whose build script only knows CUDA toolkits **≤ 13.2**. A newer toolkit (13.3) trips two failures; both are handled:

1. **cudarc version panic** (`Unsupported cuda toolkit version: 13.3`). cudarc's probe runs bare `nvcc` (found via `PATH`) and panics on an unknown version. The `cudarc/fallback-latest` feature (wired into `antumbra-train`'s `cuda` feature) makes it fall back to its newest supported version (13.2, ABI- compatible with the 13.3 libs), but _only when the `nvcc` probe fails to run_. So **keep `nvcc` off `PATH` at build time**; candle-kernels still finds it via `CUDA_PATH`.
2. **CCCL preprocessor error** (`C1189: MSVC/cl.exe with traditional
   preprocessor`). CUDA 13's CCCL headers require the conforming preprocessor; forward it to `cl.exe` with `NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor`.

The validated build environment (all set before `cargo`; **`%CUDA_PATH%\bin` is deliberately NOT on `PATH`** so cudarc falls back):

```bat
call "...\VC\Auxiliary\Build\vcvars64.bat"            rem cl.exe + INCLUDE/LIB for nvcc
set "CUDA_PATH=...\CUDA\v13.3"
set "CUDA_ROOT=...\CUDA\v13.3"                         rem candle-kernels finds nvcc + libs here
set "NVCC_PREPEND_FLAGS=-Xcompiler=/Zc:preprocessor"  rem CUDA 13 CCCL needs the conforming preprocessor
cargo build -p antumbra-cli --features models,cuda
```

**At run time** (not build time) the CUDA runtime DLLs must be loadable, so put both on `PATH` then run the built binary directly (avoids a cargo rebuild that would re-trip the probe):

```bat
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"  rem bin\x64 = runtime DLLs (CUDA 13 moved them)
target\debug\antumbra.exe --url surrealkv://./data/antumbra.skv train --corpus corpora/arith.json --generations 1
```

Release builds also need the `surrealdb` / `surrealdb-core` `opt-level = 1` overrides in the root `Cargo.toml` (rustc ICEs optimizing them at opt 3).

**Python for verifiers.** The example corpora's verifiers shell out to `python`. On Windows the `python` command is often the **Store alias stub** (prints "Python was not found" and fails _every_ verify). For `train` this means nothing graduates. For `consolidate`/`capture` the failure is more insidious: graduation still happens (the gate is internal, no python), but capture only trains on corrections it can _re-verify_, so a broken verifier yields zero winners, an empty SFT batch, and a saved adapter that is the **untrained init** -- `internalized
0.00` and `ask` returns the base model's prose, even though the run looked successful. (capture now prints a `warning: 0 of N correction(s) verified` line for exactly this.) Two fixes:

- Point the verifier straight at a real interpreter with **`ANTUMBRA_PYTHON`** (the `CommandVerifier` substitutes it for `python`/`python3`), e.g. `set "ANTUMBRA_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"`, more robust than fighting `PATH` order.
- Or, to validate the GPU path with no Python at all, use a corpus whose `verify` is `cmd /C exit 0` (e.g. `corpora/smoke.json`): RAFT treats every completion as a pass, which still trains and serves a real adapter (generation quality just isn't gated).

**Driver requirement:** the GPU driver must support the _toolkit_ version, or PTX load fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Check `nvidia-smi` (driver 610.47 here covers CUDA 13.3). If the driver is older than the toolkit, update it or install a matching (lower) toolkit.

**Note (2026-06-10):** a Developer-shell alternative to the `vcvars64.bat` + `fallback-latest` recipe above also works and is what the autonomous-server validation used: enter a VS dev shell (`Enter-VsDevShell` via `vswhere`, which puts `cl.exe` on `PATH`), put `%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin` on `PATH`, and pin cudarc explicitly with `CUDARC_CUDA_VERSION=13020` (13.2 bindings, ABI-compatible with the 13.3 runtime) instead of relying on the off-`PATH` fallback. Either path produces the same binary; the non-negotiable is that **`cl.exe` must be on `PATH`** when candle's kernels compile (a fresh kernel build with no `cl.exe` fails as `nvcc fatal: Cannot find compiler 'cl.exe'`, sometimes surfaced as an empty `nvcc error`).

## The autonomous server (serving + consolidation)

The default Docker stack ([`docker/Dockerfile`](../docker/Dockerfile)) builds the **light** server: it does memory (store / recall / route / compartments / sync), but `answer` reports _serving not configured_ and `--auto-consolidate` is a no-op, because both need the candle/GPU half. To get real expert serving **and** autonomous consolidation (a reinforced or high-confidence memory auto-graduates its compartment into a private expert on the GPU, hot-registered so `answer` serves it without a restart), run the `models,cuda` server where the GPU is.

**Docker (Linux, or Windows via WSL2; needs the NVIDIA Container Toolkit):**

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d
```

The override ([`docker/docker-compose.gpu.yml`](../docker/docker-compose.gpu.yml)) swaps the mcp service for the [`Dockerfile.cuda`](../docker/Dockerfile.cuda) build, requests the GPU, adds `--auto-consolidate`, and mounts volumes for the base weights and trained adapters. Set `CUDA_COMPUTE_CAP` in `Dockerfile.cuda` to your card's arch (86 = RTX 30-series, 89 = 40-series). On a CDI host (NixOS, or any daemon without a named nvidia runtime) add `-f docker/docker-compose.gpu-cdi.yml` last. _The image is validated on a Linux GPU host (EXP-022) and still not built in CI, which has no GPU._

**Native (e.g. a Windows GPU box):** build `antumbra-mcp` per the CUDA section above (`--features models,cuda`), then run it with the runtime `PATH` set:

```bat
target\release\antumbra-mcp.exe --http 127.0.0.1:8081 --url ws://127.0.0.1:8000/rpc ^
  --db-user root --db-pass %SURREAL_PASS% ^
  --embedder-url http://127.0.0.1:11434/v1/embeddings --embedder-model all-minilm ^
  --auto-consolidate
```

Either way, the gate defaults are deliberately conservative (recurrence ≥ 2, confidence ≥ 0.5; `world`/`bank` facts need a check, `opinion`s graduate on a ≥ 0.9 provenance tier). The autonomous capture is light (8 rounds) since it re-fires and supersedes; the manual `antumbra consolidate-compartment` keeps the heavier 40-round capture for a deliberate one-off.

**What the server log tells you.** Every outcome of the trigger is one `[auto-consolidate]` line: `N graduated -> expert:... (internalized 1.00, now servable)` after a mint; `0 of 400 graduate (397 under-reinforced, 3 volatile)` when nothing clears the gate, said once per change and not on every write; `N graduated but the capture did not learn` when the verifier passed nothing. Recurrence is the reinforcement _count_, so memories that were each reinforced once are all under-reinforced: that one line is usually the answer to "why has nothing trained". A write that arrives while its compartment is training is not lost to the trigger; the run in flight is followed by another.

**The GPU-gated tests.** CI has no GPU, so the tests that close this loop are `#[ignore]`d there. On a GPU host, `just test-gpu` builds the `gpu-test` target of `Dockerfile.cuda` and runs the server's whole test binary with them included (pass `gpu="--gpus all"` on a daemon with a named nvidia runtime). Run it before tagging a release.

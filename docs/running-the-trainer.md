# Running the trainer (MT-3)

The candle RAFT trainer (ADR-0010) is built and compiles; this is how to run it
on a CUDA GPU (the 3090 Ti). Everything below the actual run is CPU-tested; the
first real run is where model-specific behaviour gets tuned.

## Prerequisites

- An NVIDIA GPU + CUDA toolkit (candle's `cuda` feature links it). On Windows
  with CUDA 13.x see the dedicated build section below — the toolkit version
  needs handling.
- Rust 1.90+.
- A real `python` on `PATH` *only* for corpora whose `verify` calls it (the
  Windows Store alias is not a real Python — see "No Python?" below).
- Network access for the *first* run (downloads Qwen2.5-Coder-1.5B, ~3 GB, into
  the Hugging Face cache; already cached here).

## Build

```bash
cargo build -p antumbra-cli --features models,cuda --release
```

`models` pulls in the candle Qwen2.5-Coder + LoRA stack; `cuda` selects the GPU
backend. Without `cuda` it runs on CPU (slow, but works for a smoke test).

## Run

```bash
cargo run -p antumbra-cli --features models,cuda --release -- \
  --url surrealkv://./data/antumbra.skv \
  train --corpus corpora/example-tasks.json --generations 1
```

Each generation: spawn a shadow → sample `K` completions per task → run each
task's `verify` command (exit 0 = pass) → SFT the LoRA adapter on the winners →
graduate if the pass-rate clears the threshold → persist the adapter + lineage
to SurrealDB. Then inspect:

```bash
cargo run -p antumbra-cli -- --url surrealkv://./data/antumbra.skv status
cargo run -p antumbra-cli -- --url surrealkv://./data/antumbra.skv experts
```

## Corpus format

A JSON array of tasks. `verify.program`/`args` run after generation with the
candidate completion in `$ANTUMBRA_COMPLETION`; `extract_code: true` pulls the
code out of a markdown fence first.

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
- generation: `max_new_tokens` and `DEFAULT_STOPS` in `decode.rs` end the
  completion at the function boundary.

## Generation quality (the validated recipe, 2026-06-05)

Getting clean, *correct* output from a small LoRA over few examples took four
research-grounded fixes (each diagnosed against a real GPU failure, citations in
the ADRs / experiment ledger). The end state: a 10-example "deno install &lt;pkg&gt;"
corpus trains an expert that emits the right command for trained **and held-out**
packages.

1. **Decode policy** (`decode.rs::DecodePolicy`): a repetition penalty + no-repeat
   n-gram + nucleus top-p, applied to the generated continuation only. Without it
   greedy decoding loops (`axios axios axios…`). `RaftConfig::for_serving` turns
   it on for `ask`/`serve`/`answer`; training leaves it off.
2. **Instruct base + chat template**: the default base is
   `Qwen2.5-Coder-1.5B-Instruct`; `QwenCausalLm` detects the `-Instruct` name and
   wraps every prompt in the Qwen chat template (and stops at `<|im_end|>`). The
   raw completion base rambles word-salad; the instruct base produces clean,
   well-formed commands.
3. **Shuffle the SFT order** every step (`sft_step`): per-example SGD over a
   fixed-order corpus collapses to the *last* example. Shuffling lets the LoRA
   learn the prompt-conditioned mapping instead of memorizing the tail.
4. **Learning rate**: batch-of-1 SGD + shuffle is unstable at `lr 1e-3` (it
   over-updates into `den den` fragments). `3e-4` (the new default) is stable
   ("Beware of the Batch Size" effect).

Open levers if quality is still short: true gradient accumulation (one step per
batch, not per example), more examples per skill, denser-than-binary rewards, and
a proper eval harness (pass@k at low temperature, not a single `ask`).

## Known first-run caveats

- **Precision:** v0 trains in f32 (so autograd flows through the frozen base
  into the LoRA factors). Fine for 1.5 B on 24 GB; if memory is tight, lower K
  or sequence length before reaching for quantization (MT-4).
- **Prompt shape:** the default base is now the instruct variant with the chat
  template applied automatically (see "Generation quality" above), so prompts no
  longer need to be bare comments. To train on the raw completion base instead,
  set a non-`Instruct` `base_model` and the chat wrapping switches off.
- **Speed:** a full pass is `K × tasks × rounds` generations; start with
  `--generations 1`, small K.
- **Verifier safety:** `verify` runs real commands - point it at a sandboxed
  corpus, not arbitrary input.

## Building for CUDA on Windows (CUDA 13.3 + MSVC) — validated 2026-06-05

candle 0.10 compiles its CUDA kernels with `nvcc` (which needs the MSVC host
compiler) and pins **cudarc 0.19**, whose build script only knows CUDA toolkits
**≤ 13.2**. A newer toolkit (13.3) trips two failures; both are handled:

1. **cudarc version panic** (`Unsupported cuda toolkit version: 13.3`). cudarc's
   probe runs bare `nvcc` (found via `PATH`) and panics on an unknown version.
   The `cudarc/fallback-latest` feature (wired into `antumbra-train`'s `cuda`
   feature) makes it fall back to its newest supported version (13.2, ABI-
   compatible with the 13.3 libs) — but *only when the `nvcc` probe fails to
   run*. So **keep `nvcc` off `PATH` at build time**; candle-kernels still finds
   it via `CUDA_PATH`.
2. **CCCL preprocessor error** (`C1189: MSVC/cl.exe with traditional
   preprocessor`). CUDA 13's CCCL headers require the conforming preprocessor;
   forward it to `cl.exe` with `NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor`.

The validated build environment (all set before `cargo`; **`%CUDA_PATH%\bin` is
deliberately NOT on `PATH`** so cudarc falls back):

```bat
call "...\VC\Auxiliary\Build\vcvars64.bat"            rem cl.exe + INCLUDE/LIB for nvcc
set "CUDA_PATH=...\CUDA\v13.3"
set "CUDA_ROOT=...\CUDA\v13.3"                         rem candle-kernels finds nvcc + libs here
set "NVCC_PREPEND_FLAGS=-Xcompiler=/Zc:preprocessor"  rem CUDA 13 CCCL needs the conforming preprocessor
cargo build -p antumbra-cli --features models,cuda
```

**At run time** (not build time) the CUDA runtime DLLs must be loadable, so put
both on `PATH` then run the built binary directly (avoids a cargo rebuild that
would re-trip the probe):

```bat
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"  rem bin\x64 = runtime DLLs (CUDA 13 moved them)
target\debug\antumbra.exe --url surrealkv://./data/antumbra.skv train --corpus corpora/arith.json --generations 1
```

Release builds also need the `surrealdb` / `surrealdb-core` `opt-level = 1`
overrides in the root `Cargo.toml` (rustc ICEs optimizing them at opt 3).

**Python for verifiers.** The example corpora's verifiers shell out to `python`.
On Windows the `python` command is often the **Store alias stub** (prints "Python
was not found" and fails *every* verify, so nothing graduates). Two fixes:

- Point the verifier straight at a real interpreter with **`ANTUMBRA_PYTHON`**
  (the `CommandVerifier` substitutes it for `python`/`python3`), e.g.
  `set "ANTUMBRA_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"` —
  more robust than fighting `PATH` order.
- Or, to validate the GPU path with no Python at all, use a corpus whose `verify`
  is `cmd /C exit 0` (e.g. `corpora/smoke.json`): RAFT treats every completion as
  a pass, which still trains and serves a real adapter (generation quality just
  isn't gated).

**Driver requirement:** the GPU driver must support the *toolkit* version, or PTX
load fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Check `nvidia-smi` (driver
610.47 here covers CUDA 13.3). If the driver is older than the toolkit, update it
or install a matching (lower) toolkit.

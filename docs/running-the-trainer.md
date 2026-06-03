# Running the trainer (MT-3)

The candle RAFT trainer (ADR-0010) is built and compiles; this is how to run it
on a CUDA GPU (the 3090 Ti). Everything below the actual run is CPU-tested; the
first real run is where model-specific behaviour gets tuned.

## Prerequisites

- An NVIDIA GPU + CUDA toolkit (candle's `cuda` feature links it).
- Rust 1.90+.
- `python` on `PATH` (only for the example corpus' verifier).
- Network access for the first run (downloads Qwen2.5-Coder-1.5B, ~3 GB, into
  the Hugging Face cache).

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

## Known first-run caveats

- **Precision:** v0 trains in f32 (so autograd flows through the frozen base
  into the LoRA factors). Fine for 1.5 B on 24 GB; if memory is tight, lower K
  or sequence length before reaching for quantization (MT-4).
- **Prompt shape:** the base completion model is prompted with a comment; if it
  rambles, tighten `DEFAULT_STOPS` or the prompt. An instruct variant would need
  a chat template.
- **Speed:** a full pass is `K × tasks × rounds` generations; start with
  `--generations 1`, small K.
- **Verifier safety:** `verify` runs real commands - point it at a sandboxed
  corpus, not arbitrary input.

## Building for CUDA on Windows (CUDA 13 + MSVC)

candle 0.10 builds its CUDA kernels with `nvcc` at compile time, which on Windows
needs the MSVC host compiler. The validated environment (all set before `cargo`):

```bat
call "...\VC\Auxiliary\Build\vcvars64.bat"            rem cl.exe + INCLUDE/LIB for nvcc
set "CUDA_PATH=...\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"  rem bin=nvcc, bin\x64=runtime DLLs (CUDA 13 moved them)
set "CUDARC_CUDA_VERSION=13000"                        rem cudarc tops out at 13.2; pin 13.0 (ABI-compatible)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"   rem CUDA 13 CCCL headers reject MSVC's traditional preprocessor
cargo run -p antumbra-cli --features models,cuda --release -- train --corpus corpora/smoke.json ...
```

Release builds also need the `surrealdb` / `surrealdb-core` `opt-level = 1`
overrides in the root `Cargo.toml` (rustc 1.96 ICEs optimizing them at opt 3).

**Driver requirement:** the GPU driver must support the *toolkit* version, or PTX
load fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. Check `nvidia-smi` ("CUDA
Version" = the driver's max); it must be **≥ the installed toolkit**. If it's
lower, either update the NVIDIA driver or install a matching (lower) toolkit and
set `CUDA_PATH` / `CUDARC_CUDA_VERSION` to it.

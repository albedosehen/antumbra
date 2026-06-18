@echo off
REM Distil the sentiment specialist on the local GPU (RTX 3090 Ti, CUDA 13.3).
REM
REM Loads the MSVC toolchain (vcvars64) so nvcc finds cl.exe, then sets the three
REM env vars the candle/cudarc CUDA build needs on this box, and runs `antumbra teach`
REM over the sentiment corpus with bf16 (f16 overflows to NaN logits on this stack).
REM
REM Usage:  scripts\teach_sentiment_gpu.cmd
REM Output: adapters\expert_sentiment-g0.safetensors  (+ the minted expert in mem:// store)

setlocal

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1

REM CCCL + MSVC traditional-preprocessor workaround, and pin cudarc to its newest
REM supported toolkit (13.2, ABI-compatible with the installed 13.3 libs).
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"
set "CUDARC_CUDA_VERSION=13020"

REM The verifier shells out to `python`; point it at a real interpreter so corrections
REM actually verify (else 0 verify -> UNTRAINED adapter, a silent no-op).
set "ANTUMBRA_PYTHON=C:\Users\shonp\AppData\Local\Programs\Python\Python312\python.exe"

REM `--features models,cuda`: `cuda` selects the GPU backend and now also pulls in `models`
REM (the cfg that gates the candle trainer into `teach`), so `--features cuda` alone is
REM equivalent; both are spelled out here for clarity. `--url` is an antumbra global arg, so
REM it goes AFTER the `--` separator (before the subcommand), not to cargo. mem:// =
REM ephemeral store; the run proves the loop, the minted adapter lands in adapters\.
cargo run --release -p antumbra-cli --features models,cuda -- --url mem:// teach --corpus corpora\sentiment.json --run expert:sentiment-g0 --rounds 40 --samples 8 --max-new-tokens 96 --lr 3e-4

endlocal

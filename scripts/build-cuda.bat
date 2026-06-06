@echo off
rem Validated CUDA 13.3 + MSVC build (see docs/running-the-trainer.md).
rem CUDA bin is deliberately kept OFF PATH so cudarc 0.19 falls back to 13.2.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "NVCC_PREPEND_FLAGS=-Xcompiler=/Zc:preprocessor"
cargo build -p antumbra-cli --features models,cuda

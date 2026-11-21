# antumbra-mcp

# --- build stage: pinned toolchain, release binary -------------------------
FROM rust:1.96-slim-bookworm AS builder
WORKDIR /build
# pkg-config covers crates that probe the system; the TLS stack is pure-Rust
# (rustls), so no OpenSSL is needed. g++ is for the tokenizer the model
# backend reads prompts with (its suffix-array crate is C++). The main
# workspace has no git dependencies, so the build needs no credentials of any
# kind.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config g++ \
    && rm -rf /var/lib/apt/lists/*
COPY . .
# `models` is the model backend: the `answer` tool serves through experts, on
# the CPU from a quantized base through Antumbra's own kernels (AVX2 or
# AVX-512 on x86-64, NEON on arm64). The base is read only when the first
# answer has an expert to serve, so a server that has none never fetches it.
# `--build-arg FEATURES=` builds the light server, whose every answer
# escalates.
ARG FEATURES=models
RUN cargo build --release -p antumbra-mcp --locked ${FEATURES:+--features "$FEATURES"}
# The two places the server writes, made here because the runtime image has
# no shell: base weights (the Hugging Face cache) and adapters. Owned by the
# nonroot user, so a named volume mounted on either starts out writable.
RUN mkdir -p /out/weights /out/adapters

# --- runtime stage: runs as uid 65532 (nonroot) ----------------
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /build/target/release/antumbra-mcp /usr/local/bin/antumbra-mcp
COPY --from=builder --chown=65532:65532 /out/weights /weights
COPY --from=builder --chown=65532:65532 /out/adapters /adapters
# The base (Qwen2.5-Coder-1.5B-Instruct, about 3 GB) downloads into /weights
# the first time an expert is served; adapters a server trained, pulled or read
# back from its archive land under /adapters, the working directory, as the
# GPU image keeps them. Compose mounts a named volume on each.
ENV HF_HOME=/weights
WORKDIR /adapters
EXPOSE 8081
# The base image already runs as the nonroot user; the binary verifies request
# JWTs and signs in per request, so it never needs elevated privileges.
ENTRYPOINT ["/usr/local/bin/antumbra-mcp"]

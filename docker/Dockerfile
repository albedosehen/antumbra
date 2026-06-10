# antumbra-mcp, built reproducibly and shipped on a minimal NON-ROOT runtime.
# Build context is the repo root (so the whole workspace + Cargo.lock are present):
#   docker build -f docker/Dockerfile -t antumbra-mcp:local .

# --- build stage: pinned toolchain, release binary -------------------------
FROM rust:1.96-slim-bookworm AS builder
WORKDIR /build
# pkg-config covers crates that probe the system; the TLS stack is pure-Rust
# (rustls), so no OpenSSL is needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config \
    && rm -rf /var/lib/apt/lists/*
# Copy the whole workspace; --locked pins to the committed Cargo.lock.
COPY . .
RUN cargo build --release -p antumbra-mcp --locked

# --- runtime stage: distroless, runs as uid 65532 (nonroot) ----------------
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /build/target/release/antumbra-mcp /usr/local/bin/antumbra-mcp
EXPOSE 8081
# The base image already runs as the nonroot user; the binary verifies request
# JWTs and signs in per request, so it never needs elevated privileges.
ENTRYPOINT ["/usr/local/bin/antumbra-mcp"]

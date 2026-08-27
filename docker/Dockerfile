# antumbra-mcp

# --- build stage: pinned toolchain, release binary -------------------------
FROM rust:1.96-slim-bookworm AS builder
WORKDIR /build
# pkg-config covers crates that probe the system; the TLS stack is pure-Rust
# (rustls), so no OpenSSL is needed. git is for the private oneiriq-kayak
# dependency: resolving the workspace reads every member's manifest, so even
# this kayak-free binary's build must fetch it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends pkg-config git \
    && rm -rf /var/lib/apt/lists/*
COPY . .
# The private git dependency fetches through a BuildKit secret (a read
# token), which never lands in a layer; without one, the build works only
# where the dependency cache is already warm.
RUN --mount=type=secret,id=oneiriq_token \
    set -e; \
    if [ -s /run/secrets/oneiriq_token ]; then \
        token="$(cat /run/secrets/oneiriq_token)"; \
        git config --global --add \
            url."https://x-access-token:${token}@github.com/Oneiriq/".insteadOf \
            "ssh://git@github.com/Oneiriq/"; \
        git config --global --add \
            url."https://x-access-token:${token}@github.com/Oneiriq/".insteadOf \
            "https://github.com/Oneiriq/"; \
    fi; \
    CARGO_NET_GIT_FETCH_WITH_CLI=true cargo build --release -p antumbra-mcp --locked

# --- runtime stage: runs as uid 65532 (nonroot) ----------------
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /build/target/release/antumbra-mcp /usr/local/bin/antumbra-mcp
EXPOSE 8081
# The base image already runs as the nonroot user; the binary verifies request
# JWTs and signs in per request, so it never needs elevated privileges.
ENTRYPOINT ["/usr/local/bin/antumbra-mcp"]

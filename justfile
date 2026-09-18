default:
    @just --list

install:
    cargo install --path crates/antumbra-cli --locked
    cargo install --path crates/antumbra-tui --locked
    cargo install --path crates/antumbra-mcp --locked

uninstall:
    cargo uninstall antumbra-cli antumbra-tui antumbra-mcp

test:
    cargo test --workspace

lint:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt --all

# Format the Markdown docs: requires `dprint` on PATH and the config in `./dprint.json`.
fmt-md:
    dprint fmt

# Preview the release artifacts cargo-dist would build for the current commit.
# Requires `dist` (cargo-dist) on PATH.
dist-plan:
    dist plan

# The hosted control plane is its own cargo workspace (see the root Cargo.toml);
# these run its checks, including the kayak contract gate under `contract`.
test-control:
    cd crates/antumbra-control-server && cargo test --features contract

lint-control:
    cd crates/antumbra-control-server && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features contract -- -D warnings

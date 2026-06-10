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

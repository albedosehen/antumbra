# Antumbra task runner. Run `just` to list recipes, `just <recipe>` to run one.

# List the available recipes (default).
default:
    @just --list

# Build and install the three Antumbra binaries from source into ~/.cargo/bin:
# `antumbra` (CLI), `antumbra-tui` (operator console), `antumbra-mcp` (MCP server).
# Needs the Rust toolchain. End users without Rust should prefer the prebuilt
# installer in the README instead.
install:
    cargo install --path crates/antumbra-cli --locked
    cargo install --path crates/antumbra-tui --locked
    cargo install --path crates/antumbra-mcp --locked

# Remove the installed binaries again.
uninstall:
    cargo uninstall antumbra-cli antumbra-tui antumbra-mcp

# Whole-workspace test, mirrors CI.
test:
    cargo test --workspace

# Clippy as an error gate, mirrors CI.
lint:
    cargo clippy --workspace --all-targets -- -D warnings

# Format the whole workspace.
fmt:
    cargo fmt --all

# Format the Markdown docs: unwrap prose so each paragraph is one soft-wrapping
# line (readable raw, clean diffs). Leaves code and Mermaid blocks alone. Needs
# `dprint` on PATH (cargo install dprint).
fmt-md:
    dprint fmt

# Preview the release artifacts cargo-dist would build for the current commit.
# Requires `dist` (cargo-dist) on PATH.
dist-plan:
    dist plan

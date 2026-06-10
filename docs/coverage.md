# Test coverage

Measured with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) (source-based LLVM coverage).

```bash
# The headline number (excludes binary entrypoints + the TUI render layer; see below).
# The [\\/] char class matches both path separators (CI is unix, Windows uses `\`):
cargo llvm-cov --workspace --ignore-filename-regex '(main\.rs$|antumbra-tui[\\/]src[\\/]ui[\\/]|antumbra-tui[\\/]src[\\/]snapshot\.rs)'

# Everything, no exclusions:
cargo llvm-cov --workspace --summary-only
```

**Result (2026-06-10, gated suite included):** **93.4% line / 91.3% region / 91.6% function** over the measured surface, which **includes** the `antumbra-tui` logic (only the TUI render layer + binary entrypoints are excluded). This run also folds in the previously network-gated paths: with the gated suite's env set and `-- --include-ignored` (see [Running the gated suite](#running-the-gated-suite)), the store's `signin_root` root-credential branch, the sync worker's push/pull cycle, and `antumbra-embed`'s live `ureq` request are exercised against a real SurrealDB v3 container and a local embeddings endpoint. The owner-view store reads (`edge`/`generation`/`evaluation`/`loop_control`) are 100% line-covered and `app.rs` sits at ~90%. The remaining uncovered residue is structural and intentional: the feature-gated raster tier branches in `antumbra-tui/render.rs` (only a `raster` build constructs them), the Windows-FFI monitor detection in `pacing.rs` (can't run on CI), and the `models`-gated GPU trainer/serve code (not compiled into the default build). Running the default suite without the gated env yields a slightly lower number, since those tests skip.

The paths that need a live `ws://` SurrealDB or a remote embeddings endpoint now have **gated integration tests** that pass against a real SurrealDB v3 container and a local OpenAI-compatible endpoint (see [Running the gated suite](#running-the-gated-suite)):

- the store's `signin_root` root-credential branch, exercised over `ws://` by `antumbra-store/tests/ws_owner.rs` (on `mem://` the owner view is plain `invalidate`, so the embedded suite cannot reach the `RootCredentials` arm);
- the sync worker's incremental push / pull / converge cycle against a networked authoritative store, by `antumbra-sync/tests/ws_incremental.rs`;
- the `antumbra-embed` live `ureq` request path, by the `#[ignore]`d `real_endpoint_returns_the_right_dimension` (the `EmbedTransport` seam is also mock-tested offline).

Each skips when its env var is unset, so the default `cargo test` stays network-free. The only surface still outside the measurement is the **`models`-gated candle trainer/serve code**, which needs a GPU and is not compiled into the default build (its CPU-testable logic is already measured over fakes).

## Running the gated suite

These exercise the live `ws://` and network paths. Bring up two throwaway SurrealDB v3 containers (separate databases, so the store and sync tests never share state) and a 384-dim embeddings endpoint, then run with the env set:

```bash
docker run --rm -d -p 8002:8000 surrealdb/surrealdb:v3.0.5 start --user root --pass root --bind 0.0.0.0:8000 memory
docker run --rm -d -p 8003:8000 surrealdb/surrealdb:v3.0.5 start --user root --pass root --bind 0.0.0.0:8000 memory
ollama pull all-minilm   # a 384-dim model that matches EMBED_DIM

ANTUMBRA_STORE_WS=ws://127.0.0.1:8002/rpc \
ANTUMBRA_SYNC_WS=ws://127.0.0.1:8003/rpc \
ANTUMBRA_EMBED_URL=http://127.0.0.1:11434/v1/embeddings ANTUMBRA_EMBED_MODEL=all-minilm \
cargo test --workspace -- --include-ignored
```

To fold the gated paths into the headline coverage, run the same env in front of the `cargo llvm-cov` command above (also with `-- --include-ignored`).

## What is excluded from the headline metric, and why

- **`*/main.rs`**: binary entrypoints, namely `fn main`, the tokio/runtime bootstrap, and the top-level command dispatch. This is glue; the logic each arm calls lives in the library crates, which are measured. (The CLI's _command-handler_ modules `ops.rs`/`commands.rs` are **not** excluded; they are counted, and partially covered by `crates/antumbra-cli/tests/cli_smoke.rs`, which drives the real binary through its no-GPU commands.)
- **`antumbra-tui/src/ui/` + `snapshot.rs`**: the terminal UI **render layer** (presentation), namely the ratatui drawing, the canvas graph, and the headless buffer→text/PNG snapshot scaffolding. The crate's **logic** is no longer excluded: the app state machine (`app.rs`), fuzzy command/filter matching (`command.rs`), the store-change event diff (`events.rs`), frame pacing/monitor detection (`pacing.rs`), scrolling (`scroll.rs`), and theme/overlay/transition helpers are now measured, covered by the 40+ snapshot/unit/store-backed tests.
- **GPU / `models`-gated code**: the real candle Qwen + LoRA model stack is behind the `models` feature and is not compiled in the default build, so it is not part of the coverage surface at all. Its CPU-testable logic (decode policy, config, RAFT/consolidation orchestration over fakes, etc.) _is_ measured.

## Notes

- `cargo-llvm-cov` captures coverage of subprocesses, so `cli_smoke.rs` (which spawns the `antumbra` binary) contributes real coverage of the dispatch and command code, though some binary-only paths still read low because not every command is exercised.
- Re-run after adding tests; the instrumented build is a full rebuild.

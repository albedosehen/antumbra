# Test coverage

Measured with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
(source-based LLVM coverage).

```bash
# The headline number (excludes binary entrypoints + the TUI render layer; see below).
# The [\\/] char class matches both path separators (CI is unix, Windows uses `\`):
cargo llvm-cov --workspace --ignore-filename-regex '(main\.rs$|antumbra-tui[\\/]src[\\/]ui[\\/]|antumbra-tui[\\/]src[\\/]snapshot\.rs)'

# Everything, no exclusions:
cargo llvm-cov --workspace --summary-only
```

**Result (2026-06-07):** **92.6% line / 90.6% region / 90.8% function** over the
measured surface, which **includes** the `antumbra-tui` logic (only the TUI
render layer + binary entrypoints are excluded). The measured surface grew
substantially with the console build-out (operator actions, the
memory/loop/evals pages, the drill-downs, the loop graceful-stop) yet coverage
held: the new owner-view store reads (`edge`/`generation`/`evaluation`/
`loop_control`) are 100% line-covered, and `app.rs` sits at ~90%. Most library
crates are 89% to 100%; the low outliers are `antumbra-embed` (~67%, its live HTTP
path is network-gated, see below) and `antumbra-tui/render.rs` (~69%, the
feature-gated raster probe + the `Raster`/`HalfBlock` tier branches that only a
`raster` build constructs). Within the console, `events.rs`/`theme.rs` are 100%,
`overlay.rs`/`scroll.rs`/`command.rs` ~98% to 99%, `transition.rs` 95%, and
`pacing.rs` 89% (the remainder is Windows-FFI monitor detection that can't run
on CI).

The main untested remainders are paths that need a live `ws://` SurrealDB, a
remote HTTP endpoint, or a GPU, and so belong to the docker-/network-/`models`-gated
suites: the store's `signin_root` root-credential branch; the sync worker's
push/pull cycle (an in-memory store is fresh per connect, so reconcile always
moves nothing); the `antumbra-embed` HTTP client's live request path (the
`EmbedTransport` seam is unit-tested, but the real `ureq` call needs a server);
and the `models`-gated candle trainer code (not compiled in the default build).

## What is excluded from the headline metric, and why

- **`*/main.rs`**: binary entrypoints, namely `fn main`, the tokio/runtime bootstrap, and
  the top-level command dispatch. This is glue; the logic each arm calls lives in
  the library crates, which are measured. (The CLI's *command-handler* modules
  `ops.rs`/`commands.rs` are **not** excluded; they are counted, and partially
  covered by `crates/antumbra-cli/tests/cli_smoke.rs`, which drives the real
  binary through its no-GPU commands.)
- **`antumbra-tui/src/ui/` + `snapshot.rs`**: the terminal UI **render layer**
  (presentation), namely the ratatui drawing, the canvas graph, and the headless
  buffer→text/PNG snapshot scaffolding. The crate's **logic** is no longer excluded:
  the app state machine (`app.rs`), fuzzy command/filter matching (`command.rs`),
  the store-change event diff (`events.rs`), frame pacing/monitor detection
  (`pacing.rs`), scrolling (`scroll.rs`), and theme/overlay/transition helpers are
  now measured, covered by the 40+ snapshot/unit/store-backed tests.
- **GPU / `models`-gated code**: the real candle Qwen + LoRA model stack is behind
  the `models` feature and is not compiled in the default build, so it is not part
  of the coverage surface at all. Its CPU-testable logic (decode policy, config,
  RAFT/consolidation orchestration over fakes, etc.) *is* measured.

## Notes

- `cargo-llvm-cov` captures coverage of subprocesses, so `cli_smoke.rs` (which
  spawns the `antumbra` binary) contributes real coverage of the dispatch and
  command code, though some binary-only paths still read low because not every
  command is exercised.
- Re-run after adding tests; the instrumented build is a full rebuild.

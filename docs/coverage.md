# Test coverage

Measured with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov)
(source-based LLVM coverage).

```bash
# The headline number (excludes binary entrypoints + the TUI render layer; see below):
cargo llvm-cov --workspace --ignore-filename-regex '(main\.rs$|antumbra-tui/src/ui/|antumbra-tui/src/snapshot\.rs)'

# Everything, no exclusions:
cargo llvm-cov --workspace --summary-only
```

**Result (2026-06-06):** **92.4% line / 90.3% region / 90.3% function** over the
measured surface (84.6% line with nothing excluded). The library crates are
91–100% each. *(This figure predates folding the `antumbra-tui` logic files into
the surface — re-run the headline command to refresh it now that the console's
state machine is measured rather than wholly excluded.)*

## What is excluded from the headline metric, and why

- **`*/main.rs`** — binary entrypoints: `fn main`, the tokio/runtime bootstrap, and
  the top-level command dispatch. This is glue; the logic each arm calls lives in
  the library crates, which are measured. (The CLI's *command-handler* modules
  `ops.rs`/`commands.rs` are **not** excluded — they are counted, and partially
  covered by `crates/antumbra-cli/tests/cli_smoke.rs`, which drives the real
  binary through its no-GPU commands.)
- **`antumbra-tui/src/ui/` + `snapshot.rs`** — the terminal UI **render layer**
  (presentation): the ratatui drawing, the canvas graph, and the headless
  buffer→text/PNG snapshot scaffolding. The crate's **logic** is no longer excluded —
  the app state machine (`app.rs`), fuzzy command/filter matching (`command.rs`),
  the store-change event diff (`events.rs`), frame pacing/monitor detection
  (`pacing.rs`), scrolling (`scroll.rs`), and theme/overlay/transition helpers are
  now measured, covered by the 40+ snapshot/unit/store-backed tests.
- **GPU / `models`-gated code** — the real candle Qwen + LoRA model stack is behind
  the `models` feature and is not compiled in the default build, so it is not part
  of the coverage surface at all. Its CPU-testable logic (decode policy, config,
  RAFT/consolidation orchestration over fakes, etc.) *is* measured.

## Notes

- `cargo-llvm-cov` captures coverage of subprocesses, so `cli_smoke.rs` (which
  spawns the `antumbra` binary) contributes real coverage of the dispatch and
  command code — though some binary-only paths still read low because not every
  command is exercised.
- Re-run after adding tests; the instrumented build is a full rebuild.

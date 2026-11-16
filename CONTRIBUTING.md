# Contributing to Antumbra

Antumbra is before 1.0 and moves quickly. These are the conventions that keep that workable.

## Issues

Open one through the forms under **New issue**: a bug report, an improvement, or a feature request. Blank issues are off. Each form asks for the component, a reproduction or a problem statement, the expected behaviour and your `antumbra --version`. A new issue is labelled `triage` on arrival; a maintainer replaces that with the labels that apply (`bug`, `enhancement`, `security`, `stability`, `refactor`, `tests`, `documentation`) and adds `has pr` once a fix is open.

One issue per defect. A pull request that fixes several unrelated things is harder to review and to revert than three that fix one each.

Security problems go through [a private advisory](https://github.com/albedosehen/antumbra/security/advisories/new), not an issue. They are fixed first and published with the release that carries the fix.

Issues are public. Before you submit, replace internal hostnames, private addresses, tokens and personal paths with placeholders such as `workspace.example`.

## Branches and pull requests

- Branch names say what they carry: `fix/<slug>`, `feat/<slug>`, `refactor/<slug>`, `docs/<slug>`, `chore/<slug>`, `release/vX.Y.Z`.
- Titles are conventional commits: `fix(mcp): ...`, `refactor(store): ...`, `test(cli): ...`, `docs: ...`. The scope is the crate without its `antumbra-` prefix.
- When a release branch is open (`release/vX.Y.Z`), pull requests for that release target it; the release branch merges to `main` as one pull request that bumps every crate's version and adds the `CHANGELOG.md` entry. Otherwise target `main`.
- The pull request template asks for the motivation, what the change does, how you tested it, its security considerations and the issue it closes (`Closes #N`). Fill it in; an empty section is a question the reviewer has to ask.
- One pull request per issue.

## Code

- **No `unwrap` or `expect` without a reason**, in production code or in tests. Propagate with `?`, `let .. else`, `match` or a typed error. A test is `async fn name() -> anyhow::Result<()>` (or the crate's `Result`) ending in `Ok(())`; an `Option` that must be `Some` is `let Some(x) = .. else { anyhow::bail!("why") }`; an expected error is matched, not `unwrap_err`'d. The few justified cases (an HMAC key of any length, a value filtered to `Some` on the line above) carry a one-line comment saying why.
- Errors follow the crate: `antumbra_core::{AntumbraError, Result}` in the library crates, `thiserror` enums where a crate defines its own, `anyhow` in the binaries.
- Before you push: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test` for the crates you touched. CI runs the same on the pinned toolchain in `rust-toolchain.toml`, plus `clippy --features models` for the model path and the hook-script tests.
- Tests for every change. A fix carries the test that fails without it; a refactor carries the test that proves behaviour did not move.
- The hook scripts come in pairs (`.sh` and `.ps1`); a change to one is a change to both, with a case in `scripts/hooks/tests/` for each.

## Releases

A release is a branch `release/vX.Y.Z` with one commit `release: vX.Y.Z` that moves every crate's version and both lockfiles and adds the changelog entry, opened as a pull request to `main`. After it merges, the merge commit is tagged `vX.Y.Z`; the Release workflow builds the artifacts and publishes the GitHub release from the changelog entry.

Thank you for the pull request.

<!-- Title: a conventional commit, e.g. `fix(mcp): ...`, `refactor(store): ...`, `docs: ...`.
     Base: the current release branch (`release/vX.Y.Z`) when one is open, otherwise `main`.
     Add the `breaking-change` label if a user or operator must change something to upgrade. -->

## What is the motivation?

<!-- The problem, in one or two sentences, with the issue's reproduction if there is one. -->

## What does this change do?

<!-- The change, the files it touches, and any behaviour that changes for a user or an operator. -->

## What is your testing strategy?

<!-- The tests added or extended, by name, and what each asserts. The commands you ran and their result.
     `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo test` for the touched crates are expected. -->

## Security Considerations

<!-- Does this touch authentication, tenant or user isolation, secrets, input that reaches the store, error or log output, or a resource limit?
     If so, say what changes and what guards it. If not, say why in one line (e.g. "documentation only"). -->

## Is this related to any issues?

<!-- `Closes #N` or `Fixes #N` closes the issue when this merges. -->

* [ ] No related issues

## Checklist

* [ ] Tests return `Result` and use `?`; no `unwrap`/`expect` without a one-line reason beside it
* [ ] Nothing here names an internal host, a private address, a token or a personal path
* [ ] I have read [CONTRIBUTING.md](https://github.com/albedosehen/antumbra/blob/main/CONTRIBUTING.md)

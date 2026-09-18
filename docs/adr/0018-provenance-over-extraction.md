# ADR-0018: Provenance over extraction

**Status:** Accepted · **Date:** 2026-09-18 · **Related:** 0004 (the boundary: branch as a governing feature), 0012 (Penumbra memory), 0015 (MCP runtime surface), P-3 (knowledge documents)

## Context

The obvious way to give an agent an inventory of a codebase (what routes a service exposes, which modules exist, who owns what) is static extraction: parse every repository with per-language grammars, store the symbols, and serve them. Two costs come with it, and both compound:

1. **The parsers are owned forever.** Every extractor is a per-language, per-framework heuristic that goes stale when an idiom changes, and every grammar pin is a break waiting for a version bump. The maintenance is not the model's; it is the operator's.
2. **The symbols go stale silently.** An index is a snapshot. With many concurrent branches, the snapshot of a feature branch freezes at extraction time; when the branch is renamed, merged, or deleted, nothing in the index says so unless a lifecycle tracker is built and run on every event. An agent that reads the stale snapshot produces coherent, plausible, wrong code. A comparable system examined in September 2026 contained the hazard only by making branch tracking opt-in and defaulting reads to `main`; a tracked feature branch was still a snapshot from track time with no age warning and no deletion signal.

Antumbra already refuses to own parsers (the scope decision behind keeping tree-sitter in Kushtakas and out of this workspace). The question was how to answer inventory questions anyway.

## Decision

**Staleness is a property of provenance, evaluated at recall, not a mutation problem solved by re-extraction and garbage collection.**

1. **Memories about code carry a git anchor.** `store_memory` takes an optional structured `provenance` (repo slug, commit, branch, path), stored as one evidence entry `git:<repo>@<commit>[#<branch>][:<path>]` (`antumbra_core::provenance`). Every memory view returns it parsed.
2. **Recall is scoped to where the caller is.** `recall_memories` takes the caller's `repo` and `branch`; each hit is judged `in_scope`, `other_branch`, `other_repo`, or `unknown`, and out-of-scope hits are demoted below in-scope ones, never hidden. This is ADR-0004's rule with the branch as the governing feature: a memory scoped to one branch is inhibited on another, the way a repo-scoped convention is inhibited one repo over.
3. **The session hook judges the anchor with git in hand.** The server knows only what the caller says; the bootstrap hook runs inside the repository and asks git whether each anchor's commit is an ancestor of HEAD and whether its branch still exists, tagging hits `[live]`, `[not-on-head]`, or `[orphaned]`, and optionally penalizing orphans. Nothing is re-extracted; a stale memory is visible instead of silently wrong.
4. **Inventory by execution, not parsing.** The truthful route table is what the framework prints. `antumbra ingest --title routes -- <the framework's own lister>` stores the command's output as a knowledge document stamped with the same anchor; the framework maintains its lister, and re-ingesting a title replaces its chunks in place. This is the critic's principle (the environment is the truth) applied to inventory, and it is the parser-free cold start for an organization: run each service's self-describer once and ingest it.
5. **Git-derived facts on demand.** Ownership, hotspots, and co-change need no parser: `antumbra git-facts` derives them from `git log` over a window and stores each as a `world` memory whose evidence names the commit range, so the fact is self-invalidating.
6. **Consume symbols, do not own parsers.** Where symbol-level answers are genuinely needed, the seam is a symbol source (LSP `documentSymbol`/`references`, or a ctags provider) whose *answers* become provenance-stamped memories. The index itself is never Antumbra's to keep fresh. Deferred until a real need names it; an unused port would be dead code.

Inventory questions are never answered from expert weights. Provenance-backed memory and documents answer *what exists*; experts answer *how to do it*. That line is the one place a learning system could otherwise produce exactly the coherent-but-wrong output this decision exists to prevent.

## Consequences

- **Positive:** no grammar pins, no extractor treadmill, no branch-lifecycle tracker; freshness is exact (git ancestry) rather than scheduled; the anchor survives sync and tombstones like any other evidence; the same mechanism scopes conventions and inventory alike.
- **Negative:** coverage is lazy, it grows where agents work and where listers were run, so an untouched service is unknown until someone ingests its self-description. The hook's branch check knows only what the local clone knows (fetch to be current). Frameworks without a lister need the deferred symbol source.
- **Neutral:** the server's `scope` and the hook's tags are two layers on purpose: the server has no git, the hook has no store; each judges what it can see.

## Validation

- `antumbra-core::provenance` tests: every evidence shape round-trips; malformed or foreign entries parse to nothing; scope follows repo then branch; the ssh, scp, and https spellings of a remote normalize to one slug; demotion is stable and hides nothing.
- `antumbra-mcp` (`server/tests.rs::provenance_is_stored_and_scopes_recall`): a structured anchor becomes evidence, comes back parsed on every view, a recall given repo + branch tags every hit and puts the in-scope one first, and a malformed anchor is refused rather than stored as junk.
- `antumbra-cli` (`ops::tests`, `gitfacts::tests`, `tests/cli_smoke.rs`): `ingest` stores anchored chunks from a file and from a command and replaces a title in place; with a copal archive configured it uploads the original first and stamps every chunk with the file id + digest, and fails closed storing nothing when copal is down; `git-facts` derives ownership, hotspots, and co-change from a fixture log; the hook scripts are exercised in CI without a server and on this repository with git present.

_Kill criterion:_ an agent that asks "what does this service expose on main?" after a branch was deleted still gets the deleted branch's answer with no tag, or gets nothing for a service whose lister was ingested. Either means the anchor is not doing its job.

# ADR-0019: Native GitHub integration, and a dependency graph made of evidence

**Status:** Proposed · **Date:** 2026-09-19 · **Related:** 0004 (the boundary), 0012 (Penumbra memory), 0018 (provenance over extraction)

## Context

ADR-0018 anchors memories to commits and judges them at recall. The events that make an anchor stale, a merge, a branch deletion, a pull request closing, all originate in the hosting platform, and today Antumbra learns of them only when a session-start hook happens to run inside a checkout. Two consequences follow. First, the squash-merge re-anchor is an unbuilt step that has to live somewhere, and the merge event is the natural place. Second, an organization evaluating Antumbra against a static code-context service asks what remains that only static extraction can provide, and the honest answer is the cross-service dependency graph, because every other capability is reachable from platform events plus the existing ingest.

The obvious way to close that last gap is to adopt tree-sitter extraction. Antumbra has deliberately refused to own parsers (0018): every extractor is a per-language, per-framework heuristic maintained forever, and its output is a snapshot that goes stale silently. A graph built that way would import into Antumbra the very failure mode the rest of the system is designed to avoid.

## Decision

### 1. Antumbra ships a native GitHub integration

A GitHub App, installed once per organization, with read access to contents and subscriptions to push, pull request, branch create and delete, and installation events. Its handlers are thin and each one feeds something that already exists:

- **Merge** triggers the lister ingest and the generated-document ingest for the repository, anchored to the merge commit, and re-anchors memories from the merged head to that commit, which closes the squash-merge gap at the source event.
- **Branch delete** tags that branch's memories orphaned server-side, before any session starts.
- **Pull request merged** stores an anchored memory (title, files, reviewers, linked issues) with the pull request as evidence.
- **Installation** on a new repository is the cold start: run the listers once, ingest, done.
- **A schedule** runs `git-facts` across the fleet, so ownership, hotspots, and co-change exist for every repository the App sees.

The integration is also a product surface of its own. A **knowledge diff** on every pull request, posted as a check run, lists the memories and documents the change touches by path anchor, the facts it contradicts, and the memories its branch deletion will orphan. Institutional knowledge becomes visible at review time, which no static context service does.

### 2. A dependency edge is a claim with evidence, never a parsed fact

The graph is built from four evidence sources, each an ordinary memory with provenance, confidence, reinforcement, and decay:

| Source | Evidence | Confidence |
| --- | --- | --- |
| Declared | Package manifests and lockfiles naming client SDKs; OpenAPI client configuration; event catalogs; infrastructure as code and the cloud resource graph (queue and topic subscriptions, function bindings) | High; the file or the cloud API is the evidence |
| Observed | An APM service map, gateway logs, mesh telemetry, over a time window; traffic is the weight | High while the window is recent; decays as it ages |
| Learned | Cross-repository co-change, deploy ordering, incidents naming two services, from the GitHub integration and `git-facts` | Medium; grows with recurrence |
| Claimed and verified | A model reads a service once and claims its dependencies; a claim is stored above a low floor only when a declared or observed source corroborates it | Low until corroborated; the verifier decides |

Blast radius is a weighted traversal over the union that returns each edge's evidence with the answer. An edge reinforced by a fresh observation or declaration gains standing; an edge nobody has seen for a window fades and is eventually tombstoned. The graph stays honest without a cron job that re-extracts it.

### 3. What Antumbra does not do

It does not own grammars or query files. Where a customer already runs a symbol indexer (SCIP or an LSP), its cross-repository references may be ingested as one more declared source, consumed rather than maintained.

## Consequences

- **Positive:** the two open items of 0018 (the re-anchor step, the cold start for an organization) become event handlers; the graph gap closes without a parser; the knowledge diff is a differentiator no extraction-based service can offer, because it needs anchored memories to exist.
- **Negative:** a GitHub App is a new operational surface (installation flow, webhook verification, rate limits, retries). The observed source depends on the customer having telemetry; without it, the learned and claimed sources carry the cold start, and they take time to earn confidence.
- **Neutral:** this is a design record, not an implementation. The webhook receiver, the check run, the edge memory type, and the fleet runner are queued work (roadmap P-7).

## Validation

A pilot against a static extraction service: three services with real cross-service traffic, declared and observed edges enabled, one month. Compare blast-radius answers edge by edge. _Kill criterion:_ the evidence graph misses a dependency the static extractor finds, and that dependency mattered in an incident during the window. If that happens, the claimed-and-verified source was not enough and a consumed symbol index becomes mandatory rather than optional.

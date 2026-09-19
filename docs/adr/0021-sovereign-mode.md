# ADR-0021: Sovereign mode, or what a coding agent loses when you stop it phoning home

**Status:** Accepted (in progress: detection and the doctor are in; instruction injection, the conventions compartment and the MCP schema lint are queued) · **Date:** 2026-09-19 · **Related:** 0013 (identity), 0015 (MCP runtime surface), 0017 (the memory fabric), 0018 (provenance over extraction), 0020 (sovereign artifacts)

## Context

Antumbra exists so that a person's work stays on their side. The first thing such a person does to their coding agent is turn its telemetry off. In Claude Code that is one environment variable, and it does more than it says: `DISABLE_TELEMETRY` also turns off feature-flag fetching, and a list of features that have nothing to do with telemetry are gated on those flags. `DO_NOT_TRACK`, `DISABLE_GROWTHBOOK` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` do the same, and so does running on a third-party provider (Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry, Claude Platform on AWS). There is no documented way to opt out of metrics and keep the flags; only `DISABLE_ERROR_REPORTING` is free of the side effect.

What goes, per the vendor's own documentation (read 2026-09-19 against Claude Code 2.1.278): reading `AGENTS.md` as project instructions, starting in auto mode by default, `/auto-mode-setup`, Remote Control and messaging sessions on other machines, `claude import`, `/skill-doctor`, the sync of skills and plugins from the hosted account, the advisor tool, comments on hosted artifacts, a newer MCP protocol probe, the PowerShell tool on Windows when Git Bash is installed, vendor-bound drafted feedback, and the exclusion of MCP tools whose input schema the API would reject, which otherwise fails every request with a 400 that names the tool only by its position.

None of this is announced. A repository whose only instruction file is `AGENTS.md` silently stops instructing the agent. The documented list is also incomplete: in the session where the variable was first set, four tools left the running session at once, and two of them (`Monitor`, `PushNotification`) are on no list.

Call the state **sovereign mode**. It is the state Antumbra's users are most likely to be in, so it is the normal case here and not an edge case.

## Decision

### 1. Principles

1. **Detect, do not assume.** One pure function decides whether a session is in sovereign mode, from the environment, the agent's settings files and the provider. Every compensation is gated on it, so nothing is supplied twice when the flags are available.
2. **Rules are supplied verbatim from an authoritative source, never recalled.** An instruction file is a rule. Embedding recall is lossy and ranked, so rule text goes into context byte for byte, with its digest. The authoritative source is the working tree for a repository's rules and the archive for rules that have no repository (0020). A chunk is never a source of rules.
3. **Anchored to a version.** The gated list changes between releases of the agent. Every rule names the version it was verified against, the page that says so and the date, and is re-verified when the installed version changes.
4. **No phoning home to compensate.** Antumbra does not fetch the flags itself and does not proxy a vendor-hosted feature.
5. **Name the losses.** What lives on the vendor's infrastructure is an accepted loss and is reported as one.
6. **Fail open.** A hook that errors or times out never blocks a session (the standing hook contract).
7. **The host's dominant shell decides.** PowerShell on Windows, zsh on NixOS, bash on WSL, zsh or bash on macOS. Antumbra holds no shell preference and never stops a user who picks another: every hook ships as a `.ps1` and a `.sh` with identical behavior, and the doctor expects the PowerShell tool on Windows only.

### 2. The matrix

Three classes. **Restored**: Antumbra supplies it. **Setting**: a local setting brings it back, and the doctor checks the setting. **Accepted loss**: it lives on the vendor's side.

| Lost with the flags off | Class | What Antumbra does |
| --- | --- | --- |
| `AGENTS.md` read as project instructions | Restored | The session-start hook supplies it verbatim, a hook on file reads covers subdirectories, and a hook on subagent start covers subagents, which receive no session-start context. Until that ships, the doctor names any project whose `AGENTS.md` is not being read |
| MCP tools with a schema the API rejects are excluded | Restored | An MCP schema lint, run outside the agent, names the offenders and prints the deny rule. A deny rule on a bare tool name removes the tool from the request entirely, so it is a complete fix. It runs outside the agent because a bad schema makes every request inside it fail |
| `/auto-mode-setup` drafting trust entries | Restored | Drafted from Antumbra's own memories of the user's infrastructure, not from session transcripts |
| `/skill-doctor` unused-skill report | Restored | A hook counts skill use; the TUI shows it |
| Remote Control, messaging sessions on other machines | Restored, in part, later | An asynchronous handoff compartment over the networked tier. No live control |
| Skills and plugins synced from the hosted account | Accepted loss by intent | Off is the sovereign default |
| PowerShell tool on Windows with Git Bash installed | Setting | `CLAUDE_CODE_USE_POWERSHELL_TOOL=1`, required on Windows by principle 7 |
| MCP protocol probe | Setting, advisory | `MCP_PROTOCOL_NEGOTIATION=auto` |
| Sessions start in auto mode | Setting, advisory | Only the built-in default falls back. An explicit `permissions.defaultMode: "auto"` in the user's own settings is an earlier step and is still honored (probed: three headless runs under `DISABLE_TELEMETRY=1`). Ignored in project settings |
| The VS Code extension reading settings for its starting mode | Accepted loss | Documented: with the flags off it ignores every settings file |
| The advisor tool | Accepted loss | It sends the whole conversation, every tool call and result, to a stronger model on the vendor's infrastructure. That is escalation upward and off the machine, the opposite of `route` and `answer`. No analog is claimed |
| Comments on hosted artifacts | Accepted loss now | The sovereign answer is 0020: a comment becomes a memory |
| Vendor-bound drafted feedback | Accepted loss | Friction is kept locally as `bank` memories by the capture hook |
| `claude import` | Accepted loss | A one-time migration of configuration from other agents. Nothing to compensate |

### 3. The doctor

`antumbra claude doctor` reads the environment and the agent's settings files (the user's, then the project's and the project's local file), decides whether the session is in sovereign mode and why, and prints each rule with its state. It exits non-zero when a required setting is missing. It runs outside the agent, which matters: it is the way back in when a bad MCP schema has made every request fail.

Detection follows the documented semantics exactly, because they differ. `DISABLE_TELEMETRY` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` count when set to any non-empty value, `0` and `false` included. `DO_NOT_TRACK` and `DISABLE_GROWTHBOOK` are ordinary booleans. A provider switch counts unless the host platform has declared that it manages the provider.

It does not edit the agent's settings. The file is the user's, its key order is theirs, and Antumbra's JSON handling would re-sort it; the doctor prints the exact lines to add. An `apply` that writes them after a backup is left for later, and will never touch the permission lists.

### 4. Rules as memories

Each rule in the matrix is also a `world` memory in a `claude-code` compartment, with the version, the page and the date as evidence, so an agent can be told at session start what is different about the session it is in (artifact comments are not available; `AGENTS.md` is supplied by Antumbra; prefer the host's shell). When the installed version changes, the doctor marks every rule unverified until it has been re-checked, and a rule the new documentation no longer supports is penalized, not deleted.

The three levels of rule that have no repository map onto 0017's hierarchy: a device's rules sit on its `device_profile`, a user's follow the user fabric to every node they run, and an organization's are a document a member offers into the hive and the owner accepts. That last is stricter than a vendor's managed instruction file: an owner can publish rules and cannot conscript a member into them.

### Order of work

1. Detection, the matrix and the doctor (this record's first increment).
2. Instruction injection: session start, file reads, subagent start; both hook pairs.
3. The conventions compartment and the session-start block.
4. The MCP schema lint, after establishing which schema constraints the API rejects.
5. Trust entries drafted from memory; skill usage.
6. A read-only panel in the TUI.
7. `apply`, and the handoff compartment as a roadmap entry.

## Consequences

- **Positive:** turning telemetry off stops costing a user features they were never told were attached to it. The doctor is useful to anyone in this state, with or without the rest of Antumbra. The rules become recallable knowledge with an expiry instead of tribal knowledge.
- **Negative:** Antumbra takes on tracking another product's release notes, and will be wrong for a while after each release until someone re-verifies. Supplied context does not carry the standing of native project instructions and does not appear in the agent's own memory view; adherence has to be measured and not assumed.
- **Neutral:** the documented list understates the loss, so the matrix is built from observation as well as documentation. A probe that diffs the tool list with and without the variables would settle it, at the cost of one telemetry-on session; it is not run without the user's say-so.

## Alternatives considered

- **Leave telemetry on and keep the features.** Not a choice Antumbra gets to make for its users, and not one most of them would make.
- **Recall the rules instead of injecting them.** A rule that is sometimes ranked fourth is not a rule.
- **Write a `CLAUDE.md` that imports `AGENTS.md` into every repository.** It is the vendor's documented bridge and it works, but it changes repositories the user may not own and leaves a file behind in each. It stays available as an explicit, per-repository command and is never automatic.
- **Have the doctor fix the settings itself.** The settings file grants the agent its permissions. A tool that rewrites it on its own initiative is a tool to distrust, however narrow the edit.

## Validation

- **Detection.** A table test over every trigger, with the values that must and must not count for each (`DISABLE_TELEMETRY=0` counts; `DO_NOT_TRACK=0` does not), the provider switches, and the host-managed exemption. _Kill criterion:_ a documented trigger the detector misses, or a value it misreads, on a release the matrix claims to be verified against.
- **The doctor.** On a Windows host in sovereign mode without the PowerShell variable it exits non-zero and says which line to add; with it, zero. In a project whose only instruction file is `AGENTS.md` it says the file is not being read. _Kill criterion:_ it reports a session as healthy in which the agent is demonstrably not reading the project's instructions.
- **Injection (next increment).** Against the documented default, case by case: `AGENTS.md` only, both files, a local `CLAUDE.md` present, a subdirectory's file, an import, the excluded names, an oversized file. _Kill criterion:_ the agent follows natively loaded instructions measurably better than the same text supplied by the hook. Then injection is the wrong mechanism and the bridge file becomes the recommendation.

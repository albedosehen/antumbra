# ADR-0021: Sovereign mode, or what a coding agent loses when you stop it phoning home

**Status:** Accepted (in progress: detection, the doctor, the `AGENTS.md` bridge, the session-start block and the conventions compartment are in; the MCP schema lint is next) · **Date:** 2026-09-19 · **Related:** 0013 (identity), 0015 (MCP runtime surface), 0017 (the memory fabric), 0018 (provenance over extraction), 0020 (sovereign artifacts)

> **`AGENTS.md` is bridged, not injected (2026-09-19).** This record first chose to supply the file through the session-start hook. That was wrong, and the vendor's own page says why: a hook's context is capped at 10,000 characters, and past the cap the agent receives a file path and a 2,000-character preview that it is never asked to open. The first real instruction file measured was 11,468 characters. It would have been cut to a fifth, silently. Hook context is also a system reminder and not project instructions, it is not restored after compaction, and it does not reach subagents, which is why the first plan needed three hooks.
>
> A `CLAUDE.local.md` beside the file, containing `@AGENTS.md`, has none of those problems: it is read natively with no cap short of the agent's own 4 MiB, at and above the working directory and in subdirectories as files there are read, it is re-read from disk after compaction, and subagents load it like any project instruction. The objection recorded below against a bridge file was that it changes repositories the user may not own. `CLAUDE.local.md` is the file the vendor designates for instructions that are _not_ committed, and one line in `.git/info/exclude`, which is itself untracked, keeps it out of `git status`. Nothing a repository tracks changes. `antumbra claude bridge` writes them, never on its own initiative, and `--remove` deletes only a file that is still a bridge and nothing else.
>
> Measured, not argued: a throwaway repository with a 23,403-character `AGENTS.md` whose last line held a passphrase, asked for it in a single turn, headless, under `DISABLE_TELEMETRY=1`. Without the bridge the agent answered `NONE`, which is also the first direct evidence of the loss this record is about. With it, the passphrase.

## Context

Antumbra exists so that a person's work stays on their side. The first thing such a person does to their coding agent is turn its telemetry off. In Claude Code that is one environment variable, and it does more than it says: `DISABLE_TELEMETRY` also turns off feature-flag fetching, and a list of features that have nothing to do with telemetry are gated on those flags. `DO_NOT_TRACK`, `DISABLE_GROWTHBOOK` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` do the same, and so does running on a third-party provider (Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry, Claude Platform on AWS). There is no documented way to opt out of metrics and keep the flags; only `DISABLE_ERROR_REPORTING` is free of the side effect.

What goes, per the vendor's own documentation (read 2026-09-19 against Claude Code 2.1.278): reading `AGENTS.md` as project instructions, starting in auto mode by default, `/auto-mode-setup`, Remote Control and messaging sessions on other machines, `claude import`, `/skill-doctor`, the sync of skills and plugins from the hosted account, the advisor tool, comments on hosted artifacts, a newer MCP protocol probe, the PowerShell tool on Windows when Git Bash is installed, vendor-bound drafted feedback, and the exclusion of MCP tools whose input schema the API would reject, which otherwise fails every request with a 400 that names the tool only by its position.

None of this is announced. A repository whose only instruction file is `AGENTS.md` silently stops instructing the agent. The documented list is also incomplete: in the session where the variable was first set, four tools left the running session at once, and two of them (`Monitor`, `PushNotification`) are on no list.

Call the state **sovereign mode**. It is the state Antumbra's users are most likely to be in, so it is the normal case here and not an edge case.

## Decision

### 1. Principles

1. **Detect, do not assume.** One pure function decides whether a session is in sovereign mode, from the environment, the agent's settings files and the provider. Every compensation is gated on it, so nothing is supplied twice when the flags are available.
2. **Rules reach the agent verbatim from an authoritative source, never recalled.** An instruction file is a rule. Embedding recall is lossy and ranked, so a rule is never a chunk. A repository's rules are read natively from the working tree, through an import when the agent will not read the file itself; rules that have no repository come from the archive, with their digest (0020).
3. **Anchored to a version.** The gated list changes between releases of the agent. Every rule names the version it was verified against, the page that says so and the date, and is re-verified when the installed version changes.
4. **No phoning home to compensate.** Antumbra does not fetch the flags itself and does not proxy a vendor-hosted feature.
5. **Name the losses.** What lives on the vendor's infrastructure is an accepted loss and is reported as one.
6. **Fail open.** A hook that errors or times out never blocks a session (the standing hook contract).
7. **The host's dominant shell decides.** PowerShell on Windows, zsh on NixOS, bash on WSL, zsh or bash on macOS. Antumbra holds no shell preference and never stops a user who picks another: every hook ships as a `.ps1` and a `.sh` with identical behavior, and the doctor expects the PowerShell tool on Windows only.

### 2. The matrix

Three classes. **Restored**: Antumbra supplies it. **Setting**: a local setting brings it back, and the doctor checks the setting. **Accepted loss**: it lives on the vendor's side.

| Lost with the flags off | Class | What Antumbra does |
| --- | --- | --- |
| `AGENTS.md` read as project instructions | Restored | `antumbra claude bridge` writes an untracked `CLAUDE.local.md` beside each `AGENTS.md` the agent would have read, importing it, so it is read natively again: at launch, in subdirectories, after compaction, and by subagents. The doctor names any project whose `AGENTS.md` is not being read |
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

Telling and remembering are two things, built as two.

- **Telling: `antumbra claude brief`.** The session-start hook passes on a block rendered from the compiled matrix, never recalled (principle 2), and only in sovereign mode (principle 1). It says what changes what an agent should do and nothing else: whether `AGENTS.md` reached it, which shell is the host's, and what it must not offer because it is gone. When `AGENTS.md` is not bridged it tells the agent to read the file before anything else. A pointer fits in a hook where the file does not. It is a fallback for the session in front of the user, not the fix: it reaches neither subagents nor the session after compaction, which the bridge does. The block never asks the agent for its version, because that starts the agent's runtime inside a hook with ten seconds to live.
- **Remembering: `antumbra claude remember`.** It writes the rules through the MCP surface, the way an agent writes, so they are embedded, owned by the user and private until the user shares the compartment. Two things had to be decided. The memories are **volatile**: the list changes with each release, and a volatile memory never graduates, so no expert is trained on a vendor's release notes. And the command is **idempotent** although the surface is not (`store_memory` gives every write a fresh id): every memory opens with a key, `[claude-code:<rule>]`, and the plan is made from what is already there. A rule that is present is kept, whatever its confidence, since a user who penalized it has made a judgment and storing it again would be arguing with them. One whose text changed is stored again and the old text penalized once. One that left the matrix is penalized once. Nothing is deleted.

Building this turned up a defect in the hook that was already there. Nothing bounded what it emitted, and twelve recalled memories can pass the 10,000-character cap, after which the agent receives a preview and no notice. The hook now holds its whole output under 9,500 characters: the block and the git line first, memories best first while they fit, and a closing line that counts what was left out so the agent knows to recall it.

The three levels of rule that have no repository map onto 0017's hierarchy: a device's rules sit on its `device_profile`, a user's follow the user fabric to every node they run, and an organization's are a document a member offers into the hive and the owner accepts. That last is stricter than a vendor's managed instruction file: an owner can publish rules and cannot conscript a member into them.

### Order of work

1. Detection, the matrix and the doctor (this record's first increment).
2. The `AGENTS.md` bridge (in).
3. The conventions compartment and the session-start block (in).
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
- **Supply `AGENTS.md` through the session-start hook.** This record's first choice. A hook's context is capped at 10,000 characters and degrades to a preview past it, it carries less standing than project instructions, it is lost on compaction, and it never reaches subagents. See the note at the top.
- **Write a tracked `CLAUDE.md` that imports `AGENTS.md`.** The vendor's documented bridge, and it works, but it changes what a repository tracks, in repositories the user may not own. The untracked `CLAUDE.local.md` does the same job without that cost.
- **Have the doctor fix the settings itself.** The settings file grants the agent its permissions. A tool that rewrites it on its own initiative is a tool to distrust, however narrow the edit.

## Validation

- **Detection.** A table test over every trigger, with the values that must and must not count for each (`DISABLE_TELEMETRY=0` counts; `DO_NOT_TRACK=0` does not), the provider switches, and the host-managed exemption. _Kill criterion:_ a documented trigger the detector misses, or a value it misreads, on a release the matrix claims to be verified against.
- **The doctor.** On a Windows host in sovereign mode without the PowerShell variable it exits non-zero and says which line to add; with it, zero. In a project whose only instruction file is `AGENTS.md` it says the file is not being read. _Kill criterion:_ it reports a session as healthy in which the agent is demonstrably not reading the project's instructions.
- **The bridge.** Planned against the documented default, case by case: `AGENTS.md` only, one at and one above the working directory, the `.claude/` form, a `CLAUDE.md` at or above (left alone: the agent never read `AGENTS.md` there), the user's own `CLAUDE.local.md` (left alone), a subdirectory judged by itself, the names the agent never reads, nothing above the repository. A bridge is transparent to the planner, so a second run changes nothing, and removal never deletes a file the user has added to. End to end, the passphrase experiment in the note at the top. _Kill criterion:_ a bridged `AGENTS.md` that the agent does not act on, or a bridge that shows up in `git status` or changes a tracked file. (The earlier criterion for hook injection, that natively loaded instructions would be followed better than supplied ones, was met before anything was built: the cap decides it.)
- **The block.** Table-tested over the standings that change it: nothing outside sovereign mode, an unread file named with the order to read it, a bridge explained and protected, the shell following the host, every rule an agent can reach for listed and no other, and the worst case inside its share of the cap. End to end, a throwaway repository with a 31,026-character `AGENTS.md`, unbridged, whose last line requires every reply to end with a marker, asked for one unrelated word, headless, under `DISABLE_TELEMETRY=1`. Hooks off: `READY`, in one turn, the rule never seen. With the block: the file read, the marker on the reply, and the user told about the bridge. Two runs each, the same both times; a first attempt that asked for the passphrase outright proved nothing, because an agent with tool turns went looking for it unprompted. _Kill criterion:_ an agent told to read an unbridged file that does not, or a block that speaks outside sovereign mode.
- **The hook's limit.** Both siblings, against twelve memories of 1,500 characters, with no server and no `antumbra` installed: under 10,000 characters, the block before any memory, the best memory kept, and kept plus counted-as-left-out equal to twelve. Both tests fail when the limit is raised out of reach. _Kill criterion:_ a bootstrap the agent truncates.
- **The compartment.** The plan is table-tested against a small stand-in for the surface: a first run stores every rule volatile in a compartment it creates, a second changes nothing, a changed rule is stored and its old text retired exactly once, a rule that left the matrix is retired and nothing deleted, a current rule the user penalized is not stored again, other memories are never touched, and a dry run only reads. Against a real `antumbra-mcp` over HTTP on an in-memory store: a dry run left zero memories, the first run stored fifteen, the second kept fifteen, one compartment existed, another member of the workspace could see and recall none of them, and the server never considered consolidating. _Kill criterion:_ a second run that writes, or a rule that reaches a member it was not shared with.

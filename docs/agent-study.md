# A coding agent with and without Antumbra

Every result Antumbra had measured was about its own parts: how often recall
finds the right memory, how often its small model follows a rule after
training, how fast it decodes. None of that says what changes for the agent a
person uses. This study measures that: the same coding agent, on the same
tasks, with no memory, with Antumbra, and with the same conventions written by
hand into the agent's instruction file. Issue
[#573](https://github.com/albedosehen/antumbra/issues/573) has the history.

This page is the preregistration. It was written, and committed, before the
first run, and it says what is measured, how, and when the study stops. What a
run finds is reported against it, including where the finding is no difference.

## The question

When a person tells a coding agent how they work, and corrects it when it gets
that wrong, does the agent keep doing it right in later sessions, at what cost
in tokens, and does Antumbra change that compared with having no memory and with
an instruction file written by hand?

## The arms

All three run the same agent and model, pinned for the whole study: Claude
Code in headless mode (`claude -p`), its version recorded with every run. Each
session runs with `--setting-sources project,local`, `--strict-mcp-config` and
a settings file of its own that turns Claude Code's own memory off
(`autoMemoryEnabled: false`) and leaves out the operator's own instruction
files (`claudeMdExcludes`). So nothing of the person running the study reaches
any arm: not their hooks, not their MCP servers, not their memory. Checked
before the first run: a session started this way named no user instruction,
saw no recall block and had no Antumbra tool.

| Arm | What the agent has |
| --- | --- |
| **None** | The repository and the prompt. Nothing carries from one session to the next but the repository itself. |
| **Antumbra** | Antumbra's hooks and MCP server, as `antumbra setup local` wires them, against a store made fresh for the run and thrown away after it. The stack runs on a separate host (`remote-stack.sh`): the release branch's server image, an in-memory SurrealDB, and the model backend's in-process all-MiniLM-L6-v2 embedder, where setup's stack embeds the same model through ollama. |
| **Instructions** | The conventions the first session states, written into a `CLAUDE.md` at the repository's root before the first session, as a person who maintains such a file by hand would. |

The instructions arm is the fair comparison. It is what a careful person does
today, and it costs the time to write and keep up the file, which the study
does not count.

### What the harness trials changed

Before the pilot, the harness was run on one and two sessions of the first
series to find its own faults. Those runs are not counted. They changed five
things, each the same for every arm:

- **Runs live outside `~/.claude`.** Claude Code refuses edits to files under
  it as sensitive, so an agent there could change nothing; and the exclusion of
  the operator's instruction files would have hidden the instructions arm's own.
- **The operator's `ANTUMBRA_*` variables are cleared.** They reach every
  process through the environment, and the hooks read `ANTUMBRA_TOKEN` before
  the token file the arm names, so the antumbra arm's hooks presented the
  operator's token to the study's store and were refused.
- **The antumbra arm's capture hook answers with a block** (`ANTUMBRA_CAPTURE_NUDGE=block`).
  `claude -p` ends a turn on a Stop hook's feedback where an interactive session
  goes on with it, so the reminder to store what was learned never reached the
  agent; the block keeps the headless turn going, as the feedback keeps an
  interactive one going.
- **Read-only commands may be piped.** Each arm may run `cargo`, `git` and the
  read-only cmdlets and tools agents pipe output through; files change only
  through the edit tools. A pipe through `Select-String` was refused before.
- **The ledger's date is read from the file.** A session after one that set it
  to the day has nothing to change, and the check had read the diff alone.

## A series

A series is several sessions on one repository, checked out at a pinned commit,
one after another in the same checkout, so each session starts where the last
one left the code. The first session states the conventions in the person's
own words, beside its task. The later sessions give a task and nothing else.

After each session the harness checks every convention with a program, never
by judgment. For each one broken it sends one fixed correction, the same words
in every arm, into the same session, then checks again. Then it runs hidden
tests for the task that the agent never sees, and removes them.

The first series is on a private repository, a Rust trading engine, in its
risk crate: five small, exactly specified changes, and four conventions:

1. After a change, add a row for it to the repository's work ledger and set
   its "Last updated" date to the day.
2. Commit as `type(scope): subject` with the crate as the scope, and with no
   `Co-Authored-By` or "Generated with" trailers.
3. Run the formatter and the linter, warnings as errors, before saying it is
   done.
4. Name tests for their behavior in a sentence, never `test_...`.

Its tasks, hidden tests and raw runs are kept private with the repository. What
is reported is the aggregate, below.

## What is measured

Per session:

- **Conventions broken before any correction**, each by name. The primary
  measure is the number broken in sessions 2 onward, where a convention was
  stated in session 1 and corrected when it was broken since: a convention
  broken there is a correction the person has to make again.
- **Corrections sent**, and conventions still broken after the correction.
- **Task success**: the hidden tests pass, after the correction.
- **Tokens**, input, output and cache, **cost** as Claude Code reports it,
  **turns** and **wall time**, for the session and its correction apart.
- In the Antumbra arm, what recall put in front of the agent, from the hooks'
  own record.

## Repetitions, and when it stops

- **Pilot:** one run of each arm through the first series, to check the
  harness and to price a full run. Its numbers are reported as a pilot and
  not pooled with the rest.
- **The study:** three runs of each arm per series, the arms in a rotated
  order so time of day and the provider's load do not line up with an arm.
  Medians are reported with the range of the three.
- **Stop** after three runs of each arm on each series that is run, or when
  the study's spend reaches the budget its operator set before starting it,
  whichever is first. A run is never repeated to replace one that came out
  badly; a run that fails for a reason outside the agent (the provider down, a
  crash of the harness) is recorded as such and run again.

## What it does not claim

- One agent and one model. Codex is the next agent to run it.
- The repositories chosen, and conventions a program can check. Most of what a
  person teaches an agent is not this checkable.
- Cost as Claude Code reports it, which is list price on a subscription.
- A store made fresh for each run. A person's real store holds months of other
  memories, which recall has to find these among.

## Running it

`scripts/agent-study/run-series.ps1 -Series <series.json> -Arm none|antumbra|instructions -Run <n> -Out <dir>`
runs one arm through one series and writes every session's event stream,
checks and usage under the run's folder. `-StudyHost <ssh alias> -StudyAddress <ip>`
runs the antumbra arm's throwaway stack on that host through
`scripts/agent-study/remote-stack.sh`, every name `agentstudy-*` and its port
18081, so nothing the host already runs is touched. The series file names the repository, its
commit, the sessions, the conventions with their checks and corrections, and
the hidden tests.

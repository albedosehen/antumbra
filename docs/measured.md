# Measured: a coding agent with Antumbra and without it

One study, run on 2026-11-22. It asks one thing: when a user has told a coding agent how they work, does the agent still know in a later session?

## The short answer

Over five later sessions that needed conventions given in earlier ones, the same agent broke a convention it had already been given:

| | Conventions broken again | Tasks done | Corrections needed | Cost of a run |
| --- | --- | --- | --- | --- |
| No memory | 3 of 22 (1 to 4) | 5 of 5 | 2 (1 to 3) | $0.64 |
| **Antumbra** | **0 of 22 (0 to 0)** | 5 of 5 | 0 | $0.57 |
| The conventions written into the instruction file by hand | 0 of 22 (0 to 0) | 5 of 5 | 0 | $0.40 |

Each cell is the median of three runs, with the least and the most. Across the three runs that is 8 of 66 with no memory, and none of 66 with Antumbra or with the file.

So Antumbra did what a carefully written instruction file does, without anyone writing the file. The file is still cheaper to read, by about a third.

## What was run

- **One agent, one model:** Claude Code 2.1.330, headless, with `claude-sonnet-5-5`, the same in every arm.
- **Three arms.** No memory: the agent's own file memory off, no hooks, no Antumbra. Antumbra: the hooks and the MCP server exactly as `antumbra setup` wires them, against the published 0.5.1 server image with its default embedder, in a workspace that starts empty. By hand: no Antumbra, and all six conventions written into the repository's instruction file before the first session.
- **A series of eight sessions** on one small Python repository, each a fresh agent, so only the repository (and, with Antumbra, its server) carries anything from one session to the next. Sessions 1 to 3 each state two conventions in passing, as a user would. Sessions 4 to 8 state none and need them; those five are what is counted.
- **Six conventions:** the form of a commit message; money as integer cents with no floats or `/`; no built-in exceptions; how tests are named; no `print`; time only through the project's clock.
- **Checked by pattern, not by opinion.** After a session's first turn, each convention that applies is checked against the lines the session added. A broken one is answered with a fixed correction, as a user would give it. Whether the task was done is judged by tests the agent never sees.
- **Decided beforehand:** what is counted, and when to stop (three runs an arm; more only if the no-memory and Antumbra ranges overlapped, which they did not).

## What it found beyond the table

- **Every task was done in every arm.** The difference is not whether the work got done but whether it was done the user's way without being told again.
- **With no memory the agent kept four conventions of six anyway,** by reading the repository: the code written in the early sessions follows them. The two it broke are the two the code shows least: how tests are named, and that a discount must not be worked out with a float.
- **With Antumbra, recall had what was needed every time.** Before each counted session the server returned a memory of every convention that session needed, 22 of 22 in each run.
- **Antumbra cost less than no memory here** ($0.57 against $0.64 a run), because being corrected costs more than recalling. It took a few more agent turns (56 against 48).

## What it does not show

- **It is small:** one series, eight sessions, six conventions, three runs an arm, one agent and one model. It has not been run under another agent.
- **The repository helps the no-memory arm.** Conventions that leave no trace in the code would show a wider gap; conventions that are all visible in the code would show none.
- **The file was written once, correctly, by someone who knew all six conventions.** What it costs to keep such a file true as conventions change is not measured.
- **A pattern checks the letter of a convention,** not its spirit.
- Nothing here measures how the work feels.

## A note on the runs

The first three Antumbra runs were void and were run again: the harness had given every run the same user id, the server refused it, and the hooks, which are made to fail open, let the sessions go on with no memory. Those three behaved as the no-memory arm does (3, 2 and 2 broken again). They are kept with the other records and counted with neither arm.

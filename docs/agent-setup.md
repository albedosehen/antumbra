# Setting up Antumbra: a guide for coding agents

You are setting up Antumbra for the person you are working with. Antumbra gives their coding agent a memory that carries across sessions: it recalls what matters at the start of each session and with each prompt, and stores what was learned at the end. Setup is one command, `antumbra setup`, which you run on their behalf. This guide tells you what to check, what to ask, and how to confirm it worked.

The steps are subject to change while Antumbra is in private preview. If a command here fails in a way this guide does not describe, run `antumbra setup --help` and follow the CLI over this page.

## Rules

- **Ask before installing anything system-wide** (Docker, ollama, Rust, jq). Say what it is for. Never install silently.
- **Never print a token.** Do not echo it, paste it into the chat, or write it anywhere but the file setup writes (`~/.antumbra/token.txt`). Pass it with `--token-file`, never on a command line you show.
- **Do not edit Claude Code's `settings.json` yourself.** Setup does it safely: it backs the file up, only adds, never rewrites what is already there, and refuses an edit that would change anything else.
- **Re-running is safe.** Every step keeps what is already in place. When a step fails, its message says what to fix: fix it (or ask the person to), then run the same command again.
- Pass `--yes` so setup does not wait on a question you cannot answer. Use `--dry-run` first if the person wants to see what will change.

## 1. Ask which setup they want

Ask: **"Should Antumbra run on this machine, or connect to a hosted workspace you have an address for?"**

| They say | Use | They need |
| --- | --- | --- |
| This machine | `antumbra setup local` | Docker and ollama, about 10 minutes the first time |
| Hosted | `antumbra setup hosted <url>` | The workspace address (`https://...`) and a token or a sign-in link |

## 2. Make sure the `antumbra` command exists

```sh
antumbra --version
```

If it is missing, build it from the repository (until the first release ships installers). This needs git and Rust through [rustup](https://rustup.rs); ask before installing either.

```sh
git clone https://github.com/albedosehen/antumbra.git
cd antumbra
cargo install --path crates/antumbra-cli --locked
```

The first build takes several minutes. The repository is private during the preview, so the clone needs the person's GitHub access.

## 3a. On this machine

Check the two prerequisites:

```sh
docker version --format '{{.Server.Version}}'
curl -s http://127.0.0.1:11434/api/tags
```

- Docker missing or not running: [Docker Desktop](https://docs.docker.com/get-docker/) on macOS and Windows, Docker Engine on Linux. It must be running.
- ollama missing: [ollama.com](https://ollama.com). It must be running.
- On macOS and Linux the hooks also need `jq` and `curl`.

Then, from inside the clone:

```sh
antumbra setup local --yes
```

It checks the prerequisites again, pulls the embedding model, generates the stack's secrets into `docker/.env`, builds and starts the store and the server in Docker (the slow part, once), mints a token, proves recall works with it, and connects Claude Code. The last lines read `Antumbra is set up.`

On Linux, if setup says ollama only listens on 127.0.0.1, the person needs to change ollama's service (the message gives the exact `systemctl` steps; they need sudo). Ask them to, then run setup again.

## 3b. A hosted workspace

The person has an address and either a token or a sign-in link from an email. Ask them to save the token into a file themselves, so it never passes through the chat. Your shell cannot take what they type, so they run it: in Claude Code, by typing `!` followed by the command. For example:

```sh
# macOS and Linux: they paste the token, then press Ctrl-D
cat > ~/antumbra-token.txt
```

```powershell
# Windows: they paste the token at the prompt
Read-Host 'Token' | Set-Content $HOME\antumbra-token.txt
```

Then:

```sh
antumbra setup hosted https://their-workspace.example --token-file ~/antumbra-token.txt --yes
```

Setup proves the server accepts the token, reads the workspace from it, saves it to `~/.antumbra/token.txt`, and connects Claude Code. Delete the temporary file afterwards.

With a sign-in link instead: `antumbra setup hosted <url> --signin-link '<link>' --yes`. A link works once and lasts 15 minutes; if it has expired, they ask for a new one.

## 4. Confirm it

```sh
antumbra setup check
```

Exit code 0 and `Everything is in place.` means done. Anything else is listed with what to do about it.

## 5. Tell the person

Claude Code loads hooks and MCP servers when a session starts, so **they need to start a new Claude Code session** (or restart it). The new session opens with what Antumbra remembers, and what they work on goes back in.

## What setup changed

Say this if they ask:

- `~/.antumbra/`: `token.txt` (only they can read it), `hooks/` (the scripts Claude Code runs), `setup.json` (what was set up).
- `~/.claude/settings.json`: hooks on SessionStart, UserPromptSubmit, Stop and PreCompact; `ANTUMBRA_URL`, `ANTUMBRA_WORKSPACE_ID` and (when absent) `ANTUMBRA_HOST_ID` under `env`; `autoMemoryEnabled: false` when the file did not say (pass `--keep-auto-memory` to skip). A backup sits beside it, `settings.json.antumbra-backup-<time>`.
- Claude Code's MCP servers: `antumbra`, user scope, reading the token from its file on each connection.
- The server: this machine is listed among their devices under its `ANTUMBRA_HOST_ID`, as a memory node. Each session start names it again, which keeps "last seen" current. If they want a particular name, set `ANTUMBRA_HOST_ID` in the settings' `env` before running setup, since setup keeps a name that is already there.
- Local only: `docker/.env` in the clone, the containers `antumbra-surrealdb` and `antumbra-mcp` (they restart with Docker), and the database in `~/.antumbra/surrealdb`.

## Undo

```sh
claude mcp remove antumbra --scope user
docker compose -f docker/docker-compose.yml down      # local only, from the clone
```

Then restore the settings backup over `~/.claude/settings.json` and delete `~/.antumbra`.

## When something goes wrong

| Message | What to do |
| --- | --- |
| `Docker is not installed, or not running` | Install or start Docker, then run setup again. |
| `ollama is not answering` | Install or start ollama. |
| `is not inside a clone of the Antumbra repository` | `cd` into the clone, or pass `--repo <path>`. |
| `ollama only listens on 127.0.0.1` (Linux) | The person runs the `systemctl` steps in the message. |
| `the server ... refused the token` | The token expired or is for another server; ask for a new one. |
| `the claude command is not on your PATH` | Claude Code is not installed where setup can find it; setup prints the one `claude mcp add-json` command to run once it is. |
| `settings.json is not valid JSON` | The file was already broken; show the person and fix it together. |

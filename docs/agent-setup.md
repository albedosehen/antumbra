# Setting up Antumbra: a guide for coding agents

You are setting up Antumbra for the person you are working with. Antumbra gives their coding agent a memory that carries across sessions: it recalls what matters at the start of each session and with each prompt, and stores what was learned at the end. Setup is one command, `antumbra setup`, which you run on their behalf. This guide tells you what to check, what to ask, and how to confirm it worked.

Setup connects Claude Code, and every other agent it knows that it finds on the machine (Gemini CLI, Codex, Copilot CLI, goose, OpenCode). To connect only some, name them: `--agent gemini`, or `--agent claude,codex`.

Antumbra is before 1.0, so the steps can change between releases. If a command here fails in a way this guide does not describe, run `antumbra setup --help` and follow the CLI over this page.

## Rules

- **Ask before installing anything system-wide** (Docker, ollama, Rust, jq). Say what it is for. Never install silently.
- **Never print a token.** Do not echo it, paste it into the chat, or write it anywhere but the file setup writes (`~/.antumbra/token.txt`). Pass it with `--token-file`, never on a command line you show.
- **Do not edit Claude Code's `settings.json` yourself.** Setup does it safely: it backs the file up, only adds, never rewrites what is already there, and refuses an edit that would change anything else. For every other agent it writes only files of its own, and edits none of the person's.
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

If it is missing, install it from a release with the installer for their OS. The command names a release tag rather than `latest`, so what it runs is a published, unchanging asset; the installer downloads a prebuilt `antumbra` (no Rust needed) into `~/.cargo/bin`, checks it against the sha256 built into the script, and adds that directory to the PATH, so a new shell may be needed before the command is found. Ask before running it.

```sh
# macOS and Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/albedosehen/antumbra/releases/download/v0.5.2/antumbra-cli-installer.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/albedosehen/antumbra/releases/download/v0.5.2/antumbra-cli-installer.ps1 | iex
```

To check the archive before anything runs before anything runs, download it and the `.sha256` the release carries beside it, compare, and unpack the binary onto your PATH (the installer does the same comparison, against the sums built into the pinned script):

```bash
v=v0.5.2; t=aarch64-apple-darwin   # or x86_64-apple-darwin, x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu
curl --proto '=https' --tlsv1.2 -LsSfO "https://github.com/albedosehen/antumbra/releases/download/$v/antumbra-cli-$t.tar.xz"{,.sha256}
shasum -a 256 -c "antumbra-cli-$t.tar.xz.sha256" && tar -xJf "antumbra-cli-$t.tar.xz"
```

A setup on this machine does not need the repository. Outside a clone, setup pulls the server image published for the CLI's version. Inside one, it builds the server from the source instead. Only the GPU build (`--gpu`) needs a clone, and so git (ask before installing it).

To build the command from a clone instead of using the installer, they need Rust through [rustup](https://rustup.rs) (ask before installing it): `cargo install --path crates/antumbra-cli --locked`. The first build takes several minutes.

## 3a. On this machine

Check the two prerequisites:

```sh
docker version --format '{{.Server.Version}}'
curl -s http://127.0.0.1:11434/api/tags
```

- Docker missing or not running: [Docker Desktop](https://docs.docker.com/get-docker/) on macOS and Windows, Docker Engine on Linux (in WSL, either Docker Engine in the distro or Docker Desktop's WSL integration). It must be running.
- ollama missing: [ollama.com](https://ollama.com). It must be running.
- On macOS and Linux the hooks also need `jq` and `curl`.

Then, from any directory:

```sh
antumbra setup local --yes
```

It checks the prerequisites again, pulls the embedding model, writes the stack's compose file to `~/.antumbra/stack/docker` and generates its secrets into the `.env` beside it, pulls the server image and starts the store and the server in Docker, mints a token, proves recall works with it, and connects Claude Code. The last lines read `Antumbra is set up.` Run from inside a clone, it uses the clone's `docker/` instead and builds the server from the source, which takes several minutes the first time.

If setup says the server's container cannot reach ollama, the message names the fix for the kind of Docker this is. Ask the person to make it, then run setup again:

- Linux with Docker Engine: ollama only listens on 127.0.0.1, so its service needs changing (the message gives the exact `systemctl` steps; they need sudo).
- WSL with Docker Desktop: containers run in Docker Desktop's VM, which reaches the Windows host rather than the WSL distro. Either run ollama on Windows (and stop the copy in WSL), or turn on WSL's mirrored networking (`networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown` from Windows) and start ollama in WSL with `OLLAMA_HOST=0.0.0.0`. Both are changes on the Windows side.

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

An agent loads hooks and MCP servers when a session starts, so **they need to start a new session** in each agent setup connected (its last lines name them, and say which it connected because it found them on the machine). The new session opens with what Antumbra remembers, and what they work on goes back in.

Gemini CLI connects no MCP server in a folder the person has not trusted, and asks before an Antumbra tool runs. Both are theirs to decide; `docs/integration.md` has the policy file that allows the tools.

On Windows goose runs its hooks with `sh`; when setup says there is none on the PATH, tell the person (Git for Windows has one). Ask before installing it.

**Codex runs none of the hooks until the person approves them.** Tell them: start Codex, type `/hooks`, approve Antumbra's three. You cannot do this for them, and setup's last lines say when it is needed.

**Unity Gateway.** When the person starts their agents with `ug`, setup has written for those too, and says so. One case needs them: when setup or the check says `ug claude` relays a Claude subscription, that launch leaves their own settings out, so they start it with the command printed, `ug claude --settings ~/.antumbra/claude-settings.json`. And when the check says an organization's or a workspace's file stops the hooks or the server, that file is not yours or theirs to edit: tell them what it says, and who it says to ask. `docs/integration.md` has both under "Agents started through Unity Gateway".

## What setup changed

Say this if they ask:

- `~/.antumbra/`: `token.txt` (only they can read it), `hooks/` (the scripts the agents' hooks run), `setup.json` (what was set up, and for which agents).
- `~/.claude/settings.json`: hooks on SessionStart, UserPromptSubmit, Stop and PreCompact; `ANTUMBRA_URL`, `ANTUMBRA_WORKSPACE_ID` and (when absent) `ANTUMBRA_HOST_ID` under `env`; `autoMemoryEnabled: false` when the file did not say (pass `--keep-auto-memory` to skip). A backup sits beside it, `settings.json.antumbra-backup-<time>`.
- Claude Code's MCP servers: `antumbra`, user scope, reading the token from its file on each connection.
- Gemini CLI, when it is connected: the extension `~/.gemini/extensions/antumbra` (a manifest and `hooks/hooks.json`). Its `settings.json` is not touched.
- OpenCode, when it is connected: one file, `~/.config/opencode/plugins/antumbra.js`. Its `opencode.json` is not touched.
- goose, when it is connected: the plugin `~/.agents/plugins/antumbra` (a manifest, `.mcp.json` and `hooks/hooks.json`); one line, `GOOSE_MOIM_MESSAGE_FILE`, added to goose's `config.yaml` when it names no such file, with a backup beside it; and `~/.antumbra/goose/`, where the hooks keep what goose is to read.
- Copilot CLI, when it is connected: `~/.copilot/hooks/antumbra.json`, and `mcpServers.antumbra` added to `~/.copilot/mcp-config.json`, with a backup beside it, `mcp-config.json.antumbra-backup-<time>`.
- Codex, when it is connected: `[mcp_servers.antumbra]` in `~/.codex/config.toml`, written by `codex mcp add`, and hooks on SessionStart, UserPromptSubmit and Stop added to `~/.codex/hooks.json`, with a backup beside it, `hooks.json.antumbra-backup-<time>`.
- On a machine with Unity Gateway (`ug`), for the agents it starts from a directory of its own: the same Gemini CLI extension in `~/.ucode/.gemini-home/.gemini/extensions/antumbra`, the same OpenCode plugin file in `~/.ucode/opencode-xdg/opencode/plugins/`, and `~/.antumbra/claude-settings.json`, Claude Code settings that hold Antumbra's hooks and nothing of theirs. Nothing `ug` wrote is edited.
- The server: this machine is listed among their devices under its `ANTUMBRA_HOST_ID`, as a memory node. Each session start names it again, which keeps "last seen" current. If they want a particular name, set `ANTUMBRA_HOST_ID` in the settings' `env` before running setup, since setup keeps a name that is already there.
- Local only: the stack's compose file and `.env` in `~/.antumbra/stack/docker` (or `docker/.env` in the clone, when setup ran from one), the containers `antumbra-surrealdb` and `antumbra-mcp` (they restart with Docker), and the database in `~/.antumbra/surrealdb`.

## Undo

```sh
claude mcp remove antumbra --scope user
docker compose -f ~/.antumbra/stack/docker/docker-compose.yml down   # local only (from a clone: docker/docker-compose.yml)
```

Then restore the settings backup over `~/.claude/settings.json` and delete `~/.antumbra`. For Gemini CLI: `gemini extensions uninstall antumbra`, or delete `~/.gemini/extensions/antumbra`. For OpenCode: delete `~/.config/opencode/plugins/antumbra.js`. For goose: delete `~/.agents/plugins/antumbra` and the `GOOSE_MOIM_MESSAGE_FILE` line of its `config.yaml`. For Copilot CLI: delete `~/.copilot/hooks/antumbra.json` and the `antumbra` entry of `~/.copilot/mcp-config.json` (`copilot mcp remove antumbra`). For Codex: `codex mcp remove antumbra`, and restore the backup over `~/.codex/hooks.json` (or delete the three entries whose command is `antumbra hook codex ...`). For what was written beside Unity Gateway's files: delete `~/.ucode/.gemini-home/.gemini/extensions/antumbra` and `~/.ucode/opencode-xdg/opencode/plugins/antumbra.js`; `claude-settings.json` goes with `~/.antumbra`.

## When something goes wrong

| Message | What to do |
| --- | --- |
| `Docker is not installed, or not running` | Install or start Docker, then run setup again. |
| `ollama is not answering` | Install or start ollama. |
| `is not inside a clone of the Antumbra repository` | Only `--repo` and `--gpu` need a clone: point `--repo` at the clone, or leave both out to run the published image. |
| `docker pull ... failed` | Check the machine is online. A CLI built from source has no published image at its version: run setup inside the clone, or install the CLI from the release. |
| `the server's container cannot reach ollama` | Follow the steps the message prints for their setup (see 3a). |
| `ollama only listens on 127.0.0.1` (Linux) | The person runs the `systemctl` steps in the message. |
| `the server ... refused the token` | The token expired or is for another server; ask for a new one. |
| `the claude command is not on your PATH` | Claude Code is not installed where setup can find it; setup prints the one `claude mcp add-json` command to run once it is. |
| `settings.json is not valid JSON` | The file was already broken; show the person and fix it together. |
| `is a plain http:// address, so the token would travel in cleartext` | Use the workspace's `https://` address. Only for a lab the person vouches for, run setup again with `--allow-http`; `setup check` then reports the address as accepted. |

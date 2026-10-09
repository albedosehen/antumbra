# Getting started

Antumbra is a private memory and specialist substrate your coding agent plugs into over MCP. This guide stands it up end to end on macOS, Windows, or Linux. Where a step differs by OS, look for the platform note.

## Architecture in one line

A SurrealDB v3 database and the Antumbra MCP server run in Docker (hardened, non-root); the operator console (`antumbra-tui`) and CLI (`antumbra`) run as native binaries; embeddings come from a local `ollama`. Everything talks to one `ws://` database, so the server, console, CLI, and your agent can all connect at once.

## The quick way

With [Docker](https://docs.docker.com/get-docker/) and [ollama](https://ollama.com) running, install the CLI from the latest release (no Rust needed):

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/albedosehen/antumbra/releases/latest/download/antumbra-cli-installer.sh | sh
```

```powershell
# Windows (PowerShell)
irm https://github.com/albedosehen/antumbra/releases/latest/download/antumbra-cli-installer.ps1 | iex
```

Then, from a clone of the repository (the local stack is built from it):

```bash
git clone https://github.com/albedosehen/antumbra.git
cd antumbra
antumbra setup local
```

With [Rust](https://rustup.rs) you can build the CLI from the clone instead of using the installer: `cargo install --path crates/antumbra-cli --locked`.

`antumbra setup local` does every step in the next section for you: it checks Docker and ollama, pulls the embedding model, generates the secrets into `docker/.env`, starts the stack, mints a token into `~/.antumbra/token.txt`, proves recall works with it, writes the hooks to `~/.antumbra/hooks`, wires them into Claude Code's `settings.json` (backed up, added to, never rewritten), and registers the MCP server. Run it again any time; it keeps what is already in place. `--dry-run` shows what it would change.

Somebody else runs the server? Connect to it instead, with the token or the sign-in link you were given:

```bash
antumbra setup hosted https://your-workspace.example --token-file ~/antumbra-token.txt
```

Then check every piece, with what to do about anything that is not right:

```bash
antumbra setup check
```

Want your agent to do it? Point it at **[agent-setup.md](agent-setup.md)** (also served at <https://antumbrahq.ai/setup.md>): "Set up Antumbra for me by following https://antumbrahq.ai/setup.md".

## By hand

What `antumbra setup local` automates, step by step, for when you want to see or change each piece.

### Prerequisites

- **git**
- **Rust** via [rustup](https://rustup.rs). The repo pins its toolchain, so rustup fetches the right version automatically.
- **Docker**: Docker Desktop on macOS/Windows, Docker Engine on Linux.
- **ollama**: <https://ollama.com>, the local embedder.

### 1. Clone and pull the embedding model

```bash
git clone https://github.com/albedosehen/antumbra.git
cd antumbra
ollama pull all-minilm
```

`all-minilm` is a 384-dimensional model that matches the store's vector dimension.

- **Windows:** `ollama` is a Windows app; run `ollama pull all-minilm` in PowerShell or CMD (it is often not on a Git-Bash PATH).
- **Linux:** so the Dockerized server can reach ollama, start it listening on all interfaces: `OLLAMA_HOST=0.0.0.0 ollama serve` (or set `OLLAMA_HOST=0.0.0.0` in its service unit). On macOS/Windows Docker Desktop this is automatic.

### 2. Install the console and CLI

```bash
cargo install --path crates/antumbra-cli --locked   # the `antumbra` CLI
cargo install --path crates/antumbra-tui --locked   # the `antumbra-tui` console
```

If you have [`just`](https://github.com/casey/just), `just install` builds all three binaries instead. The first build compiles the whole dependency graph and is slow; it happens once. Both land in `~/.cargo/bin`, which rustup puts on your PATH.

### 3. Configure secrets

```bash
cp docker/.env.example docker/.env
```

Edit `docker/.env`:

- `SURREAL_PASS`: a strong password (the database root login).
- `ANTUMBRA_JWT_SECRET`: 32 random bytes, base64. Generate one:
  - macOS/Linux: `openssl rand -base64 32`
  - Windows (PowerShell): `[Convert]::ToBase64String((1..32 | % { Get-Random -Max 256 }))`
- `ANTUMBRA_SURREAL_DATA`: an absolute host directory for the database files, kept outside the repo. Create it first:
  - macOS/Linux: `mkdir -p ~/.antumbra/surrealdb`
  - Windows: use forward slashes, e.g. `C:/Users/<you>/.antumbra/surrealdb`
  - **Linux:** the container runs as uid 65532, so make the directory writable by it: `sudo chown -R 65532:65532 ~/.antumbra/surrealdb`. (Not needed on Docker Desktop, which maps permissions for you.)
- Leave the `ANTUMBRA_EMBEDDER_*` defaults (host ollama, `all-minilm`).

### 4. Start the stack

```bash
docker compose -f docker/docker-compose.yml up -d
```

The first run builds the `antumbra-mcp` image (a one-time, slow Rust build). Then check both services:

```bash
docker compose -f docker/docker-compose.yml ps
```

`antumbra-surrealdb` should be `healthy` and `antumbra-mcp` `Up`. The database listens on `ws://127.0.0.1:8000` and the MCP server on `http://127.0.0.1:8081`, both bound to localhost only.

### 5. Mint an access token

The MCP server verifies a signed token on every call. Mint a long-lived one (it reuses the secret from `docker/.env`):

```bash
docker compose -f docker/docker-compose.yml run --rm antumbra-mcp \
  --mint-token --tenant ws:default --user user:default --token-ttl-days 365
```

Copy the printed JWT; it is your `ANTUMBRA_TOKEN`.

### 6. Open the console and CLI

Both connect to the database over `ws://` with the root login from your `.env`:

```bash
# macOS/Linux
PASS=$(grep '^SURREAL_PASS=' docker/.env | cut -d= -f2-)
antumbra-tui --url ws://127.0.0.1:8000/rpc --db-user root --db-pass "$PASS"
antumbra     --url ws://127.0.0.1:8000/rpc --db-user root --db-pass "$PASS" status
```

```powershell
# Windows (PowerShell)
$PASS = (Get-Content docker/.env | ? { $_ -like 'SURREAL_PASS=*' }) -replace '^SURREAL_PASS=',''
antumbra-tui --url ws://127.0.0.1:8000/rpc --db-user root --db-pass $PASS
antumbra     --url ws://127.0.0.1:8000/rpc --db-user root --db-pass $PASS status
```

In the console, press `?` for the keymap and `c` for the "connect your agent" panel. It is empty until your agent (or `antumbra seed`) writes memories.

### 7. Connect your coding agent

This is the point of Antumbra: your agent (Claude Code, Cursor, any MCP client) reads and writes Antumbra over its lifecycle. Set these in the agent's environment:

```
ANTUMBRA_URL=http://127.0.0.1:8081
ANTUMBRA_WORKSPACE_ID=ws:default
ANTUMBRA_TOKEN=<the token from step 5>
ANTUMBRA_HOST_ID=<this machine's name>
```

then add the lifecycle hooks. **[`scripts/hooks/README.md`](../scripts/hooks/README.md)** has copy-paste `settings.json` blocks: a `bash` / `.sh` set for macOS/Linux and a `pwsh` / `.ps1` set for Windows. (`antumbra setup` writes these for you.)

## Optional: serving + autonomy (needs an NVIDIA GPU)

The stack above is the default, GPU-free server: memory, recall, routing. To also have your agent get **answers served from trained experts** (the `answer` tool) and have Antumbra **train the behaviors you accept into your own standing expert**, run the GPU build on a machine with an NVIDIA card (Linux, or Windows via WSL2):

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up -d
```

See [Running the trainer → the GPU server](running-the-trainer.md#the-gpu-server-serving--standing-experts) for the build details, the native (non-Docker) recipe, and standing experts. Without a GPU, `answer` escalates cleanly and everything else works as normal.

## Just kicking the tires?

You need none of the above to look around. From a clone:

```bash
cargo run -p antumbra-tui -- --demo
```

renders a seeded console against a throwaway in-memory store.

## Troubleshooting

- **`antumbra-tui: command not found`**: `~/.cargo/bin` is not on your PATH yet. Open a new shell after installing rustup, or run the binary from `target/release/`.
- **Console or CLI hangs or errors connecting to `ws://`**: the stack is not up. Check `docker compose -f docker/docker-compose.yml ps` and `... logs`.
- **Memories store but recall is nonsense**: the embedder is unreachable or mismatched. Confirm ollama is running and `ollama pull all-minilm` succeeded; on Linux confirm ollama is bound with `OLLAMA_HOST=0.0.0.0`.
- **`antumbra-surrealdb` keeps restarting on Linux**: the data directory is not writable by uid 65532. Run `sudo chown -R 65532:65532 <ANTUMBRA_SURREAL_DATA>`.
- **Windows: `ollama pull` is not found in Git Bash**: run it in PowerShell or CMD instead.

See **[`docker/README.md`](../docker/README.md)** for the stack's security model and **[Using Antumbra](integration.md)** for the agent integration in depth.

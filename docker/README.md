# Antumbra local Docker stack

A hardened, **non-root** local stack: SurrealDB v3 (Antumbra needs v3) plus the `antumbra-mcp` server, with embeddings served by a host ollama. Everything binds to `127.0.0.1` and every container runs as uid 65532 with capabilities dropped and `no-new-privileges`.

## Prerequisites

- Docker (Desktop or Engine).
- A host embeddings endpoint returning 384-dim vectors (matches the store's `EMBED_DIM`). With ollama: `ollama pull all-minilm`.

## Setup

```bash
cp docker/.env.example docker/.env     # fill in SURREAL_PASS, ANTUMBRA_JWT_SECRET, ANTUMBRA_SURREAL_DATA
docker compose -f docker/docker-compose.yml up -d
```

- **SurrealDB** comes up on `ws://127.0.0.1:8000/rpc` (namespace `antumbra`, database `main`, isolated from anything else on the server).
- **antumbra-mcp** comes up on `http://127.0.0.1:8081` once SurrealDB is healthy.

Because the database is a real `ws://` server, the MCP server, the `antumbra-tui` console, the CLI, and your coding agent can all connect at the same time, unlike the single-writer embedded `surrealkv` file.

## Why these choices

- **Non-root, zero privilege.** The SurrealDB image already runs as uid 65532; the `antumbra-mcp` image ships on `distroless/cc:nonroot`. Both drop all capabilities and set `no-new-privileges`. `antumbra-mcp` runs read-only (it talks to the DB over the wire).
- **Host bind-mount for the data.** A named volume is created root-owned, which the non-root server cannot write. A host bind-mount is writable by the container on Docker Desktop. On a **Linux** host, `chown -R 65532:65532` the `ANTUMBRA_SURREAL_DATA` directory once.
- **rocksdb backend.** A standard server KV backend; the v3 client is agnostic to it over `ws://`.

## Migrating memories in

With the stack up, mint a token and import an export (see `scripts/kushtaka-export.ps1` / `scripts/kushtaka-import.ps1`):

```powershell
# point the import at the dockerized server
$env:ANTUMBRA_URL = 'http://127.0.0.1:8081'
$env:ANTUMBRA_TOKEN = '<token minted with the same ANTUMBRA_JWT_SECRET>'
pwsh scripts/kushtaka-import.ps1 -In kushtaka-export.json
```

## Notes

- `docker/.env` holds secrets and is gitignored. Move these to a secret manager (Doppler) for anything beyond local use.
- The first `antumbra-mcp` image build compiles the workspace and is slow; rebuilds are cached.

# Antumbra local Docker stack

A hardened, **non-root** local stack: SurrealDB v3 (Antumbra needs v3) plus the `antumbra-mcp` server, with embeddings served by a host ollama, and an optional `antumbra-control-server` for hosted signup. Everything binds to `127.0.0.1` and every container runs as uid 65532 with capabilities dropped and `no-new-privileges`.

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

## Hosted signup (optional)

`antumbra-control-server` is the control plane: invite-gated signup and magic-link login that mint RS256 tokens. Skip it entirely to stay on the offline HS256 `mint-token` flow.

One-time setup -- generate the RS256 issuer keypair into `docker/keys/` (gitignored) and set the control-plane variables in `docker/.env`:

```bash
openssl genrsa -out docker/keys/control-signing.pem 2048
openssl rsa -in docker/keys/control-signing.pem -pubout -out docker/keys/control-signing.pub.pem
# in docker/.env: ANTUMBRA_MAGIC_SECRET (openssl rand -base64 32), ANTUMBRA_CONTROL_BASE_URL
docker compose -f docker/docker-compose.yml --profile control up -d antumbra-control-server
```

The server refuses to start while `ANTUMBRA_MAGIC_SECRET` is the `.env.example` placeholder (or shorter than 32 chars) -- a copied example file must not ship forgeable links.

The flow: mint an invite (`antumbra-control-server mint-invite`, expires in 14 days by default; `--ttl-days 0` for non-expiring), `POST /signup {email, invite}`, follow the emailed magic link (`GET /magic/verify?token=...`), receive the RS256 JWT. Every magic link is **single-use** (a replayed link is refused even inside its 15-minute window) and link requests sit behind a per-email cooldown (default 60s; `--link-cooldown-secs`). `list-invites` / `revoke-invite <code>` manage outstanding codes. Without `SMTP_*` set, the dev mailer logs the magic link to the container's stderr (`docker logs antumbra-control-server`); set `SMTP_HOST/PORT/USER/PASS/FROM` in `docker/.env` for real delivery. To make `antumbra-mcp` accept these tokens, run it with `--jwt-public-key /keys/control-signing.pub.pem --jwt-audience antumbra` (mount `./keys` into that container too).

### Least-privilege database access (recommended)

By default the control plane signs in as the SurrealDB instance root. Scope it to a **database-level** user instead, so a compromise of this internet-adjacent service is contained to the `antumbra/main` database:

```bash
# one-time DDL, as root (a ROLES OWNER database user can still run the schema DDL):
echo "DEFINE USER antumbra_control ON DATABASE PASSWORD '<generated>' ROLES OWNER;" \
  | docker exec -i antumbra-surrealdb /surreal sql \
      --endpoint http://localhost:8000 --user root --pass "$SURREAL_PASS" \
      --ns antumbra --db main --hide-welcome
# then in docker/.env:
#   ANTUMBRA_CONTROL_DB_USER=antumbra_control
#   ANTUMBRA_CONTROL_DB_PASS=<generated>
#   ANTUMBRA_CONTROL_DB_AUTH=database
docker compose -f docker/docker-compose.yml --profile control up -d antumbra-control-server
```

## Notes

- `docker/.env` holds secrets and is gitignored. Move these to a secret manager (Doppler) for anything beyond local use. Compose passes them as environment, never as command-line arguments; a deployment that mounts secrets as files (Docker secrets, Kubernetes, a Key Vault CSI mount) passes `--db-pass-file` and `--jwt-secret-file` to `antumbra-mcp` instead, so they appear in neither the process arguments nor the environment.
- The first image build of `antumbra-mcp` or `antumbra-control-server` compiles the workspace and is slow; rebuilds are cached.
- The RS256 private key never enters an image; it is bind-mounted read-only at runtime.

## GPU build on a CDI host

A Docker that hands out GPUs through the Container Device Interface (NixOS with `hardware.nvidia-container-toolkit`, or any daemon with CDI enabled and no named `nvidia` runtime) refuses the classic device request in `docker-compose.gpu.yml` with "could not select device driver nvidia". Layer `docker-compose.gpu-cdi.yml` last; it resets that block and asks for `nvidia.com/gpu=all` by CDI name:

```sh
docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml -f docker/docker-compose.gpu-cdi.yml up -d surrealdb antumbra-mcp
```

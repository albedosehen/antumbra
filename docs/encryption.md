# Encryption at rest

Status: **design, not built.** Antumbra 0.5.3 writes everything in the clear, and its changelog says so ("the database is not encrypted"). This document is the plan for changing that. It states what the encryption protects against, where it sits, how keys are made, kept and retired, and in what order the work lands.

The aim is one sentence: **a user who installs Antumbra gets their data encrypted on disk without doing anything, and without encrypting their disk.**

## What it protects against, and what it does not

[security.md](security.md) protects tenants from each other on a running server. This document adds a second adversary: **someone who has Antumbra's files but not the running machine.**

Protected:

- A lost or stolen laptop, or a disk pulled from a server.
- A copy of the data directory: a backup, a folder synced to OneDrive, iCloud or Dropbox, a container volume snapshot, a disk image, a decommissioned drive.
- Another account on the same machine that can read the files but cannot act as the user.

Not protected:

- Malware running as the user while Antumbra is unlocked. The operating system releases the key to that user's processes, so this is the same boundary that protects a browser's saved passwords.
- An administrator or root on the running machine, or anyone who can read the process's memory.
- The operator of a hosted server. They hold the key by design, as [security.md](security.md) already places the operator out of scope.
- Plaintext the operating system writes on its own: swap, a hibernation file, crash dumps. Antumbra locks its key material in memory where it can and turns off core dumps on a server, but full protection here needs the operating system's own swap encryption.
- Data that was on disk in the clear before encryption was turned on. On an SSD, deleting the old files does not reliably erase them.

## Where the encryption sits

### The database: the storage engine encrypts it

Antumbra cannot encrypt fields itself. The database reads them in the clear:

- **Full-text search:** BM25 indexes on `memory.content` and `document_chunk.content`.
- **Vector search:** HNSW indexes on every embedding.
- **Grouping and lookups:** `GROUP BY title` and lookups by host, evidence and slug.
- **Row permissions:** the engine evaluates `tenant_id`, `compartment`, `owner` and `grantee` on every row.

A ciphertext field would break all of these, and an embedding left in the clear leaks the text it was made from.

So the database is encrypted underneath, by the storage engine. **SurrealKV 1.4** (2026-11-11) encrypts every file it writes:

- the manifest;
- every table file (SSTable);
- the value log;
- the write-ahead log.

How it does it:

- **Per-file keys.** Each file gets its own key, derived from a root key, a random salt and the file's identity.
- **Bound blocks.** Every block is sealed with AES-256-GCM, or XChaCha20-Poly1305 on a CPU without AES instructions. Each block is bound to its database, its file, its key id and its position, so blocks cannot be swapped, reordered or moved between files.
- **Pluggable root keys.** The application supplies them through a `KeyManager`. Each key id is `Active`, `DecryptOnly` or `Retired`, which is what rotation and erasure need (below).

Queries, indexes, permissions and sync work unchanged, because the engine decrypts below them.

Two things stand in the way:

1. **Antumbra is on SurrealDB 3.3, whose engine has no encryption.** Moving to SurrealDB 4 comes first. SurrealDB 4 also removes RocksDB, which the Docker setup's database server uses today.
2. **SurrealDB 4.0.1 does not pass a key manager to SurrealKV.** Its `surrealdb-kvs-surrealkv` crate builds the engine's options with no key (upstream issue surrealdb/surrealdb#88 is open). The plan is to fix this upstream, with two additions:
   - **An in-process hook for embedded use.** A program registers its `KeyManager` for a store's path before opening it. The key never reaches disk or a configuration string.
   - **A datastore option for servers.** For example `surrealkv_encryption_key_file`, which `surreal start` reads from a mounted secret.

   Until both are released, Antumbra carries the first as a small `[patch]` of that one crate.

### Files Antumbra writes itself: a sealed-file format

Some user-derived files live outside the database:

| File                                                                                   | Why it is sensitive                                               |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| LoRA adapters (`adapters/*.safetensors`)                                               | trained on a user's behaviors and memories, so they can leak them |
| training corpora (`corpora/_imported.json`, `_metabolized.json`, `_consolidated.json`) | memory and trace text                                             |
| exports (`antumbra-export-*.jsonl`)                                                    | every memory of the caller                                        |
| Goose session files (`~/.antumbra/goose/*.md`)                                         | recalled memory text                                              |

Each of these is written as a **sealed file**. Its header is `ASF1`, a version, the cipher suite, a key id, a 32-byte random salt and the file's purpose. Its body is the STREAM construction:

- the plaintext in 64 KiB chunks, each sealed with XChaCha20-Poly1305;
- each nonce carries the chunk's counter and a flag on the last chunk, so truncation, reordering and appending are all refused;
- the header is the associated data of every chunk.

The file's key is `HKDF-SHA-256(file key, salt, "antumbra/file/v1" ‖ purpose ‖ workspace)`.

This is the same shape as Copal's sealed blobs and SurrealKV's files, so one way of thinking covers all three.

How this fits the files' existing behavior:

- **Adapters.** They are already read whole into memory to be served, so decrypting them costs no new code path. The sha256 on the expert card stays a digest of the plaintext, so the existing integrity checks, the expert source and the Copal archive keep their meaning.
- **Base models.** Weights downloaded from Hugging Face are public and are not encrypted.

### Secrets: the operating system's credential store

Today these are plain files:

- `~/.antumbra/token.txt` and the workspace tokens under `~/.antumbra/workspaces/`;
- the waiting secrets in `join.json` and `login.json`;
- `--copal-keys`;
- the Docker stack's `.env`.

Each moves into the platform's credential store (below), with the plain file kept only where no store exists.

The hooks change with them. They read the token today with `cat`; they will ask `antumbra auth header`, which reads the store.

The Docker stack's secrets will be held in the store too. `antumbra setup local` hands them to `docker compose` in its environment when it starts the stack, so they are not left in a file.

## Keys

### The hierarchy

```
platform protector (DPAPI, Keychain, Secret Service, TPM, KMS, passphrase, recovery key)
        │ unwraps
        ▼
keyring master key (256-bit, random)
        │ unseals
        ▼
keyring:  database root keys   { id → key, state }      → SurrealKV KeyManager
          file keys            { generation → key }      → sealed files (HKDF per file)
          workspace file keys  { workspace → key }       → a workspace's adapters and exports
```

- **Every data key is random**, never derived from a password.
- **The keyring** is one small file next to the data, `keyring.asf`. It is sealed with XChaCha20-Poly1305 under a random master key, and its header is the associated data.
- **Slots wrap the master key.** The master key is stored only wrapped, once per **slot**, as LUKS2 does. Any one slot opens the keyring, so a slot can be added or removed without touching the data.

### Slots, in the order Antumbra tries them

| Platform      | Default slot                                                                                                     | Optional hardware binding                                                                                                   |
| ------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Windows       | Windows Credential Manager (DPAPI, unlocked by the user's sign-in)                                               | a TPM key through the CNG Platform Crypto Provider, so the disk is useless on another machine even with the user's password |
| macOS         | the data-protection Keychain, `AfterFirstUnlockThisDeviceOnly`, never synced to iCloud                           | a Secure Enclave key (signed builds only, since it needs an entitlement)                                                    |
| Linux desktop | Secret Service (GNOME Keyring, KWallet)                                                                          | a TPM 2.0 key                                                                                                               |
| Linux server  | `systemd-creds` with the TPM, delivered to the service as a credential                                           | —                                                                                                                           |
| container     | a key file mounted as a secret (`/run/secrets/...`, a Kubernetes Secret, Doppler), never an environment variable | a cloud KMS or Vault Transit unwraps the master key                                                                         |
| anywhere      | a passphrase through Argon2id (64 MiB, 3 passes, 4 lanes; RFC 9106's second recommendation)                      | —                                                                                                                           |

The credential stores are reached through the `keyring` crate, which covers Windows, macOS and Linux behind one API.

Where no store exists (a headless Linux box with neither TPM nor Secret Service), Antumbra says so once and writes the master key to `~/.config/antumbra/master.key` at mode 0600, apart from the data. That protects a copied data directory but not a stolen disk, and `antumbra keys status` reports it.

### The recovery key

Setup also creates a **recovery slot**: a random 256-bit key, shown once as 52 base32 characters in groups. This is how a user gets their data back when the machine that held the default slot is gone (a new laptop, a cleared TPM, a reinstalled OS).

It is the only thing Antumbra asks a user to keep, and keeping it is optional. Without it, losing the machine's slot loses the data. `antumbra keys recovery --new` replaces the recovery slot.

### Rotation and erasure

- **Rotating a protector is instant.** Re-wrapping the master key in a new slot and dropping the old one touches no data.
- **Rotating a database key** adds a key id as `Active` and moves the old one to `DecryptOnly`. New files are sealed under the new key at once. Old files keep their key until they are rewritten: by compaction (table files), by the value log's garbage collection, and as write-ahead segments roll over.

  `antumbra keys rotate` forces a full compaction, then watches SurrealKV's encryption status, which counts the files and bytes still under each id without opening any. When the old id covers nothing, it is set to `Retired` and its key is destroyed.

  Destroying it also erases what deleting rows leaves behind: old versions in table files that compaction has not reached are unreadable once their key is gone. `antumbra keys rotate` runs the whole cycle.
- **Rotating file keys** starts a new generation for new files. `antumbra keys rotate --files` re-seals the old files, then drops the old generation.
- **Removing a workspace** destroys its workspace file key, which makes its adapters and exports unreadable at once wherever copies went. Its rows are deleted as today, and the next database key rotation erases their remnants.

### Keys in memory

- **Wiped when dropped:** keys live in `Zeroizing` buffers, and their types hide them from `Debug`, as secret-bearing types already do.
- **Locked where possible:** the keyring's pages are locked against swapping where the platform allows it.
- **No core dumps on a server:** the server process turns them off (`PR_SET_DUMPABLE` on Linux).
- **Never logged:** no key, wrapped or not, is ever logged. `antumbra keys status` prints key ids and states, never key material.

### Algorithms, and the quantum question

Every algorithm here is standard, current and implemented in audited RustCrypto crates, most of them already in Antumbra's dependency tree:

- **Ciphers:** XChaCha20-Poly1305, and AES-256-GCM inside SurrealKV;
- **Key derivation:** HKDF-SHA-256 for subkeys;
- **Passphrases:** Argon2id;
- **Randomness:** the operating system's generator through `getrandom`.

**Post-quantum.** At-rest encryption with 256-bit symmetric keys is not threatened by quantum computers. The only public-key steps are the device-bound platform wraps (TPM, Secure Enclave), which never leave the machine.

The first feature that would send a wrapped key off the machine is enrolling a second device or escrowing a key. Any such feature will use hybrid post-quantum key exchange: HPKE (RFC 9180) with X-Wing, which combines X25519 and ML-KEM-768. Classical public-key wrapping alone will not be used.

## What a user sees

- **New installs.** Nothing changes in how Antumbra is used. Setup prints one line, "Antumbra encrypts its data on this machine; the key is kept in <store>", and shows the recovery key once.
- **Status.** `antumbra keys status` shows:
  - which slots exist;
  - the database key ids and their states;
  - the bytes still under each database key;
  - any file found in the clear where a sealed one belongs.
- **Existing installs.** `antumbra store encrypt` moves a store over:
  1. exports the plaintext store;
  2. imports it into a new encrypted one;
  3. compares the counts;
  4. swaps the directories, keeping the old store until the user removes it;
  5. seals the files listed above.

  SurrealKV settles whether a database is encrypted when the database is created, and refuses a file in the clear inside an encrypted one. So a copy is the only way. Setup offers this once; it never runs unasked.
- **Backups** of an encrypted data directory are safe to keep anywhere. Restoring one needs the machine's slot or the recovery key.

### Servers and Docker

- **`antumbra setup local`** stores the database key in the user's credential store and starts the stack with it. The SurrealDB container reads it from a secret file that lives only for the container's lifetime. That needs the datastore option above; until SurrealDB ships it, the Docker stack stays unencrypted and `antumbra keys status` says so.
- **Hosted and Kubernetes deployments** mount the key as a secret file, or unwrap it through a KMS, so that a stolen volume or snapshot is useless without the key service. The operator still holds the key.

## Order of work

1. **SurrealDB 4.** Move Antumbra to SurrealDB 4 and SurrealKV 1.4, including the Docker database's move off RocksDB. Encryption is impossible before this.
2. **`antumbra-crypto`.** A new crate:
   - the keyring and its slots;
   - the platform protectors;
   - the recovery key;
   - the sealed-file format;
   - a `KeyManager` over the keyring.

   It needs nothing from step 1, so it can be built in parallel and tested on its own, including format tests and tests that tampering is refused.
3. **Upstream.** Propose the SurrealDB changes (the in-process hook and the server option), and carry the hook as a `[patch]` until they are released.
4. **The embedded store, encrypted by default** for new installs; `antumbra store encrypt` and `antumbra keys status`.
5. **The files:** adapters, corpora, exports and the Goose files sealed.
6. **The secrets:** into the credential store, with `antumbra auth header` for the hooks.
7. **Servers:** the Docker stack's key and secrets, then the KMS protectors.
8. **Hardware binding:** TPM and Secure Enclave slots, and an optional "ask me first" unlock through Windows Hello or Touch ID.

## Decisions still open

- **The recovery key.** Shown at setup and optional (this design), or required to be confirmed before setup finishes.
- **The Docker stack's database.** Keep the separate SurrealDB container and wait for the upstream server option, or embed the store in `antumbra-mcp` so encryption does not depend on SurrealDB's server.
- **The upstream route.** Carry the `[patch]` from the start (this design), or wait for SurrealDB to release the hook.

//! The keyring: every key antumbra encrypts with, in one file.
//!
//! ```json
//! { "format": "antumbra-keyring", "version": 1, "id": "<hex>",
//!   "slots": [ { "kind": "recovery", "id": "<hex>", "params": {}, "wrapped": "<base64>" } ],
//!   "payload": "<base64>" }
//! ```
//!
//! The payload (the keys) is sealed with XChaCha20-Poly1305 under a random master key. The master
//! key is stored only wrapped, once per slot, as LUKS2 does: any one slot opens the keyring, and a
//! slot is added or removed without touching anything else. Each slot's wrap is bound to the
//! keyring's id, the slot's kind, id and parameters, so none of them can be changed or moved to
//! another keyring.
//!
//! Keys live in scopes (`files`, or a workspace such as `ws:acme`), each with numbered
//! generations of which one is active. A [`KeyRef`] (`files/1`) names one; sealed files carry it
//! in their header. Rotating a scope starts a new generation for new files; retiring a generation
//! destroys its key, and with it everything still sealed under it.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, XNonce};
use data_encoding::{BASE64, HEXLOWER};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, Zeroizing};

use crate::key::random;
use crate::slot::Protector;
use crate::{Error, Key, Result};

const FORMAT: &str = "antumbra-keyring";
const VERSION: u32 = 1;
const ID_LEN: usize = 16;
const NONCE_LEN: usize = 24;
const PAYLOAD_LABEL: &[u8] = b"antumbra-keyring/v1/payload\0";
const SLOT_LABEL: &[u8] = b"antumbra-keyring/v1/slot\0";

/// The scope of the keys for files that belong to no workspace.
pub const FILES_SCOPE: &str = "files";

/// One key of the keyring: a scope and a generation, written `scope/generation`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyRef {
    scope: String,
    generation: u32,
}

impl KeyRef {
    pub fn new(scope: &str, generation: u32) -> Result<Self> {
        check_scope(scope)?;
        if generation == 0 {
            return Err(Error::format("key reference", "generations start at 1"));
        }
        Ok(Self {
            scope: scope.to_owned(),
            generation,
        })
    }

    pub fn scope(&self) -> &str {
        &self.scope
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }
}

impl fmt::Display for KeyRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.scope, self.generation)
    }
}

impl FromStr for KeyRef {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let (scope, generation) = text
            .rsplit_once('/')
            .ok_or_else(|| Error::format("key reference", format!("{text:?} has no '/'")))?;
        let generation = generation
            .parse()
            .map_err(|_| Error::format("key reference", format!("{text:?} has no generation")))?;
        Self::new(scope, generation)
    }
}

/// A scope is a short name of letters, digits and `:` `.` `_` `-`, so a key reference parses
/// back the same and fits a sealed file's header.
fn check_scope(scope: &str) -> Result<()> {
    let fine = !scope.is_empty()
        && scope.len() <= 200
        && scope
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '.' | '_' | '-'));
    if fine {
        Ok(())
    } else {
        Err(Error::format(
            "key scope",
            format!("{scope:?} is not a scope name"),
        ))
    }
}

/// A slot as `status` shows it: its kind and id, never its key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotInfo {
    pub kind: String,
    pub id: String,
}

struct Scope {
    active: u32,
    keys: BTreeMap<u32, Key>,
}

/// Antumbra's keys, opened. See the module documentation.
pub struct Keyring {
    id: [u8; ID_LEN],
    master: Key,
    scopes: BTreeMap<String, Scope>,
    slots: Vec<SlotRecord>,
}

impl fmt::Debug for Keyring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Keyring")
            .field("id", &self.id_hex())
            .field("scopes", &self.scopes.keys().collect::<Vec<_>>())
            .field("slots", &self.slots().len())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
struct KeyringFile {
    format: String,
    version: u32,
    id: String,
    slots: Vec<SlotRecord>,
    payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlotRecord {
    kind: String,
    id: String,
    #[serde(default)]
    params: BTreeMap<String, String>,
    wrapped: String,
}

#[derive(Serialize, Deserialize)]
struct PayloadFile {
    scopes: BTreeMap<String, ScopeFile>,
}

#[derive(Serialize, Deserialize)]
struct ScopeFile {
    active: u32,
    keys: BTreeMap<u32, String>,
}

impl Drop for PayloadFile {
    fn drop(&mut self) {
        for scope in self.scopes.values_mut() {
            for key in scope.keys.values_mut() {
                key.zeroize();
            }
        }
    }
}

impl Keyring {
    /// A new keyring with a fresh master key and one active key for files. It has no slot yet:
    /// add at least one before saving it.
    pub fn create() -> Result<Self> {
        let mut scopes = BTreeMap::new();
        scopes.insert(FILES_SCOPE.to_owned(), Scope::fresh()?);
        Ok(Self {
            id: random()?,
            master: Key::generate()?,
            scopes,
            slots: Vec::new(),
        })
    }

    /// The keyring's id, in hex. Public: it names the keyring, it opens nothing.
    pub fn id_hex(&self) -> String {
        HEXLOWER.encode(&self.id)
    }

    /// Adds a slot that `protector` opens, and returns its id.
    pub fn add_slot(&mut self, protector: &dyn Protector) -> Result<String> {
        let slot_id = HEXLOWER.encode(&random::<4>()?);
        let (kek, params) = protector.enroll(&self.id, &slot_id)?;
        let aad = slot_aad(&self.id, protector.kind(), &slot_id, &params)?;
        let wrapped = seal_blob(&kek, self.master.expose(), &aad)?;
        self.slots.push(SlotRecord {
            kind: protector.kind().to_owned(),
            id: slot_id.clone(),
            params,
            wrapped,
        });
        Ok(slot_id)
    }

    /// Removes a slot. The last slot cannot be removed: nothing would open the keyring.
    pub fn remove_slot(&mut self, slot_id: &str) -> Result<SlotInfo> {
        let at = self
            .slots
            .iter()
            .position(|slot| slot.id == slot_id)
            .ok_or_else(|| Error::Refused(format!("no slot {slot_id}")))?;
        if self.slots.len() == 1 {
            return Err(Error::Refused(
                "the last slot cannot be removed: nothing would open the keyring".into(),
            ));
        }
        let slot = self.slots.remove(at);
        Ok(SlotInfo {
            kind: slot.kind,
            id: slot.id,
        })
    }

    pub fn slots(&self) -> Vec<SlotInfo> {
        self.slots
            .iter()
            .map(|slot| SlotInfo {
                kind: slot.kind.clone(),
                id: slot.id.clone(),
            })
            .collect()
    }

    /// The keyring as the bytes of its file.
    pub fn to_bytes(&self) -> Result<Zeroizing<Vec<u8>>> {
        if self.slots.is_empty() {
            return Err(Error::Refused(
                "a keyring with no slot could never be opened: add one first".into(),
            ));
        }
        let payload = PayloadFile {
            scopes: self
                .scopes
                .iter()
                .map(|(name, scope)| {
                    let keys = scope
                        .keys
                        .iter()
                        .map(|(generation, key)| (*generation, BASE64.encode(key.expose())))
                        .collect();
                    (
                        name.clone(),
                        ScopeFile {
                            active: scope.active,
                            keys,
                        },
                    )
                })
                .collect(),
        };
        let plain = Zeroizing::new(serde_json::to_vec(&payload).map_err(json_error)?);
        let aad = payload_aad(&self.id);
        let file = KeyringFile {
            format: FORMAT.to_owned(),
            version: VERSION,
            id: self.id_hex(),
            slots: self.slots.clone(),
            payload: seal_blob(&self.master, &plain, &aad)?,
        };
        let mut out = serde_json::to_vec_pretty(&file).map_err(json_error)?;
        out.push(b'\n');
        Ok(Zeroizing::new(out))
    }

    /// Opens a keyring from the bytes of its file with `protector`, trying each of its slots of
    /// that kind.
    pub fn from_bytes(bytes: &[u8], protector: &dyn Protector) -> Result<Self> {
        let file: KeyringFile = serde_json::from_slice(bytes).map_err(json_error)?;
        if file.format != FORMAT {
            return Err(Error::format(
                "keyring",
                format!("format {:?}", file.format),
            ));
        }
        if file.version != VERSION {
            return Err(Error::format(
                "keyring",
                format!("version {} is not one this build reads", file.version),
            ));
        }
        let id_bytes = HEXLOWER
            .decode(file.id.as_bytes())
            .map_err(|_| Error::format("keyring", "its id is not hex"))?;
        let id: [u8; ID_LEN] = id_bytes
            .try_into()
            .map_err(|_| Error::format("keyring", "its id is not 16 bytes"))?;

        let master = file
            .slots
            .iter()
            .filter(|slot| slot.kind == protector.kind())
            .find_map(|slot| {
                let kek = protector.unlock(&id, &slot.id, &slot.params).ok()?;
                let aad = slot_aad(&id, &slot.kind, &slot.id, &slot.params).ok()?;
                let master = open_blob(&kek, &slot.wrapped, &aad).ok()?;
                Key::from_slice(&master).ok()
            })
            .ok_or_else(|| Error::NoSlotOpened {
                kind: protector.kind().to_owned(),
            })?;

        let plain = open_blob(&master, &file.payload, &payload_aad(&id))?;
        let payload: PayloadFile = serde_json::from_slice(&plain).map_err(json_error)?;
        let mut scopes = BTreeMap::new();
        for (name, scope) in &payload.scopes {
            check_scope(name)?;
            let mut keys = BTreeMap::new();
            for (generation, encoded) in &scope.keys {
                let bytes = Zeroizing::new(
                    BASE64
                        .decode(encoded.as_bytes())
                        .map_err(|_| Error::format("keyring", "a key is not base64"))?,
                );
                keys.insert(*generation, Key::from_slice(&bytes)?);
            }
            if !keys.contains_key(&scope.active) {
                return Err(Error::format(
                    "keyring",
                    format!("scope {name} has no key for its active generation"),
                ));
            }
            scopes.insert(
                name.clone(),
                Scope {
                    active: scope.active,
                    keys,
                },
            );
        }
        Ok(Self {
            id,
            master,
            scopes,
            slots: file.slots,
        })
    }

    /// Writes the keyring to `path` atomically, readable by its owner only.
    pub fn save(&self, path: &Path) -> Result<()> {
        write_atomically(path, &self.to_bytes()?)
    }

    /// Opens the keyring at `path` with `protector`.
    pub fn open(path: &Path, protector: &dyn Protector) -> Result<Self> {
        let bytes = Zeroizing::new(fs::read(path)?);
        Self::from_bytes(&bytes, protector)
    }

    /// The key `key_ref` names.
    pub fn key(&self, key_ref: &KeyRef) -> Result<Key> {
        self.scopes
            .get(key_ref.scope())
            .and_then(|scope| scope.keys.get(&key_ref.generation()))
            .cloned()
            .ok_or_else(|| Error::UnknownKey(key_ref.to_string()))
    }

    /// The active key of `scope`, which new files are sealed under.
    pub fn active(&self, scope: &str) -> Result<(KeyRef, Key)> {
        let found = self
            .scopes
            .get(scope)
            .ok_or_else(|| Error::UnknownKey(format!("{scope}/*")))?;
        let key_ref = KeyRef::new(scope, found.active)?;
        let key = self.key(&key_ref)?;
        Ok((key_ref, key))
    }

    /// Makes `scope` with a fresh key if it does not exist. Returns whether it was made, in which
    /// case the keyring must be saved.
    pub fn ensure_scope(&mut self, scope: &str) -> Result<bool> {
        check_scope(scope)?;
        if self.scopes.contains_key(scope) {
            return Ok(false);
        }
        self.scopes.insert(scope.to_owned(), Scope::fresh()?);
        Ok(true)
    }

    /// Starts a new generation of `scope` for new files. Older generations still open what was
    /// sealed under them until they are retired.
    pub fn rotate(&mut self, scope: &str) -> Result<KeyRef> {
        let found = self
            .scopes
            .get_mut(scope)
            .ok_or_else(|| Error::UnknownKey(format!("{scope}/*")))?;
        let next = found
            .keys
            .keys()
            .next_back()
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| Error::Refused(format!("scope {scope} has no generation left")))?;
        found.keys.insert(next, Key::generate()?);
        found.active = next;
        KeyRef::new(scope, next)
    }

    /// Destroys a generation that is no longer active. Whatever is still sealed under it can
    /// never be opened again.
    pub fn retire(&mut self, key_ref: &KeyRef) -> Result<()> {
        let found = self
            .scopes
            .get_mut(key_ref.scope())
            .ok_or_else(|| Error::UnknownKey(key_ref.to_string()))?;
        if found.active == key_ref.generation() {
            return Err(Error::Refused(format!(
                "{key_ref} is the active generation; rotate first"
            )));
        }
        found
            .keys
            .remove(&key_ref.generation())
            .map(drop)
            .ok_or_else(|| Error::UnknownKey(key_ref.to_string()))
    }

    /// Destroys every key of `scope`, a workspace's when the workspace is removed: everything
    /// sealed under them, wherever copies went, can never be opened again.
    pub fn destroy_scope(&mut self, scope: &str) -> Result<()> {
        if scope == FILES_SCOPE {
            return Err(Error::Refused(format!(
                "the {FILES_SCOPE} scope is never destroyed"
            )));
        }
        self.scopes
            .remove(scope)
            .map(drop)
            .ok_or_else(|| Error::UnknownKey(format!("{scope}/*")))
    }

    /// Each scope with its active generation and every generation it still holds.
    pub fn scopes(&self) -> Vec<(String, u32, Vec<u32>)> {
        self.scopes
            .iter()
            .map(|(name, scope)| {
                (
                    name.clone(),
                    scope.active,
                    scope.keys.keys().copied().collect(),
                )
            })
            .collect()
    }
}

impl Scope {
    fn fresh() -> Result<Self> {
        let mut keys = BTreeMap::new();
        keys.insert(1, Key::generate()?);
        Ok(Self { active: 1, keys })
    }
}

fn payload_aad(id: &[u8]) -> Vec<u8> {
    [PAYLOAD_LABEL, id].concat()
}

fn slot_aad(
    id: &[u8],
    kind: &str,
    slot_id: &str,
    params: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let params = serde_json::to_vec(params).map_err(json_error)?;
    Ok([
        SLOT_LABEL,
        id,
        b"\0",
        kind.as_bytes(),
        b"\0",
        slot_id.as_bytes(),
        b"\0",
        &params,
    ]
    .concat())
}

/// `base64(nonce || XChaCha20-Poly1305(key, nonce, plain, aad))` with a random 24-byte nonce.
fn seal_blob(key: &Key, plain: &[u8], aad: &[u8]) -> Result<String> {
    let cipher = XChaCha20Poly1305::new_from_slice(key.expose())
        .map_err(|_| Error::Kdf("key has the wrong length".into()))?;
    let nonce: [u8; NONCE_LEN] = random()?;
    let sealed = cipher
        .encrypt(&XNonce::from(nonce), Payload { msg: plain, aad })
        .map_err(|_| Error::Authentication)?;
    Ok(BASE64.encode(&[nonce.as_slice(), &sealed].concat()))
}

fn open_blob(key: &Key, encoded: &str, aad: &[u8]) -> Result<Zeroizing<Vec<u8>>> {
    let bytes = BASE64
        .decode(encoded.as_bytes())
        .map_err(|_| Error::format("keyring", "a sealed field is not base64"))?;
    if bytes.len() < NONCE_LEN {
        return Err(Error::Authentication);
    }
    let (nonce, sealed) = bytes.split_at(NONCE_LEN);
    let nonce: [u8; NONCE_LEN] = nonce.try_into().map_err(|_| Error::Authentication)?;
    let cipher = XChaCha20Poly1305::new_from_slice(key.expose())
        .map_err(|_| Error::Kdf("key has the wrong length".into()))?;
    cipher
        .decrypt(&XNonce::from(nonce), Payload { msg: sealed, aad })
        .map(Zeroizing::new)
        .map_err(|_| Error::Authentication)
}

fn json_error(err: serde_json::Error) -> Error {
    Error::format("keyring", err.to_string())
}

/// Writes `bytes` to `path` through a file beside it that is renamed over it, so a crash leaves
/// the old file or the new one. On Unix the file is made readable by its owner only.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => std::path::PathBuf::from("."),
    };
    fs::create_dir_all(&dir)?;
    let name = path
        .file_name()
        .ok_or_else(|| Error::Refused(format!("{} names no file", path.display())))?;
    let mut temp_name = name.to_os_string();
    temp_name.push(format!(".{}.tmp", HEXLOWER.encode(&random::<6>()?)));
    let temp = dir.join(temp_name);
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let written = (|| -> Result<()> {
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = written {
        let _ = fs::remove_file(&temp);
        return Err(err);
    }
    if let Err(err) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(err.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed;
    use crate::slot::{KeyFile, Passphrase, PassphraseParams, RecoveryKey};

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    /// Small Argon2id parameters, so the tests are quick. Real slots use the defaults.
    fn quick(text: &str) -> Passphrase {
        Passphrase::with_params(
            text,
            PassphraseParams {
                memory_kib: 256,
                passes: 1,
                lanes: 1,
            },
        )
    }

    #[test]
    fn a_key_ref_prints_and_parses_back() -> TestResult {
        let key_ref: KeyRef = "ws:acme/3".parse()?;
        assert_eq!(key_ref.scope(), "ws:acme");
        assert_eq!(key_ref.generation(), 3);
        assert_eq!(key_ref.to_string(), "ws:acme/3");
        assert!("files".parse::<KeyRef>().is_err());
        assert!("files/0".parse::<KeyRef>().is_err());
        assert!("a b/1".parse::<KeyRef>().is_err());
        assert!("/1".parse::<KeyRef>().is_err());
        Ok(())
    }

    #[test]
    fn every_slot_opens_the_same_keys() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let key_file = KeyFile::from_key(Key::generate()?);
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&recovery)?;
        keyring.add_slot(&quick("correct horse battery staple"))?;
        keyring.add_slot(&key_file)?;
        let (key_ref, key) = keyring.active(FILES_SCOPE)?;
        let bytes = keyring.to_bytes()?;

        for opened in [
            Keyring::from_bytes(&bytes, &recovery)?,
            Keyring::from_bytes(&bytes, &quick("correct horse battery staple"))?,
            Keyring::from_bytes(&bytes, &key_file)?,
        ] {
            assert_eq!(opened.id_hex(), keyring.id_hex());
            assert_eq!(opened.key(&key_ref)?, key);
            assert_eq!(opened.slots().len(), 3);
        }
        Ok(())
    }

    #[test]
    fn the_wrong_secret_opens_nothing() -> TestResult {
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&quick("right"))?;
        keyring.add_slot(&RecoveryKey::generate()?)?;
        let bytes = keyring.to_bytes()?;
        assert!(matches!(
            Keyring::from_bytes(&bytes, &quick("wrong")),
            Err(Error::NoSlotOpened { .. })
        ));
        assert!(matches!(
            Keyring::from_bytes(&bytes, &RecoveryKey::generate()?),
            Err(Error::NoSlotOpened { .. })
        ));
        assert!(matches!(
            Keyring::from_bytes(&bytes, &KeyFile::from_key(Key::generate()?)),
            Err(Error::NoSlotOpened { .. })
        ));
        Ok(())
    }

    #[test]
    fn a_changed_payload_or_slot_is_refused() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&recovery)?;
        let bytes = keyring.to_bytes()?;
        let file: serde_json::Value = serde_json::from_slice(&bytes)?;

        let mut changed = file.clone();
        let payload = changed["payload"].as_str().ok_or("no payload")?.to_owned();
        let mut raw = BASE64.decode(payload.as_bytes())?;
        let last = raw.len() - 1;
        raw[last] ^= 1;
        changed["payload"] = BASE64.encode(&raw).into();
        assert!(matches!(
            Keyring::from_bytes(&serde_json::to_vec(&changed)?, &recovery),
            Err(Error::Authentication)
        ));

        let mut moved = file.clone();
        moved["slots"][0]["params"]["note"] = "added".into();
        assert!(matches!(
            Keyring::from_bytes(&serde_json::to_vec(&moved)?, &recovery),
            Err(Error::NoSlotOpened { .. })
        ));

        let mut other_id = file;
        other_id["id"] = HEXLOWER.encode(&[7u8; ID_LEN]).into();
        assert!(matches!(
            Keyring::from_bytes(&serde_json::to_vec(&other_id)?, &recovery),
            Err(Error::NoSlotOpened { .. })
        ));
        Ok(())
    }

    #[test]
    fn slots_come_and_go_but_never_the_last() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let mut keyring = Keyring::create()?;
        assert!(
            keyring.to_bytes().is_err(),
            "a keyring with no slot is never written"
        );
        let first = keyring.add_slot(&recovery)?;
        let second = keyring.add_slot(&quick("pass"))?;
        assert!(keyring.remove_slot(&first).is_ok());
        assert!(matches!(
            keyring.remove_slot(&second),
            Err(Error::Refused(_))
        ));
        let bytes = keyring.to_bytes()?;
        assert!(Keyring::from_bytes(&bytes, &recovery).is_err());
        assert!(Keyring::from_bytes(&bytes, &quick("pass")).is_ok());
        Ok(())
    }

    #[test]
    fn rotation_keeps_old_files_open_until_retired() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&recovery)?;
        let (old_ref, old_key) = keyring.active(FILES_SCOPE)?;
        let sealed_old = sealed::seal(b"old", &old_key, &old_ref, "export")?;

        let new_ref = keyring.rotate(FILES_SCOPE)?;
        assert_eq!(new_ref.generation(), 2);
        assert_eq!(keyring.active(FILES_SCOPE)?.0, new_ref);
        let reopened = Keyring::from_bytes(&keyring.to_bytes()?, &recovery)?;
        let plain = sealed::open(&sealed_old, "export", |r| reopened.key(r))?;
        assert_eq!(plain.as_slice(), b"old");

        assert!(matches!(keyring.retire(&new_ref), Err(Error::Refused(_))));
        keyring.retire(&old_ref)?;
        let retired = Keyring::from_bytes(&keyring.to_bytes()?, &recovery)?;
        assert!(matches!(
            sealed::open(&sealed_old, "export", |r| retired.key(r)),
            Err(Error::UnknownKey(_))
        ));
        Ok(())
    }

    #[test]
    fn destroying_a_workspace_scope_erases_its_files() -> TestResult {
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&RecoveryKey::generate()?)?;
        assert!(keyring.ensure_scope("ws:acme")?);
        assert!(!keyring.ensure_scope("ws:acme")?);
        let (key_ref, key) = keyring.active("ws:acme")?;
        let sealed_file = sealed::seal(b"adapter", &key, &key_ref, "adapter")?;
        keyring.destroy_scope("ws:acme")?;
        assert!(matches!(
            sealed::open(&sealed_file, "adapter", |r| keyring.key(r)),
            Err(Error::UnknownKey(_))
        ));
        assert!(matches!(
            keyring.destroy_scope(FILES_SCOPE),
            Err(Error::Refused(_))
        ));
        Ok(())
    }

    #[test]
    fn save_and_open_round_trip_through_a_file() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&recovery)?;
        let dir = std::env::temp_dir().join(format!(
            "antumbra-keyring-test-{}",
            HEXLOWER.encode(&random::<8>()?)
        ));
        let path = dir.join("keyring.json");
        keyring.save(&path)?;
        keyring.rotate(FILES_SCOPE)?;
        keyring.save(&path)?;
        let opened = Keyring::open(&path, &recovery)?;
        assert_eq!(opened.active(FILES_SCOPE)?.0.generation(), 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o777, 0o600);
        }
        let leftovers = fs::read_dir(&dir)?.count();
        assert_eq!(leftovers, 1, "no temporary file is left behind");
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn debug_shows_no_key() -> TestResult {
        let mut keyring = Keyring::create()?;
        keyring.add_slot(&RecoveryKey::generate()?)?;
        let (_, key) = keyring.active(FILES_SCOPE)?;
        let shown = format!("{keyring:?}");
        assert!(!shown.contains(&BASE64.encode(key.expose())));
        assert!(!shown.contains(&HEXLOWER.encode(key.expose())));
        Ok(())
    }
}

#[cfg(all(test, feature = "os-store"))]
mod os_store_tests {
    use super::*;
    use crate::slot::OsStore;

    /// Writes to the real credential store of the machine running it, under a test service
    /// name, and deletes what it wrote. Run it on purpose:
    /// `cargo test -p antumbra-crypto --features os-store -- --ignored`.
    #[test]
    #[ignore = "writes to this machine's credential store"]
    fn a_keyring_opens_through_the_platform_credential_store() -> Result<()> {
        let store = OsStore::with_service("antumbra-crypto-test");
        let mut keyring = Keyring::create()?;
        let slot = keyring.add_slot(&store)?;
        let (key_ref, key) = keyring.active(FILES_SCOPE)?;
        let opened = Keyring::from_bytes(&keyring.to_bytes()?, &store);
        let forgotten = store.forget(&keyring.id, &slot);
        let opened = opened?;
        forgotten?;
        assert_eq!(opened.key(&key_ref)?, key);
        assert!(
            Keyring::from_bytes(&keyring.to_bytes()?, &store).is_err(),
            "once the secret is forgotten, the slot no longer opens"
        );
        Ok(())
    }
}

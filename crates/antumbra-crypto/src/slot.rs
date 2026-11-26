//! What opens a keyring slot. A [`Protector`] turns something the user or the platform holds
//! into the key that wraps the keyring's master key.
//!
//! | kind | holds | for |
//! |---|---|---|
//! | `recovery` | [`RecoveryKey`], 256 random bits shown as 52 base32 characters | getting the data back without the machine |
//! | `passphrase` | [`Passphrase`], through Argon2id | a machine with no credential store, or a user who wants one |
//! | `key-file` | [`KeyFile`], 32 bytes in a file | a container given its key as a mounted secret |
//! | `os-store` | `OsStore`, a random secret in the platform's credential store | the default on a desktop (feature `os-store`) |

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use data_encoding::{BASE32_NOPAD, BASE64, HEXLOWER_PERMISSIVE};
use zeroize::Zeroizing;

use crate::kdf::{argon2id, subkey};
use crate::key::random;
use crate::{Error, Key, Result, KEY_LEN};

/// Something that opens a kind of keyring slot.
pub trait Protector {
    /// The slot kind this protector makes and opens.
    fn kind(&self) -> &'static str;

    /// For a new slot `slot_id` of keyring `keyring_id`: the key that wraps the master key, and
    /// the parameters to store with the slot (never secret).
    fn enroll(&self, keyring_id: &[u8], slot_id: &str) -> Result<(Key, BTreeMap<String, String>)>;

    /// The wrapping key of an existing slot, from its stored parameters.
    fn unlock(
        &self,
        keyring_id: &[u8],
        slot_id: &str,
        params: &BTreeMap<String, String>,
    ) -> Result<Key>;
}

fn slot_label(kind: &str) -> Vec<u8> {
    [
        b"antumbra-keyring/v1/slot-key/".as_slice(),
        kind.as_bytes(),
        b"\0",
    ]
    .concat()
}

/// The key a user keeps to open their data without the machine that holds the usual slot: 256
/// random bits, shown once as 52 base32 characters in groups of four.
#[derive(Clone)]
pub struct RecoveryKey(Key);

impl RecoveryKey {
    pub fn generate() -> Result<Self> {
        Ok(Self(Key::generate()?))
    }

    /// Reads a recovery key as a user types it back: case, spaces and dashes do not matter.
    pub fn parse(text: &str) -> Result<Self> {
        let cleaned: Zeroizing<String> = Zeroizing::new(
            text.chars()
                .filter(|c| !c.is_whitespace() && *c != '-')
                .map(|c| c.to_ascii_uppercase())
                .collect(),
        );
        let bytes = Zeroizing::new(
            BASE32_NOPAD
                .decode(cleaned.as_bytes())
                .map_err(|_| Error::format("recovery key", "it is not base32"))?,
        );
        Ok(Self(Key::from_slice(&bytes)?))
    }

    /// The key as the user is shown it: `ABCD-EFGH-...`, 13 groups.
    pub fn display(&self) -> Zeroizing<String> {
        let plain = Zeroizing::new(BASE32_NOPAD.encode(self.0.expose()));
        let groups: Vec<&str> = plain
            .as_bytes()
            .chunks(4)
            .filter_map(|group| std::str::from_utf8(group).ok())
            .collect();
        Zeroizing::new(groups.join("-"))
    }
}

impl fmt::Debug for RecoveryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RecoveryKey(<redacted>)")
    }
}

impl Protector for RecoveryKey {
    fn kind(&self) -> &'static str {
        "recovery"
    }

    fn enroll(&self, keyring_id: &[u8], slot_id: &str) -> Result<(Key, BTreeMap<String, String>)> {
        Ok((
            self.unlock(keyring_id, slot_id, &BTreeMap::new())?,
            BTreeMap::new(),
        ))
    }

    fn unlock(
        &self,
        keyring_id: &[u8],
        slot_id: &str,
        _: &BTreeMap<String, String>,
    ) -> Result<Key> {
        subkey(
            &self.0,
            keyring_id,
            &[&slot_label(self.kind()), slot_id.as_bytes()],
        )
    }
}

/// Argon2id's cost: memory in KiB, passes, and lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassphraseParams {
    pub memory_kib: u32,
    pub passes: u32,
    pub lanes: u32,
}

impl Default for PassphraseParams {
    /// RFC 9106's second recommended option: 64 MiB, 3 passes, 4 lanes.
    fn default() -> Self {
        Self {
            memory_kib: 64 * 1024,
            passes: 3,
            lanes: 4,
        }
    }
}

/// The most a stored slot may ask of Argon2id, so a crafted keyring cannot make opening it take
/// all the memory there is.
const MAX_MEMORY_KIB: u32 = 4 * 1024 * 1024;
const MAX_PASSES: u32 = 64;
const MAX_LANES: u32 = 64;
const SALT_LEN: usize = 16;

/// A passphrase, stretched with Argon2id.
pub struct Passphrase {
    text: Zeroizing<String>,
    params: PassphraseParams,
}

impl Passphrase {
    /// A passphrase with the default cost for new slots.
    pub fn new(text: &str) -> Self {
        Self::with_params(text, PassphraseParams::default())
    }

    /// A passphrase with a chosen cost for new slots. Opening a slot always uses the cost stored
    /// with it.
    pub fn with_params(text: &str, params: PassphraseParams) -> Self {
        Self {
            text: Zeroizing::new(text.to_owned()),
            params,
        }
    }
}

impl fmt::Debug for Passphrase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Passphrase")
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

fn param(params: &BTreeMap<String, String>, name: &str, max: u32) -> Result<u32> {
    let value: u32 = params
        .get(name)
        .ok_or_else(|| Error::format("passphrase slot", format!("no {name}")))?
        .parse()
        .map_err(|_| Error::format("passphrase slot", format!("{name} is not a number")))?;
    if value == 0 || value > max {
        return Err(Error::format(
            "passphrase slot",
            format!("{name} {value} is out of range"),
        ));
    }
    Ok(value)
}

impl Protector for Passphrase {
    fn kind(&self) -> &'static str {
        "passphrase"
    }

    fn enroll(&self, keyring_id: &[u8], slot_id: &str) -> Result<(Key, BTreeMap<String, String>)> {
        let salt: [u8; SALT_LEN] = random()?;
        let mut params = BTreeMap::new();
        params.insert("salt".to_owned(), BASE64.encode(&salt));
        params.insert("memory_kib".to_owned(), self.params.memory_kib.to_string());
        params.insert("passes".to_owned(), self.params.passes.to_string());
        params.insert("lanes".to_owned(), self.params.lanes.to_string());
        let key = self.unlock(keyring_id, slot_id, &params)?;
        Ok((key, params))
    }

    fn unlock(
        &self,
        keyring_id: &[u8],
        slot_id: &str,
        params: &BTreeMap<String, String>,
    ) -> Result<Key> {
        let salt = BASE64
            .decode(
                params
                    .get("salt")
                    .ok_or_else(|| Error::format("passphrase slot", "no salt"))?
                    .as_bytes(),
            )
            .map_err(|_| Error::format("passphrase slot", "its salt is not base64"))?;
        let stretched = argon2id(
            self.text.as_bytes(),
            &salt,
            param(params, "memory_kib", MAX_MEMORY_KIB)?,
            param(params, "passes", MAX_PASSES)?,
            param(params, "lanes", MAX_LANES)?,
        )?;
        subkey(
            &stretched,
            keyring_id,
            &[&slot_label(self.kind()), slot_id.as_bytes()],
        )
    }
}

/// A 32-byte key held in a file, as a container is given a secret. The file holds the key as
/// 32 raw bytes, 64 hex characters or base64; surrounding whitespace is ignored.
pub struct KeyFile(Key);

impl KeyFile {
    pub fn from_key(key: Key) -> Self {
        Self(key)
    }

    pub fn read(path: &Path) -> Result<Self> {
        let bytes = Zeroizing::new(fs::read(path)?);
        if bytes.len() == KEY_LEN {
            return Ok(Self(Key::from_slice(&bytes)?));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| Error::format("key file", "it is neither 32 bytes nor text"))?
            .trim();
        let decoded = HEXLOWER_PERMISSIVE
            .decode(text.as_bytes())
            .or_else(|_| BASE64.decode(text.as_bytes()))
            .map_err(|_| Error::format("key file", "it is neither hex nor base64"))?;
        let decoded = Zeroizing::new(decoded);
        Ok(Self(Key::from_slice(&decoded)?))
    }

    /// Makes a new key and writes it to `path` as 64 hex characters, readable by its owner only.
    pub fn generate_to(path: &Path) -> Result<Self> {
        let key = Key::generate()?;
        let mut text = Zeroizing::new(data_encoding::HEXLOWER.encode(key.expose()));
        text.push('\n');
        crate::keyring::write_atomically(path, text.as_bytes())?;
        Ok(Self(key))
    }
}

impl fmt::Debug for KeyFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("KeyFile(<redacted>)")
    }
}

impl Protector for KeyFile {
    fn kind(&self) -> &'static str {
        "key-file"
    }

    fn enroll(&self, keyring_id: &[u8], slot_id: &str) -> Result<(Key, BTreeMap<String, String>)> {
        Ok((
            self.unlock(keyring_id, slot_id, &BTreeMap::new())?,
            BTreeMap::new(),
        ))
    }

    fn unlock(
        &self,
        keyring_id: &[u8],
        slot_id: &str,
        _: &BTreeMap<String, String>,
    ) -> Result<Key> {
        subkey(
            &self.0,
            keyring_id,
            &[&slot_label(self.kind()), slot_id.as_bytes()],
        )
    }
}

/// A random secret in the platform's credential store: Windows Credential Manager, the macOS
/// Keychain, Secret Service on Linux. The store releases it to the signed-in user, so opening
/// the keyring asks nothing of them.
#[cfg(feature = "os-store")]
#[derive(Debug, Clone)]
pub struct OsStore {
    service: String,
}

#[cfg(feature = "os-store")]
impl Default for OsStore {
    fn default() -> Self {
        Self {
            service: "antumbra".to_owned(),
        }
    }
}

#[cfg(feature = "os-store")]
impl OsStore {
    /// A store whose entries are filed under `service` (default `antumbra`).
    pub fn with_service(service: &str) -> Self {
        Self {
            service: service.to_owned(),
        }
    }

    fn account(keyring_id: &[u8], slot_id: &str) -> String {
        format!(
            "keyring-{}-{slot_id}",
            data_encoding::HEXLOWER.encode(keyring_id)
        )
    }

    fn entry(service: &str, account: &str) -> Result<keyring::Entry> {
        keyring::Entry::new(service, account).map_err(|err| Error::CredentialStore(err.to_string()))
    }

    /// Deletes the secret of a slot that was removed from its keyring.
    pub fn forget(&self, keyring_id: &[u8], slot_id: &str) -> Result<()> {
        Self::entry(&self.service, &Self::account(keyring_id, slot_id))?
            .delete_credential()
            .map_err(|err| Error::CredentialStore(err.to_string()))
    }
}

#[cfg(feature = "os-store")]
impl Protector for OsStore {
    fn kind(&self) -> &'static str {
        "os-store"
    }

    fn enroll(&self, keyring_id: &[u8], slot_id: &str) -> Result<(Key, BTreeMap<String, String>)> {
        let secret = Key::generate()?;
        let account = Self::account(keyring_id, slot_id);
        Self::entry(&self.service, &account)?
            .set_secret(secret.expose())
            .map_err(|err| Error::CredentialStore(err.to_string()))?;
        let mut params = BTreeMap::new();
        params.insert("service".to_owned(), self.service.clone());
        params.insert("account".to_owned(), account);
        let key = subkey(
            &secret,
            keyring_id,
            &[&slot_label(self.kind()), slot_id.as_bytes()],
        )?;
        Ok((key, params))
    }

    fn unlock(
        &self,
        keyring_id: &[u8],
        slot_id: &str,
        params: &BTreeMap<String, String>,
    ) -> Result<Key> {
        let service = params
            .get("service")
            .map_or(self.service.as_str(), String::as_str);
        let account = params
            .get("account")
            .cloned()
            .unwrap_or_else(|| Self::account(keyring_id, slot_id));
        let secret = Zeroizing::new(
            Self::entry(service, &account)?
                .get_secret()
                .map_err(|err| Error::CredentialStore(err.to_string()))?,
        );
        let secret = Key::from_slice(&secret)?;
        subkey(
            &secret,
            keyring_id,
            &[&slot_label(self.kind()), slot_id.as_bytes()],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn a_recovery_key_reads_back_however_it_is_typed() -> TestResult {
        let key = RecoveryKey::generate()?;
        let shown = key.display();
        assert_eq!(shown.len(), 52 + 12, "52 characters in 13 groups");
        assert_eq!(shown.split('-').count(), 13);
        let typed = shown.to_lowercase().replace('-', " ");
        let back = RecoveryKey::parse(&typed)?;
        assert_eq!(back.0, key.0);
        assert!(RecoveryKey::parse("not a key").is_err());
        assert!(RecoveryKey::parse(&shown[..40]).is_err());
        Ok(())
    }

    #[test]
    fn protectors_never_print_their_secret() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        assert_eq!(format!("{recovery:?}"), "RecoveryKey(<redacted>)");
        let passphrase = Passphrase::new("hunter2-and-more");
        assert!(!format!("{passphrase:?}").contains("hunter2"));
        assert_eq!(
            format!("{:?}", KeyFile::from_key(Key::generate()?)),
            "KeyFile(<redacted>)"
        );
        Ok(())
    }

    #[test]
    fn the_slot_key_depends_on_the_keyring_and_the_slot() -> TestResult {
        let recovery = RecoveryKey::generate()?;
        let none = BTreeMap::new();
        let a = recovery.unlock(b"keyring-a", "slot1", &none)?;
        assert_eq!(a, recovery.unlock(b"keyring-a", "slot1", &none)?);
        assert_ne!(a, recovery.unlock(b"keyring-b", "slot1", &none)?);
        assert_ne!(a, recovery.unlock(b"keyring-a", "slot2", &none)?);
        Ok(())
    }

    #[test]
    fn a_passphrase_slot_refuses_costs_out_of_range() -> TestResult {
        let passphrase = Passphrase::with_params(
            "pass",
            PassphraseParams {
                memory_kib: 256,
                passes: 1,
                lanes: 1,
            },
        );
        let (key, mut params) = passphrase.enroll(b"keyring", "slot")?;
        assert_eq!(key, passphrase.unlock(b"keyring", "slot", &params)?);
        params.insert("memory_kib".into(), (MAX_MEMORY_KIB + 1).to_string());
        assert!(passphrase.unlock(b"keyring", "slot", &params).is_err());
        params.insert("memory_kib".into(), "0".into());
        assert!(passphrase.unlock(b"keyring", "slot", &params).is_err());
        Ok(())
    }

    #[test]
    fn a_key_file_reads_raw_hex_and_base64() -> TestResult {
        let dir = std::env::temp_dir().join(format!(
            "antumbra-keyfile-test-{}",
            data_encoding::HEXLOWER.encode(&random::<8>()?)
        ));
        fs::create_dir_all(&dir)?;
        let made = KeyFile::generate_to(&dir.join("generated.key"))?;
        let raw = [5u8; KEY_LEN];
        fs::write(dir.join("raw.key"), raw)?;
        fs::write(
            dir.join("hex.key"),
            format!("  {}\n", data_encoding::HEXLOWER.encode(&raw)),
        )?;
        fs::write(dir.join("b64.key"), format!("{}\n", BASE64.encode(&raw)))?;
        fs::write(dir.join("short.key"), "abcd")?;

        assert_eq!(KeyFile::read(&dir.join("generated.key"))?.0, made.0);
        let expected = Key::from_slice(&raw)?;
        for name in ["raw.key", "hex.key", "b64.key"] {
            assert_eq!(KeyFile::read(&dir.join(name))?.0, expected, "{name}");
        }
        assert!(KeyFile::read(&dir.join("short.key")).is_err());
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}

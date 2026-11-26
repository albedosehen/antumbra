//! Encryption at rest for antumbra, as `docs/encryption.md` designs it.
//!
//! - [`Keyring`]: antumbra's keys, sealed in one file under a random master key that one or
//!   more slots open (the platform credential store, a recovery key, a passphrase, a key file).
//! - [`sealed`]: the format of the files antumbra writes outside its database, adapters,
//!   corpora and exports among them: the STREAM construction over XChaCha20-Poly1305, so that a
//!   changed, truncated, reordered or extended file is refused.
//! - [`RecoveryKey`]: the key a user keeps to get their data back without the machine.
//!
//! Every key is random, lives in memory that is wiped when it is dropped, and prints as
//! `<redacted>`.

mod error;
mod kdf;
mod key;
pub mod keyring;
pub mod sealed;
pub mod slot;

pub use error::{Error, Result};
pub use key::{Key, KEY_LEN};
pub use keyring::{KeyRef, Keyring, SlotInfo};
pub use slot::{KeyFile, Passphrase, PassphraseParams, Protector, RecoveryKey};

#[cfg(feature = "os-store")]
pub use slot::OsStore;

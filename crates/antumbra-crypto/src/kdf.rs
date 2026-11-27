//! Key derivation: HKDF-SHA-256 for subkeys of random keys, Argon2id for passphrases.

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

use crate::{Error, Key, Result, KEY_LEN};

/// HKDF-SHA-256 of `ikm` with `salt`, expanded under the concatenation of `info` into `out`.
/// Every caller passes a distinct label first in `info`, so no two uses share a derived key.
pub(crate) fn hkdf(ikm: &[u8], salt: &[u8], info: &[&[u8]], out: &mut [u8]) -> Result<()> {
    let joined: Vec<u8> = info.concat();
    Hkdf::<Sha256>::new(Some(salt), ikm)
        .expand(&joined, out)
        .map_err(|err| Error::Kdf(err.to_string()))
}

/// A 256-bit subkey of `ikm`.
pub(crate) fn subkey(ikm: &Key, salt: &[u8], info: &[&[u8]]) -> Result<Key> {
    let mut out = Zeroizing::new([0u8; KEY_LEN]);
    hkdf(ikm.expose(), salt, info, out.as_mut())?;
    Key::from_slice(out.as_ref())
}

/// Argon2id of a passphrase into a 256-bit key.
pub(crate) fn argon2id(
    passphrase: &[u8],
    salt: &[u8],
    memory_kib: u32,
    passes: u32,
    lanes: u32,
) -> Result<Key> {
    let params = argon2::Params::new(memory_kib, passes, lanes, Some(KEY_LEN))
        .map_err(|err| Error::Kdf(err.to_string()))?;
    let argon = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut out = Zeroizing::new([0u8; KEY_LEN]);
    argon
        .hash_password_into(passphrase, salt, out.as_mut())
        .map_err(|err| Error::Kdf(err.to_string()))?;
    Key::from_slice(out.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_label_separates_subkeys() -> Result<()> {
        let root = Key::from_slice(&[7; KEY_LEN])?;
        let a = subkey(&root, b"salt", &[b"antumbra/a"])?;
        let b = subkey(&root, b"salt", &[b"antumbra/b"])?;
        assert_ne!(a, b);
        assert_eq!(a, subkey(&root, b"salt", &[b"antumbra/a"])?);
        Ok(())
    }

    #[test]
    fn argon2id_depends_on_the_passphrase_and_the_salt() -> Result<()> {
        let a = argon2id(b"correct horse", b"0123456789abcdef", 256, 1, 1)?;
        assert_eq!(
            a,
            argon2id(b"correct horse", b"0123456789abcdef", 256, 1, 1)?
        );
        assert_ne!(
            a,
            argon2id(b"correct horsf", b"0123456789abcdef", 256, 1, 1)?
        );
        assert_ne!(
            a,
            argon2id(b"correct horse", b"0123456789abcdeg", 256, 1, 1)?
        );
        Ok(())
    }
}

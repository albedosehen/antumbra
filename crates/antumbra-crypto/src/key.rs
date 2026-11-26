use std::fmt;

use zeroize::Zeroizing;

use crate::{Error, Result};

/// The length of every key antumbra makes: 256 bits.
pub const KEY_LEN: usize = 32;

/// A 256-bit key. The bytes are wiped when the last copy is dropped, and `Debug` never shows
/// them.
#[derive(Clone)]
pub struct Key(Zeroizing<[u8; KEY_LEN]>);

impl Key {
    /// A fresh key from the operating system's random number generator.
    pub fn generate() -> Result<Self> {
        let mut bytes = Zeroizing::new([0u8; KEY_LEN]);
        fill_random(bytes.as_mut())?;
        Ok(Self(bytes))
    }

    /// A key from exactly [`KEY_LEN`] bytes.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        if bytes.len() != KEY_LEN {
            return Err(Error::format(
                "key",
                format!("{} bytes, not {KEY_LEN}", bytes.len()),
            ));
        }
        key.copy_from_slice(bytes);
        Ok(Self(key))
    }

    /// The key's bytes. Copy them only into memory that is wiped too.
    pub fn expose(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Key(<redacted>)")
    }
}

impl PartialEq for Key {
    /// Constant-time, so a comparison never tells how many bytes matched.
    fn eq(&self, other: &Self) -> bool {
        let mut diff = 0u8;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

impl Eq for Key {}

/// Fill `out` from the operating system's random number generator.
pub(crate) fn fill_random(out: &mut [u8]) -> Result<()> {
    getrandom::fill(out).map_err(|err| Error::Random(err.to_string()))
}

/// `N` random bytes.
pub(crate) fn random<const N: usize>() -> Result<[u8; N]> {
    let mut out = [0u8; N];
    fill_random(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_never_shows_the_bytes() -> Result<()> {
        let key = Key::from_slice(&[0xAB; KEY_LEN])?;
        let shown = format!("{key:?}");
        assert_eq!(shown, "Key(<redacted>)");
        assert!(!shown.to_lowercase().contains("ab"));
        Ok(())
    }

    #[test]
    fn generated_keys_differ() -> Result<()> {
        assert_ne!(Key::generate()?, Key::generate()?);
        Ok(())
    }

    #[test]
    fn a_key_must_be_32_bytes() {
        assert!(Key::from_slice(&[0; 31]).is_err());
        assert!(Key::from_slice(&[0; 33]).is_err());
    }
}

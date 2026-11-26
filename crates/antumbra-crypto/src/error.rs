use std::io;

/// What can go wrong. A failed authentication tag is one coarse variant on purpose: whether a
/// file was changed, truncated, reordered or sealed under another key is not told apart.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o: {0}")]
    Io(io::Error),

    /// The bytes are not in the expected format: a wrong magic, an unknown version, a field
    /// out of range.
    #[error("not a valid {what}: {detail}")]
    Format { what: &'static str, detail: String },

    /// An authentication tag did not verify: the data was changed, truncated, reordered or
    /// extended, or the key is not the one it was sealed with.
    #[error(
        "authentication failed: the data was changed, or the key is not the one it was sealed with"
    )]
    Authentication,

    /// A sealed file was opened for a purpose other than the one it was sealed for.
    #[error("sealed for {found:?}, opened as {expected:?}")]
    Purpose { expected: String, found: String },

    /// No slot of the keyring opened with what was offered.
    #[error("no {kind} slot of this keyring opened")]
    NoSlotOpened { kind: String },

    /// The keyring has no key under this reference: never made, or destroyed.
    #[error("no key {0}")]
    UnknownKey(String),

    /// A request the keyring refuses, such as removing its last slot.
    #[error("{0}")]
    Refused(String),

    #[error("the operating system's random number generator failed: {0}")]
    Random(String),

    #[error("key derivation failed: {0}")]
    Kdf(String),

    #[error("credential store: {0}")]
    CredentialStore(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn format(what: &'static str, detail: impl Into<String>) -> Self {
        Self::Format {
            what,
            detail: detail.into(),
        }
    }
}

impl From<io::Error> for Error {
    /// An error of this crate that went through `std::io` (a [`crate::sealed::OpenReader`]
    /// answers its reads with them) comes back as itself, not as an i/o error.
    fn from(err: io::Error) -> Self {
        if !err.get_ref().is_some_and(|inner| inner.is::<Error>()) {
            return Self::Io(err);
        }
        match err.into_inner().map(|inner| inner.downcast::<Error>()) {
            Some(Ok(ours)) => *ours,
            _ => Self::Io(io::Error::other(
                "an error was lost on its way through std::io",
            )),
        }
    }
}

impl From<Error> for io::Error {
    fn from(err: Error) -> Self {
        match err {
            Error::Io(io) => io,
            other => io::Error::new(io::ErrorKind::InvalidData, other),
        }
    }
}

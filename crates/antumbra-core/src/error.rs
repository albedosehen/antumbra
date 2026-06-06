//! Crate-wide error type. Kept storage-agnostic: the store layer maps driver
//! errors into [`AntumbraError::Store`] so the domain core never depends on a
//! database crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AntumbraError {
    /// A lifecycle entity was asked to make a transition its state machine
    /// forbids (e.g. a pruned shadow graduating).
    #[error("invalid {entity} transition: {from} -> {to}")]
    InvalidTransition {
        entity: &'static str,
        from: String,
        to: String,
    },

    /// A capability that exists as a seam but is not built in this revision
    /// (e.g. real candle QLoRA training, llama.cpp serving).
    #[error("not yet implemented: {0}")]
    Unimplemented(&'static str),

    /// Anything raised by the persistence layer, flattened to a string so the
    /// core stays free of a database dependency.
    #[error("store: {0}")]
    Store(String),

    /// A vector argument did not match the configured embedding dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dimension { expected: usize, got: usize },

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

impl AntumbraError {
    pub fn store(msg: impl Into<String>) -> Self {
        Self::Store(msg.into())
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

pub type Result<T> = core::result::Result<T, AntumbraError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_render_their_message() {
        assert_eq!(AntumbraError::store("boom").to_string(), "store: boom");
        assert_eq!(AntumbraError::other("nope").to_string(), "nope");
        assert_eq!(
            AntumbraError::Unimplemented("serving").to_string(),
            "not yet implemented: serving"
        );
        assert_eq!(
            AntumbraError::InvalidTransition {
                entity: "shadow",
                from: "pruned".into(),
                to: "graduated".into(),
            }
            .to_string(),
            "invalid shadow transition: pruned -> graduated"
        );
        assert_eq!(
            AntumbraError::Dimension {
                expected: 384,
                got: 2
            }
            .to_string(),
            "dimension mismatch: expected 384, got 2"
        );
    }

    #[test]
    fn serde_errors_convert_via_from() {
        let err: AntumbraError = serde_json::from_str::<i32>("not json").unwrap_err().into();
        assert!(err.to_string().starts_with("serde:"));
    }
}

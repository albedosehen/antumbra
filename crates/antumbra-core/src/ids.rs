//! Newtype identifiers and the generation counter.
//!
//! Ids are strings (SurrealDB record ids serialize cleanly as such) — this also
//! sidesteps the documented u64 round-trip loss in the SurrealDB serde codec by
//! never modelling an identity as a large native integer.

use serde::{Deserialize, Serialize};

macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

string_id!(
    /// A frozen expert (in v0, a LoRA adapter over the shared base). ADR-0001.
    ExpertId
);
string_id!(
    /// A short-lived trainable shadow (the penumbra). ADR-0002.
    ShadowId
);
string_id!(
    /// A learned counterfactual boundary (the antumbra / keystone). ADR-0004.
    BoundaryId
);
string_id!(
    /// A durable orchestration or generational run. ADR-0005 / ADR-0008.
    RunId
);

/// Monotonic generation counter for the durable loop (ADR-0008).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Generation(pub u32);

impl Generation {
    pub const ZERO: Generation = Generation(0);

    pub fn next(self) -> Self {
        Generation(self.0 + 1)
    }
}

impl core::fmt::Display for Generation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "g{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_roundtrip_as_transparent_strings() {
        let id = ExpertId::new("expert:deno-conv");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"expert:deno-conv\"");
        let back: ExpertId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn generation_advances() {
        assert_eq!(Generation::ZERO.next(), Generation(1));
        assert!(Generation(2) > Generation(1));
    }
}

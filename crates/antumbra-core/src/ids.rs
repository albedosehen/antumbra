//! Newtype identifiers and the generation counter.
//!
//! Ids are strings (SurrealDB record ids serialize cleanly as such); this also
//! sidesteps the documented u64 round-trip loss in the SurrealDB serde codec by
//! never modeling an identity as a large native integer.

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
    /// shared base
    ExpertId
);
string_id!(
    /// trainable shadow (penumbra).
    ShadowId
);
string_id!(
    /// learned counterfactual boundary (antumbra).
    BoundaryId
);
string_id!(
    /// durable orchestration or generational run.
    RunId
);
string_id!(
    /// Memory trace
    MemoryId
);
string_id!(
    /// One knowledge document embedded chunk, distinct from an episodic [`MemoryId`].
    DocumentChunkId
);
string_id!(
    /// Engine-enforced via `PERMISSIONS ... WHERE tenant_id =
    /// $auth.tenant`
    TenantId
);
string_id!(
    /// User (person, team, or agent) identifier
    UserId
);

/// The user the GitHub App acts as in a workspace: a system identity, the
/// same in every workspace the App is installed in. It writes memories on the
/// workspace's behalf and owns nothing private.
pub const GITHUB_SYSTEM_USER: &str = "user:github";

/// The user ids that may belong to more than one workspace. Every other user
/// id names one user in one workspace (`principal::provision` refuses a
/// second), so a rule that matches on the user alone, as the expert table's
/// read rule does, never matches another workspace's.
pub const SYSTEM_USERS: &[&str] = &[GITHUB_SYSTEM_USER];

impl UserId {
    /// Whether this is a system identity that may belong to many workspaces.
    pub fn is_system(&self) -> bool {
        SYSTEM_USERS.contains(&self.as_str())
    }
}
string_id!(
    /// Named latent-space of memories: the unit of
    /// organization, sharing, deletion, and reference-scope within a tenant.
    CompartmentId
);
string_id!(
    /// A verifier, addressed by the hash of what it checks with.
    VerifierId
);

/// Monotonic generation counter for the durable loop.
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
    fn only_a_system_identity_may_belong_to_many_workspaces() {
        assert!(UserId::new(GITHUB_SYSTEM_USER).is_system());
        assert!(!UserId::new("user:owner").is_system());
        assert!(!UserId::new("user:github-fan").is_system());
    }

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

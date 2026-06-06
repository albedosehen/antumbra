//! Compartments — named "latent-spaces" of memory within a tenant.
//!
//! A compartment is the unit of organization, sharing, deletion, and
//! reference-scope. It is owned by a [`UserId`] and lives inside a [`TenantId`]
//! (the hard isolation boundary). Compartments come into being two ways: a user
//! creates one explicitly, or the antumbra *proposes* one by clustering the
//! penumbra into a competence-coherent region (the user then keeps / names /
//! shares it). A mature compartment is the natural training unit — it
//! consolidates into a private expert.
//!
//! Sharing is a capability **grant** between users *within the same tenant*
//! (intra-tenant, user-to-user). The grant graph is what the engine-enforced
//! compartment `PERMISSIONS` evaluate.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{CompartmentId, TenantId, UserId};

/// What a grantee may do with a shared compartment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Capability {
    /// Recall / read the compartment's memories.
    Reference,
    /// Create graph edges *into* the compartment's memories (implies
    /// `Reference`).
    Link,
}

impl Capability {
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Reference => "reference",
            Capability::Link => "link",
        }
    }

    /// Whether this capability permits recall (both do).
    pub fn allows_reference(self) -> bool {
        matches!(self, Capability::Reference | Capability::Link)
    }

    /// Whether this capability permits linking (only `Link`).
    pub fn allows_link(self) -> bool {
        matches!(self, Capability::Link)
    }
}

/// How a compartment came to exist — for surfacing the antumbra's proposals
/// distinctly from user-created spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    /// Created explicitly by a user.
    User,
    /// Proposed by the antumbra (competence clustering); awaits curation.
    Proposed,
}

impl Origin {
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::User => "user",
            Origin::Proposed => "proposed",
        }
    }
}

/// A named latent-space of memory owned by a user within a tenant.
#[derive(Debug, Clone, PartialEq)]
pub struct Compartment {
    pub id: CompartmentId,
    pub tenant: TenantId,
    pub owner: UserId,
    pub name: String,
    pub origin: Origin,
    pub created_at: DateTime<Utc>,
    /// Bumped on any change (notably deletion), so a delete is the newest version
    /// and wins under last-write-wins sync.
    pub updated_at: DateTime<Utc>,
    /// When set, the compartment is **deleted** (a tombstone): read paths hide it
    /// and the engine ACL stops treating the owner as owning it (so its memories
    /// become invisible), and the deletion propagates across the fleet instead of
    /// resurrecting from another replica that still has the live row.
    pub deleted_at: Option<DateTime<Utc>>,
}

impl Compartment {
    pub fn new(
        id: impl Into<CompartmentId>,
        tenant: impl Into<TenantId>,
        owner: impl Into<UserId>,
        name: impl Into<String>,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            id: id.into(),
            tenant: tenant.into(),
            owner: owner.into(),
            name: name.into(),
            origin: Origin::User,
            created_at: now,
            updated_at: now,
            deleted_at: None,
        }
    }

    /// Mark this compartment as an antumbra proposal (awaiting curation).
    pub fn proposed(mut self) -> Self {
        self.origin = Origin::Proposed;
        self
    }

    /// Delete the compartment as a tombstone at `now` (stamping `updated_at`), so
    /// the deletion is the newest version and propagates rather than resurrecting.
    pub fn soft_delete(&mut self, now: DateTime<Utc>) {
        self.deleted_at = Some(now);
        self.updated_at = now;
    }

    /// `true` if this compartment is a tombstone (deleted).
    pub fn is_deleted(&self) -> bool {
        self.deleted_at.is_some()
    }
}

/// A capability grant: `compartment` is shared to `grantee` with `capability`,
/// recorded by `granted_by`. Intra-tenant (the tenant is the hard boundary), so
/// grantee and owner are members of the same `tenant`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grant {
    pub tenant: TenantId,
    pub compartment: CompartmentId,
    pub grantee: UserId,
    pub capability: Capability,
    pub granted_by: UserId,
    pub created_at: DateTime<Utc>,
    /// Bumped whenever the grant changes (notably on revoke), so a revocation is
    /// the grant's newest version and wins under last-write-wins sync — a stale
    /// live copy on another device cannot out-rank it.
    pub updated_at: DateTime<Utc>,
    /// When set, the grant is **revoked** (a tombstone): the engine ACL excludes
    /// it (access ends at once, locally), and the revocation propagates across the
    /// fleet rather than leaving a stale grant that keeps the grantee in.
    pub deleted_at: Option<DateTime<Utc>>,
}

impl Grant {
    pub fn new(
        tenant: impl Into<TenantId>,
        compartment: impl Into<CompartmentId>,
        grantee: impl Into<UserId>,
        capability: Capability,
        granted_by: impl Into<UserId>,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            tenant: tenant.into(),
            compartment: compartment.into(),
            grantee: grantee.into(),
            capability,
            granted_by: granted_by.into(),
            created_at: now,
            updated_at: now,
            deleted_at: None,
        }
    }

    /// Revoke the grant as a tombstone at `now` (stamping `updated_at`, so the
    /// revocation is the newest version). Access ends immediately where the engine
    /// ACL runs, and the revocation propagates.
    pub fn revoke(&mut self, now: DateTime<Utc>) {
        self.deleted_at = Some(now);
        self.updated_at = now;
    }

    /// `true` if this grant has been revoked (a tombstone).
    pub fn is_revoked(&self) -> bool {
        self.deleted_at.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_implications() {
        assert!(Capability::Reference.allows_reference());
        assert!(!Capability::Reference.allows_link());
        assert!(Capability::Link.allows_reference());
        assert!(Capability::Link.allows_link());
    }

    #[test]
    fn capability_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&Capability::Link).unwrap(),
            "\"link\""
        );
    }

    #[test]
    fn compartment_origin_defaults_user_and_can_be_proposed() {
        let c = Compartment::new("c:1", "ws:1", "user:a", "deno work", Utc::now());
        assert_eq!(c.origin, Origin::User);
        assert_eq!(c.proposed().origin, Origin::Proposed);
    }

    #[test]
    fn grant_round_trips_fields() {
        let g = Grant::new(
            "ws:1",
            "c:1",
            "user:b",
            Capability::Reference,
            "user:a",
            Utc::now(),
        );
        assert_eq!(g.grantee, UserId::new("user:b"));
        assert_eq!(g.granted_by, UserId::new("user:a"));
        assert!(g.capability.allows_reference());
    }
}

//! The tenant hive: a third brain the org shares, assembled from
//! what members choose to offer and the owner chooses to accept.
//!
//! Two gates, and neither alone opens it. The owner enables the hive for the
//! tenant; each member opts their own nodes in. So an owner cannot conscript a
//! member's memory, and a member cannot publish into a hive the owner has not
//! opened. That symmetry is the point: the alternative designs are an owner who
//! can take, or a member who can push into an org that never asked.
//!
//! Contribution is the member's and curation is the owner's. A member offers a
//! unit; the owner sees the whole offered set and narrows it. The active hive is
//! exactly what the owner has accepted -- never what was merely offered.
//!
//! What the hive is not: authority. It pools knowledge and confers no power over
//! another user's agents. The owner's right to read across users exists for
//! curation and training; curating a hive is not directing anyone's agents.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{TenantId, UserId};

/// What a member can offer into the hive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OfferedKind {
    /// A compartment, and so the memories in it.
    Compartment,
    /// A knowledge document.
    Document,
    /// A consolidated expert, which joins the shared umbra.
    Expert,
}

impl OfferedKind {
    pub fn as_str(self) -> &'static str {
        match self {
            OfferedKind::Compartment => "compartment",
            OfferedKind::Document => "document",
            OfferedKind::Expert => "expert",
        }
    }
}

impl std::str::FromStr for OfferedKind {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "compartment" => Ok(OfferedKind::Compartment),
            "document" => Ok(OfferedKind::Document),
            "expert" => Ok(OfferedKind::Expert),
            other => Err(crate::AntumbraError::other(format!(
                "unknown offered kind `{other}` (use compartment | document | expert)"
            ))),
        }
    }
}

/// Where an offer stands. Only `Accepted` is in the hive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OfferStatus {
    /// The member has offered it and the owner has not ruled.
    Offered,
    /// The owner accepted. This, and only this, is the active hive.
    Accepted,
    /// The owner looked and declined. Kept rather than deleted so a member can
    /// see that their offer was considered, which an absent row cannot say.
    Declined,
}

impl OfferStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            OfferStatus::Offered => "offered",
            OfferStatus::Accepted => "accepted",
            OfferStatus::Declined => "declined",
        }
    }

    pub fn is_active(self) -> bool {
        matches!(self, OfferStatus::Accepted)
    }
}

impl std::str::FromStr for OfferStatus {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "offered" => Ok(OfferStatus::Offered),
            "accepted" => Ok(OfferStatus::Accepted),
            "declined" => Ok(OfferStatus::Declined),
            other => Err(crate::AntumbraError::other(format!(
                "unknown offer status `{other}` (use offered | accepted | declined)"
            ))),
        }
    }
}

/// The tenant's gate. One row per tenant, written only by the owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hive {
    pub tenant_id: TenantId,
    pub enabled: bool,
    pub updated_at: DateTime<Utc>,
}

impl Hive {
    pub fn new(tenant: TenantId, enabled: bool, now: DateTime<Utc>) -> Self {
        Self {
            tenant_id: tenant,
            enabled,
            updated_at: now,
        }
    }
}

/// A member's gate. One row per (tenant, user), written only by that user.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiveMembership {
    pub tenant_id: TenantId,
    pub user: UserId,
    pub opted_in: bool,
    pub updated_at: DateTime<Utc>,
}

impl HiveMembership {
    pub fn new(tenant: TenantId, user: UserId, opted_in: bool, now: DateTime<Utc>) -> Self {
        Self {
            tenant_id: tenant,
            user,
            opted_in,
            updated_at: now,
        }
    }
}

/// One unit a member has offered, and what the owner made of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiveOffer {
    pub id: String,
    pub tenant_id: TenantId,
    pub subject_kind: OfferedKind,
    /// The compartment key, document title or expert id being offered. Free
    /// text, because the three kinds name their subjects differently and the
    /// hive is not the place that resolves them.
    pub subject_id: String,
    pub offered_by: UserId,
    pub status: OfferStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl HiveOffer {
    pub fn new(
        tenant: TenantId,
        subject_kind: OfferedKind,
        subject_id: impl Into<String>,
        offered_by: UserId,
        now: DateTime<Utc>,
    ) -> Self {
        let subject_id = subject_id.into();
        Self {
            id: Self::id_for(&tenant, subject_kind, &subject_id),
            tenant_id: tenant,
            subject_kind,
            subject_id,
            offered_by,
            status: OfferStatus::Offered,
            created_at: now,
            updated_at: now,
        }
    }

    /// Keyed on (tenant, kind, subject) and deliberately **not** on the offering
    /// member: one compartment is one thing, and two members offering the same
    /// subject is one decision for the owner rather than two competing rows.
    pub fn id_for(tenant: &TenantId, kind: OfferedKind, subject_id: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tenant.as_str().hash(&mut hasher);
        kind.as_str().hash(&mut hasher);
        subject_id.hash(&mut hasher);
        format!("hive_offer:{:x}", hasher.finish())
    }

    pub fn with_status(mut self, status: OfferStatus, now: DateTime<Utc>) -> Self {
        self.status = status;
        self.updated_at = now;
        self
    }
}

/// Whether the hive is actually open to this member's contribution.
///
/// Both gates, always. A missing row on either side reads as closed, because
/// the absence of a decision is not consent: an owner who has not enabled the
/// hive has not opened it, and a member who has not opted in has not joined.
pub fn is_open(hive: Option<&Hive>, membership: Option<&HiveMembership>) -> bool {
    hive.is_some_and(|h| h.enabled) && membership.is_some_and(|m| m.opted_in)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> DateTime<Utc> {
        Utc::now()
    }

    fn tenant() -> TenantId {
        TenantId::new("ws:org")
    }

    #[test]
    fn neither_gate_alone_opens_the_hive() {
        let on = Hive::new(tenant(), true, at());
        let off = Hive::new(tenant(), false, at());
        let joined = HiveMembership::new(tenant(), UserId::new("user:lily"), true, at());
        let out = HiveMembership::new(tenant(), UserId::new("user:lily"), false, at());

        assert!(is_open(Some(&on), Some(&joined)));
        assert!(
            !is_open(Some(&on), Some(&out)),
            "an owner cannot conscript a member's memory"
        );
        assert!(
            !is_open(Some(&off), Some(&joined)),
            "a member cannot publish into a hive the owner has not opened"
        );
        assert!(!is_open(Some(&off), Some(&out)));
    }

    /// The absence of a decision is not consent.
    #[test]
    fn a_missing_gate_reads_as_closed() {
        let on = Hive::new(tenant(), true, at());
        let joined = HiveMembership::new(tenant(), UserId::new("user:lily"), true, at());
        assert!(!is_open(None, Some(&joined)), "no tenant decision");
        assert!(!is_open(Some(&on), None), "no member decision");
        assert!(!is_open(None, None));
    }

    #[test]
    fn only_an_accepted_offer_is_in_the_hive() {
        let now = at();
        let offer = HiveOffer::new(
            tenant(),
            OfferedKind::Compartment,
            "comp:brand",
            UserId::new("user:lily"),
            now,
        );
        assert_eq!(offer.status, OfferStatus::Offered);
        assert!(!offer.status.is_active(), "offered is not accepted");
        assert!(offer
            .clone()
            .with_status(OfferStatus::Accepted, now)
            .status
            .is_active());
        assert!(!offer
            .clone()
            .with_status(OfferStatus::Declined, now)
            .status
            .is_active());
    }

    /// One subject is one decision for the owner, whoever offered it.
    #[test]
    fn an_offer_is_keyed_on_its_subject_and_not_its_offerer() {
        let now = at();
        let lily = HiveOffer::new(
            tenant(),
            OfferedKind::Compartment,
            "comp:brand",
            UserId::new("user:lily"),
            now,
        );
        let oslo = HiveOffer::new(
            tenant(),
            OfferedKind::Compartment,
            "comp:brand",
            UserId::new("user:oslo"),
            now,
        );
        assert_eq!(lily.id, oslo.id, "one subject, one decision");
        // A different kind or a different subject is a different decision, and
        // a tenant boundary is a boundary here as everywhere.
        assert_ne!(
            HiveOffer::new(
                tenant(),
                OfferedKind::Document,
                "comp:brand",
                UserId::new("user:lily"),
                now
            )
            .id,
            lily.id
        );
        assert_ne!(
            HiveOffer::new(
                TenantId::new("ws:other"),
                OfferedKind::Compartment,
                "comp:brand",
                UserId::new("user:lily"),
                now
            )
            .id,
            lily.id
        );
    }

    #[test]
    fn the_wire_spellings_round_trip() -> crate::Result<()> {
        for kind in [
            OfferedKind::Compartment,
            OfferedKind::Document,
            OfferedKind::Expert,
        ] {
            assert_eq!(kind.as_str().parse::<OfferedKind>()?, kind);
        }
        for status in [
            OfferStatus::Offered,
            OfferStatus::Accepted,
            OfferStatus::Declined,
        ] {
            assert_eq!(status.as_str().parse::<OfferStatus>()?, status);
        }
        assert!("shadow".parse::<OfferedKind>().is_err());
        assert!("pending".parse::<OfferStatus>().is_err());
        Ok(())
    }
}

//! A genesis run that the node holding the work could not do itself: a
//! memory-only node escalates instead of failing.
//!
//! Escalation has to leave something behind. A node that only logged "this
//! belongs on the rig" would have failed quietly in a way that looks, from the
//! outside, exactly like a compartment that never cleared the gate. The request
//! is a row, keyed per (tenant, user, compartment), so a compartment that goes
//! on being reinforced asks once rather than piling up, and so the machine that
//! can train has something to find.

use serde::{Deserialize, Serialize};

use crate::ids::{CompartmentId, TenantId, UserId};

/// How far a request has got. A node that takes one claims it first, so two
/// trainers in a user's fabric do not both run it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenesisStatus {
    /// Asked, and waiting for the machine that can run it.
    Pending,
    /// A node has taken it and is training.
    Claimed,
    /// Run. Kept rather than deleted, so the asking node can see what became of
    /// what it asked for.
    Done,
}

impl GenesisStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            GenesisStatus::Pending => "pending",
            GenesisStatus::Claimed => "claimed",
            GenesisStatus::Done => "done",
        }
    }
}

impl std::str::FromStr for GenesisStatus {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "pending" => Ok(GenesisStatus::Pending),
            "claimed" => Ok(GenesisStatus::Claimed),
            "done" => Ok(GenesisStatus::Done),
            other => Err(crate::AntumbraError::other(format!(
                "unknown genesis status `{other}` (use pending | claimed | done)"
            ))),
        }
    }
}

/// One compartment, owed a genesis run somewhere other than where it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisRequest {
    pub id: String,
    pub tenant_id: TenantId,
    pub user: UserId,
    pub compartment: CompartmentId,
    /// The machine that could not run it.
    pub from_host: String,
    /// The machine the fabric named when the request was made. A hint, not an
    /// assignment: by the time anything reads this, the user's fabric may have
    /// a different trainer in it, and that one should take the work rather than
    /// leave it for a machine that is no longer there.
    pub to_host: String,
    pub status: GenesisStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl GenesisRequest {
    pub fn new(
        tenant: TenantId,
        user: UserId,
        compartment: CompartmentId,
        from_host: impl Into<String>,
        to_host: impl Into<String>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        Self {
            id: Self::id_for(&tenant, &user, &compartment),
            tenant_id: tenant,
            user,
            compartment,
            from_host: from_host.into(),
            to_host: to_host.into(),
            status: GenesisStatus::Pending,
            created_at: now,
            updated_at: now,
        }
    }

    /// Keyed per (tenant, user, compartment): a compartment that is reinforced
    /// again while its request is outstanding re-asks the same question, and a
    /// question already asked should not become a second row.
    pub fn id_for(tenant: &TenantId, user: &UserId, compartment: &CompartmentId) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tenant.as_str().hash(&mut hasher);
        user.as_str().hash(&mut hasher);
        compartment.as_str().hash(&mut hasher);
        format!("genesis_request:{:x}", hasher.finish())
    }

    pub fn is_open(&self) -> bool {
        matches!(self.status, GenesisStatus::Pending | GenesisStatus::Claimed)
    }

    pub fn with_status(
        mut self,
        status: GenesisStatus,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        self.status = status;
        self.updated_at = now;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(user: &str, compartment: &str) -> GenesisRequest {
        GenesisRequest::new(
            TenantId::new("ws:t"),
            UserId::new(user),
            CompartmentId::new(compartment),
            "laptop",
            "rig",
            chrono::Utc::now(),
        )
    }

    #[test]
    fn asking_twice_about_one_compartment_asks_once() {
        let first = request("user:a", "comp:rust");
        assert_eq!(request("user:a", "comp:rust").id, first.id);
        // A different compartment is a different question, and another user's
        // compartment is another user's question.
        assert_ne!(request("user:a", "comp:surql").id, first.id);
        assert_ne!(request("user:b", "comp:rust").id, first.id);
    }

    #[test]
    fn a_request_is_open_until_it_has_been_run() -> crate::Result<()> {
        let now = chrono::Utc::now();
        let asked = request("user:a", "comp:rust");
        assert_eq!(asked.status, GenesisStatus::Pending);
        assert!(asked.is_open());
        let claimed = asked.clone().with_status(GenesisStatus::Claimed, now);
        assert!(claimed.is_open(), "taken is not the same as finished");
        assert!(!claimed.with_status(GenesisStatus::Done, now).is_open());

        for status in [
            GenesisStatus::Pending,
            GenesisStatus::Claimed,
            GenesisStatus::Done,
        ] {
            assert_eq!(status.as_str().parse::<GenesisStatus>()?, status);
        }
        assert!("running".parse::<GenesisStatus>().is_err());
        Ok(())
    }
}

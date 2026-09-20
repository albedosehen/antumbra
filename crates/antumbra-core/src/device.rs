//! A node in a user's fabric (ADR-0017): which machine an agent is running on,
//! what it can do, and therefore where training goes.
//!
//! ADR-0013 says a user's agents share one memory. ADR-0017 says those agents
//! run on several machines, and that the machines are not interchangeable: a
//! laptop can recall and store, and cannot train. A node registers itself here
//! so the fabric knows which of the user's machines is the one that can.
//!
//! The row is the user's, not the host's. Memory follows the user (ADR-0017
//! section A), so two people on one machine have two profiles, and one person
//! on three machines has three.

use serde::{Deserialize, Serialize};

use crate::ids::{TenantId, UserId};

/// What a node is for. A user's fabric has many `Memory` nodes and, when they
/// own hardware that can train, one `Genesis`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceRole {
    /// Recall, store and route, locally. Training is dispatched elsewhere, and
    /// a node in this role escalates rather than failing (ADR-0017 section A).
    Memory,
    /// Can train. Genesis is dispatched here: a compartment that clears the
    /// consolidation gate (ADR-0012), or an explicit `train`.
    Genesis,
}

impl DeviceRole {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceRole::Memory => "memory",
            DeviceRole::Genesis => "genesis",
        }
    }
}

impl std::str::FromStr for DeviceRole {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "memory" => Ok(DeviceRole::Memory),
            "genesis" => Ok(DeviceRole::Genesis),
            other => Err(crate::AntumbraError::other(format!(
                "unknown device role `{other}` (use memory | genesis)"
            ))),
        }
    }
}

/// The video memory a node needs before genesis is dispatched to it. ADR-0006
/// names the fleet: the 3090 Ti 24 GB is the v0 training target, the M4 Pro has
/// 48 GB, and the 8 GB cards (a Pascal 1080, a Jetson Orin Nano) are called out
/// there as weak at low-bit. 16 GiB is the line that admits the first two and
/// the 16 GB variant of the 3080 mobile, and leaves the 8 GB machines as memory
/// nodes, which is what they are.
pub const GENESIS_MIN_VRAM_MIB: u64 = 16 * 1024;

/// Whether a backend can train at all. This is the backend the *build* can
/// drive, not merely the silicon present: a CUDA box running a binary compiled
/// without the CUDA backend cannot train, and a node that says otherwise
/// collects dispatches it will only escalate.
pub fn backend_can_train(backend: &str) -> bool {
    matches!(
        backend.trim().to_ascii_lowercase().as_str(),
        "cuda" | "metal" | "mlx"
    )
}

/// The role a node's own hardware earns it (ADR-0017 A2: "derived from backend
/// and VRAM").
///
/// Unknown VRAM does not demote a training backend. `None` means the node could
/// not tell, and treating "did not know" as "has none" would make every machine
/// whose memory we cannot read a memory node, which is the wrong default for
/// the one field most likely to be unreadable. A *known* figure below the floor
/// does demote: that is a measurement, not an absence.
pub fn role_for(backend: &str, vram_mib: Option<u64>) -> DeviceRole {
    if !backend_can_train(backend) {
        return DeviceRole::Memory;
    }
    match vram_mib {
        Some(mib) if mib < GENESIS_MIN_VRAM_MIB => DeviceRole::Memory,
        _ => DeviceRole::Genesis,
    }
}

/// One of a user's machines, as that machine describes itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// `device_profile:<something stable for this (tenant, user, host)>`.
    pub id: String,
    pub tenant_id: TenantId,
    /// Whose node this is. A machine two people use has a row for each.
    pub user: UserId,
    /// What the machine calls itself, the same name a memory's provenance
    /// carries as its host.
    pub host: String,
    /// The serving backend the node found: `cuda`, `metal`, `cpu`. Free text
    /// rather than an enum, because the set grows with the hardware and a node
    /// reporting something this version has not heard of should still register.
    pub backend: String,
    /// Video memory in mebibytes, when the node could tell. `None` is not zero:
    /// it means the node did not know, and a role should not be inferred from it.
    pub vram_mib: Option<u64>,
    pub role: DeviceRole,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl DeviceProfile {
    /// A node describing itself. The id is derived from (tenant, user, host) so
    /// the same machine re-registering updates its row instead of adding one.
    pub fn new(
        tenant: TenantId,
        user: UserId,
        host: impl Into<String>,
        backend: impl Into<String>,
        role: DeviceRole,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        let host = host.into();
        Self {
            id: Self::id_for(&tenant, &user, &host),
            tenant_id: tenant,
            user,
            host,
            backend: backend.into(),
            vram_mib: None,
            role,
            updated_at: now,
        }
    }

    /// A node describing itself from what it found, taking the role that
    /// follows from it ([`role_for`]) rather than being told one.
    pub fn detected(
        tenant: TenantId,
        user: UserId,
        host: impl Into<String>,
        backend: impl Into<String>,
        vram_mib: Option<u64>,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Self {
        let backend = backend.into();
        let role = role_for(&backend, vram_mib);
        Self {
            vram_mib,
            ..Self::new(tenant, user, host, backend, role, now)
        }
    }

    pub fn with_vram(mut self, mib: u64) -> Self {
        self.vram_mib = Some(mib);
        self
    }

    /// Stable across restarts, and distinct per user on a shared machine.
    pub fn id_for(tenant: &TenantId, user: &UserId, host: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tenant.as_str().hash(&mut hasher);
        user.as_str().hash(&mut hasher);
        host.hash(&mut hasher);
        format!("device_profile:{:x}", hasher.finish())
    }

    pub fn is_genesis(&self) -> bool {
        self.role == DeviceRole::Genesis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> chrono::DateTime<chrono::Utc> {
        chrono::Utc::now()
    }

    fn profile(tenant: &str, user: &str, host: &str) -> DeviceProfile {
        DeviceProfile::new(
            TenantId::new(tenant),
            UserId::new(user),
            host,
            "cpu",
            DeviceRole::Memory,
            at(),
        )
    }

    #[test]
    fn a_machine_re_registering_keeps_its_row_and_two_users_on_it_do_not_share_one() {
        let again = profile("ws:t", "user:a", "laptop");
        assert_eq!(profile("ws:t", "user:a", "laptop").id, again.id);
        // The row is the user's, not the host's.
        assert_ne!(profile("ws:t", "user:b", "laptop").id, again.id);
        // And a tenant boundary is a boundary here as everywhere.
        assert_ne!(profile("ws:other", "user:a", "laptop").id, again.id);
        // One user, several machines, several rows.
        assert_ne!(profile("ws:t", "user:a", "workstation").id, again.id);
    }

    #[test]
    fn a_role_survives_the_round_trip_it_takes_through_the_store() -> crate::Result<()> {
        for role in [DeviceRole::Memory, DeviceRole::Genesis] {
            assert_eq!(role.as_str().parse::<DeviceRole>()?, role);
            // Schemaless rows come back as whatever was written, so the wire
            // spelling and the parsed spelling have to be the same one.
            let wire = serde_json::to_string(&role)
                .map_err(|e| crate::AntumbraError::other(e.to_string()))?;
            assert_eq!(wire, format!("\"{}\"", role.as_str()));
        }
        assert!("trainer".parse::<DeviceRole>().is_err());
        Ok(())
    }

    #[test]
    fn a_role_is_earned_by_a_backend_that_can_train_and_memory_enough_to_do_it() {
        // ADR-0006's fleet, as each machine would report itself.
        assert_eq!(role_for("cuda", Some(24_576)), DeviceRole::Genesis); // 3090 Ti
        assert_eq!(role_for("metal", Some(49_152)), DeviceRole::Genesis); // M4 Pro
        assert_eq!(role_for("cuda", Some(16_384)), DeviceRole::Genesis); // 3080 mobile 16
        assert_eq!(role_for("cuda", Some(8_192)), DeviceRole::Memory); // 1080, Jetson
                                                                       // A build that cannot drive the silicon cannot train on it, however
                                                                       // much of it there is.
        assert_eq!(role_for("cpu", Some(131_072)), DeviceRole::Memory);
        assert_eq!(role_for("rocm", Some(24_576)), DeviceRole::Memory);
        // Not knowing is not the same as having none.
        assert_eq!(role_for("cuda", None), DeviceRole::Genesis);
        assert_eq!(role_for("cpu", None), DeviceRole::Memory);
        // What a node reports is what it found, however it spelled it.
        assert_eq!(role_for("CUDA", None), DeviceRole::Genesis);
    }

    #[test]
    fn a_detected_node_takes_the_role_its_hardware_earns_rather_than_one_it_is_told() {
        let at = at();
        let laptop = DeviceProfile::detected(
            TenantId::new("ws:t"),
            UserId::new("user:a"),
            "laptop",
            "cpu",
            Some(8_192),
            at,
        );
        assert_eq!(laptop.role, DeviceRole::Memory);
        assert_eq!(laptop.vram_mib, Some(8_192));

        let rig = DeviceProfile::detected(
            TenantId::new("ws:t"),
            UserId::new("user:a"),
            "rig",
            "cuda",
            Some(24_576),
            at,
        );
        assert!(rig.is_genesis());
        // The same machine keeps the same row whichever constructor found it.
        assert_eq!(rig.id, profile("ws:t", "user:a", "rig").id);
    }

    #[test]
    fn not_knowing_the_vram_is_not_the_same_as_having_none() {
        let unknown = profile("ws:t", "user:a", "laptop");
        assert_eq!(unknown.vram_mib, None);
        assert_eq!(unknown.clone().with_vram(0).vram_mib, Some(0));
        assert!(!unknown.is_genesis());
    }
}

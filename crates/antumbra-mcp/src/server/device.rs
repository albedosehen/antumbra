//! Devices: the machines a user's sessions run on, listed in the
//! user's fabric whether or not a server runs on them.
//!
//! Its own router, joined to the others in `engine.rs`. A server registers its
//! own machine: a stdio server when it starts, an HTTP server when an identity
//! first calls it. A laptop whose agent talks to a hosted hub runs no server,
//! so before this nothing ever named it, and the fabric listed the hub and not
//! the machines working against it. `register_device` is how such a session
//! names its machine. The row it writes is always a memory node: the server
//! cannot see that machine's hardware, and the machine runs no trainer.

use super::*;

use antumbra_core::handoff::{self, ANY};
use antumbra_core::DeviceProfile;
use antumbra_store::repo::device as store_device;

/// What a machine with no `ANTUMBRA_HOST_ID` calls itself. Every unnamed
/// machine says it, so a row for it would be one row standing for all of them.
const UNNAMED: &str = "local";

/// The longest name a client may give its machine. A host name is at most 253
/// characters; this leaves room for one a person chose and refuses a header
/// that is something else.
const MAX_NAME_CHARS: usize = 128;

/// `raw` as the name of one machine, normalized as handoffs compare names
/// (trimmed, lowercased), or `None` when it names no machine in particular
/// (blank, `any`, the unnamed `local`) or is not a name (too long, or holding a
/// control character).
///
/// What a client sends in `X-Antumbra-Host` passes through this before
/// anything is stamped with it (#193). The name is provenance, not
/// authorization: a client can only say which of its own identity's machines
/// it is, so a refused name costs the stamp, never the call.
pub(crate) fn machine_name(raw: &str) -> Option<String> {
    let name = handoff::normalize_host(raw);
    let names_one = name != ANY && name != UNNAMED;
    let is_name = name.chars().count() <= MAX_NAME_CHARS && !name.chars().any(char::is_control);
    (names_one && is_name).then_some(name)
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct RegisterDeviceParams {
    /// The machine this session runs on, by host name: the name its handoffs
    /// are addressed to (`ANTUMBRA_HOST_ID`, which the session-start block
    /// names).
    pub(super) host: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct RegisteredDeviceOut {
    /// The machine, normalized (trimmed, lowercased).
    pub(super) host: String,
    /// Whether this call wrote the machine's row. False when a server running
    /// on that machine registers it itself: its row stays as the server wrote it.
    pub(super) registered: bool,
    /// The role the machine holds: `memory` for one a session named.
    pub(super) role: String,
    /// Why nothing was written, when nothing was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) note: Option<String>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DeviceView {
    pub(super) host: String,
    /// `memory`, or `genesis` for a machine that can train.
    pub(super) role: String,
    /// What the machine reported: the backend its server can drive (`cuda`,
    /// `metal`, `cpu`), or `client` for a machine a session named, which runs
    /// no server.
    pub(super) backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) vram_mib: Option<u64>,
    /// When it last registered: a server when it starts or an identity first
    /// calls it, a client at each session start. A machine no longer in use
    /// keeps the last time it was.
    pub(super) last_seen: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct DevicesOut {
    /// Your machines, most recently seen first.
    pub(super) devices: Vec<DeviceView>,
    /// The machine that trains for you, when one of them can.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) trainer: Option<String>,
    /// The machine this call came from, as what this session writes is
    /// stamped (`author_host` on a memory): the name its client sent in
    /// `X-Antumbra-Host`, else this server's own.
    pub(super) this_device: String,
    /// Whether the client named it. False means the server's own name stands
    /// in: right for a server on the session's own machine, and the hub's name
    /// for every machine that reaches a hosted server without naming itself.
    pub(super) named_by_client: bool,
}

#[tool_router(router = device_router, vis = "pub(super)")]
impl McpServer {
    /// Name the machine this session runs on, so the fabric lists it.
    #[tool(
        description = "Register the machine this session runs on in your fabric, by host name (ANTUMBRA_HOST_ID), so it is listed among your devices and can be addressed by handoffs. A machine named this way is a memory node. A server running on a machine registers that machine itself, and its row is left as it is. The session-start hook calls this."
    )]
    pub(super) async fn register_device(
        &self,
        Parameters(p): Parameters<RegisterDeviceParams>,
    ) -> Result<Json<RegisteredDeviceOut>, ErrorData> {
        let host = handoff::normalize_host(&p.host);
        if host == ANY || host == UNNAMED {
            return Err(ErrorData::invalid_params(
                format!(
                    "`{}` names no machine in particular; set ANTUMBRA_HOST_ID on it and register that name",
                    p.host.trim()
                ),
                None,
            ));
        }
        let mine = store_device::list_for_user(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        let existing = mine
            .iter()
            .find(|d| handoff::normalize_host(&d.host) == host);
        // A row a server wrote about its own machine knows that machine's
        // hardware, and a session naming the machine does not. Overwriting it
        // would demote a trainer to a memory node.
        if let Some(own) = existing.filter(|d| !d.is_client()) {
            return Ok(Json(RegisteredDeviceOut {
                host,
                registered: false,
                role: own.role.as_str().to_string(),
                note: Some("a server runs on this machine and registers it itself".to_string()),
            }));
        }
        // This server's own machine, when its own registration did not land:
        // the next start writes it, with the hardware this process can see.
        if host == handoff::normalize_host(&self.host) {
            return Ok(Json(RegisteredDeviceOut {
                host,
                registered: false,
                role: crate::hardware::role().as_str().to_string(),
                note: Some(
                    "this is the server's own machine; it registers itself when it starts"
                        .to_string(),
                ),
            }));
        }
        let profile = DeviceProfile::client(
            self.tenant.clone(),
            self.user.clone(),
            host.clone(),
            Utc::now(),
        );
        store_device::upsert(&self.store, &profile)
            .await
            .map_err(err)?;
        Ok(Json(RegisteredDeviceOut {
            host,
            registered: true,
            role: profile.role.as_str().to_string(),
            note: None,
        }))
    }

    /// The machines in your fabric.
    #[tool(
        description = "List your machines: every one registered in your fabric, by host name, with its role (memory, or genesis for one that can train), what it reported, and when it was last seen, most recent first. These are the names handoffs are addressed to, and the names list_memories and recall_memories filter by `host`. Also says which machine this session's writes are stamped with (this_device) and whether its client named it."
    )]
    pub(super) async fn devices(&self) -> Result<Json<DevicesOut>, ErrorData> {
        let mine = store_device::list_for_user(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        // The same rule the store uses to name the trainer: the freshest
        // genesis row. `mine` is newest first already.
        let trainer = mine.iter().find(|d| d.is_genesis()).map(|d| d.host.clone());
        let devices = mine
            .into_iter()
            .map(|d| DeviceView {
                host: d.host,
                role: d.role.as_str().to_string(),
                backend: d.backend,
                vram_mib: d.vram_mib,
                last_seen: d.updated_at.to_rfc3339(),
            })
            .collect();
        Ok(Json(DevicesOut {
            devices,
            trainer,
            this_device: self.device(),
            named_by_client: self.device.is_some(),
        }))
    }
}

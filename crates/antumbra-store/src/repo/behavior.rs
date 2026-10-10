//! Behaviors in the store: a memory in the user's own `behavior`
//! compartment, its spec in the content and its state in the evidence. One
//! writer for the MCP tool and the CLI's import, so both store a behavior the
//! same way. Validation (`Spec::problems`) and embedding are the caller's.

use chrono::Utc;

use antumbra_core::behavior::{self, Spec, State, Status};
use antumbra_core::{
    Compartment, CompartmentId, Memory, MemoryId, MemoryNetwork, Result, TenantId, UserId,
};

use crate::repo::{compartment, memory};
use crate::store::Store;

/// What [`record`] stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded {
    pub id: MemoryId,
    /// Whether the behavior named to supersede was found and retired.
    pub superseded: Option<bool>,
}

/// The user's behavior compartment, created the first time it is needed.
pub async fn compartment_of(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
) -> Result<CompartmentId> {
    let id = behavior::compartment_id(tenant, user);
    let exists = compartment::list_owned(store, tenant, user)
        .await?
        .iter()
        .any(|c| c.id == id);
    if !exists {
        compartment::create(
            store,
            &Compartment::new(
                id.clone(),
                tenant.clone(),
                user.clone(),
                behavior::COMPARTMENT_NAME,
                Utc::now(),
            ),
        )
        .await?;
    }
    Ok(id)
}

/// Store a behavior as `id`, authored by `user` on `host`, with its content's
/// `embedding`. A behavior is volatile: it trains through its tasks and check
/// (`antumbra behave`), never through write-time consolidation, which would
/// teach it to echo itself. When `supersedes` names one of the user's
/// behaviors, that one is retired. `sources` are the memories it was drawn
/// from, when it came from the store rather than from the user; each is kept
/// in the evidence, and no longer offered as a candidate.
#[allow(clippy::too_many_arguments)]
pub async fn record(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    host: &str,
    id: MemoryId,
    spec: &Spec,
    status: Status,
    scope: &str,
    supersedes: Option<&str>,
    sources: &[String],
    embedding: Vec<f32>,
) -> Result<Recorded> {
    let mut evidence = vec![
        behavior::status_evidence(status),
        behavior::scope_evidence(scope),
    ];
    if let Some(old) = supersedes {
        evidence.push(behavior::supersedes_evidence(old));
    }
    let mut cited: Vec<&str> = sources
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    cited.sort_unstable();
    cited.dedup();
    evidence.extend(cited.into_iter().map(behavior::source_evidence));
    let compartment = compartment_of(store, tenant, user).await?;
    let m = Memory::new(
        id.as_str(),
        tenant.clone(),
        MemoryNetwork::Opinion,
        behavior::content(spec),
        1.0,
        Utc::now(),
    )
    .with_embedding(embedding)
    .in_compartment(compartment)
    .by(user.clone(), host.to_string())
    .with_evidence(evidence)
    .volatile(true);
    memory::upsert(store, &m).await?;
    let superseded = match supersedes {
        Some(old) => Some(
            set_status(store, tenant, user, old, Status::Retired)
                .await?
                .is_some(),
        ),
        None => None,
    };
    Ok(Recorded { id, superseded })
}

/// Set one of the user's behaviors to `status`. `None` when `id` is not a
/// behavior in their behavior compartment.
pub async fn set_status(
    store: &Store,
    tenant: &TenantId,
    user: &UserId,
    id: &str,
    status: Status,
) -> Result<Option<Status>> {
    let compartment = behavior::compartment_id(tenant, user);
    let found = memory::get(store, tenant, &MemoryId::new(id))
        .await?
        .filter(|m| m.compartment.as_ref() == Some(&compartment))
        .filter(|m| State::of(&m.evidence).is_some());
    let Some(mut m) = found else {
        return Ok(None);
    };
    behavior::set_status(&mut m.evidence, status);
    m.updated_at = Utc::now();
    memory::upsert(store, &m).await?;
    Ok(Some(status))
}

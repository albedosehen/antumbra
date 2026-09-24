//! Verifier repository (ADR-0022 S-4): the verifier namespace, every
//! measurement of trust, and every change of a verifier's state.
//!
//! A verifier is keyed by its content address and its spec is stored as
//! canonical text, so the database cannot reorder or retype what was
//! measured. Every read checks the address against the content and refuses a
//! verifier whose check is not the one that was measured.
//!
//! Measurements and state changes are rows of their own, appended and
//! numbered per verifier, never rewritten. The only way into trusted is
//! [`record_measurement`] with a sound measurement; [`transition`] is for a
//! person, and the state machine refuses it trust. The tables carry no
//! permissions clause, so no tenant session reads or writes them, and nothing
//! that trains holds more than [`Registry`], which only reads.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use surql::query::builder::Query;
use surql::query::crud::{create_record, first, query_records};
use surql::types::operators::eq;

use antumbra_core::ports::TrustedVerifiers;
use antumbra_core::{
    after_measurement, canonical_json, current_trust, grants_reward, AntumbraError, Result,
    TrustCause, TrustMeasurement, TrustState, VerifierId, VerifierOrigin, VerifierRecord,
    VerifierTier, VerifierTransition,
};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "verifier";
const TRANSITIONS: &str = "verifier_transition";
const MEASUREMENTS: &str = "verifier_measurement";

#[derive(Serialize, Deserialize)]
struct VerifierRow {
    key: String,
    domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    task: Option<String>,
    tier: VerifierTier,
    origin: VerifierOrigin,
    /// The spec as canonical text.
    spec: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposed_by: Option<String>,
    created_at: DateTime<Utc>,
}

impl VerifierRow {
    fn from_domain(r: &VerifierRecord) -> Self {
        VerifierRow {
            key: r.id.as_str().to_string(),
            domain: r.domain.clone(),
            task: r.task.clone(),
            tier: r.tier,
            origin: r.origin,
            spec: canonical_json(&r.spec),
            proposed_by: r.proposed_by.clone(),
            created_at: r.created_at,
        }
    }

    fn into_domain(self) -> Result<VerifierRecord> {
        let record = VerifierRecord {
            id: VerifierId::new(self.key),
            domain: self.domain,
            task: self.task,
            tier: self.tier,
            origin: self.origin,
            spec: serde_json::from_str(&self.spec)?,
            proposed_by: self.proposed_by,
            created_at: self.created_at,
        };
        if !record.is_intact() {
            return Err(AntumbraError::rejected(format!(
                "verifier {} does not match its address: its check is not the one that was measured",
                record.id
            )));
        }
        Ok(record)
    }
}

#[derive(Serialize, Deserialize)]
struct TransitionRow {
    #[serde(flatten)]
    transition: VerifierTransition,
    seq: u32,
}

#[derive(Serialize, Deserialize)]
struct MeasurementRow {
    #[serde(flatten)]
    measurement: TrustMeasurement,
    seq: u32,
}

/// Add a verifier to the namespace, or return the one already there with the
/// same address: the same content is the same check, whoever proposed it.
pub async fn propose(store: &Store, record: &VerifierRecord) -> Result<VerifierRecord> {
    if !record.is_intact() {
        return Err(AntumbraError::rejected(format!(
            "verifier {} does not match its address",
            record.id
        )));
    }
    if let Some(existing) = get(store, &record.id).await? {
        return Ok(existing);
    }
    let row = VerifierRow::from_domain(record);
    create_record(store.client(), TABLE, serde_json::to_value(row)?)
        .await
        .map_err(map)?;
    Ok(record.clone())
}

/// One verifier by address.
pub async fn get(store: &Store, id: &VerifierId) -> Result<Option<VerifierRecord>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("key", id.as_str()));
    let row: Option<VerifierRow> = first(store.client(), &query).await.map_err(map)?;
    row.map(VerifierRow::into_domain).transpose()
}

/// Every verifier, in address order.
pub async fn list(store: &Store) -> Result<Vec<VerifierRecord>> {
    let query = Query::new().select(None).from_table(TABLE).map_err(map)?;
    let rows: Vec<VerifierRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut all = rows
        .into_iter()
        .map(VerifierRow::into_domain)
        .collect::<Result<Vec<_>>>()?;
    all.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    Ok(all)
}

/// Every state change of verifier `id`, in order.
pub async fn history(store: &Store, id: &VerifierId) -> Result<Vec<VerifierTransition>> {
    let query = Query::new()
        .select(None)
        .from_table(TRANSITIONS)
        .map_err(map)?
        .where_(eq("verifier", id.as_str()))
        .order_by("seq", "ASC")
        .map_err(map)?;
    let rows: Vec<TransitionRow> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows.into_iter().map(|r| r.transition).collect())
}

/// Where a verifier stands.
pub async fn state_of(store: &Store, record: &VerifierRecord) -> Result<TrustState> {
    Ok(current_trust(
        record.origin,
        &history(store, &record.id).await?,
    ))
}

/// Every measurement of verifier `id`, oldest first.
pub async fn measurements(store: &Store, id: &VerifierId) -> Result<Vec<TrustMeasurement>> {
    let query = Query::new()
        .select(None)
        .from_table(MEASUREMENTS)
        .map_err(map)?
        .where_(eq("verifier", id.as_str()))
        .order_by("seq", "ASC")
        .map_err(map)?;
    let rows: Vec<MeasurementRow> = query_records(store.client(), &query).await.map_err(map)?;
    Ok(rows.into_iter().map(|r| r.measurement).collect())
}

/// Record a measurement of `record` and make the move it calls for, if any:
/// trust for a proposed verifier found sound, revocation for one rejected
/// outright, quarantine for a trusted one found unsound.
pub async fn record_measurement(
    store: &Store,
    record: &VerifierRecord,
    measurement: &TrustMeasurement,
) -> Result<Option<VerifierTransition>> {
    if measurement.verifier != record.id {
        return Err(AntumbraError::rejected(format!(
            "a measurement of {} cannot be recorded against {}",
            measurement.verifier, record.id
        )));
    }
    let seq = measurements(store, &record.id).await?.len();
    let row = MeasurementRow {
        measurement: measurement.clone(),
        seq: u32::try_from(seq).unwrap_or(u32::MAX),
    };
    create_record(store.client(), MEASUREMENTS, serde_json::to_value(row)?)
        .await
        .map_err(map)?;
    let past = history(store, &record.id).await?;
    let from = current_trust(record.origin, &past);
    let Some(to) = after_measurement(from, &measurement.verdict) else {
        return Ok(None);
    };
    let cause = TrustCause::Measured {
        at: measurement.at,
        verdict: measurement.verdict.clone(),
    };
    append(store, &record.id, &past, from, to, cause)
        .await
        .map(Some)
}

/// A person's move: quarantine or revocation. The state machine refuses a
/// person trust, which only a sound measurement gives.
pub async fn transition(
    store: &Store,
    id: &VerifierId,
    to: TrustState,
    note: Option<String>,
) -> Result<VerifierTransition> {
    let Some(record) = get(store, id).await? else {
        return Err(AntumbraError::rejected(format!("no verifier {id}")));
    };
    let past = history(store, id).await?;
    let from = current_trust(record.origin, &past);
    append(store, id, &past, from, to, TrustCause::Operator { note }).await
}

async fn append(
    store: &Store,
    id: &VerifierId,
    past: &[VerifierTransition],
    from: TrustState,
    to: TrustState,
    cause: TrustCause,
) -> Result<VerifierTransition> {
    from.transition(to, &cause)?;
    let transition = VerifierTransition {
        verifier: id.clone(),
        from,
        to,
        cause,
        at: Utc::now(),
    };
    let row = TransitionRow {
        transition: transition.clone(),
        seq: u32::try_from(past.len()).unwrap_or(u32::MAX),
    };
    create_record(store.client(), TRANSITIONS, serde_json::to_value(row)?)
        .await
        .map_err(map)?;
    Ok(transition)
}

/// Whether verifier `record` may grant reward at `now`.
pub async fn grants(store: &Store, record: &VerifierRecord, now: DateTime<Utc>) -> Result<bool> {
    let state = state_of(store, record).await?;
    let last_sound = measurements(store, &record.id)
        .await?
        .into_iter()
        .rev()
        .find(|m| m.verdict.is_sound());
    Ok(grants_reward(
        record.origin,
        state,
        last_sound.as_ref(),
        now,
    ))
}

/// The namespace as the reward path reads it: read-only, and asked afresh
/// on every check, so a quarantine stops a verifier's reward at once.
#[derive(Clone)]
pub struct Registry {
    store: Store,
}

impl Registry {
    pub fn new(store: Store) -> Self {
        Registry { store }
    }
}

#[async_trait]
impl TrustedVerifiers for Registry {
    async fn trusted_spec(&self, id: &VerifierId, task: &str) -> Result<Option<serde_json::Value>> {
        let Some(record) = get(&self.store, id).await? else {
            return Ok(None);
        };
        if !record.applies_to(task) {
            return Ok(None);
        }
        Ok(grants(&self.store, &record, Utc::now())
            .await?
            .then_some(record.spec))
    }
}

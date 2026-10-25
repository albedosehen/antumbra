//! The relic registry: what the gallery lists and what a view resolves.
//!
//! One row per relic, keyed by its slug, so the slug is unique across the
//! server and a public relic can be found by it alone. The engine's rule
//! scopes reads to the relic's compartment audience and writes to its owner
//! (`relic_perms`); a write the rule refuses persists nothing and raises no
//! error, so [`put`] reads the row back before it reports success.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{get_record, query_records, upsert_record};
use surql::types::operators::{and_, eq};
use surql::types::RecordID;

use antumbra_core::{CompartmentId, Relic, RelicVersion, Result, TenantId, UserId, Visibility};

use crate::dto::parse_dt;
use crate::error::map;
use crate::store::Store;

const TABLE: &str = "relic";

#[derive(Serialize, Deserialize)]
struct VersionRow {
    number: u32,
    copal_file: String,
    copal_digest: String,
    created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author_host: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct RelicRow {
    key: String,
    tenant_id: String,
    owner: String,
    compartment: String,
    title: String,
    // The wire word, so a row in a visibility this build does not know is
    // left out of a listing rather than failing it.
    visibility: String,
    versions: Vec<VersionRow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    created_at: String,
    updated_at: String,
}

impl RelicRow {
    fn from_domain(r: &Relic) -> Self {
        RelicRow {
            key: r.slug.clone(),
            tenant_id: r.tenant.as_str().to_string(),
            owner: r.owner.as_str().to_string(),
            compartment: r.compartment.as_str().to_string(),
            title: r.title.clone(),
            visibility: r.visibility.as_str().to_string(),
            versions: r
                .versions
                .iter()
                .map(|v| VersionRow {
                    number: v.number,
                    copal_file: v.copal_file.clone(),
                    copal_digest: v.copal_digest.clone(),
                    created_at: v.created_at.to_rfc3339(),
                    author_host: v.author_host.clone(),
                })
                .collect(),
            source: r.source.clone(),
            created_at: r.created_at.to_rfc3339(),
            updated_at: r.updated_at.to_rfc3339(),
        }
    }

    fn into_domain(self) -> Result<Option<Relic>> {
        let Ok(visibility) = self.visibility.parse::<Visibility>() else {
            return Ok(None);
        };
        let versions = self
            .versions
            .into_iter()
            .map(|v| {
                Ok(RelicVersion {
                    number: v.number,
                    copal_file: v.copal_file,
                    copal_digest: v.copal_digest,
                    created_at: parse_dt(&v.created_at)?,
                    author_host: v.author_host,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(Relic {
            slug: self.key,
            tenant: TenantId::new(self.tenant_id),
            owner: UserId::new(self.owner),
            compartment: CompartmentId::new(self.compartment),
            title: self.title,
            visibility,
            versions,
            source: self.source,
            created_at: parse_dt(&self.created_at)?,
            updated_at: parse_dt(&self.updated_at)?,
        }))
    }
}

fn rid(slug: &str) -> Result<RecordID<()>> {
    RecordID::<()>::new(TABLE, slug).map_err(map)
}

/// Write `relic` whole, creating or replacing its row, and report whether it
/// landed. `false` means the engine refused the write: the slug belongs to
/// someone else, or the session may not write into the relic's compartment.
pub async fn put(store: &Store, relic: &Relic) -> Result<bool> {
    let data: Value = serde_json::to_value(RelicRow::from_domain(relic))?;
    // A refused upsert can still come back as an engine error rather than an
    // empty result; either way the read-back below is what decides.
    let _ = upsert_record(store.client(), &rid(&relic.slug)?, data).await;
    // The whole row, not a field or two: a refused rewrite leaves the
    // previous row standing, which can match on owner and version count.
    let landed = get(store, &relic.tenant, &relic.slug).await?;
    Ok(landed.as_ref() == Some(relic))
}

/// One relic by slug, as this session may see it: `None` when it does not
/// exist, belongs to another tenant, or is outside the session's audience,
/// indistinguishably.
pub async fn get(store: &Store, tenant: &TenantId, slug: &str) -> Result<Option<Relic>> {
    let Some(row) = get_record(store.client(), &rid(slug)?).await.map_err(map)? else {
        return Ok(None);
    };
    let row: RelicRow = serde_json::from_value(row)?;
    if row.tenant_id != tenant.as_str() {
        return Ok(None);
    }
    row.into_domain()
}

/// Every relic this session may see in `tenant`, most recently updated first.
pub async fn list(store: &Store, tenant: &TenantId) -> Result<Vec<Relic>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<RelicRow> = query_records(store.client(), &query).await.map_err(map)?;
    let mut relics: Vec<Relic> = rows
        .into_iter()
        .map(RelicRow::into_domain)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    relics.sort_by_key(|r| std::cmp::Reverse(r.updated_at));
    Ok(relics)
}

/// A public relic by slug, in any tenant: what the unauthenticated view
/// resolves. Call it only on a root connection; under a record session the
/// engine rule would hide other tenants' rows anyway, and this read must not
/// depend on who is signed in.
pub async fn get_public(store: &Store, slug: &str) -> Result<Option<Relic>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(and_(eq("key", slug), eq("visibility", "public")))
        .limit(1)
        .map_err(map)?;
    let rows: Vec<RelicRow> = query_records(store.client(), &query).await.map_err(map)?;
    match rows.into_iter().next() {
        Some(row) => row.into_domain(),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use chrono::Utc;

    fn relic(slug: &str, tenant: &str, visibility: Visibility) -> Relic {
        let now = Utc::now();
        Relic {
            slug: slug.into(),
            tenant: TenantId::new(tenant),
            owner: UserId::new("user:lily"),
            compartment: CompartmentId::new("comp:lily"),
            title: "The Ceiling".into(),
            visibility,
            versions: vec![RelicVersion {
                number: 1,
                copal_file: "01aaa".into(),
                copal_digest: "d1".into(),
                created_at: now,
                author_host: Some("laptop".into()),
            }],
            source: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn a_relic_round_trips_and_a_new_version_replaces_its_row() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let tenant = TenantId::new("ws:t");
        let mut r = relic("ceiling", "ws:t", Visibility::Private);
        assert!(put(&s, &r).await?);
        assert_eq!(get(&s, &tenant, "ceiling").await?, Some(r.clone()));

        r.versions.push(RelicVersion {
            number: 2,
            copal_file: "01bbb".into(),
            copal_digest: "d2".into(),
            created_at: Utc::now(),
            author_host: None,
        });
        assert!(put(&s, &r).await?);
        let back = get(&s, &tenant, "ceiling").await?.unwrap();
        assert_eq!(back.current().map(|v| v.number), Some(2));
        assert_eq!(back.next_version(), 3);
        assert_eq!(list(&s, &tenant).await?.len(), 1, "one relic, one row");
        Ok(())
    }

    #[tokio::test]
    async fn listings_are_per_tenant_and_newest_first() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        let mut older = relic("older", "ws:t", Visibility::Private);
        older.updated_at = Utc::now() - chrono::Duration::hours(1);
        put(&s, &older).await?;
        put(&s, &relic("newer", "ws:t", Visibility::Public)).await?;
        put(&s, &relic("elsewhere", "ws:other", Visibility::Public)).await?;

        let slugs: Vec<String> = list(&s, &TenantId::new("ws:t"))
            .await?
            .into_iter()
            .map(|r| r.slug)
            .collect();
        assert_eq!(slugs, vec!["newer", "older"]);
        assert!(get(&s, &TenantId::new("ws:t"), "elsewhere")
            .await?
            .is_none());
        Ok(())
    }

    #[tokio::test]
    async fn only_public_relics_resolve_without_a_tenant() -> Result<()> {
        let s = Store::connect_memory(EMBED_DIM).await?;
        put(&s, &relic("open", "ws:a", Visibility::Public)).await?;
        put(&s, &relic("closed", "ws:b", Visibility::Private)).await?;
        assert_eq!(
            get_public(&s, "open").await?.map(|r| r.slug),
            Some("open".into())
        );
        assert!(get_public(&s, "closed").await?.is_none());
        assert!(get_public(&s, "missing").await?.is_none());
        Ok(())
    }
}

//! What each repository's manifests declared, as last read (the evidence
//! graph's declared source, through the GitHub App): one row per (workspace,
//! repository), replaced on every reading, so the edges into and out of a
//! repository can be worked out again from every repository's last reading
//! without fetching them all.
//!
//! Written only by the server in owner mode, from what GitHub returned; a
//! member's session may read the rows but not write them (`schema.rs`), so
//! nobody can forge a manifest into the graph's evidence.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use surql::query::builder::Query;
use surql::query::crud::{query_records, upsert_record};
use surql::types::operators::eq;
use surql::types::RecordID;

use antumbra_core::manifest::{Manifest, ManifestSet};
use antumbra_core::provenance::normalize_repo;
use antumbra_core::{Result, TenantId};

use crate::error::map;
use crate::store::Store;

const TABLE: &str = "manifest_set";

#[derive(Serialize, Deserialize)]
struct Row {
    tenant_id: String,
    repo: String,
    commit: String,
    #[serde(default)]
    branch: Option<String>,
    manifests: Vec<Manifest>,
    read_at: String,
}

/// One row per repository in a workspace. A record key is one key across
/// every workspace, so the workspace is part of it.
fn key(tenant: &TenantId, repo: &str) -> String {
    format!("{}|{}", tenant.as_str(), normalize_repo(repo))
}

/// Keep `set` as the repository's latest reading, replacing the one before.
pub async fn upsert(
    store: &Store,
    tenant: &TenantId,
    set: &ManifestSet,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    let rid = RecordID::<()>::new(TABLE, key(tenant, &set.repo).as_str()).map_err(map)?;
    let row = Row {
        tenant_id: tenant.as_str().to_string(),
        repo: normalize_repo(&set.repo),
        commit: set.commit.clone(),
        branch: set.branch.clone(),
        manifests: set.manifests.clone(),
        read_at: now.to_rfc3339(),
    };
    let data: Value = serde_json::to_value(row)?;
    upsert_record(store.client(), &rid, data)
        .await
        .map_err(map)?;
    Ok(())
}

/// Every repository's latest reading in a workspace, in repository order.
pub async fn list(store: &Store, tenant: &TenantId) -> Result<Vec<ManifestSet>> {
    let query = Query::new()
        .select(None)
        .from_table(TABLE)
        .map_err(map)?
        .where_(eq("tenant_id", tenant.as_str()));
    let rows: Vec<Row> = query_records(store.client(), &query).await.map_err(map)?;
    let mut sets: Vec<ManifestSet> = rows
        .into_iter()
        .map(|r| ManifestSet {
            repo: r.repo,
            commit: r.commit,
            branch: r.branch,
            manifests: r.manifests,
        })
        .collect();
    sets.sort_by(|a, b| a.repo.cmp(&b.repo));
    Ok(sets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::EMBED_DIM;
    use antumbra_core::manifest;

    fn set(repo: &str, commit: &str, text: &str) -> ManifestSet {
        ManifestSet {
            repo: repo.to_string(),
            commit: commit.to_string(),
            branch: Some("main".to_string()),
            manifests: manifest::read("Cargo.toml", text).into_iter().collect(),
        }
    }

    /// A reading replaces the one before for that repository, and one
    /// workspace never sees another's rows.
    #[tokio::test]
    async fn the_latest_reading_replaces_the_last_and_stays_in_its_workspace() -> Result<()> {
        let store = Store::connect_memory(EMBED_DIM).await?;
        let acme = TenantId::new("ws:acme");
        let other = TenantId::new("ws:other");
        let now = chrono::Utc::now();

        upsert(
            &store,
            &acme,
            &set(
                "github.com/Acme/Web",
                "aaaaaaa",
                "[package]\nname = \"web\"\n",
            ),
            now,
        )
        .await?;
        upsert(
            &store,
            &acme,
            &set(
                "github.com/acme/web",
                "bbbbbbb",
                "[package]\nname = \"web\"\n\n[dependencies]\norders = \"1\"\n",
            ),
            now,
        )
        .await?;
        upsert(
            &store,
            &acme,
            &set(
                "github.com/acme/orders",
                "ccccccc",
                "[package]\nname = \"orders\"\n",
            ),
            now,
        )
        .await?;
        upsert(
            &store,
            &other,
            &set(
                "github.com/acme/web",
                "ddddddd",
                "[package]\nname = \"web\"\n",
            ),
            now,
        )
        .await?;

        let sets = list(&store, &acme).await?;
        let read: Vec<(&str, &str)> = sets
            .iter()
            .map(|s| (s.repo.as_str(), s.commit.as_str()))
            .collect();
        assert_eq!(
            read,
            [
                ("github.com/acme/orders", "ccccccc"),
                ("github.com/acme/web", "bbbbbbb")
            ]
        );
        assert_eq!(sets[1].manifests[0].depends, ["orders"]);
        assert_eq!(list(&store, &other).await?.len(), 1);
        Ok(())
    }
}

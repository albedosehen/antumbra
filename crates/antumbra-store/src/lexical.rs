//! The lexical leg of recall, for memories and document chunks alike.
//!
//! The leg matches a row on ANY word of the query, ranked by BM25. SurrealDB's
//! plain `@@` matches on every word, which a fragment of a memory's own text
//! satisfies and a question in the caller's words almost never does: on
//! kuskokwim "dependency graph servers" matched one memory and "the dependency
//! graph of my MCP servers" none, so for a natural prompt the leg returned
//! nothing and recall ran on the dense leg alone. `@1,OR@` matches any word,
//! and the score still ranks the rows that hold more of them, and rarer ones,
//! first. Stopwords are dropped first ([`antumbra_core::query::lexical_terms`]),
//! since a word in every row ranks nothing and costs a score per row.

use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde_json::json;

use antumbra_core::query::lexical_terms;
use antumbra_core::{Result, TenantId};

use crate::store::Store;

/// The `k` rows of `table` in `tenant` (and `network`, when given) whose
/// `content` best matches any word of `query_text`, best first, each with its
/// BM25 `score`. `fields` is the projection: `*` for whole rows, or only what
/// the caller reads, since a whole memory carries its embedding. A blank query
/// returns nothing.
pub(crate) async fn any_word<T: DeserializeOwned>(
    store: &Store,
    table: &'static str,
    fields: &'static str,
    tenant: &TenantId,
    query_text: &str,
    k: usize,
    network: Option<&str>,
) -> Result<Vec<T>> {
    let terms = lexical_terms(query_text);
    if terms.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    // The ORDER BY is load-bearing: the match returns every row that matches at
    // all, in record order, so a LIMIT without it keeps an arbitrary k of them.
    let also = if network.is_some() {
        " AND network = $network"
    } else {
        ""
    };
    let surql = format!(
        "SELECT {fields}, search::score(1) AS score FROM {table} \
         WHERE content @1,OR@ $terms AND tenant_id = $tenant{also} \
         ORDER BY score DESC LIMIT {k}"
    );
    let mut vars = BTreeMap::from([
        ("terms".to_string(), json!(terms)),
        ("tenant".to_string(), json!(tenant.as_str())),
    ]);
    if let Some(network) = network {
        vars.insert("network".to_string(), json!(network));
    }
    store.query_rows(&surql, vars).await
}

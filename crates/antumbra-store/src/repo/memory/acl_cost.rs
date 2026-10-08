//! What `memory`'s read rule costs a recall under a record session, measured
//! on the embedded engine at about the size of the user's store: 7,000
//! memories in the session user's own compartment, a dozen compartments of
//! other users beside it, random unit vectors and text full of common words.
//!
//! Each variant of the rule is installed as the owner, then the session signs
//! in as the tenant (the per-request record session the HTTP surface uses)
//! and runs the dense leg, the lexical leg and the whole ranking for fifteen
//! queries. Every variant must rank the same keys as the current rule.
//!
//! A timing harness, so `#[ignore]`d; run it by hand:
//! `cargo test -p antumbra-store --release --lib acl_cost -- --ignored --nocapture`
//! (`ANTUMBRA_ACL_BENCH_N` sets the memory count).

use std::collections::BTreeMap;
use std::time::Instant;

use super::*;
use crate::repo::{compartment, principal};
use crate::schema::{memory_select_rule, EMBED_DIM, OWN_COMPARTMENT};
use antumbra_core::{Compartment, CompartmentId, UserId};

/// xorshift64*, so the corpus is the same on every run without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn vector(&mut self) -> Vec<f32> {
        let v: Vec<f32> = (0..EMBED_DIM)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }

    fn text(&mut self, words: usize) -> String {
        (0..words)
            .map(|_| WORDS[self.below(WORDS.len())])
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Mostly words that recur across a working store, so the lexical leg matches
/// as many rows as it does on the real one.
const WORDS: &[&str] = &[
    "deploy",
    "script",
    "service",
    "memory",
    "recall",
    "shaman",
    "kuskokwim",
    "build",
    "image",
    "commit",
    "branch",
    "merge",
    "test",
    "config",
    "rollback",
    "handle",
    "router",
    "port",
    "dashboard",
    "trace",
    "span",
    "query",
    "index",
    "tenant",
    "session",
    "user",
    "compartment",
    "grant",
    "token",
    "secret",
    "doppler",
    "flux",
    "kubernetes",
    "pod",
    "container",
    "docker",
    "nginx",
    "edge",
    "header",
    "policy",
    "site",
    "page",
    "editor",
    "pixel",
    "stroke",
    "oneiric",
    "antumbra",
    "openobserve",
    "collector",
    "metric",
    "alert",
    "verified",
    "fixed",
    "broke",
    "failed",
    "error",
    "latency",
    "slow",
    "fast",
    "release",
    "version",
    "update",
    "schema",
    "migration",
    "field",
    "table",
    "record",
    "vector",
    "embedding",
    "rerank",
    "floor",
    "chunk",
    "lexical",
    "dense",
    "fusion",
    "pool",
    "score",
    "shon",
    "decision",
    "rule",
    "standing",
    "convention",
    "project",
    "task",
    "active",
];

/// The owner branch as the rule asked it before reading the compartment record
/// directly: a query over the compartment table, run per row. (Limiting that
/// query to the session's tenant, so it can use `compartment_owner_idx`, was
/// measured too and bought nothing: the cost is running a query per row.)
const SUBQUERY_OWNER_BRANCH: &str =
    "compartment IN (SELECT VALUE key FROM compartment WHERE owner = $auth.user AND deleted_at IS NONE)";

/// The rule as it was, with the owner branch put back as a subquery.
fn with_subquery_owner_branch(rule: &str) -> String {
    let lookup =
        format!("({OWN_COMPARTMENT}.owner = $auth.user AND {OWN_COMPARTMENT}.deleted_at IS NONE)");
    rule.replace(&lookup, SUBQUERY_OWNER_BRANCH)
}

/// `rule` installed as memory's select permission: the table's own definition
/// with its select clause swapped.
async fn install(store: &Store, definition: &str, rule: &str) -> Result<()> {
    let start = definition
        .find("FOR select WHERE ")
        .expect("a select permission")
        + 17;
    let end = start
        + definition[start..]
            .find(", FOR create")
            .expect("a create permission");
    let ddl = format!(
        "{}{}{}",
        definition[..start].replacen("DEFINE TABLE memory", "DEFINE TABLE OVERWRITE memory", 1),
        rule,
        &definition[end..]
    );
    store.signin_root().await?;
    store
        .client()
        .query_with_vars(&ddl, BTreeMap::new())
        .await
        .map_err(map)?;
    Ok(())
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a timing harness; run by hand"]
async fn read_rule_cost() -> Result<()> {
    let n: usize = std::env::var("ANTUMBRA_ACL_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7000);
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:bench");
    let user = UserId::new("user:bench");
    principal::provision(&store, &tenant, &user).await?;
    let now = Utc::now();
    for i in 0..12 {
        let other = Compartment::new(
            CompartmentId::new(format!("comp:other-{i}")),
            tenant.clone(),
            UserId::new(format!("user:o{i}")),
            "other",
            now,
        );
        compartment::create(&store, &other).await?;
    }
    let mine = CompartmentId::new("comp:ws:bench:user:bench:default");
    compartment::create(
        &store,
        &Compartment::new(mine.clone(), tenant.clone(), user.clone(), "default", now),
    )
    .await?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let ids: Vec<usize> = (0..n).collect();
    for batch in ids.chunks(200) {
        let memories: Vec<Memory> = batch
            .iter()
            .map(|i| {
                Memory::new(
                    format!("memory:bench-{i}"),
                    tenant.as_str(),
                    MemoryNetwork::World,
                    rng.text(60),
                    0.8,
                    now,
                )
                .in_compartment(mine.as_str())
                .with_embedding(rng.vector())
            })
            .collect();
        futures::future::try_join_all(memories.iter().map(|m| upsert(&store, m))).await?;
    }
    let queries: Vec<(String, Vec<f32>)> = (0..15).map(|_| (rng.text(6), rng.vector())).collect();

    let definition: Vec<String> = store
        .query_rows("RETURN [(INFO FOR DB).tables.memory]", BTreeMap::new())
        .await?;
    let definition = &definition[0];
    let current = memory_select_rule();
    let before = with_subquery_owner_branch(&current);
    assert_ne!(before, current, "the owner branch must be found to swap it");
    let variants = [
        ("owner branch as a subquery (before)", before),
        ("owner branch by record lookup (now)", current),
        (
            "tenant only (lower bound)",
            "tenant_id = $auth.tenant".to_string(),
        ),
    ];

    let k = 30;
    let pool = candidate_pool(k);
    let mut reference: Option<Vec<Vec<String>>> = None;
    println!(
        "{n} memories; k {k}, pool {pool}; medians over {} queries",
        queries.len()
    );
    for (name, rule) in &variants {
        install(&store, definition, rule).await?;
        store.signin(&tenant, &user).await?;
        hybrid_keys(&store, &tenant, &queries[0].0, &queries[0].1, k, None, &[]).await?;
        let (mut dense, mut lexical, mut whole) = (Vec::new(), Vec::new(), Vec::new());
        let mut ranked = Vec::new();
        for (text, vector) in &queries {
            let t = Instant::now();
            let hits = dense_scored(&store, &tenant, vector, pool, None, &[]).await?;
            dense.push(t.elapsed().as_secs_f64() * 1e3);
            assert!(
                !hits.is_empty(),
                "{name}: the session must see its own memories"
            );
            let t = Instant::now();
            sparse_keys(&store, &tenant, text, pool, None).await?;
            lexical.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            ranked.push(hybrid_keys(&store, &tenant, text, vector, k, None, &[]).await?);
            whole.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let same = match &reference {
            None => {
                reference = Some(ranked);
                "reference".to_string()
            }
            Some(r) => format!("same keys as before: {}", r == &ranked),
        };
        println!(
            "{name:<34} dense {:>6.1} ms  lexical {:>6.1} ms  ranking {:>6.1} ms  ({same})",
            median(dense),
            median(lexical),
            median(whole)
        );
    }
    store.signin_root().await?;
    let (mut dense, mut whole) = (Vec::new(), Vec::new());
    for (text, vector) in &queries {
        let t = Instant::now();
        dense_scored(&store, &tenant, vector, pool, None, &[]).await?;
        dense.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        hybrid_keys(&store, &tenant, text, vector, k, None, &[]).await?;
        whole.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{:<34} dense {:>6.1} ms  ranking {:>6.1} ms",
        "owner session, no rule (root)",
        median(dense),
        median(whole)
    );
    Ok(())
}

/// What listing one small compartment costs beside a large one, under the
/// record session: the handoffs a session start announces, read from a
/// compartment of six among 7,000 memories. Without an index on the
/// compartment the read examines (and runs the read rule on) every memory of
/// the workspace; production traced `tool handoffs` at 190-370 ms.
/// `cargo test -p antumbra-store --release --lib compartment_list_cost -- --ignored --nocapture`
#[tokio::test(flavor = "multi_thread")]
#[ignore = "a timing harness; run by hand"]
async fn compartment_list_cost() -> Result<()> {
    let n: usize = std::env::var("ANTUMBRA_ACL_BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7000);
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:bench");
    let user = UserId::new("user:bench");
    principal::provision(&store, &tenant, &user).await?;
    let now = Utc::now();
    let mine = CompartmentId::new("comp:ws:bench:user:bench:default");
    let handoffs = CompartmentId::new("comp:ws:bench:user:bench:handoff");
    for (id, name) in [(&mine, "default"), (&handoffs, "handoff")] {
        compartment::create(
            &store,
            &Compartment::new(id.clone(), tenant.clone(), user.clone(), name, now),
        )
        .await?;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let ids: Vec<usize> = (0..n + 6).collect();
    for batch in ids.chunks(200) {
        let memories: Vec<Memory> = batch
            .iter()
            .map(|&i| {
                let home = if i < n { &mine } else { &handoffs };
                Memory::new(
                    format!("memory:bench-{i}"),
                    tenant.as_str(),
                    MemoryNetwork::World,
                    rng.text(60),
                    0.8,
                    now,
                )
                .in_compartment(home.as_str())
                .with_embedding(rng.vector())
            })
            .collect();
        futures::future::try_join_all(memories.iter().map(|m| upsert(&store, m))).await?;
    }

    async fn timed(
        store: &Store,
        tenant: &TenantId,
        compartment: &CompartmentId,
    ) -> Result<(f64, usize)> {
        list_by_compartment(store, tenant, compartment).await?;
        let mut ms = Vec::new();
        let mut found = 0;
        for _ in 0..15 {
            let t = Instant::now();
            found = list_by_compartment(store, tenant, compartment).await?.len();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        Ok((median(ms), found))
    }

    println!("{n} memories in one compartment, 6 in another; medians of 15 lists");
    store.signin_root().await?;
    store
        .client()
        .query_with_vars(
            "REMOVE INDEX IF EXISTS memory_tenant_compartment_idx ON memory",
            BTreeMap::new(),
        )
        .await
        .map_err(map)?;
    store.signin(&tenant, &user).await?;
    let (without, found) = timed(&store, &tenant, &handoffs).await?;
    assert_eq!(found, 6);
    println!("{:<34} {without:>7.1} ms", "no compartment index (before)");

    store.signin_root().await?;
    store.ensure_schema().await?;
    store.signin(&tenant, &user).await?;
    let (with, found) = timed(&store, &tenant, &handoffs).await?;
    assert_eq!(found, 6);
    println!("{:<34} {with:>7.1} ms", "tenant+compartment index (now)");
    Ok(())
}

//! What the chunk search costs a session that can read little of its
//! workspace, measured on the embedded engine: thousands of memories cut into
//! pieces in one user's own compartment, a handful in the shared pool, and a
//! second member of the workspace who owns none of them.
//!
//! On kuskokwim a recall by an identity that could see almost nothing spent
//! 3.8 to 4.6 s in `recall chunks`, against 180 ms for the workspace's owner,
//! while its dense leg over whole memories took 20 ms.
//!
//! Timing harnesses, so `#[ignore]`d; run them by hand:
//! `cargo test -p antumbra-store --release --lib chunk_acl_cost -- --ignored --nocapture`
//! `cargo test -p antumbra-store --release --lib readable_set_cost -- --ignored --nocapture`
//! (`ANTUMBRA_CHUNK_BENCH_MEMORIES` sets the memory count, eight pieces each).

use std::time::Instant;

use super::*;
use crate::repo::{compartment, memory, principal};
use crate::schema::EMBED_DIM;
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

    fn vector(&mut self) -> Vec<f32> {
        let v: Vec<f32> = (0..EMBED_DIM)
            .map(|_| (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / norm).collect()
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

const PIECES: usize = 8;

/// The corpus both harnesses measure: `n` memories in the owner's own
/// compartment and ten in the shared pool, eight pieces each, and nine
/// query vectors.
struct Corpus {
    store: Store,
    tenant: TenantId,
    owner: UserId,
    member: UserId,
    queries: Vec<Vec<f32>>,
}

async fn corpus() -> Result<Corpus> {
    let n: usize = std::env::var("ANTUMBRA_CHUNK_BENCH_MEMORIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let store = Store::connect_memory(EMBED_DIM).await?;
    let tenant = TenantId::new("ws:bench");
    let owner = UserId::new("user:owner");
    let member = UserId::new("user:member");
    principal::provision(&store, &tenant, &owner).await?;
    principal::provision(&store, &tenant, &member).await?;
    let now = Utc::now();
    let mine = CompartmentId::new("comp:ws:bench:user:owner:default");
    compartment::create(
        &store,
        &Compartment::new(mine.clone(), tenant.clone(), owner.clone(), "default", now),
    )
    .await?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let shared = 10;
    let started = Instant::now();
    for i in 0..n + shared {
        let mut m = Memory::new(
            format!("memory:bench-{i}"),
            tenant.as_str(),
            MemoryNetwork::World,
            format!("bench memory {i}"),
            0.8,
            now,
        )
        .with_embedding(rng.vector());
        if i < n {
            m = m.in_compartment(mine.as_str());
        }
        memory::upsert(&store, &m).await?;
        let vectors = (0..PIECES).map(|_| rng.vector()).collect();
        replace(&store, &m, "hash", vectors).await?;
    }
    println!(
        "{} memories, {} pieces ({} in the shared pool), written in {:.0} s",
        n + shared,
        (n + shared) * PIECES,
        shared * PIECES,
        started.elapsed().as_secs_f64()
    );
    let queries = (0..9).map(|_| rng.vector()).collect();
    Ok(Corpus {
        store,
        tenant,
        owner,
        member,
        queries,
    })
}

/// The index walk alone, for each session: what the read rule costs it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "a timing harness; run by hand"]
async fn chunk_acl_cost() -> Result<()> {
    let Corpus {
        store,
        tenant,
        owner,
        member,
        queries,
    } = corpus().await?;
    let k = 30;
    let rows = k * ROWS_PER_MEMORY;
    let long = Duration::from_secs(60);

    for (name, who) in [
        ("owner (reads every piece)", Some(&owner)),
        ("member (reads the shared pool only)", Some(&member)),
        ("root (no rule)", None),
    ] {
        match who {
            Some(user) => store.signin(&tenant, user).await?,
            None => store.signin_root().await?,
        }
        walk(&store, &tenant, &queries[0], rows, None, long).await?;
        let mut times = Vec::new();
        let mut found = 0;
        for q in &queries {
            let t = Instant::now();
            found = walk(&store, &tenant, q, rows, None, long).await?.len();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{name:<38} walk {:>8.1} ms  ({found} pieces)",
            median(times)
        );
    }
    store.signin_root().await?;
    Ok(())
}

/// The chunk search a session gets, end to end: the exact search over the
/// shared pool and the compartments it reads when they are few, the walk
/// otherwise. Prints which index each exact query reads through, then the
/// first search for each session (for the owner, the one that finds it reads
/// too much and remembers it) and the median of the rest.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "a timing harness; run by hand"]
async fn readable_set_cost() -> Result<()> {
    let Corpus {
        store,
        tenant,
        owner,
        member,
        queries,
    } = corpus().await?;
    for (name, user) in [("owner", &owner), ("member", &member)] {
        store.signin(&tenant, user).await?;
        let readable = readable_compartments(&store, &tenant, user).await?;
        for (label, place) in [
            ("shared pool", "compartment = NONE"),
            ("compartments", "compartment INSIDE $readable"),
        ] {
            let surql = format!(
                "SELECT memory FROM memory_chunk WHERE tenant_id = $tenant AND {place} EXPLAIN"
            );
            let vars = BTreeMap::from([
                ("tenant".to_string(), json!(tenant.as_str())),
                ("readable".to_string(), json!(readable)),
            ]);
            let plan = serde_json::to_string(
                &store
                    .client()
                    .query_with_vars(&surql, vars)
                    .await
                    .map_err(map)?,
            )?;
            let index = plan
                .split("\"index\":\"")
                .nth(1)
                .and_then(|s| s.split('"').next())
                .unwrap_or("none");
            println!("{name:<7} {label:<13} reads through {index}");
        }
        let t = Instant::now();
        let first = nearest_scored(&store, &tenant, &queries[0], 30, None).await?;
        let first_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut rest = Vec::new();
        for q in &queries[1..] {
            let t = Instant::now();
            nearest_scored(&store, &tenant, q, 30, None).await?;
            rest.push(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{name:<7} {} compartments  first search {first_ms:>7.1} ms ({} pieces)  then {:>7.1} ms",
            readable.len(),
            first.len(),
            median(rest)
        );
    }
    store.signin_root().await?;
    Ok(())
}

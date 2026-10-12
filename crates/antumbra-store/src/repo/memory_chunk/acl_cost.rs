//! What the chunk search costs a session that can read little of its
//! workspace, measured on the embedded engine: thousands of memories cut into
//! pieces in one user's own compartment, a handful in the shared pool, and a
//! second member of the workspace who owns none of them.
//!
//! On kuskokwim a recall by an identity that could see almost nothing spent
//! 3.8 to 4.6 s in `recall chunks`, against 180 ms for the workspace's owner,
//! while its dense leg over whole memories took 20 ms.
//!
//! A timing harness, so `#[ignore]`d; run it by hand:
//! `cargo test -p antumbra-store --release --lib chunk_acl_cost -- --ignored --nocapture`
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

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a timing harness; run by hand"]
async fn chunk_acl_cost() -> Result<()> {
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
    let queries: Vec<Vec<f32>> = (0..9).map(|_| rng.vector()).collect();
    let k = 30;

    for (name, who) in [
        ("owner (reads every piece)", Some(&owner)),
        ("member (reads the shared pool only)", Some(&member)),
        ("root (no rule)", None),
    ] {
        match who {
            Some(user) => store.signin(&tenant, user).await?,
            None => store.signin_root().await?,
        }
        nearest(&store, &tenant, &queries[0], k, None).await?;
        let (mut full, mut scored) = (Vec::new(), Vec::new());
        let mut found = 0;
        for q in &queries {
            let t = Instant::now();
            found = nearest(&store, &tenant, q, k, None).await?.len();
            full.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            nearest_scored(&store, &tenant, q, k, None).await?;
            scored.push(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{name:<38} nearest {:>8.1} ms  nearest_scored {:>8.1} ms  ({found} pieces)",
            median(full),
            median(scored)
        );
    }
    store.signin_root().await?;
    Ok(())
}

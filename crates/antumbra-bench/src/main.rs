//! # antumbra-bench — multi-domain retrieval-quality benchmark
//!
//! A small evaluation harness (a binary, **not** a criterion micro-bench) that
//! compares *embedder configurations* on retrieval quality over a labeled
//! multi-domain corpus, reporting `recall@k` and MRR. It exists so the
//! **Matryoshka generalist embedder is a selectable config, not an exclusive
//! swap**: the same battery scores the built-in deterministic baseline and a
//! real Matryoshka HTTP endpoint side by side, so an operator can see which
//! serves a given scenario better instead of guessing.
//!
//! ## What it measures
//!
//! The corpus (`corpus/multidomain.json`, embedded via [`include_str!`]) spans
//! three deliberately different retrieval regimes:
//! - **code** — NL → code identifiers (e.g. "function that reverses a string"
//!   → `fn reverse_string`),
//! - **finance** — company/ticker recall (e.g. "Apple quarterly earnings" →
//!   the `AAPL` doc),
//! - **news** — narrative / insider entities (a person/org → the doc naming
//!   them).
//!
//! Each domain carries a handful of `(query, relevant_doc_id)` labels. For each
//! embedder config the harness spins up a fresh in-memory store
//! ([`Store::connect_memory`]), embeds + stores every doc as a memory, then for
//! each labeled query embeds it and runs [`recall_hybrid`] (dense HNSW fused
//! with the BM25 sparse leg via RRF), and computes the metrics.
//!
//! ## Configs
//!
//! - The built-in [`FixedEmbedder`] is always run as a **deterministic
//!   baseline**, so the bench (and its test) execute with no network. Fused
//!   with the lexical sparse leg it produces stable, sane numbers; it is the
//!   config CI exercises.
//! - A real Matryoshka HTTP endpoint is **opt-in via environment variables**
//!   (`ANTUMBRA_BENCH_EMBED_URL` / `_MODEL` / `_KEY` / `_SOURCE_DIM`). When the
//!   URL is unset the HTTP config is skipped and only the baseline runs. With
//!   `_SOURCE_DIM` set, the endpoint is treated as a Matryoshka model whose
//!   renormalized `EMBED_DIM` prefix is stored — the very path this feature
//!   adds.
//!
//! ## A corpus of your own memories
//!
//! `ANTUMBRA_BENCH_LABELS` names a label file from `scripts/d2-labels.sh`
//! instead: every distinct memory becomes a document, and every positive pair
//! a query whose answer is its memory. The query is twelve words cut from about
//! 60% of the way through the memory and excised from it, so neither leg can
//! match it verbatim, and it sits past the first few hundred tokens a short
//! embedder reads. Each config is scored twice, hybrid as recall runs and dense
//! alone, so what the embedder itself contributes shows.
//!
//! ## Chunks (ADR-0025)
//!
//! `ANTUMBRA_BENCH_CHUNK_CHARS=<n>` stores each document as pieces of about
//! `n` characters, each a memory of its own, and scores a document at the rank
//! of its best piece: the measurement the chunk index was decided on.
//! `ANTUMBRA_BENCH_CHUNK_INDEX=1` stores it the way the server does instead:
//! whole, with its pieces in the chunk index at the server's size, recalled
//! through the same dense leg, so the number is the deployed path's.
//!
//! ## Questions, and embedders that want prefixes
//!
//! `scripts/question-labels.sh` writes a label file of the same shape whose
//! queries are questions a chat model wrote for each memory, in its own words,
//! with distractor memories that have no question. Every labeled run also
//! scores BM25 alone, the no-model baseline. Embedders trained for asymmetric
//! search read a marker before the text: `ANTUMBRA_BENCH_QUERY_PREFIX` goes
//! before every query (`query: ` for e5) and `ANTUMBRA_BENCH_DOC_PREFIX`
//! before every document and piece (`passage: `).
//!
//! `ANTUMBRA_BENCH_CALIBRATE=1` ranks the dense leg as the server does, by
//! each memory's similarity above its own baseline over the probe texts,
//! instead of by raw cosine.
//!
//! ## Run
//!
//! ```text
//! cargo run -p antumbra-bench
//! # with a real Matryoshka endpoint:
//! ANTUMBRA_BENCH_EMBED_URL=http://localhost:8080/v1/embeddings \
//! ANTUMBRA_BENCH_EMBED_MODEL=bge-m3 ANTUMBRA_BENCH_EMBED_SOURCE_DIM=1024 \
//!   cargo run -p antumbra-bench
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::Deserialize;

use antumbra_core::calibrate::PROBE_TEXTS;
use antumbra_core::chunk::{cut_hash, split, MEMORY_CHUNK_CHARS};
use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{Memory, MemoryNetwork, Result, TenantId};
use antumbra_store::repo::{memory, memory_chunk};
use antumbra_store::{Store, EMBED_DIM};

/// The labeled corpus, baked into the binary so the harness is self-contained
/// (no working-directory dependence for the run or the test).
const CORPUS_JSON: &str = include_str!("../corpus/multidomain.json");

/// The `recall@k` cutoffs reported, plus the deepest pool the harness requests
/// from recall. Ascending; the largest also bounds how many candidates we fetch.
const K_VALUES: [usize; 5] = [1, 3, 5, 10, 30];

/// The id prefix under which corpus docs are stored as memories; stripped to map
/// a recalled `Memory` back to its labeled doc id.
const ID_PREFIX: &str = "memory:";

// --- corpus ----------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct Corpus {
    domains: Vec<Domain>,
}

#[derive(Debug, Clone, Deserialize)]
struct Domain {
    name: String,
    documents: Vec<Document>,
    labels: Vec<Label>,
}

#[derive(Debug, Clone, Deserialize)]
struct Document {
    id: String,
    text: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Label {
    query: String,
    relevant_doc_id: String,
}

impl Corpus {
    /// Parse the embedded corpus.
    fn load() -> Result<Self> {
        serde_json::from_str(CORPUS_JSON)
            .map_err(|e| antumbra_core::AntumbraError::other(format!("bad bench corpus: {e}")))
    }

    /// A corpus from a D-2 label file: every distinct memory is a document,
    /// and every positive pair a query whose answer is its memory.
    fn from_labels(text: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Pair {
            query: String,
            memory: String,
            relevant: bool,
        }
        let pairs: Vec<Pair> = serde_json::from_str(text)
            .map_err(|e| antumbra_core::AntumbraError::other(format!("bad label file: {e}")))?;
        let mut ids: BTreeMap<String, String> = BTreeMap::new();
        let mut documents = Vec::new();
        for pair in &pairs {
            if !ids.contains_key(&pair.memory) {
                let id = format!("m{}", ids.len());
                ids.insert(pair.memory.clone(), id.clone());
                documents.push(Document {
                    id,
                    text: pair.memory.clone(),
                });
            }
        }
        let labels = pairs
            .iter()
            .filter(|pair| pair.relevant)
            .map(|pair| Label {
                query: pair.query.clone(),
                relevant_doc_id: ids[&pair.memory].clone(),
            })
            .collect();
        Ok(Corpus {
            domains: vec![Domain {
                name: "memories".into(),
                documents,
                labels,
            }],
        })
    }
}

/// Which retrieval the queries run through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Dense and BM25 fused by rank, as recall runs.
    Hybrid,
    /// The embedding alone: what the embedder itself contributes.
    Dense,
    /// BM25 alone, no model: the baseline a constructed benchmark is read
    /// against before any model is.
    Lexical,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Hybrid => "hybrid",
            Mode::Dense => "dense",
            Mode::Lexical => "lexical, no model",
        }
    }
}

// --- metrics ----------------------------------------------------------------

/// Accumulated retrieval metrics over a set of queries. `recall@k` is the
/// fraction of queries whose relevant doc appears in the top `k`; MRR is the
/// mean reciprocal rank of the relevant doc (0 when it is outside the deepest
/// pool considered).
#[derive(Debug, Clone, Default)]
struct Metrics {
    queries: usize,
    /// Hit counts keyed by `k` (a query "hits@k" when its relevant doc ranks at
    /// position `<= k`).
    hits_at_k: BTreeMap<usize, usize>,
    reciprocal_rank_sum: f64,
}

impl Metrics {
    /// Fold in one query whose relevant doc landed at `rank` (1-based) in the
    /// recalled list, or `None` if it was not recalled at all.
    fn record(&mut self, rank: Option<usize>) {
        self.queries += 1;
        if let Some(r) = rank {
            for &k in &K_VALUES {
                if r <= k {
                    *self.hits_at_k.entry(k).or_default() += 1;
                }
            }
            self.reciprocal_rank_sum += 1.0 / r as f64;
        } else {
            // Ensure every k has an entry even when nothing hit, so the table is
            // rectangular.
            for &k in &K_VALUES {
                self.hits_at_k.entry(k).or_default();
            }
        }
    }

    /// `recall@k` in `[0, 1]` (0 when no queries were scored).
    fn recall_at(&self, k: usize) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        *self.hits_at_k.get(&k).unwrap_or(&0) as f64 / self.queries as f64
    }

    /// Mean reciprocal rank in `[0, 1]`.
    fn mrr(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.reciprocal_rank_sum / self.queries as f64
    }

    /// Fold another `Metrics` into this one (for the overall rollup).
    fn merge(&mut self, other: &Metrics) {
        self.queries += other.queries;
        self.reciprocal_rank_sum += other.reciprocal_rank_sum;
        for &k in &K_VALUES {
            *self.hits_at_k.entry(k).or_default() += other.hits_at_k.get(&k).copied().unwrap_or(0);
        }
    }
}

/// The full result of scoring one embedder config: per-domain metrics keyed by
/// domain name, plus the overall rollup.
#[derive(Debug, Clone)]
struct ConfigReport {
    config_name: String,
    per_domain: BTreeMap<String, Metrics>,
    overall: Metrics,
}

// --- evaluation -------------------------------------------------------------

/// Score one embedder over the whole corpus: load a fresh in-memory store,
/// embed and store every doc, then run hybrid recall for each labeled query and
/// tally the metrics per domain and overall.
async fn evaluate(
    config_name: &str,
    embedder: &dyn Embedder,
    corpus: &Corpus,
    mode: Mode,
) -> Result<ConfigReport> {
    let deepest = K_VALUES.iter().copied().max().unwrap_or(10);
    let tenant = TenantId::new("ws:bench");
    let now = chrono::Utc::now();

    let mut per_domain: BTreeMap<String, Metrics> = BTreeMap::new();
    let mut overall = Metrics::default();

    for domain in &corpus.domains {
        // Each domain gets its own store so recall is confined to in-domain docs
        // (a clean per-domain measurement; the overall rollup sums the domains).
        let store = Store::connect_memory(EMBED_DIM).await?;

        for doc in &domain.documents {
            if chunk_index() {
                index(&store, embedder, &tenant, doc, now).await?;
                continue;
            }
            // Whole, or in chunks each stored on its own, `doc~j`, when
            // ANTUMBRA_BENCH_CHUNK_CHARS asks: a long memory's one vector blurs
            // a passage from its middle, a chunk's vector does not.
            let pieces = match chunk_chars() {
                Some(size) => split(&doc.text, size),
                None => vec![doc.text.clone()],
            };
            let whole = pieces.len() == 1;
            for (j, piece) in pieces.into_iter().enumerate() {
                let emb = embedder.embed(&as_document(&piece)).await?;
                let id = if whole {
                    format!("{ID_PREFIX}{}", doc.id)
                } else {
                    format!("{ID_PREFIX}{}{CHUNK_MARK}{j}", doc.id)
                };
                let mem = Memory::new(id, tenant.clone(), MemoryNetwork::World, piece, 1.0, now)
                    .with_embedding(emb);
                memory::upsert(&store, &mem).await?;
            }
        }

        // Chunks of one document compete for the same ranks, so fetch deeper and
        // keep each document's best chunk.
        let fetch = if chunk_chars().is_some() && !chunk_index() {
            deepest * 10
        } else {
            deepest
        };
        // The probes the server calibrates its dense leg with, when asked.
        let probes = if calibrated() {
            let mut out = Vec::with_capacity(PROBE_TEXTS.len());
            for text in PROBE_TEXTS {
                out.push(embedder.embed(&as_query(text)).await?);
            }
            out
        } else {
            Vec::new()
        };
        let mut metrics = Metrics::default();
        for label in &domain.labels {
            let qv = embedder.embed(&as_query(&label.query)).await?;
            let hits = match mode {
                Mode::Lexical => {
                    memory::sparse_recall(&store, &tenant, &label.query, fetch, None).await?
                }
                Mode::Hybrid => {
                    memory::recall_hybrid(&store, &tenant, &label.query, &qv, fetch, None, &probes)
                        .await?
                }
                // A blank query text is the dense leg alone, chunk leg and all.
                Mode::Dense if chunk_index() => {
                    memory::recall_hybrid(&store, &tenant, "", &qv, fetch, None, &probes).await?
                }
                Mode::Dense => memory::recall(&store, &tenant, &qv, fetch, None).await?,
            };
            let rank = rank_of(&documents_of(&hits), &label.relevant_doc_id);
            metrics.record(rank);
        }

        overall.merge(&metrics);
        per_domain.insert(domain.name.clone(), metrics);
    }

    let chunked = if chunk_index() {
        format!(", the chunk index ({MEMORY_CHUNK_CHARS} chars)")
    } else {
        chunk_chars().map_or(String::new(), |n| format!(", chunks of {n} chars"))
    };
    Ok(ConfigReport {
        config_name: format!("{config_name} / {}{chunked}{}", mode.name(), prefixes()),
        per_domain,
        overall,
    })
}

/// What separates a document's id from its chunk's number.
const CHUNK_MARK: char = '~';

/// Whether to rank the dense leg as the server does, by each memory's
/// similarity above its own baseline over the probe texts
/// (`antumbra_core::calibrate`), rather than by raw cosine.
fn calibrated() -> bool {
    std::env::var("ANTUMBRA_BENCH_CALIBRATE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The prefixes in force, and the calibration, for a report's name.
fn prefixes() -> String {
    let query = std::env::var("ANTUMBRA_BENCH_QUERY_PREFIX").unwrap_or_default();
    let document = std::env::var("ANTUMBRA_BENCH_DOC_PREFIX").unwrap_or_default();
    let calibration = if calibrated() { ", calibrated" } else { "" };
    if query.is_empty() && document.is_empty() {
        calibration.to_string()
    } else {
        format!(", prefixes {query:?} / {document:?}{calibration}")
    }
}

/// `text` as a query, behind `ANTUMBRA_BENCH_QUERY_PREFIX` when one is set.
fn as_query(text: &str) -> String {
    format!(
        "{}{text}",
        std::env::var("ANTUMBRA_BENCH_QUERY_PREFIX").unwrap_or_default()
    )
}

/// `text` as a document, behind `ANTUMBRA_BENCH_DOC_PREFIX` when one is set.
fn as_document(text: &str) -> String {
    format!(
        "{}{text}",
        std::env::var("ANTUMBRA_BENCH_DOC_PREFIX").unwrap_or_default()
    )
}

/// Whether to store documents the way the server does: whole, with their
/// pieces in the chunk index.
fn chunk_index() -> bool {
    std::env::var("ANTUMBRA_BENCH_CHUNK_INDEX").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Store `doc` the way the server does: whole, and, when it is longer than one
/// piece, its pieces in the chunk index, as the server's keeper cuts them.
async fn index(
    store: &Store,
    embedder: &dyn Embedder,
    tenant: &TenantId,
    doc: &Document,
    now: chrono::DateTime<chrono::Utc>,
) -> Result<()> {
    let whole = embedder.embed(&as_document(&doc.text)).await?;
    let mem = Memory::new(
        format!("{ID_PREFIX}{}", doc.id),
        tenant.clone(),
        MemoryNetwork::World,
        doc.text.clone(),
        1.0,
        now,
    )
    .with_embedding(whole);
    memory::upsert(store, &mem).await?;
    let pieces = split(&doc.text, MEMORY_CHUNK_CHARS);
    if pieces.len() > 1 {
        let mut vectors = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            vectors.push(embedder.embed(&as_document(piece)).await?);
        }
        memory_chunk::replace(store, &mem, &cut_hash(&doc.text), vectors).await?;
    }
    Ok(())
}

fn chunk_chars() -> Option<usize> {
    std::env::var("ANTUMBRA_BENCH_CHUNK_CHARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n: &usize| n > 0)
}

/// A recalled list as documents, each at the rank of its best chunk.
fn documents_of(hits: &[Memory]) -> Vec<Memory> {
    let mut seen = std::collections::BTreeSet::new();
    hits.iter()
        .filter(|m| {
            let doc =
                m.id.as_str()
                    .split(CHUNK_MARK)
                    .next()
                    .unwrap_or("")
                    .to_string();
            seen.insert(doc)
        })
        .map(|m| {
            let mut m = m.clone();
            let doc =
                m.id.as_str()
                    .split(CHUNK_MARK)
                    .next()
                    .unwrap_or("")
                    .to_string();
            m.id = doc.into();
            m
        })
        .collect()
}

/// The 1-based rank of `relevant_doc_id` in a recalled list (mapping each
/// memory id back to its doc id by stripping the store prefix), or `None` if it
/// is absent.
fn rank_of(hits: &[Memory], relevant_doc_id: &str) -> Option<usize> {
    hits.iter()
        .position(|m| {
            m.id.as_str()
                .strip_prefix(ID_PREFIX)
                .unwrap_or(m.id.as_str())
                == relevant_doc_id
        })
        .map(|i| i + 1)
}

// --- reporting --------------------------------------------------------------

/// Render the comparison table: one block per config, rows = domains (+ ALL),
/// columns = recall@k and MRR.
fn render_table(reports: &[ConfigReport]) -> String {
    let mut out = String::new();
    out.push_str("\nAntumbra multi-domain retrieval benchmark (recall@k + MRR)\n");
    out.push_str("higher is better; metrics in [0,1]\n");

    for report in reports {
        out.push_str(&format!("\n== config: {} ==\n", report.config_name));
        // Header.
        out.push_str(&format!("{:<10}", "domain"));
        for &k in &K_VALUES {
            out.push_str(&format!("  R@{:<5}", k));
        }
        out.push_str(&format!("  {:<6}\n", "MRR"));

        // Per-domain rows (BTreeMap keeps domains in a stable order).
        for (domain, m) in &report.per_domain {
            out.push_str(&row(domain, m));
        }
        // Overall rollup.
        out.push_str(&row("ALL", &report.overall));
    }
    out
}

/// One formatted metrics row.
fn row(label: &str, m: &Metrics) -> String {
    let mut s = format!("{label:<10}");
    for &k in &K_VALUES {
        s.push_str(&format!("  {:<6.3}", m.recall_at(k)));
    }
    s.push_str(&format!("  {:<6.3}\n", m.mrr()));
    s
}

// --- config selection -------------------------------------------------------

/// Build the list of embedder configs to score. The deterministic baseline is
/// always present; the Matryoshka HTTP config is appended only when
/// `ANTUMBRA_BENCH_EMBED_URL` is set (so the bench runs offline by default).
fn configs() -> Vec<(String, Arc<dyn Embedder>)> {
    let mut v: Vec<(String, Arc<dyn Embedder>)> = vec![(
        "FixedEmbedder(baseline)".to_string(),
        Arc::new(FixedEmbedder::new(EMBED_DIM)) as Arc<dyn Embedder>,
    )];

    if let Ok(url) = std::env::var("ANTUMBRA_BENCH_EMBED_URL") {
        let model = std::env::var("ANTUMBRA_BENCH_EMBED_MODEL")
            .unwrap_or_else(|_| "all-MiniLM-L6-v2".to_string());
        let key = std::env::var("ANTUMBRA_BENCH_EMBED_KEY").ok();
        let source_dim = std::env::var("ANTUMBRA_BENCH_EMBED_SOURCE_DIM")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let label = match source_dim {
            Some(n) => format!("HttpEmbedder(matryoshka src={n}) {model}"),
            None => format!("HttpEmbedder(strict) {model}"),
        };
        v.push((
            label,
            Arc::new(antumbra_embed::HttpEmbedder::new_with_dim(
                url, model, key, source_dim,
            )) as Arc<dyn Embedder>,
        ));
    }
    v
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let corpus = match std::env::var("ANTUMBRA_BENCH_LABELS") {
        Ok(path) => Corpus::from_labels(
            &std::fs::read_to_string(&path)
                .map_err(|e| antumbra_core::AntumbraError::other(format!("read {path}: {e}")))?,
        )?,
        Err(_) => Corpus::load()?,
    };
    let configs = configs();

    let mut reports = Vec::with_capacity(configs.len() * 2 + 1);
    if std::env::var("ANTUMBRA_BENCH_LABELS").is_ok() {
        // The lexical leg reads no vector, so the baseline embedder stands in.
        let (_, baseline) = &configs[0];
        reports.push(evaluate("BM25", baseline.as_ref(), &corpus, Mode::Lexical).await?);
    }
    // On a label file the deterministic baseline scores near zero and says
    // nothing, and a run is a recall per query per config: an endpoint's run
    // leaves it out. Without an endpoint it is the only config, and runs.
    let labeled = std::env::var("ANTUMBRA_BENCH_LABELS").is_ok();
    let skip = usize::from(labeled && configs.len() > 1);
    for (name, embedder) in configs.iter().skip(skip) {
        for mode in [Mode::Hybrid, Mode::Dense] {
            reports.push(evaluate(name, embedder.as_ref(), &corpus, mode).await?);
        }
    }

    print!("{}", render_table(&reports));
    if configs.len() == 1 {
        println!(
            "\n(only the deterministic baseline ran; set ANTUMBRA_BENCH_EMBED_URL \
             — and ANTUMBRA_BENCH_EMBED_SOURCE_DIM for a Matryoshka endpoint — to compare)"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The harness runs end-to-end over the FixedEmbedder baseline and produces
    /// sane metrics: every metric is in `[0, 1]`, every domain is present, and
    /// `recall@k` is monotonic non-decreasing in `k`.
    #[tokio::test]
    async fn baseline_runs_and_reports_sane_metrics() {
        let corpus = Corpus::load().expect("corpus parses");
        assert_eq!(corpus.domains.len(), 3, "code / finance / news");

        let embedder = FixedEmbedder::new(EMBED_DIM);
        let report = evaluate("baseline", &embedder, &corpus, Mode::Hybrid)
            .await
            .expect("eval runs");

        // All three domains scored.
        for d in ["code", "finance", "news"] {
            let m = report.per_domain.get(d).expect("domain present");
            assert!(m.queries > 0, "{d} had queries");
            // recall@k in [0,1] and monotonic non-decreasing in k; MRR in [0,1].
            let mut prev = 0.0;
            for &k in &K_VALUES {
                let r = m.recall_at(k);
                assert!((0.0..=1.0).contains(&r), "recall@{k}={r} in range for {d}");
                assert!(r + 1e-9 >= prev, "recall@{k} monotonic for {d}");
                prev = r;
            }
            let mrr = m.mrr();
            assert!((0.0..=1.0).contains(&mrr), "mrr={mrr} in range for {d}");
        }

        // Overall rollup sums the per-domain query counts and stays in range.
        let total_q: usize = report.per_domain.values().map(|m| m.queries).sum();
        assert_eq!(report.overall.queries, total_q, "rollup sums queries");
        for &k in &K_VALUES {
            let r = report.overall.recall_at(k);
            assert!((0.0..=1.0).contains(&r), "overall recall@{k} in range");
        }

        // The fused (dense + BM25) baseline should actually retrieve something:
        // the lexically-grounded corpus means recall@10 overall must be > 0.
        assert!(
            report.overall.recall_at(10) > 0.0,
            "baseline recalls at least some relevant docs"
        );

        // And the rendered table mentions every domain + the rollup, so the
        // operator-facing output is well-formed.
        let table = render_table(&[report]);
        for needle in ["code", "finance", "news", "ALL", "MRR"] {
            assert!(table.contains(needle), "table mentions {needle}");
        }
    }

    /// A label file becomes one domain: each distinct memory a document, each
    /// positive pair a query answered by its memory; a negative pair adds no
    /// query. Both modes run over it.
    #[tokio::test]
    async fn a_label_file_is_a_corpus_and_both_modes_score_it() {
        let labels = serde_json::json!([
            {"query": "the outbox pattern", "memory": "orders write through an outbox table", "relevant": true},
            {"query": "the outbox pattern", "memory": "parcels leave the warehouse at noon", "relevant": false},
            {"query": "parcels leave", "memory": "parcels leave the warehouse at noon", "relevant": true},
            {"query": "parcels leave", "memory": "orders write through an outbox table", "relevant": false}
        ]);
        let corpus = Corpus::from_labels(&labels.to_string()).expect("label file parses");
        let domain = &corpus.domains[0];
        assert_eq!(domain.documents.len(), 2);
        assert_eq!(domain.labels.len(), 2);
        assert_eq!(domain.labels[1].relevant_doc_id, "m1");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        for mode in [Mode::Hybrid, Mode::Dense, Mode::Lexical] {
            let report = evaluate("baseline", &embedder, &corpus, mode)
                .await
                .expect("eval runs");
            assert_eq!(report.overall.queries, 2);
            assert!(report.config_name.ends_with(mode.name()));
            assert!(
                (report.overall.recall_at(30) - 1.0).abs() < 1e-9,
                "two documents: everything is in the top 30"
            );
        }
    }

    /// The server's way of storing a document: whole, and its pieces in the
    /// chunk index when it is longer than one; a piece's own vector finds it
    /// through the dense leg.
    #[tokio::test]
    async fn the_chunk_index_mode_stores_a_document_as_the_server_does() {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        let tenant = TenantId::new("ws:bench");
        let embedder = FixedEmbedder::new(EMBED_DIM);
        let now = chrono::Utc::now();
        let long = (0..200)
            .map(|i| format!("word{i:03}"))
            .collect::<Vec<_>>()
            .join(" ");
        for (id, text) in [("long", long.clone()), ("short", "a short one".into())] {
            let doc = Document {
                id: id.into(),
                text,
            };
            index(&store, &embedder, &tenant, &doc, now).await.unwrap();
        }
        let held = memory_chunk::indexed(&store, &tenant).await.unwrap();
        assert_eq!(held.len(), 1, "the short document is one piece");
        assert_eq!(held["memory:long"].content_hash, cut_hash(&long));

        let piece = &split(&long, MEMORY_CHUNK_CHARS)[2];
        let qv = embedder.embed(piece).await.unwrap();
        let hits = memory::recall_hybrid(&store, &tenant, "", &qv, 1, None, &[])
            .await
            .unwrap();
        assert_eq!(rank_of(&hits, "long"), Some(1));
    }

    /// `rank_of` maps a stored memory id back to its labeled doc id and reports a
    /// 1-based rank (or `None` when absent).
    #[tokio::test]
    async fn rank_of_strips_prefix_and_is_one_based() {
        let now = chrono::Utc::now();
        let mk = |doc_id: &str| {
            Memory::new(
                format!("{ID_PREFIX}{doc_id}"),
                TenantId::new("ws:bench"),
                MemoryNetwork::World,
                "x",
                1.0,
                now,
            )
        };
        let hits = vec![mk("fin-aapl"), mk("fin-msft")];
        assert_eq!(rank_of(&hits, "fin-aapl"), Some(1));
        assert_eq!(rank_of(&hits, "fin-msft"), Some(2));
        assert_eq!(rank_of(&hits, "fin-nvda"), None);
    }
}

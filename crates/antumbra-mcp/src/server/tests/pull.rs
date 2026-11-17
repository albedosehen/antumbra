//! `answer` on a server that does not hold an expert's adapter: with a source
//! it pulls the adapter once and serves through it; without one the expert
//! stays out of the answer, as it was.

use super::*;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use antumbra_core::adapter::{self, Seal};

use crate::server::pull::{ExpertSource, ADAPTER_PATH};

const WEIGHTS: &[u8] = b"the admitted weights";

/// An engine that knows only what was registered with it, and answers with
/// the prompt.
#[derive(Default)]
struct Engine {
    registered: std::sync::Mutex<HashMap<ExpertId, String>>,
}

impl Engine {
    fn file_of(&self, expert: &ExpertId) -> Option<String> {
        self.registered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(expert)
            .cloned()
    }
}

#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Engine {
    fn can_serve(&self, expert: &ExpertId) -> bool {
        self.file_of(expert).is_some()
    }

    fn register_expert(&self, expert: &ExpertId, adapter_uri: &str) {
        self.registered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(expert.clone(), adapter_uri.to_string());
    }

    async fn act(&self, req: ActRequest) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Ok(antumbra_core::ports::ActOutput {
            steps: Vec::new(),
            final_output: req.prompt,
        })
    }
}

/// A directory of this test's own, removed when it ends.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> anyhow::Result<Self> {
        let dir =
            std::env::temp_dir().join(format!("antumbra-mcp-pull-{name}-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn adapters(&self) -> String {
        self.0.to_string_lossy().replace('\\', "/")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover scratch directory fails no test.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `user:test`'s standing expert for everywhere, trained to [`WEIGHTS`] on
/// `host`: its row, as every server on the database reads it.
fn standing_on(host: &str) -> Expert {
    let tenant = TenantId::new("ws:test");
    let user = UserId::new("user:test");
    let now = Utc::now();
    Expert {
        id: ExpertId::new("expert:ws:test:user:test:behavior:everywhere"),
        name: "everywhere".into(),
        base_model: "base".into(),
        artifact_uri: "adapters/on-the-trainer.safetensors".into(),
        capability_card: serde_json::json!({
            "standing": true, "private": true, "scope": "everywhere",
            adapter::CARD_ADAPTER: Seal::of_bytes(WEIGHTS).to_card(),
        }),
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(user.clone()),
        compartment: Some(antumbra_core::behavior::compartment_id(&tenant, &user)),
        placed_on: Some(host.into()),
        created_at: now,
    }
}

/// A source that answers every pull with [`WEIGHTS`] and counts them: its
/// address and its count.
async fn source() -> anyhow::Result<(String, Arc<AtomicUsize>)> {
    let pulls = Arc::new(AtomicUsize::new(0));
    let counted = pulls.clone();
    let app = axum::Router::new().route(
        ADAPTER_PATH,
        axum::routing::get(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            async { WEIGHTS.to_vec() }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        // Ends with the test's runtime.
        let _ = axum::serve(listener, app).await;
    });
    Ok((format!("http://{addr}"), pulls))
}

/// A server on the machine `here` for `user:test`, over a store that holds
/// `expert`'s row.
async fn server_on(here: &str, expert: &Expert) -> anyhow::Result<(McpServer, Arc<Engine>)> {
    let store = Store::connect_memory(EMBED_DIM).await?;
    expert::insert(&store, expert).await?;
    let engine = Arc::new(Engine::default());
    let server = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        here.into(),
        CompartmentId::new("comp:test:default"),
        Some(engine.clone()),
    );
    Ok((server, engine))
}

/// Ask through the user's standing experts alone: which ones answered.
async fn answered_by(server: &McpServer) -> anyhow::Result<Vec<String>> {
    let asked = Parameters(AnswerParams {
        task: "anything".into(),
        repo: None,
        standing_only: true,
    });
    match server.answer(asked).await {
        Ok(Json(out)) => Ok(out.standing),
        Err(e) => anyhow::bail!("answer failed: {}", e.message),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answer_pulls_a_standing_expert_placed_elsewhere_once() -> anyhow::Result<()> {
    let scratch = Scratch::new("answer")?;
    let expert = standing_on("trainer");
    let (address, pulls) = source().await?;
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::new(
        &address,
        b"a-secret-both-servers-hold",
        &scratch.adapters(),
    )));

    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);

    let kept = format!(
        "{}/{}",
        scratch.adapters(),
        adapter::file_name(&expert.id, &Seal::of_bytes(WEIGHTS).sha256)
    );
    assert_eq!(engine.file_of(&expert.id), Some(kept.clone()));
    assert_eq!(std::fs::read(&kept)?, WEIGHTS);
    // The next answer is served from what was kept.
    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);
    assert_eq!(pulls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_source_an_expert_placed_elsewhere_stays_out_of_the_answer() -> anyhow::Result<()>
{
    let expert = standing_on("trainer");
    let (server, engine) = server_on("node", &expert).await?;

    assert!(answered_by(&server).await?.is_empty());
    assert_eq!(engine.file_of(&expert.id), None);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expert_placed_here_is_never_pulled() -> anyhow::Result<()> {
    let scratch = Scratch::new("here")?;
    let expert = standing_on("node");
    let (address, pulls) = source().await?;
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::new(
        &address,
        b"a-secret-both-servers-hold",
        &scratch.adapters(),
    )));

    // Its own engine was never given the file, so it does not answer; the
    // point is that it does not ask another server for what is its own.
    assert!(answered_by(&server).await?.is_empty());
    assert_eq!(pulls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.file_of(&expert.id), None);
    Ok(())
}

/// A copal that answers every read with `content`, counts the reads, and
/// records what it was sent to keep.
struct Copal {
    content: Vec<u8>,
    reads: AtomicUsize,
    /// What an upload answers: the digest it says it holds, or a refusal.
    upload: Result<String, String>,
    kept: std::sync::Mutex<Vec<Vec<u8>>>,
}

impl Copal {
    /// A copal holding `content` that takes any upload and says it holds
    /// [`WEIGHTS`].
    fn plain(content: &[u8]) -> Self {
        Self {
            content: content.to_vec(),
            reads: AtomicUsize::new(0),
            upload: Ok(Seal::of_bytes(WEIGHTS).sha256),
            kept: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn holding(content: &[u8]) -> Arc<Self> {
        Arc::new(Self::plain(content))
    }

    fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    fn kept(&self) -> Vec<Vec<u8>> {
        self.kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl antumbra_copal::CopalTransport for Copal {
    fn post_json(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _body: &serde_json::Value,
    ) -> antumbra_core::Result<serde_json::Value> {
        Ok(serde_json::json!({ "id": "01file", "state": "draft" }))
    }

    fn put_bytes(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _content_type: &str,
        _digest: &str,
        body: &[u8],
    ) -> antumbra_core::Result<serde_json::Value> {
        self.kept
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(body.to_vec());
        match &self.upload {
            Ok(digest) => Ok(serde_json::json!({ "digest": digest, "state": "ready" })),
            Err(why) => Err(antumbra_core::AntumbraError::other(why.clone())),
        }
    }

    fn get_bytes(
        &self,
        _url: &str,
        _credential: &antumbra_copal::CopalCredential,
        _limit: u64,
    ) -> antumbra_core::Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(self.content.clone())
    }
}

fn archive(copal: Arc<Copal>) -> Arc<antumbra_copal::CopalArchive> {
    Arc::new(antumbra_copal::CopalArchive::with_transport(
        "127.0.0.1:9",
        antumbra_copal::CopalTenancy::Shared("antumbra".into()),
        copal,
    ))
}

/// `expert` with its adapter archived: its card names the copal file.
fn archived(mut expert: Expert) -> Expert {
    expert.capability_card[adapter::CARD_ADAPTER][adapter::CARD_COPAL_FILE] =
        serde_json::json!("01file");
    expert
}

/// An address nothing answers at: a pull that reaches for it fails.
const NOBODY: &str = "http://127.0.0.1:9";

/// Where a server keeping its adapters in `dir` holds `expert`'s.
fn kept_in(dir: &str, expert: &Expert) -> String {
    format!(
        "{dir}/{}",
        adapter::file_name(&expert.id, &Seal::of_bytes(WEIGHTS).sha256)
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answer_reads_the_archive_when_the_source_does_not_answer() -> anyhow::Result<()> {
    let scratch = Scratch::new("fallback")?;
    let expert = archived(standing_on("trainer"));
    let copal = Copal::holding(WEIGHTS);
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(
        ExpertSource::new(NOBODY, b"a-secret-both-servers-hold", &scratch.adapters())
            .with_archive(archive(copal.clone())),
    ));

    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);

    let kept = kept_in(&scratch.adapters(), &expert);
    assert_eq!(engine.file_of(&expert.id), Some(kept.clone()));
    assert_eq!(std::fs::read(&kept)?, WEIGHTS);
    assert_eq!(copal.reads(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_with_an_archive_and_no_source_pulls_from_the_archive() -> anyhow::Result<()> {
    let scratch = Scratch::new("archive-only")?;
    let expert = archived(standing_on("trainer"));
    let copal = Copal::holding(WEIGHTS);
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::archive_only(
        archive(copal.clone()),
        &scratch.adapters(),
    )));

    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);
    assert_eq!(
        engine.file_of(&expert.id),
        Some(kept_in(&scratch.adapters(), &expert))
    );
    // The next answer is served from what was kept.
    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);
    assert_eq!(copal.reads(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_that_trained_an_expert_restores_its_missing_adapter() -> anyhow::Result<()> {
    let scratch = Scratch::new("restore")?;
    // Trained here, its row naming a file that is no longer on the disk.
    let mut expert = archived(standing_on("node"));
    expert.artifact_uri = kept_in(&scratch.adapters(), &expert);
    let copal = Copal::holding(WEIGHTS);
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::archive_only(
        archive(copal.clone()),
        &scratch.adapters(),
    )));

    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);

    assert_eq!(std::fs::read(&expert.artifact_uri)?, WEIGHTS);
    assert_eq!(
        engine.file_of(&expert.id),
        Some(expert.artifact_uri.clone())
    );
    // With the file back, the archive is not read again.
    assert_eq!(answered_by(&server).await?, [expert.id.as_str()]);
    assert_eq!(copal.reads(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_row_naming_another_file_is_not_restored_to() -> anyhow::Result<()> {
    let scratch = Scratch::new("restore-elsewhere")?;
    let mut expert = archived(standing_on("node"));
    expert.artifact_uri = format!("{}/another-name.safetensors", scratch.adapters());
    let copal = Copal::holding(WEIGHTS);
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::archive_only(
        archive(copal.clone()),
        &scratch.adapters(),
    )));

    assert!(answered_by(&server).await?.is_empty());
    assert!(!std::path::Path::new(&expert.artifact_uri).exists());
    assert_eq!(engine.file_of(&expert.id), None);
    assert_eq!(copal.reads(), 0);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_archived_copy_that_is_not_the_adapter_is_not_kept() -> anyhow::Result<()> {
    let scratch = Scratch::new("archive-mismatch")?;
    let expert = archived(standing_on("trainer"));
    let copal = Copal::holding(b"the admitted weighty");
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::archive_only(
        archive(copal.clone()),
        &scratch.adapters(),
    )));

    assert!(answered_by(&server).await?.is_empty());

    assert_eq!(engine.file_of(&expert.id), None);
    assert!(!std::path::Path::new(&kept_in(&scratch.adapters(), &expert)).exists());
    assert_eq!(copal.reads(), 1);
    // Asked again at once, it is left alone.
    assert!(answered_by(&server).await?.is_empty());
    assert_eq!(copal.reads(), 1);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_expert_with_no_archived_copy_and_no_source_is_left_alone() -> anyhow::Result<()> {
    let scratch = Scratch::new("no-copy")?;
    // Sealed, and never archived: its card names no copal file.
    let expert = standing_on("trainer");
    let copal = Copal::holding(WEIGHTS);
    let (server, engine) = server_on("node", &expert).await?;
    let server = server.with_expert_source(Arc::new(ExpertSource::archive_only(
        archive(copal.clone()),
        &scratch.adapters(),
    )));

    assert!(answered_by(&server).await?.is_empty());
    assert_eq!(engine.file_of(&expert.id), None);
    assert_eq!(copal.reads(), 0);
    Ok(())
}

#[tokio::test]
async fn an_admitted_adapter_is_archived_and_its_copal_file_named() -> anyhow::Result<()> {
    let scratch = Scratch::new("archive-at-mint")?;
    let expert = standing_on("node");
    let path = kept_in(&scratch.adapters(), &expert);
    std::fs::create_dir_all(scratch.adapters())?;
    std::fs::write(&path, WEIGHTS)?;
    let copal = Copal::holding(WEIGHTS);

    let file = crate::server::pull::archive_adapter(
        &archive(copal.clone()),
        &TenantId::new("ws:test"),
        &expert.id,
        &Seal::of_bytes(WEIGHTS),
        &path,
    )
    .await;

    assert_eq!(file.as_deref(), Some("01file"));
    assert_eq!(copal.kept(), [WEIGHTS.to_vec()]);
    Ok(())
}

#[tokio::test]
async fn an_upload_that_fails_archives_nothing_and_is_not_an_error() -> anyhow::Result<()> {
    let scratch = Scratch::new("archive-down")?;
    let expert = standing_on("node");
    let path = kept_in(&scratch.adapters(), &expert);
    std::fs::create_dir_all(scratch.adapters())?;
    std::fs::write(&path, WEIGHTS)?;
    let seal = Seal::of_bytes(WEIGHTS);
    let tenant = TenantId::new("ws:test");

    // Copal refusing the upload.
    let down = Arc::new(Copal {
        upload: Err("copal is down".into()),
        ..Copal::plain(WEIGHTS)
    });
    let file =
        crate::server::pull::archive_adapter(&archive(down), &tenant, &expert.id, &seal, &path)
            .await;
    assert_eq!(file, None);

    // The file gone before it could be read.
    let missing = format!("{}/not-there.safetensors", scratch.adapters());
    let copal = Copal::holding(WEIGHTS);
    let file = crate::server::pull::archive_adapter(
        &archive(copal.clone()),
        &tenant,
        &expert.id,
        &seal,
        &missing,
    )
    .await;
    assert_eq!(file, None);
    assert!(copal.kept().is_empty());
    Ok(())
}

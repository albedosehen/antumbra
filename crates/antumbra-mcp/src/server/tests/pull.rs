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

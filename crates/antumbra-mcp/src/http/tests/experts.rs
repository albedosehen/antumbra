//! An expert's adapter over the real router, and a server pulling it over a
//! socket: the owner is given the file its row names and nobody is given
//! anything else, and what a server pulls is kept only when it is the bytes
//! the row names.

use super::*;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use antumbra_core::adapter::{self, Seal};
use antumbra_core::{Expert, ExpertId, Generation};
use antumbra_store::repo::expert;

use crate::http::experts::SHA256_HEADER;
use crate::server::pull::{ExpertSource, ADAPTER_PATH};

/// The workspace and user of [`valid_token`], and the host of [`state_with`].
const TENANT: &str = "ws:t";
const USER: &str = "user:t";
const HOLDER: &str = "test";

/// A directory of this test's own, removed when it ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> anyhow::Result<Self> {
        let dir = std::env::temp_dir().join(format!(
            "antumbra-mcp-experts-{name}-{}",
            std::process::id()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    /// One machine's adapter directory under it.
    fn adapters(&self, machine: &str) -> String {
        self.0.join(machine).to_string_lossy().replace('\\', "/")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover scratch directory fails no test.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `user`'s standing expert for `scope` in `tenant`, trained to `weights` on
/// `host`, with its adapter written in `dir` as admission leaves it.
fn standing(
    (tenant, user): (&str, &str),
    scope: &str,
    host: &str,
    dir: &str,
    weights: &[u8],
) -> anyhow::Result<Expert> {
    let id = ExpertId::new(format!("expert:{tenant}:{user}:behavior:{scope}"));
    let seal = Seal::of_bytes(weights);
    let path = format!("{dir}/{}", adapter::file_name(&id, &seal.sha256));
    std::fs::create_dir_all(dir)?;
    std::fs::write(&path, weights)?;
    let now = chrono::Utc::now();
    Ok(Expert {
        name: id.as_str().to_string(),
        id,
        base_model: "base".into(),
        artifact_uri: path,
        capability_card: serde_json::json!({
            "standing": true, "private": true, "scope": scope,
            adapter::CARD_ADAPTER: seal.to_card(),
        }),
        capability_vec: None,
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(UserId::new(user)),
        compartment: Some(antumbra_core::behavior::compartment_id(
            &TenantId::new(tenant),
            &UserId::new(user),
        )),
        placed_on: Some(host.into()),
        created_at: now,
    })
}

/// A server holding `experts`, written as the keeper writes them.
async fn holding(experts: &[&Expert]) -> anyhow::Result<Arc<HttpState>> {
    let st = state_with(Store::connect_memory(EMBED_DIM).await?).await;
    owner_signin(&st).await?;
    for e in experts {
        expert::insert(&st.store, e).await?;
    }
    Ok(st)
}

async fn ask(app: &Router, token: Option<&str>, id: &ExpertId) -> anyhow::Result<Response> {
    let mut request = Request::builder()
        .method("GET")
        .uri(format!("{ADAPTER_PATH}?id={}", id.as_str()));
    if let Some(token) = token {
        request = request.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    Ok(app.clone().oneshot(request.body(Body::empty())?).await?)
}

async fn bytes_of(response: Response) -> anyhow::Result<Vec<u8>> {
    Ok(axum::body::to_bytes(response.into_body(), usize::MAX)
        .await?
        .to_vec())
}

#[tokio::test]
async fn the_owner_is_given_the_adapter_its_row_names() -> anyhow::Result<()> {
    let scratch = Scratch::new("owner")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"the admitted weights",
    )?;
    let app = router(holding(&[&mine]).await?);

    let response = ask(&app, Some(&valid_token()), &mine.id).await?;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(SHA256_HEADER)
            .and_then(|v| v.to_str().ok()),
        Some(Seal::of_bytes(b"the admitted weights").sha256.as_str())
    );
    assert_eq!(bytes_of(response).await?, b"the admitted weights");
    Ok(())
}

#[tokio::test]
async fn nobody_is_given_what_is_not_theirs_or_not_what_was_admitted() -> anyhow::Result<()> {
    let scratch = Scratch::new("refused")?;
    let dir = scratch.adapters("holder");
    let me = (TENANT, USER);
    let mine = standing(me, "everywhere", HOLDER, &dir, b"mine")?;
    // Placed on another machine: this server's disk does not hold it.
    let elsewhere = standing(me, "elsewhere", "another", &dir, b"elsewhere")?;
    // Its file written over since it was admitted.
    let changed = standing(me, "changed", HOLDER, &dir, b"as admitted")?;
    std::fs::write(&changed.artifact_uri, b"written over")?;
    // Its row naming a file other than the one its sha256 determines.
    let mut misnamed = standing(me, "misnamed", HOLDER, &dir, b"misnamed")?;
    let other = format!("{dir}/another-file.safetensors");
    std::fs::copy(&misnamed.artifact_uri, &other)?;
    misnamed.artifact_uri = other;
    // Admitted before seals were kept: nothing to check its file against.
    let mut unsealed = standing(me, "unsealed", HOLDER, &dir, b"unsealed")?;
    unsealed.capability_card = serde_json::json!({ "standing": true, "scope": "unsealed" });
    let app = router(holding(&[&mine, &elsewhere, &changed, &misnamed, &unsealed]).await?);

    let unknown = ExpertId::new("expert:ws:t:user:t:behavior:never-trained");
    let refused = [
        ("another user", token(TENANT, "user:other"), &mine.id),
        ("another workspace", token("ws:o", "user:o"), &mine.id),
        ("an unknown expert", valid_token(), &unknown),
        ("an expert placed elsewhere", valid_token(), &elsewhere.id),
        ("a file written over", valid_token(), &changed.id),
        ("a row naming another file", valid_token(), &misnamed.id),
        ("an expert with no seal", valid_token(), &unsealed.id),
    ];
    let mut answers = Vec::new();
    for (what, token, id) in &refused {
        let response = ask(&app, Some(token), id).await?;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{what}");
        answers.push(bytes_of(response).await?);
    }
    // One answer for all of them: it does not say which it was.
    assert!(answers.iter().all(|a| a == &answers[0]));
    // And the owner is still given their own.
    let response = ask(&app, Some(&valid_token()), &mine.id).await?;
    assert_eq!(bytes_of(response).await?, b"mine");
    Ok(())
}

#[tokio::test]
async fn a_pull_without_a_token_is_refused() -> anyhow::Result<()> {
    let scratch = Scratch::new("anonymous")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"mine",
    )?;
    let app = router(holding(&[&mine]).await?);

    for token in [None, Some("not-a-token")] {
        let response = ask(&app, token, &mine.id).await?;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{token:?}");
    }
    Ok(())
}

#[tokio::test]
async fn pulls_past_a_client_s_budget_are_turned_away() -> anyhow::Result<()> {
    let scratch = Scratch::new("budget")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"mine",
    )?;
    let app = router(holding(&[&mine]).await?);
    let token = valid_token();

    for n in 0..crate::http::experts::PULLS_PER_CLIENT.burst {
        let response = ask(&app, Some(&token), &mine.id).await?;
        assert_eq!(response.status(), StatusCode::OK, "pull {n}");
    }
    let response = ask(&app, Some(&token), &mine.id).await?;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    Ok(())
}

/// An engine that knows only what was registered with it.
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

    async fn act(
        &self,
        req: antumbra_core::ports::ActRequest,
    ) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        Ok(antumbra_core::ports::ActOutput {
            steps: Vec::new(),
            final_output: req.prompt,
        })
    }
}

/// Serve `app` on a socket of this machine: its address.
async fn listening(app: Router) -> anyhow::Result<String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        // Ends with the test's runtime.
        let _ = axum::serve(listener, app).await;
    });
    Ok(format!("http://{addr}"))
}

/// An address nothing answers at: a pull that reaches for it fails.
const NOBODY: &str = "http://127.0.0.1:9";

fn me() -> (TenantId, UserId) {
    (TenantId::new(TENANT), UserId::new(USER))
}

/// Where a server whose adapter directory is `dir` keeps `expert`'s adapter.
fn kept(dir: &str, expert: &Expert) -> anyhow::Result<String> {
    let Some(seal) = Seal::of_card(&expert.capability_card) else {
        anyhow::bail!("{} has no seal", expert.id.as_str());
    };
    Ok(format!(
        "{dir}/{}",
        adapter::file_name(&expert.id, &seal.sha256)
    ))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_pulls_an_expert_it_does_not_hold_and_then_needs_nobody() -> anyhow::Result<()> {
    let scratch = Scratch::new("pull")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"the admitted weights",
    )?;
    let holder = listening(router(holding(&[&mine]).await?)).await?;
    let dir = scratch.adapters("node");
    let (tenant, user) = me();

    let engine = Engine::default();
    let source = ExpertSource::new(&holder, b"test-secret", &dir);
    assert!(source.hold(&engine, &mine, &tenant, &user).await?);

    let path = kept(&dir, &mine)?;
    assert_eq!(std::fs::read(&path)?, b"the admitted weights");
    assert_eq!(engine.file_of(&mine.id), Some(path.clone()));

    // Started again with the source gone: the file is found by its name,
    // checked, and registered without a request.
    let restarted = Engine::default();
    let alone = ExpertSource::new(NOBODY, b"test-secret", &dir);
    assert!(alone.hold(&restarted, &mine, &tenant, &user).await?);
    assert_eq!(restarted.file_of(&mine.id), Some(path));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pull_is_answered_only_for_the_user_it_is_signed_for() -> anyhow::Result<()> {
    let scratch = Scratch::new("signed")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"mine",
    )?;
    let holder = listening(router(holding(&[&mine]).await?)).await?;
    let dir = scratch.adapters("node");

    // A server answering another user reads the same row and asks as them.
    let engine = Engine::default();
    let source = ExpertSource::new(&holder, b"test-secret", &dir);
    let asked = source
        .hold(
            &engine,
            &mine,
            &TenantId::new(TENANT),
            &UserId::new("user:other"),
        )
        .await;

    let Err(refused) = asked else {
        anyhow::bail!("another user was given the adapter");
    };
    assert!(refused.to_string().contains("answered 404"), "{refused}");
    assert!(!Path::new(&kept(&dir, &mine)?).exists());
    assert_eq!(engine.file_of(&mine.id), None);

    // And a source that signs with another secret is not let in at all.
    let stranger = ExpertSource::new(&holder, b"another-secret", &dir);
    let (tenant, user) = me();
    let Err(refused) = stranger.hold(&engine, &mine, &tenant, &user).await else {
        anyhow::bail!("a token under another secret was accepted");
    };
    assert!(refused.to_string().contains("answered 401"), "{refused}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answer_that_is_not_the_adapter_is_not_kept() -> anyhow::Result<()> {
    let scratch = Scratch::new("mismatch")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"the admitted weights",
    )?;
    // A source that answers every pull with other bytes of the same length,
    // and one that answers with more than the row's size.
    let dir = scratch.adapters("node");
    let (tenant, user) = me();
    for (what, body) in [
        ("other bytes", b"the admitted weighty".to_vec()),
        ("too many bytes", b"the admitted weights and more".to_vec()),
    ] {
        let liar = Router::new().route(
            ADAPTER_PATH,
            axum::routing::get(move || {
                let body = body.clone();
                async move { body }
            }),
        );
        let engine = Engine::default();
        let source = ExpertSource::new(&listening(liar).await?, b"test-secret", &dir);

        let Err(refused) = source.hold(&engine, &mine, &tenant, &user).await else {
            anyhow::bail!("{what}: kept what the row does not name");
        };

        assert!(
            refused.to_string().contains("nothing was kept")
                || refused.to_string().contains("could not be read"),
            "{what}: {refused}"
        );
        assert_eq!(engine.file_of(&mine.id), None, "{what}");
        let left: Vec<_> = match std::fs::read_dir(&dir) {
            Ok(entries) => entries.filter_map(|e| e.ok()).collect(),
            Err(_) => Vec::new(),
        };
        assert!(left.is_empty(), "{what}: {left:?}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_retrained_expert_is_pulled_again_and_the_copy_before_it_removed() -> anyhow::Result<()> {
    let scratch = Scratch::new("retrained")?;
    let held = scratch.adapters("holder");
    let first = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &held,
        b"first weights",
    )?;
    let st = holding(&[&first]).await?;
    let holder = listening(router(st.clone())).await?;
    let dir = scratch.adapters("node");
    let (tenant, user) = me();
    let engine = Engine::default();
    let source = ExpertSource::new(&holder, b"test-secret", &dir);
    assert!(source.hold(&engine, &first, &tenant, &user).await?);
    let before = kept(&dir, &first)?;

    // Retrained on the holder: the same expert, another adapter on its row.
    let second = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &held,
        b"second weights",
    )?;
    owner_signin(&st).await?;
    expert::delete(&st.store, &first.id).await?;
    expert::insert(&st.store, &second).await?;

    assert!(source.hold(&engine, &second, &tenant, &user).await?);

    let after = kept(&dir, &second)?;
    assert_ne!(before, after);
    assert_eq!(engine.file_of(&second.id), Some(after.clone()));
    assert_eq!(std::fs::read(&after)?, b"second weights");
    assert!(!Path::new(&before).exists(), "the copy before it is gone");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_source_that_does_not_answer_is_not_asked_again_at_once() -> anyhow::Result<()> {
    let scratch = Scratch::new("down")?;
    let mine = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"mine",
    )?;
    let (tenant, user) = me();
    let engine = Engine::default();
    let source = ExpertSource::new(NOBODY, b"test-secret", &scratch.adapters("node"));

    let Err(down) = source.hold(&engine, &mine, &tenant, &user).await else {
        anyhow::bail!("a source nobody answers at gave an adapter");
    };
    assert!(down.to_string().contains("did not answer"), "{down}");

    // Asked for again straight away: left alone, and without an error to log.
    assert!(!source.hold(&engine, &mine, &tenant, &user).await?);
    assert_eq!(engine.file_of(&mine.id), None);
    Ok(())
}

#[tokio::test]
async fn an_expert_with_no_seal_is_not_pulled() -> anyhow::Result<()> {
    let scratch = Scratch::new("unsealed")?;
    let mut unsealed = standing(
        (TENANT, USER),
        "everywhere",
        HOLDER,
        &scratch.adapters("holder"),
        b"unsealed",
    )?;
    unsealed.capability_card = serde_json::json!({ "standing": true, "scope": "everywhere" });
    let (tenant, user) = me();

    let engine = Engine::default();
    let source = ExpertSource::new(NOBODY, b"test-secret", &scratch.adapters("node"));

    assert!(!source.hold(&engine, &unsealed, &tenant, &user).await?);
    assert_eq!(engine.file_of(&unsealed.id), None);
    Ok(())
}

//! Standing experts (ADR-0027): the user's accepted behaviors for a scope,
//! composed into every answer in that scope and never routed.

use super::*;
use std::sync::Mutex;

/// Serves the prompt back and remembers which experts each request blended.
#[derive(Default)]
struct Recording {
    blends: Mutex<Vec<Vec<String>>>,
}

#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Recording {
    async fn act(&self, req: ActRequest) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        self.blends.lock().unwrap().push(
            req.adapters
                .iter()
                .map(|e| e.as_str().to_string())
                .collect(),
        );
        Ok(antumbra_core::ports::ActOutput {
            steps: Vec::new(),
            final_output: req.prompt,
        })
    }
}

fn expert(id: &str, owner: &str, card: serde_json::Value, cap: Vec<f32>) -> Expert {
    let now = Utc::now();
    Expert {
        id: ExpertId::new(id),
        name: id.into(),
        base_model: "base".into(),
        artifact_uri: format!("mem://{id}"),
        capability_card: card,
        capability_vec: Some(cap),
        fitness: 1.0,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(UserId::new(owner)),
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

fn standing(scope: &str) -> serde_json::Value {
    serde_json::json!({ "standing": true, "private": true, "scope": scope })
}

/// A store whose router covers any task with `expert:adder`, and a server on
/// it for `user:test` that records what it serves.
async fn covered() -> (McpServer, Arc<Recording>) {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    router::save(
        &store,
        &LearnedRouter {
            weights: vec![1.0; EMBED_DIM],
            experts: vec![RouterExpert {
                id: ExpertId::new("expert:adder"),
                centroid: vec![0.0; EMBED_DIM],
            }],
            temperature: 0.1,
            floor: -1.0,
        },
    )
    .await
    .unwrap();
    serving(store).await
}

async fn serving(store: Store) -> (McpServer, Arc<Recording>) {
    let rec = Arc::new(Recording::default());
    let s = McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "h".into(),
        CompartmentId::new("comp:test:default"),
        Some(rec.clone()),
    );
    (s, rec)
}

async fn embed(text: &str) -> Vec<f32> {
    FixedEmbedder::new(EMBED_DIM).embed(text).await.unwrap()
}

fn ask(task: &str, repo: Option<&str>) -> Parameters<AnswerParams> {
    Parameters(AnswerParams {
        task: task.into(),
        repo: repo.map(str::to_string),
    })
}

#[tokio::test]
async fn an_answer_composes_the_users_standing_experts_for_everywhere_and_its_repository() {
    let (s, rec) = covered().await;
    let elsewhere = embed("nothing like the task").await;
    for e in [
        expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            elsewhere.clone(),
        ),
        expert(
            "expert:mine:repo",
            "user:test",
            standing("github.com/o/r"),
            elsewhere.clone(),
        ),
        expert(
            "expert:mine:other",
            "user:test",
            standing("github.com/o/other"),
            elsewhere.clone(),
        ),
        expert(
            "expert:theirs",
            "user:else",
            standing("everywhere"),
            elsewhere.clone(),
        ),
    ] {
        expert::insert(&s.store, &e).await.unwrap();
    }

    let out = s
        .answer(ask("add two numbers", Some("github.com/o/r")))
        .await
        .unwrap();
    assert!(!out.0.escalate);
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
    assert_eq!(
        out.0.standing,
        ["expert:mine:everywhere", "expert:mine:repo"]
    );

    let out = s.answer(ask("add two numbers", None)).await.unwrap();
    assert_eq!(out.0.standing, ["expert:mine:everywhere"]);

    assert_eq!(
        *rec.blends.lock().unwrap(),
        [
            vec!["expert:adder", "expert:mine:everywhere", "expert:mine:repo"],
            vec!["expert:adder", "expert:mine:everywhere"],
        ]
    );
}

#[tokio::test]
async fn with_nothing_routed_a_standing_expert_answers_only_a_task_like_its_own() {
    let task = "open a pull request for my branch";
    let (s, rec) = serving(Store::connect_memory(EMBED_DIM).await.unwrap()).await;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            embed(task).await,
        ),
    )
    .await
    .unwrap();
    let out = s.answer(ask(task, None)).await.unwrap();
    assert!(!out.0.escalate);
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:mine:everywhere"));
    assert_eq!(out.0.standing, ["expert:mine:everywhere"]);
    assert_eq!(rec.blends.lock().unwrap().len(), 1);

    // Taught tasks the test embedder places nowhere near this one.
    let mut unlike = vec![0.0; EMBED_DIM];
    unlike[EMBED_DIM - 1] = 1.0;
    let (s, rec) = serving(Store::connect_memory(EMBED_DIM).await.unwrap()).await;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            unlike,
        ),
    )
    .await
    .unwrap();
    let out = s.answer(ask(task, None)).await.unwrap();
    assert!(out.0.escalate, "a task unlike its own is the generalist's");
    assert!(out.0.standing.is_empty());
    assert!(rec.blends.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_standing_expert_is_never_routed() {
    let (s, _) = serving(Store::connect_memory(EMBED_DIM).await.unwrap()).await;
    let cap = embed("my private skill").await;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            cap.clone(),
        ),
    )
    .await
    .unwrap();
    expert::insert(
        &s.store,
        &expert("expert:mine", "user:test", serde_json::Value::Null, cap),
    )
    .await
    .unwrap();
    let r = s
        .route(Parameters(RouteParams {
            task: "my private skill".into(),
            top_k: Some(5),
        }))
        .await
        .unwrap();
    let ids: Vec<&str> = r.0.routes.iter().map(|h| h.expert_id.as_str()).collect();
    assert_eq!(ids, ["expert:mine"]);
}

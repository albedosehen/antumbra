//! Standing experts: the user's accepted behaviors for a scope,
//! composed into every answer in that scope and never routed.

use super::*;
use std::sync::Mutex;

/// Serves the prompt back and remembers which experts each request blended.
#[derive(Default)]
struct Recording {
    blends: Mutex<Vec<Vec<String>>>,
}

impl Recording {
    /// The experts of every request served so far, in order.
    fn blended(&self) -> Vec<Vec<String>> {
        // The fake only pushes under the lock, so a poisoned lock still holds
        // a whole list.
        self.blends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait::async_trait]
impl antumbra_core::ports::Serve for Recording {
    async fn act(&self, req: ActRequest) -> antumbra_core::Result<antumbra_core::ports::ActOutput> {
        // The fake only pushes under the lock, so a poisoned lock still holds
        // a whole list.
        self.blends
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(
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
        // Trained from the owner's behaviors in the server's workspace.
        compartment: Some(antumbra_core::behavior::compartment_id(
            &TenantId::new("ws:test"),
            &UserId::new(owner),
        )),
        placed_on: None,
        created_at: now,
    }
}

fn standing(scope: &str) -> serde_json::Value {
    serde_json::json!({ "standing": true, "private": true, "scope": scope })
}

/// A store whose router covers any task with `expert:adder`, and a server on
/// it for `user:test` that records what it serves.
async fn covered() -> anyhow::Result<(McpServer, Arc<Recording>)> {
    let store = mem_store().await?;
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
    .await?;
    Ok(serving(store).await)
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

async fn embed(text: &str) -> anyhow::Result<Vec<f32>> {
    Ok(FixedEmbedder::new(EMBED_DIM).embed(text).await?)
}

fn ask(task: &str, repo: Option<&str>) -> Parameters<AnswerParams> {
    Parameters(AnswerParams {
        task: task.into(),
        repo: repo.map(str::to_string),
        standing_only: false,
    })
}

#[tokio::test]
async fn an_answer_composes_the_users_standing_experts_for_everywhere_and_its_repository(
) -> anyhow::Result<()> {
    let (s, rec) = covered().await?;
    let elsewhere = embed("nothing like the task").await?;
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
        // The same user id in another workspace: every hosted workspace's
        // first user shares one.
        Expert {
            compartment: Some(antumbra_core::behavior::compartment_id(
                &TenantId::new("ws:other"),
                &UserId::new("user:test"),
            )),
            ..expert(
                "expert:other-workspace",
                "user:test",
                standing("everywhere"),
                elsewhere.clone(),
            )
        },
    ] {
        expert::insert(&s.store, &e).await?;
    }

    let out = s
        .answer(ask("add two numbers", Some("github.com/o/r")))
        .await?;
    assert!(!out.0.escalate);
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:adder"));
    assert_eq!(
        out.0.standing,
        ["expert:mine:everywhere", "expert:mine:repo"]
    );

    let out = s.answer(ask("add two numbers", None)).await?;
    assert_eq!(out.0.standing, ["expert:mine:everywhere"]);

    assert_eq!(
        rec.blended(),
        [
            vec!["expert:adder", "expert:mine:everywhere", "expert:mine:repo"],
            vec!["expert:adder", "expert:mine:everywhere"],
        ]
    );
    Ok(())
}

#[tokio::test]
async fn with_nothing_routed_a_standing_expert_answers_only_a_task_like_its_own(
) -> anyhow::Result<()> {
    let task = "open a pull request for my branch";
    let (s, rec) = serving(mem_store().await?).await;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            embed(task).await?,
        ),
    )
    .await?;
    let out = s.answer(ask(task, None)).await?;
    assert!(!out.0.escalate);
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:mine:everywhere"));
    assert_eq!(out.0.standing, ["expert:mine:everywhere"]);
    assert_eq!(rec.blended().len(), 1);

    // Taught tasks the test embedder places nowhere near this one.
    let mut unlike = vec![0.0; EMBED_DIM];
    unlike[EMBED_DIM - 1] = 1.0;
    let (s, rec) = serving(mem_store().await?).await;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            unlike,
        ),
    )
    .await?;
    let out = s.answer(ask(task, None)).await?;
    assert!(out.0.escalate, "a task unlike its own is the generalist's");
    assert!(out.0.standing.is_empty());
    assert!(rec.blended().is_empty());
    Ok(())
}

/// Asked for standing experts only, an answer leaves out the routed expert,
/// even where the router covers the task, and serves through the user's
/// standing experts for everywhere and the repository however unlike their
/// own the task is. With none in scope it escalates and serves nothing.
#[tokio::test]
async fn an_answer_from_standing_experts_only_skips_the_route_and_the_likeness(
) -> anyhow::Result<()> {
    let only = |repo: Option<&str>| {
        Parameters(AnswerParams {
            task: "add two numbers".into(),
            repo: repo.map(str::to_string),
            standing_only: true,
        })
    };
    let mut unlike = vec![0.0; EMBED_DIM];
    unlike[EMBED_DIM - 1] = 1.0;
    let (s, rec) = covered().await?;
    for e in [
        expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            unlike.clone(),
        ),
        expert(
            "expert:mine:repo",
            "user:test",
            standing("github.com/o/r"),
            unlike.clone(),
        ),
    ] {
        expert::insert(&s.store, &e).await?;
    }
    let out = s.answer(only(Some("github.com/o/r"))).await?;
    assert!(!out.0.escalate);
    assert_eq!(out.0.expert_id.as_deref(), Some("expert:mine:everywhere"));
    assert_eq!(
        out.0.standing,
        ["expert:mine:everywhere", "expert:mine:repo"]
    );
    assert_eq!(
        rec.blended(),
        [vec!["expert:mine:everywhere", "expert:mine:repo"]],
        "no routed expert in the blend"
    );

    let (s, rec) = covered().await?;
    let out = s.answer(only(None)).await?;
    assert!(out.0.escalate, "no standing expert in scope");
    assert!(rec.blended().is_empty());
    Ok(())
}

#[tokio::test]
async fn a_standing_expert_is_never_routed() -> anyhow::Result<()> {
    let (s, _) = serving(mem_store().await?).await;
    let cap = embed("my private skill").await?;
    expert::insert(
        &s.store,
        &expert(
            "expert:mine:everywhere",
            "user:test",
            standing("everywhere"),
            cap.clone(),
        ),
    )
    .await?;
    expert::insert(
        &s.store,
        &expert("expert:mine", "user:test", serde_json::Value::Null, cap),
    )
    .await?;
    let r = s
        .route(Parameters(RouteParams {
            task: "my private skill".into(),
            top_k: Some(5),
        }))
        .await?;
    let ids: Vec<&str> = r.0.routes.iter().map(|h| h.expert_id.as_str()).collect();
    assert_eq!(ids, ["expert:mine"]);
    Ok(())
}

/// An answer with no standing experts, which is every answer for a user
/// who has accepted no behaviors, has to carry every key its output schema
/// requires, or a client that checks results refuses the answer.
#[test]
fn an_answer_without_standing_experts_matches_the_output_schema() -> anyhow::Result<()> {
    let out = crate::server::params::AnswerOut {
        answer: "bun add react".into(),
        expert_id: None,
        escalate: false,
        note: None,
        standing: Vec::new(),
    };
    assert_required_keys_present(&out)?;
    Ok(())
}

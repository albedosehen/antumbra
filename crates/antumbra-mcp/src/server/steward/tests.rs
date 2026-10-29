//! The system governs memories by itself: recall counts what it showed, the end
//! of a turn sorts what was drawn on from what was passed over, a new memory is
//! linked to what it restates, and the knobs steer all of it.

use antumbra_core::ports::{Answer, Question, RelevanceState, TypedDecider};
use antumbra_core::testing::FixedEmbedder;
use antumbra_core::{MemoryUse, Result as CoreResult};
use antumbra_store::EMBED_DIM;

use super::*;

/// Judges a memory drawn on when the text names its first word.
struct FirstWordDecider;

#[async_trait::async_trait]
impl TypedDecider for FirstWordDecider {
    async fn decide(&self, state: &str, questions: &[Question]) -> CoreResult<Vec<Answer>> {
        let state = RelevanceState::decode(state).expect("a relevance state");
        Ok(state
            .memories
            .iter()
            .zip(questions)
            .map(|(m, _)| {
                let word = m
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .trim_matches(|c: char| !c.is_alphanumeric());
                let named = !word.is_empty() && state.query.contains(word);
                Answer::Noul {
                    probability: if named { 0.9 } else { 0.1 },
                }
            })
            .collect())
    }
}

async fn server() -> McpServer {
    let store = Store::connect_memory(EMBED_DIM).await.unwrap();
    McpServer::new(
        store,
        Arc::new(FixedEmbedder::new(EMBED_DIM)),
        TenantId::new("ws:test"),
        UserId::new("user:test"),
        "test-host".into(),
        CompartmentId::new("comp:test:default"),
        None,
    )
    .with_decider(Arc::new(FirstWordDecider))
}

async fn store(s: &McpServer, content: &str) -> StoredOut {
    s.store_memory(Parameters(StoreParams {
        provenance: None,
        content: content.into(),
        network: "world".into(),
        confidence: Some(0.8),
        evidence: None,
        volatile: None,
        compartment: None,
    }))
    .await
    .unwrap()
    .0
}

async fn recall(s: &McpServer, query: &str, session: Option<&str>) -> Vec<String> {
    s.recall_memories(Parameters(RecallParams {
        query: query.into(),
        top_k: Some(5),
        network: None,
        repo: None,
        branch: None,
        floor: Some(0.0),
        full: None,
        host: None,
        session: session.map(str::to_string),
    }))
    .await
    .unwrap()
    .0
    .memories
    .into_iter()
    .map(|v| v.id)
    .collect()
}

async fn observe(s: &McpServer, session: Option<&str>, turn: &str, text: &str) -> ObservedOut {
    s.observe_turn(Parameters(ObserveParams {
        session: session.map(str::to_string),
        turn: Some(turn.into()),
        text: text.into(),
    }))
    .await
    .unwrap()
    .0
}

async fn seen(s: &McpServer, id: &str) -> Option<MemoryUse> {
    memory_use::for_memories(&s.store, &s.tenant, &[MemoryId::new(id)])
        .await
        .unwrap()
        .pop()
}

#[tokio::test]
async fn a_turn_counts_what_it_drew_on_and_what_it_passed_over() {
    let s = server().await;
    let alpha = store(&s, "alpha: the orders service deploys blue green")
        .await
        .id;
    let beta = store(&s, "beta: billing retries a failed charge three times")
        .await
        .id;

    let shown = recall(&s, "how do we ship", Some("session-1")).await;
    assert_eq!(shown.len(), 2);
    assert_eq!(seen(&s, &alpha).await.unwrap().recalled, 1);

    let out = observe(
        &s,
        Some("session-1"),
        "turn-1",
        "Rolled it out the alpha way.",
    )
    .await;
    assert_eq!((out.judged, out.used), (2, 1));
    let a = seen(&s, &alpha).await.unwrap();
    assert_eq!((a.judged, a.used), (1, 1));
    assert!(a.last_used_at.is_some());
    let b = seen(&s, &beta).await.unwrap();
    assert_eq!((b.judged, b.used, b.passed_over()), (1, 0, 1));

    // The same turn reported again is not judged twice, and what was judged
    // is not judged again by the next turn.
    assert!(
        observe(&s, Some("session-1"), "turn-1", "alpha")
            .await
            .repeated
    );
    let next = observe(&s, Some("session-1"), "turn-2", "alpha again").await;
    assert_eq!((next.judged, next.used), (0, 0));
    assert_eq!(seen(&s, &alpha).await.unwrap().used, 1);
}

#[tokio::test]
async fn what_an_agent_recalled_for_itself_counts_only_when_used() {
    let s = server().await;
    let alpha = store(&s, "alpha: the orders service deploys blue green")
        .await
        .id;
    let beta = store(&s, "beta: billing retries a failed charge three times")
        .await
        .id;
    recall(&s, "what do we know", None).await;

    let out = observe(&s, Some("another-session"), "turn-1", "beta, three retries").await;
    assert_eq!((out.judged, out.used), (0, 1));
    let b = seen(&s, &beta).await.unwrap();
    assert_eq!((b.used, b.judged), (1, 1), "a use is a judgment too");
    let a = seen(&s, &alpha).await.unwrap();
    assert_eq!(
        a.passed_over(),
        0,
        "never held against a memory no one session saw"
    );
}

#[tokio::test]
async fn a_pass_over_still_counts_beside_an_agents_own_use() {
    let s = server().await;
    let beta = store(&s, "beta: billing retries a failed charge three times")
        .await
        .id;
    // Used once from the agent's own recall...
    recall(&s, "billing retries", None).await;
    observe(&s, Some("session-1"), "turn-1", "beta, three retries").await;
    // ...then shown to a session whose answer is about something else.
    recall(&s, "billing retries", Some("session-2")).await;
    observe(&s, Some("session-2"), "turn-2", "an unrelated answer").await;
    let b = seen(&s, &beta).await.unwrap();
    assert_eq!((b.used, b.judged), (1, 2));
    assert_eq!(b.passed_over(), 1);
}

#[tokio::test]
async fn with_the_system_switched_off_nothing_is_counted() {
    let s = server().await;
    let off = MemoryPolicy {
        auto: false,
        ..MemoryPolicy::default()
    };
    memory_policy::set(&s.store, &s.tenant, &off).await.unwrap();
    let alpha = store(&s, "alpha: the orders service deploys blue green").await;
    let again = store(&s, "alpha: the orders service deploys blue green").await;
    assert!(again.linked.is_empty(), "no links with the system off");
    recall(&s, "how do we ship", Some("session-1")).await;
    let out = observe(&s, Some("session-1"), "turn-1", "alpha").await;
    assert_eq!((out.judged, out.used), (0, 0));
    assert!(seen(&s, &alpha.id).await.is_none());
}

#[tokio::test]
async fn a_new_memory_is_linked_to_what_it_restates() {
    let s = server().await;
    let first = store(&s, "alpha: the orders service deploys blue green")
        .await
        .id;
    let second = store(&s, "alpha: the orders service deploys blue green").await;
    assert_eq!(second.linked, vec![first.clone()]);
    assert_eq!(seen(&s, &first).await.unwrap().recurred, 1);
    let edges = edge::neighbors(&s.store, &s.tenant, &MemoryId::new(&second.id), None)
        .await
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].to_id.as_str(), first);
    assert_eq!(edges[0].edge_type, EdgeType::References);
}

#[tokio::test]
async fn relating_refuses_an_end_that_does_not_exist() {
    let s = server().await;
    let real = store(&s, "alpha: the orders service deploys blue green")
        .await
        .id;
    let refused = s
        .relate_memories(Parameters(RelateParams {
            from_id: real.clone(),
            to_id: "memory:never-was".into(),
            edge_type: "supersedes".into(),
            weight: None,
        }))
        .await;
    assert!(refused.is_err());
    assert!(
        edge::neighbors(&s.store, &s.tenant, &MemoryId::new(&real), None)
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn an_agents_own_judgment_is_counted_beside_the_systems() {
    let s = server().await;
    let id = store(&s, "alpha: the orders service deploys blue green")
        .await
        .id;
    s.reinforce_memory(Parameters(IdParams {
        memory_id: id.clone(),
    }))
    .await
    .unwrap();
    s.penalize_memory(Parameters(IdParams {
        memory_id: id.clone(),
    }))
    .await
    .unwrap();
    let u = seen(&s, &id).await.unwrap();
    assert_eq!((u.good, u.bad), (1, 1));
    assert!(
        u.last_used_at.is_some(),
        "a good judgment restarts the fade"
    );
}

#[tokio::test]
async fn strength_moves_recall_order_as_far_as_its_knob_allows() {
    let s = server().await;
    let now = Utc::now();
    let weak = Memory::new(
        "memory:weak",
        "ws:test",
        MemoryNetwork::World,
        "a",
        0.1,
        now - chrono::Duration::days(720),
    );
    let strong = Memory::new(
        "memory:strong",
        "ws:test",
        MemoryNetwork::World,
        "b",
        0.9,
        now,
    );
    let other = Memory::new(
        "memory:other",
        "ws:test",
        MemoryNetwork::World,
        "c",
        0.9,
        now,
    );
    let hits = vec![weak.clone(), strong.clone(), other.clone()];

    let (kept, strengths) = s.by_strength(hits.clone(), &MemoryPolicy::default()).await;
    assert_eq!(
        kept[0].id.as_str(),
        "memory:weak",
        "the default lets relevance lead"
    );
    assert!(strengths["memory:weak"] < 0.01);

    let strength_led = MemoryPolicy {
        strength_weight: 1.0,
        ..MemoryPolicy::default()
    };
    let (kept, _) = s.by_strength(hits, &strength_led).await;
    assert_eq!(kept[0].id.as_str(), "memory:strong");
}

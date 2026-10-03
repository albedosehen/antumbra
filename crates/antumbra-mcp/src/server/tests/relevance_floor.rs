use super::*;
use antumbra_core::testing::ScriptedDecider;

async fn seeded() -> McpServer {
    let s = server().await;
    for content in [
        "the deploy runbook for the orders service",
        "how the billing reconciliation job retries",
    ] {
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
        .unwrap();
    }
    s
}

async fn recall(s: &McpServer, query: &str, floor: Option<f32>) -> MemoriesOut {
    s.recall_memories(Parameters(RecallParams {
        repo: None,
        branch: None,
        query: query.into(),
        top_k: Some(5),
        network: None,
        full: None,
        floor,
    }))
    .await
    .unwrap()
    .0
}

/// With no decider there is no floor, and the field must stay absent. A
/// deployment without a head keeps exactly the behaviour it had.
#[tokio::test]
async fn without_a_decider_nothing_is_filtered_and_nothing_is_claimed() {
    let s = seeded().await;
    let out = recall(&s, "deploy runbook", None).await;
    assert!(!out.memories.is_empty(), "rows come back as before");
    assert!(
        !out.nothing_cleared_the_floor,
        "no floor ran, so the field must not claim one did"
    );
}

/// The floor rejects everything: an EMPTY list plus the field, which is the
/// answer an agent can branch on without re-running the query.
#[tokio::test]
async fn when_nothing_clears_the_floor_the_answer_says_so() {
    // The decider judges nothing relevant, because no state contains the token.
    let s = seeded()
        .await
        .with_decider(Arc::new(ScriptedDecider::on_substring("zzz-absent", 0.9)));
    let out = recall(&s, "something unrelated", None).await;
    assert!(out.memories.is_empty(), "the floor rejected every row");
    assert!(
        out.nothing_cleared_the_floor,
        "and the answer says so, by a field rather than by inference"
    );
}

/// Rows that clear the floor come back, and the field stays absent. This is
/// the case that must NOT look like the one above.
#[tokio::test]
async fn when_something_clears_the_floor_the_rows_come_back() {
    let s = seeded()
        .await
        .with_decider(Arc::new(ScriptedDecider::on_substring(
            "deploy runbook",
            0.9,
        )));
    let out = recall(&s, "deploy runbook", None).await;
    assert!(!out.memories.is_empty(), "relevant rows survive the floor");
    assert!(
        !out.nothing_cleared_the_floor,
        "a recall that returned rows must not claim the floor emptied it"
    );
}

/// The floor is a default the caller may move. Raising it past what the
/// decider reports empties the result; lowering it takes the rows back.
#[tokio::test]
async fn the_caller_can_move_the_floor() {
    let s = seeded()
        .await
        .with_decider(Arc::new(ScriptedDecider::on_substring(
            "deploy runbook",
            0.6,
        )));
    assert!(
        recall(&s, "deploy runbook", Some(0.95))
            .await
            .memories
            .is_empty(),
        "a floor above the decider's confidence rejects everything"
    );
    assert!(
        !recall(&s, "deploy runbook", Some(0.1))
            .await
            .memories
            .is_empty(),
        "and lowering it takes the rows back -- the best of a bad lot is a choice"
    );
}

/// The floor holds for a query of several lines, which is what a pasted
/// prompt is: the calibrated floor, over a scorer that finds only the runbook
/// relevant, keeps the runbook and drops the rest. Before the state was
/// carried as JSON, the floor could not parse such a query and every one went
/// out unfiltered.
#[tokio::test]
async fn the_floor_holds_for_a_query_of_several_lines() {
    struct RunbookOnly;
    #[async_trait::async_trait]
    impl antumbra_core::ports::RelevanceScorer for RunbookOnly {
        async fn relevance(&self, _q: &str, texts: &[String]) -> antumbra_core::Result<Vec<f32>> {
            Ok(texts
                .iter()
                .map(|t| if t.contains("runbook") { 5e-3 } else { 3.7e-5 })
                .collect())
        }
    }
    let floor = antumbra_rerank::floor::CalibratedFloor::new(Arc::new(RunbookOnly));
    let s = seeded().await.with_decider(Arc::new(floor));
    let out = recall(
        &s,
        "where is the deploy\nrunbook for orders?\n\n---\nthanks",
        None,
    )
    .await;
    let kept: Vec<&str> = out.memories.iter().map(|m| m.content.as_str()).collect();
    assert_eq!(kept, ["the deploy runbook for the orders service"]);
}

/// A floor that cannot be computed must not be enforced: the rows come back
/// unfiltered, and nothing claims they were judged. ADR-0024 requires a head
/// that fails to degrade to the path it replaced rather than to nothing.
#[tokio::test]
async fn a_failing_decider_returns_rows_rather_than_swallowing_them() {
    let s = seeded()
        .await
        .with_decider(Arc::new(ScriptedDecider::failing()));
    let out = recall(&s, "deploy runbook", None).await;
    assert!(
        !out.memories.is_empty(),
        "a floor that cannot be computed must not be enforced"
    );
    assert!(!out.nothing_cleared_the_floor);
}

/// An empty store is not the same answer as a floor that rejected
/// everything, and the field must not blur them.
#[tokio::test]
async fn an_empty_store_does_not_claim_the_floor_emptied_it() {
    let s = server()
        .await
        .with_decider(Arc::new(ScriptedDecider::on_substring("zzz-absent", 0.9)));
    let out = recall(&s, "anything at all", None).await;
    assert!(out.memories.is_empty());
    assert!(
        !out.nothing_cleared_the_floor,
        "there was nothing to reject, which is a different answer"
    );
}

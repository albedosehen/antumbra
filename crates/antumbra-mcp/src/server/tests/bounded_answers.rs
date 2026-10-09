use super::*;
use crate::server::params::RECALL_CONTENT_CHARS;

/// Seed `n` memories each far longer than the bound, and recall them.
async fn recall_long(s: &McpServer, n: usize, full: Option<bool>) -> Vec<MemoryView> {
    for i in 0..n {
        // Distinct prefixes so the rows are not deduplicated, and a length
        // comfortably past the cut.
        let content = format!("memory {i} about scheduling. {}", "prose ".repeat(400));
        s.store_memory(Parameters(StoreParams {
            provenance: None,
            content,
            network: "world".into(),
            confidence: Some(0.8),
            evidence: None,
            volatile: None,
            compartment: None,
        }))
        .await
        .unwrap();
    }
    s.recall_memories(Parameters(RecallParams {
        host: None,
        repo: None,
        branch: None,
        query: "scheduling".into(),
        top_k: Some(n as u32),
        network: None,
        full,
        floor: None,
    }))
    .await
    .unwrap()
    .0
    .memories
}

/// The payload has a stated ceiling, and this test is what holds it there.
#[tokio::test]
async fn a_default_recall_of_long_memories_stays_under_a_stated_size() {
    let s = server().await;
    let hits = recall_long(&s, 5, None).await;
    assert_eq!(hits.len(), 5, "all five come back");

    let prose: usize = hits.iter().map(|m| m.content.chars().count()).sum();
    assert!(
        prose <= 5 * RECALL_CONTENT_CHARS,
        "five rows of bounded content is at most five bounds, got {prose}"
    );
    // The whole serialized payload, envelopes included, under the budget a
    // session-start hook has to live inside.
    let payload = serde_json::to_string(&hits).unwrap();
    assert!(
        payload.len() < 10_000,
        "payload {} chars must fit a hook's 10,000-character context",
        payload.len()
    );
}

/// The marker is present exactly when the cut happened, and `content_chars`
/// still reports the STORED length either way.
#[tokio::test]
async fn the_marker_is_present_exactly_when_the_content_was_cut() {
    let s = server().await;
    let short = "a short note about scheduling";
    s.store_memory(Parameters(StoreParams {
        provenance: None,
        content: short.into(),
        network: "world".into(),
        confidence: Some(0.9),
        evidence: None,
        volatile: None,
        compartment: None,
    }))
    .await
    .unwrap();
    let hits = recall_long(&s, 2, None).await;

    for m in &hits {
        if m.truncated {
            assert_eq!(
                m.content.chars().count(),
                RECALL_CONTENT_CHARS,
                "a cut row is cut to exactly the bound"
            );
            assert!(
                m.content_chars as usize > RECALL_CONTENT_CHARS,
                "content_chars reports the STORED length, not the returned one"
            );
        } else {
            assert_eq!(
                m.content.chars().count(),
                m.content_chars as usize,
                "an uncut row returns everything it says it has"
            );
        }
    }
    assert!(
        hits.iter().any(|m| m.truncated),
        "the long rows must have been cut, or this test proves nothing"
    );
    let whole = hits.iter().find(|m| m.content == short);
    if let Some(w) = whole {
        assert!(!w.truncated, "a row under the bound carries no marker");
    }
}

/// `full: true` returns the untruncated text with no marker.
#[tokio::test]
async fn full_returns_everything_and_says_nothing_was_cut() {
    let s = server().await;
    let hits = recall_long(&s, 3, Some(true)).await;
    assert!(!hits.is_empty());
    for m in &hits {
        assert!(!m.truncated, "nothing is cut when full was asked for");
        assert_eq!(
            m.content.chars().count(),
            m.content_chars as usize,
            "the whole stored text came back"
        );
        assert!(
            m.content.chars().count() > RECALL_CONTENT_CHARS,
            "these rows are longer than the bound, so this is a real check"
        );
    }
}

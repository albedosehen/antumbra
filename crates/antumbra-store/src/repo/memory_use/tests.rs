use super::*;
use crate::repo::{memory_policy, principal};
use crate::schema::EMBED_DIM;
use antumbra_core::{MemoryPolicy, UserId};

fn ids(names: &[&str]) -> Vec<MemoryId> {
    names.iter().map(|n| MemoryId::new(*n)).collect()
}

fn find<'a>(seen: &'a [MemoryUse], id: &str) -> &'a MemoryUse {
    seen.iter()
        .find(|u| u.memory.as_str() == id)
        .unwrap_or_else(|| panic!("{id} not seen"))
}

#[tokio::test]
async fn counts_start_on_first_sight_and_add_up() -> Result<()> {
    let s = Store::connect_memory(EMBED_DIM).await?;
    let ws = TenantId::new("ws:a");
    let now = Utc::now();
    record(
        &s,
        &ws,
        &ids(&["memory:1", "memory:2"]),
        Seen::RECALLED,
        now,
    )
    .await?;
    record(
        &s,
        &ws,
        &ids(&["memory:1", "memory:1"]),
        Seen::RECALLED,
        now,
    )
    .await?;
    record(&s, &ws, &ids(&["memory:1"]), Seen::USED, now).await?;
    record(&s, &ws, &ids(&["memory:2"]), Seen::PASSED_OVER, now).await?;

    let seen = all(&s, &ws).await?;
    assert_eq!(seen.len(), 2);
    let one = find(&seen, "memory:1");
    // A repeated id in one observation counts once.
    assert_eq!((one.recalled, one.judged, one.used), (2, 1, 1));
    assert!(one.last_used_at.is_some() && one.last_recalled_at.is_some());
    let two = find(&seen, "memory:2");
    assert_eq!((two.recalled, two.judged, two.used), (1, 1, 0));
    assert_eq!(two.passed_over(), 1);
    assert!(
        two.last_used_at.is_none(),
        "a pass-over does not restart the fade"
    );
    Ok(())
}

#[tokio::test]
async fn an_agents_judgment_and_a_restatement_count_as_uses() -> Result<()> {
    let s = Store::connect_memory(EMBED_DIM).await?;
    let ws = TenantId::new("ws:a");
    let now = Utc::now();
    record(&s, &ws, &ids(&["memory:1"]), Seen::GOOD, now).await?;
    record(&s, &ws, &ids(&["memory:1"]), Seen::RECURRED, now).await?;
    record(&s, &ws, &ids(&["memory:1"]), Seen::BAD, now).await?;
    record(&s, &ws, &ids(&["memory:1"]), Seen::USED, now).await?;
    let got = for_memories(&s, &ws, &ids(&["memory:1", "memory:none"])).await?;
    assert_eq!(got.len(), 1);
    let u = &got[0];
    assert_eq!(
        (u.good, u.recurred, u.bad, u.used, u.judged),
        (1, 1, 1, 1, 1)
    );
    assert_eq!(u.uses(), 3);
    assert_eq!(u.passed_over(), 0);
    Ok(())
}

#[tokio::test]
async fn a_workspace_counts_and_reads_only_its_own() -> Result<()> {
    let s = Store::connect_memory(EMBED_DIM).await?;
    let a = TenantId::new("ws:a");
    let b = TenantId::new("ws:b");
    let user = UserId::new("user:a");
    principal::provision(&s, &a, &user).await?;
    principal::provision(&s, &b, &UserId::new("user:b")).await?;
    let now = Utc::now();
    record(&s, &b, &ids(&["memory:shared"]), Seen::RECALLED, now).await?;

    s.signin(&a, &user).await?;
    record(&s, &a, &ids(&["memory:shared"]), Seen::RECALLED, now).await?;
    record(&s, &a, &ids(&["memory:shared"]), Seen::USED, now).await?;
    let mine = all(&s, &a).await?;
    assert_eq!(mine.len(), 1);
    assert_eq!((mine[0].recalled, mine[0].used), (1, 1));
    assert!(all(&s, &b).await?.is_empty(), "the engine hides another's");

    // A workspace session reads its knobs but cannot set them.
    let turned = MemoryPolicy {
        auto: false,
        ..MemoryPolicy::default()
    };
    let _ = memory_policy::set(&s, &a, &turned).await;
    s.signin_root().await?;
    assert_eq!(memory_policy::get(&s, &a).await?, MemoryPolicy::default());
    assert_eq!(find(&all(&s, &b).await?, "memory:shared").recalled, 1);
    Ok(())
}

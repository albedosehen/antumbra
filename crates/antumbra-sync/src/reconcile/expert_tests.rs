use super::*;
use antumbra_core::{
    AntumbraError, Compartment, CompartmentId, Expert, ExpertId, Generation, TenantId, UserId,
};
use antumbra_store::repo::{compartment, expert};
use antumbra_store::testing::{mem_store, members};
use antumbra_store::{tables, EMBED_DIM};
use serde_json::json;

use crate::config::Fabric;
use crate::table::PENUMBRA_TABLES;

fn tenant() -> TenantId {
    TenantId::new("ws:org")
}

fn unit() -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    v[0] = 1.0;
    v
}

/// A standing expert as a keeper mints one: a user's, from one of their
/// compartments, placed on the machine that trained it, its adapter sealed.
fn standing(id: &str, owner: &str, from: &str) -> Expert {
    Expert {
        id: ExpertId::new(id),
        name: id.to_string(),
        base_model: "base".into(),
        artifact_uri: "adapters/first.safetensors".into(),
        capability_card: json!({ "standing": true, "adapter": { "sha256": "aa", "bytes": 4 } }),
        capability_vec: Some(unit()),
        fitness: 1.0,
        frozen_at: Some(Utc::now()),
        generation: Generation::ZERO,
        owner: Some(UserId::new(owner)),
        compartment: Some(CompartmentId::new(from)),
        placed_on: Some("the-rig".into()),
        created_at: Utc::now(),
    }
}

/// An expert of the shared population: no owner, no compartment.
fn shared(id: &str) -> Expert {
    Expert {
        owner: None,
        compartment: None,
        ..standing(id, "user:nobody", "comp:none")
    }
}

/// The table's rows as they are stored, by record id, tombstones included.
async fn records(store: &Store) -> Result<BTreeMap<String, Value>> {
    Ok(index_by_id(
        row_repo::list_rows(store, tables::EXPERT).await?,
    ))
}

/// The ids of the experts `store` reads, in order.
async fn held(store: &Store) -> Result<Vec<String>> {
    let mut ids: Vec<String> = expert::list(store)
        .await?
        .into_iter()
        .map(|e| e.id.as_str().to_string())
        .collect();
    ids.sort();
    Ok(ids)
}

/// Write `e` into `store` as a server did before an expert's record was named
/// for it: under the record id `at`, as the engine would assign one, with no
/// version.
async fn insert_as_before(store: &Store, e: &Expert, at: &str) -> Result<()> {
    let scratch = mem_store().await?;
    expert::insert(&scratch, e).await?;
    let mut row = row_repo::list_rows(&scratch, tables::EXPERT)
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| AntumbraError::other("the scratch store kept no row"))?;
    let Value::Object(fields) = &mut row else {
        return Err(AntumbraError::other("a stored row is not an object"));
    };
    fields.insert("id".into(), json!(format!("{}:{at}", tables::EXPERT)));
    fields.remove("updated_at");
    match row_repo::put_row(store, &row).await? {
        row_repo::Written::Yes => Ok(()),
        other => Err(AntumbraError::other(format!(
            "the row was not written: {other:?}"
        ))),
    }
}

/// The spec the collector carries experts under.
fn experts() -> Result<&'static [TableSpec]> {
    PENUMBRA_TABLES
        .iter()
        .find(|spec| spec.name == tables::EXPERT)
        .map(std::slice::from_ref)
        .ok_or_else(|| AntumbraError::other("experts are not replicated"))
}

/// The reason the table is carried at all. A machine with its own store gets
/// the row of an expert trained elsewhere: the same record, still placed on
/// the machine that trained it, so the server that reads it pulls the adapter
/// and does not take it for a file of its own.
#[tokio::test]
async fn an_expert_crosses_as_the_record_it_is_still_placed_where_it_was_trained() -> Result<()> {
    let (laptop, rig) = (mem_store().await?, mem_store().await?);
    let trained = standing(
        "expert:ws:org:user:lily:behavior:owner/repo",
        "user:lily",
        "comp:hers",
    );
    expert::insert(&rig, &trained).await?;
    assert!(held(&laptop).await?.is_empty());

    let stats = reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    assert_eq!((stats.pulled, stats.pushed, stats.refused), (1, 0, 0));

    let landed = expert::get(&laptop, &trained.id)
        .await?
        .ok_or_else(|| AntumbraError::other("the expert did not cross"))?;
    assert_eq!(landed.placed_on.as_deref(), Some("the-rig"));
    assert_eq!(landed.artifact_uri, trained.artifact_uri);
    assert_eq!(landed.capability_card, trained.capability_card);
    assert_eq!(landed.owner, trained.owner);
    assert_eq!(landed.compartment, trained.compartment);
    // One record, the one the rig has.
    assert_eq!(
        records(&laptop).await?.keys().collect::<Vec<_>>(),
        records(&rig).await?.keys().collect::<Vec<_>>()
    );
    // And it is routed to where it landed.
    let nearest = expert::knn_by_capability(&laptop, &unit(), 1).await?;
    assert_eq!(
        nearest.first().map(|e| e.id.as_str()),
        Some(trained.id.as_str())
    );

    // Settled: the next pass moves nothing.
    assert_eq!(
        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?,
        ReconcileStats::default()
    );
    Ok(())
}

/// A retraining is a delete and an insert, which was a new record under a new
/// id: the next pass offered it to a store that held the old one and the
/// unique index refused it, failing the cycle. It is the same record now, at
/// a later version, and a collector past its first pass picks it up by that.
#[tokio::test]
async fn a_retraining_replaces_the_row_on_the_next_pass() -> Result<()> {
    let (laptop, rig) = (mem_store().await?, mem_store().await?);
    let first = standing(
        "expert:ws:org:user:lily:behavior:owner/repo",
        "user:lily",
        "comp:hers",
    );
    expert::insert(&rig, &first).await?;
    let mut cursors = Cursors::new();
    reconcile_all_since(&laptop, &rig, PENUMBRA_TABLES, &mut cursors, Duration::ZERO).await?;
    let before = records(&laptop).await?;

    // As the keeper retrains: the old one dropped, the new one written.
    let again = Expert {
        artifact_uri: "adapters/second.safetensors".into(),
        capability_card: json!({ "standing": true, "adapter": { "sha256": "bb", "bytes": 8 } }),
        ..first.clone()
    };
    expert::delete(&rig, &first.id).await?;
    expert::insert(&rig, &again).await?;

    let stats =
        reconcile_all_since(&laptop, &rig, PENUMBRA_TABLES, &mut cursors, Duration::ZERO).await?;
    assert_eq!((stats.pulled, stats.pushed), (1, 0));

    let after = records(&laptop).await?;
    assert_eq!(
        after.keys().collect::<Vec<_>>(),
        before.keys().collect::<Vec<_>>(),
        "replaced in the record it had"
    );
    let landed = expert::get(&laptop, &first.id)
        .await?
        .ok_or_else(|| AntumbraError::other("the retrained expert is not read"))?;
    assert_eq!(landed.artifact_uri, "adapters/second.safetensors");
    assert_eq!(landed.capability_card, again.capability_card);
    Ok(())
}

/// A delete crosses as a tombstone, and the copy the other store still held
/// live does not bring the expert back, however many passes run.
#[tokio::test]
async fn a_deleted_expert_is_gone_from_the_other_store_and_does_not_come_back() -> Result<()> {
    let (laptop, rig) = (mem_store().await?, mem_store().await?);
    let e = standing(
        "expert:ws:org:user:lily:behavior:owner/repo",
        "user:lily",
        "comp:hers",
    );
    expert::insert(&rig, &e).await?;
    reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    assert_eq!(held(&laptop).await?.len(), 1);

    expert::delete(&rig, &e.id).await?;
    for _ in 0..3 {
        reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    }

    for (name, store) in [("laptop", &laptop), ("rig", &rig)] {
        assert!(held(store).await?.is_empty(), "{name} still reads it");
        assert!(
            expert::knn_by_capability(store, &unit(), 1)
                .await?
                .is_empty(),
            "{name} still routes to it"
        );
        let kept = records(store).await?;
        assert_eq!(kept.len(), 1, "{name} keeps one record of it");
        assert!(
            kept.values()
                .all(|row| row.get("deleted_at").is_some_and(Value::is_string)),
            "{name}'s record is a tombstone"
        );
    }
    Ok(())
}

/// A row from before records were named for their expert is a different
/// record on every store. It is left where it is, in both directions, while
/// the rows beside it cross.
#[tokio::test]
async fn a_row_under_an_engine_assigned_id_is_carried_in_neither_direction() -> Result<()> {
    let (laptop, rig) = (mem_store().await?, mem_store().await?);
    insert_as_before(&laptop, &shared("expert:old-here"), "8k2jq0xw1v5n").await?;
    insert_as_before(&rig, &shared("expert:old-there"), "3m9d7c4b6a1z").await?;
    expert::insert(&laptop, &shared("expert:new-here")).await?;

    let stats = reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    assert_eq!((stats.pushed, stats.pulled), (1, 0));

    assert_eq!(
        held(&laptop).await?,
        vec!["expert:new-here", "expert:old-here"]
    );
    assert_eq!(
        held(&rig).await?,
        vec!["expert:new-here", "expert:old-there"]
    );

    // Opening a store moves such a row to its expert's own record, as its
    // first version; the pass after that carries it.
    laptop.ensure_schema().await?;
    rig.ensure_schema().await?;
    let stats = reconcile_all(&laptop, &rig, PENUMBRA_TABLES).await?;
    assert_eq!((stats.pushed, stats.pulled), (1, 1));
    let all = vec!["expert:new-here", "expert:old-here", "expert:old-there"];
    assert_eq!(held(&laptop).await?, all);
    assert_eq!(held(&rig).await?, all);
    Ok(())
}

/// The one way a pass over this table fails, and what ends it. A store written
/// into by a server from before, holding an expert under an engine-assigned
/// id that the other store has under its own: the row is not carried from
/// it, and the one carried to it is refused by the unique index, loudly, with
/// neither copy changed. The collector reconnects on a failed pass, which
/// opens the store and moves the row, and the pass after that settles.
#[tokio::test]
async fn a_store_not_yet_moved_refuses_the_expert_it_has_until_it_is_opened_again() -> Result<()> {
    let (laptop, rig) = (mem_store().await?, mem_store().await?);
    let old = shared("expert:both");
    insert_as_before(&laptop, &old, "8k2jq0xw1v5n").await?;
    let newer = Expert {
        artifact_uri: "adapters/second.safetensors".into(),
        ..old.clone()
    };
    expert::insert(&rig, &newer).await?;

    assert!(
        reconcile_all(&laptop, &rig, experts()?).await.is_err(),
        "the laptop was handed a second record for an expert it has"
    );
    assert_eq!(records(&laptop).await?.len(), 1);
    assert_eq!(records(&rig).await?.len(), 1);

    laptop.ensure_schema().await?;
    let stats = reconcile_all(&laptop, &rig, experts()?).await?;
    assert_eq!((stats.pulled, stats.pushed), (1, 0), "the later write wins");
    assert_eq!(records(&laptop).await?.len(), 1);
    let landed = expert::get(&laptop, &old.id)
        .await?
        .ok_or_else(|| AntumbraError::other("the expert is not read"))?;
    assert_eq!(landed.artifact_uri, "adapters/second.safetensors");
    Ok(())
}

/// Two members with a compartment each, a standing expert from each, and one
/// shared expert, all on the local store.
async fn org() -> Result<(Store, Store)> {
    let (local, remote) = (mem_store().await?, mem_store().await?);
    let lily = UserId::new("user:lily");
    let oslo = UserId::new("user:oslo");
    for store in [&local, &remote] {
        members(store, &tenant(), &[&lily, &oslo]).await?;
    }
    for (id, owner) in [("comp:hers", &lily), ("comp:his", &oslo)] {
        compartment::create(
            &local,
            &Compartment::new(id, tenant(), owner.clone(), id, Utc::now()),
        )
        .await?;
    }
    expert::insert(&local, &standing("expert:hers", "user:lily", "comp:hers")).await?;
    expert::insert(&local, &standing("expert:his", "user:oslo", "comp:his")).await?;
    expert::insert(&local, &shared("expert:shared")).await?;
    Ok((local, remote))
}

/// What belongs to one user's fabric: the expert trained from their own
/// compartment. Another member's is not theirs, and the shared population is
/// the owner's to move.
#[tokio::test]
async fn a_scoped_pass_carries_a_user_s_standing_expert_and_no_one_else_s() -> Result<()> {
    let (local, remote) = org().await?;
    let scope = Scope::resolve(&local, &Fabric::new("ws:org", "user:lily")).await?;

    let stats = reconcile_all_scoped(&local, &remote, experts()?, Some(&scope)).await?;
    assert_eq!((stats.pushed, stats.refused), (1, 0));
    assert_eq!(held(&remote).await?, vec!["expert:hers"]);

    // The control: with no scope the same pass carries the population.
    let (local, remote) = org().await?;
    reconcile_all(&local, &remote, experts()?).await?;
    assert_eq!(
        held(&remote).await?,
        vec!["expert:hers", "expert:his", "expert:shared"]
    );
    Ok(())
}

/// The limit that stands. Only an owner connection writes the population, so
/// a collector signed in as the user offers their expert and the engine
/// refuses it. That is counted as a refusal, which the collector says aloud,
/// and never as a row that crossed.
#[tokio::test]
async fn a_pass_signed_in_as_the_user_has_their_expert_refused_and_counts_it() -> Result<()> {
    let (local, remote) = org().await?;
    let fabric = Fabric::new("ws:org", "user:lily");
    fabric.bind(&local).await?;
    fabric.bind(&remote).await?;
    let scope = Scope::resolve(&local, &fabric).await?;

    let stats = reconcile_all_scoped(&local, &remote, experts()?, Some(&scope)).await?;
    assert_eq!((stats.pushed, stats.pulled, stats.refused), (0, 0, 1));

    remote.invalidate().await?;
    assert!(held(&remote).await?.is_empty(), "nothing landed");
    Ok(())
}

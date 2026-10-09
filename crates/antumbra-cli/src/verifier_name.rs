//! `antumbra verifier name`: a corpus whose tasks take their
//! reward from the namespace. A task that has a synthesized verifier able to
//! grant reward for it now names that verifier instead of carrying its own
//! spec. Every other task keeps its spec, and so does the namespace: a task's
//! authored verifier stays where it is, as the anchor the loop rechecks the
//! synthesized one against.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};

use antumbra_core::{VerifierId, VerifierOrigin};
use antumbra_store::repo::verifier;
use antumbra_store::Store;

/// For each task in `domain`, the synthesized verifier written for it that
/// may grant reward at `now`: the lowest address, when there are several.
pub(crate) async fn granting(
    store: &Store,
    domain: &str,
    now: DateTime<Utc>,
) -> anyhow::Result<BTreeMap<String, VerifierId>> {
    let mut records = verifier::list(store).await?;
    records.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    let mut named = BTreeMap::new();
    for record in records {
        if record.origin != VerifierOrigin::Synthesized || record.domain != domain {
            continue;
        }
        let Some(task) = record.task.clone() else {
            continue;
        };
        if named.contains_key(&task) || !verifier::grants(store, &record, now).await? {
            continue;
        }
        named.insert(task, record.id);
    }
    Ok(named)
}

/// Point each of `tasks` that `granting` holds a verifier for at it. Returns
/// how many were named.
pub(crate) fn name_tasks(
    tasks: &mut [serde_json::Value],
    granting: &BTreeMap<String, VerifierId>,
) -> usize {
    let mut named = 0;
    for task in tasks.iter_mut() {
        let Some(id) = task["id"].as_str().and_then(|id| granting.get(id)) else {
            continue;
        };
        task["verify"] = serde_json::json!({ "verifier": id.as_str() });
        named += 1;
    }
    named
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::{Tally, TrustPolicy, VerifierRecord, VerifierTier};

    async fn propose(
        store: &Store,
        task: &str,
        origin: VerifierOrigin,
        spec: &str,
    ) -> VerifierRecord {
        verifier::propose(
            store,
            &VerifierRecord::new(
                "strings",
                Some(task.to_string()),
                VerifierTier::Reducible,
                origin,
                serde_json::json!({ "program": spec }),
                Utc::now(),
            ),
        )
        .await
        .unwrap()
    }

    async fn trust(store: &Store, record: &VerifierRecord) {
        let sound = Tally {
            repeats: 3,
            good: 5,
            good_passed: 5,
            bad: 29,
            ..Tally::default()
        }
        .judge(&record.id, Utc::now(), &TrustPolicy::default());
        verifier::record_measurement(store, record, &sound)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn only_a_task_with_a_granting_synthesized_verifier_is_named() {
        let store = Store::connect_memory(8).await.unwrap();
        propose(&store, "swap", VerifierOrigin::Authored, "authored").await;
        let trusted = propose(&store, "swap", VerifierOrigin::Synthesized, "a").await;
        trust(&store, &trusted).await;
        // Proposed and never measured: it grants nothing.
        propose(&store, "caesar", VerifierOrigin::Synthesized, "b").await;
        let found = granting(&store, "strings", Utc::now()).await.unwrap();
        assert_eq!(
            found.into_iter().collect::<Vec<_>>(),
            vec![("swap".to_string(), trusted.id.clone())]
        );
        assert!(granting(&store, "grids", Utc::now())
            .await
            .unwrap()
            .is_empty());

        let mut tasks = vec![
            serde_json::json!({ "id": "swap", "verify": { "program": "authored" } }),
            serde_json::json!({ "id": "caesar", "verify": { "program": "own" } }),
        ];
        let map = BTreeMap::from([("swap".to_string(), trusted.id.clone())]);
        assert_eq!(name_tasks(&mut tasks, &map), 1);
        assert_eq!(tasks[0]["verify"]["verifier"], trusted.id.as_str());
        assert_eq!(tasks[1]["verify"]["program"], "own");
    }
}

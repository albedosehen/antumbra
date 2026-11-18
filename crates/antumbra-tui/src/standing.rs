//! What the server is teaching: each scope a user has behaviors or a standing
//! expert in, how many of the behaviors are taught, wait, or are only
//! proposed, and where the scope's training stands.
//!
//! The server's keeper decides this on every pass, from a user's behaviors
//! and their experts. The console reads both, so it works the same thing out,
//! with the same functions (`antumbra_core::behavior`): what a scope's expert
//! is taught, whether the expert is behind it, and whether the set as it
//! stands was refused. Nothing is asked of the server for it.

use std::collections::BTreeMap;

use antumbra_core::behavior::{self, State, Status};
use antumbra_core::{Expert, ExpertId, Memory, Shadow};
use chrono::{DateTime, Duration, Utc};

/// Where a scope's training stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// The expert holds what there is to teach.
    UpToDate,
    /// What there is to teach has changed since the expert was trained, or
    /// there is no expert yet: the keeper's next pass trains it.
    Waiting,
    /// The set as it stands was trained and not admitted. It is not trained
    /// again until a behavior in it changes.
    Refused,
    /// Nothing is left to teach: the next pass drops the expert.
    Dropping,
    /// Its behaviors are only proposed: nothing is taught until one is
    /// accepted.
    Proposed,
}

/// One scope of one user.
#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    /// `everywhere`, or a repository.
    pub scope: String,
    /// The scope's standing expert, where it has one.
    pub expert: Option<ExpertId>,
    /// Behaviors the expert was trained into.
    pub trained: usize,
    /// Behaviors accepted and not in the expert: not yet trained, or trained
    /// and not learned.
    pub accepted: usize,
    /// Behaviors proposed, waiting to be accepted.
    pub proposed: usize,
    pub state: Standing,
    /// When the scope's expert was last trained, where the console may read
    /// the trainings (an operator's token).
    pub last_trained: Option<DateTime<Utc>>,
    /// How long until the server accepts the first of its proposed behaviors
    /// that can be taught. `None` when none is waiting to be, or when the
    /// workspace leaves accepting to its users.
    pub accepts_in: Option<Duration>,
}

/// Every scope there is something to say about, user by user, each user's
/// `everywhere` first. `accept_after_hours` is the workspace's knob: how long
/// a proposed behavior waits before the server accepts it, 0 for never.
pub fn read(
    memories: &[Memory],
    experts: &[Expert],
    shadows: &[Shadow],
    accept_after_hours: f32,
    now: DateTime<Utc>,
) -> Vec<Scope> {
    // One user's behaviors are in one compartment, and their standing
    // experts name it as what they were trained from.
    let mut owners: BTreeMap<&str, (Vec<Memory>, Vec<Expert>)> = BTreeMap::new();
    for m in memories {
        if let (Some(compartment), Some(_)) = (m.compartment.as_ref(), State::of(&m.evidence)) {
            let owned = owners.entry(compartment.as_str()).or_default();
            owned.0.push(m.clone());
        }
    }
    for e in experts {
        if let (Some(compartment), Some(_)) = (e.compartment.as_ref(), e.standing_scope()) {
            let owned = owners.entry(compartment.as_str()).or_default();
            owned.1.push(e.clone());
        }
    }
    let wait = Duration::minutes((f64::from(accept_after_hours) * 60.0).round() as i64);
    let mut scopes = Vec::new();
    for (behaviors, standing) in owners.values() {
        for scope in behavior::scopes(behaviors, standing) {
            let count = |status: Status| {
                behaviors
                    .iter()
                    .filter_map(|m| State::of(&m.evidence))
                    .filter(|state| state.scope == scope && state.status == status)
                    .count()
            };
            let (trained, accepted, proposed) = (
                count(Status::Trained),
                count(Status::Accepted),
                count(Status::Proposed),
            );
            let expert = standing
                .iter()
                .find(|e| e.standing_scope() == Some(scope.as_str()));
            // `everywhere` is always among a user's scopes, with or without
            // anything in it.
            if trained + accepted + proposed == 0 && expert.is_none() {
                continue;
            }
            let taught = behavior::learnable(behaviors, &scope);
            let state = if taught.is_empty() {
                if expert.is_some() {
                    Standing::Dropping
                } else {
                    Standing::Proposed
                }
            } else if !behavior::stale(&taught, expert) {
                Standing::UpToDate
            } else if behavior::refused(&taught) {
                Standing::Refused
            } else {
                Standing::Waiting
            };
            let accepts_in = (wait > Duration::zero())
                .then(|| {
                    behaviors
                        .iter()
                        .filter(|m| {
                            State::of(&m.evidence).is_some_and(|state| {
                                state.scope == scope && state.status == Status::Proposed
                            })
                        })
                        .filter(|m| {
                            behavior::spec_of(&m.content).is_some_and(|s| s.problems().is_empty())
                        })
                        .map(|m| (m.updated_at + wait - now).max(Duration::zero()))
                        .min()
                })
                .flatten();
            scopes.push(Scope {
                last_trained: expert.and_then(|e| last_trained(shadows, &e.id, &scope)),
                expert: expert.map(|e| e.id.clone()),
                scope,
                trained,
                accepted,
                proposed,
                state,
                accepts_in,
            });
        }
    }
    scopes
}

/// When `expert`, the standing expert for `scope`, was last trained: the
/// newest of its trainings, each of which is a shadow named for it with the
/// set it trained after. An expert minted under an older form of id shares
/// no name with its trainings, and is then found by its scope.
fn last_trained(shadows: &[Shadow], expert: &ExpertId, scope: &str) -> Option<DateTime<Utc>> {
    let newest =
        |of: &dyn Fn(&Shadow) -> bool| shadows.iter().filter(|s| of(s)).map(|s| s.created_at).max();
    let named = expert.as_str().strip_prefix("expert:").unwrap_or_default();
    newest(&|s| {
        s.id.as_str()
            .strip_prefix("shadow:")
            .and_then(|rest| rest.rsplit_once(':'))
            .is_some_and(|(of, _set)| of == named)
    })
    .or_else(|| newest(&|s| behavior::shadow_scope(&s.id) == Some(scope)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use antumbra_core::behavior::{content, scope_evidence, status_evidence, Example, Spec};
    use antumbra_core::{Generation, MemoryNetwork, ShadowStatus, TenantId, UserId};

    const TENANT: &str = "ws:demo";
    const USER: &str = "user:demo";

    /// A behavior that can be taught: four examples and a check that tells
    /// them from its violation.
    fn spec(rule: &str) -> Spec {
        Spec {
            rule: rule.into(),
            must: vec!["x".into()],
            must_not: Vec::new(),
            examples: (0..4)
                .map(|i| Example {
                    task: format!("task {i}"),
                    answer: "x".into(),
                })
                .collect(),
            violations: vec!["y".into()],
        }
    }

    /// One of the demo user's behaviors, last touched `hours` ago.
    pub(crate) fn recorded(id: &str, status: Status, scope: &str, hours: i64) -> Memory {
        let (tenant, user) = (TenantId::new(TENANT), UserId::new(USER));
        let mut m = Memory::new(
            id,
            tenant.clone(),
            MemoryNetwork::Opinion,
            content(&spec(&format!("The rule of {id}."))),
            1.0,
            Utc::now() - Duration::hours(hours),
        )
        .with_evidence(vec![status_evidence(status), scope_evidence(scope)]);
        m.compartment = Some(behavior::compartment_id(&tenant, &user));
        m.updated_at = Utc::now() - Duration::hours(hours);
        m
    }

    /// The demo user's standing expert for `scope`, trained on `taught`.
    pub(crate) fn expert(scope: &str, taught: &[Memory]) -> Expert {
        let (tenant, user) = (TenantId::new(TENANT), UserId::new(USER));
        let pairs: Vec<(Memory, Spec)> = taught
            .iter()
            .filter_map(|m| Some((m.clone(), behavior::spec_of(&m.content)?)))
            .collect();
        let now = Utc::now();
        Expert {
            id: ExpertId::new(format!("expert:{TENANT}:{USER}:behavior:{scope}")),
            name: format!("expert:{TENANT}:{USER}:behavior:{scope}"),
            base_model: "base".into(),
            artifact_uri: "adapters/standing.safetensors".into(),
            capability_card: serde_json::json!({
                "standing": true,
                "scope": scope,
                "set": behavior::fingerprint(&pairs),
            }),
            capability_vec: None,
            fitness: 0.8,
            frozen_at: Some(now),
            generation: Generation::ZERO,
            owner: Some(user.clone()),
            compartment: Some(behavior::compartment_id(&tenant, &user)),
            placed_on: None,
            created_at: now,
        }
    }

    /// A training of the demo user's expert for `scope`, `days` ago.
    pub(crate) fn training(scope: &str, days: i64) -> Shadow {
        Shadow {
            id: behavior::shadow_id(
                &TenantId::new(TENANT),
                &UserId::new(USER),
                scope,
                "0123456789abcdef",
            ),
            parent_expert: None,
            adapter_uri: None,
            status: ShadowStatus::Graduated,
            generation: Generation::ZERO,
            reward_curve: vec![0.4, 0.8],
            created_at: Utc::now() - Duration::days(days),
        }
    }

    #[test]
    fn a_scope_whose_expert_holds_its_behaviors_is_up_to_date() -> anyhow::Result<()> {
        let taught = vec![recorded("memory:a", Status::Trained, "everywhere", 100)];
        let scopes = read(
            &taught,
            &[expert("everywhere", &taught)],
            &[training("everywhere", 3)],
            24.0,
            Utc::now(),
        );
        assert_eq!(scopes.len(), 1);
        let s = &scopes[0];
        assert_eq!((s.trained, s.accepted, s.proposed), (1, 0, 0));
        assert_eq!(s.state, Standing::UpToDate);
        assert_eq!(s.accepts_in, None);
        let Some(when) = s.last_trained else {
            anyhow::bail!("its training was not found");
        };
        assert_eq!((Utc::now() - when).num_days(), 3);
        Ok(())
    }

    #[test]
    fn a_behavior_accepted_since_the_training_makes_the_scope_wait() -> anyhow::Result<()> {
        let taught = vec![recorded("memory:a", Status::Trained, "everywhere", 100)];
        let mut all = taught.clone();
        all.push(recorded("memory:b", Status::Accepted, "everywhere", 1));
        let scopes = read(
            &all,
            &[expert("everywhere", &taught)],
            &[],
            24.0,
            Utc::now(),
        );
        assert_eq!(scopes[0].state, Standing::Waiting);
        assert_eq!((scopes[0].trained, scopes[0].accepted), (1, 1));
        // And a scope with something accepted and no expert yet waits too.
        let new = vec![recorded(
            "memory:c",
            Status::Accepted,
            "github.com/acme/orders",
            1,
        )];
        let scopes = read(&new, &[], &[], 24.0, Utc::now());
        assert_eq!(
            scopes.len(),
            1,
            "everywhere, with nothing in it, is not listed"
        );
        assert_eq!(scopes[0].scope, "github.com/acme/orders");
        assert_eq!(scopes[0].state, Standing::Waiting);
        assert_eq!(scopes[0].expert, None);
        Ok(())
    }

    #[test]
    fn a_proposed_behavior_says_when_the_server_accepts_it() -> anyhow::Result<()> {
        let proposed = vec![
            recorded("memory:old", Status::Proposed, "everywhere", 20),
            recorded("memory:new", Status::Proposed, "everywhere", 2),
        ];
        let scopes = read(&proposed, &[], &[], 24.0, Utc::now());
        assert_eq!(scopes[0].state, Standing::Proposed);
        assert_eq!(scopes[0].proposed, 2);
        // The one that has waited longest is accepted first.
        let Some(left) = scopes[0].accepts_in else {
            anyhow::bail!("nothing says when it is accepted");
        };
        assert!((3..=4).contains(&left.num_hours()), "{left}");
        // One that has waited past the wait is due now, not in the past.
        let scopes = read(&proposed, &[], &[], 6.0, Utc::now());
        assert_eq!(scopes[0].accepts_in, Some(Duration::zero()));
        // A workspace that leaves accepting to its users says no time.
        let scopes = read(&proposed, &[], &[], 0.0, Utc::now());
        assert_eq!(scopes[0].accepts_in, None);
        Ok(())
    }

    #[test]
    fn an_expert_with_nothing_left_to_teach_is_to_be_dropped() -> anyhow::Result<()> {
        let was = vec![recorded("memory:a", Status::Trained, "everywhere", 100)];
        let retired = vec![recorded("memory:a", Status::Retired, "everywhere", 1)];
        let scopes = read(
            &retired,
            &[expert("everywhere", &was)],
            &[],
            24.0,
            Utc::now(),
        );
        assert_eq!(scopes[0].state, Standing::Dropping);
        Ok(())
    }

    #[test]
    fn two_users_scopes_are_kept_apart() -> anyhow::Result<()> {
        let mine = recorded("memory:mine", Status::Accepted, "everywhere", 1);
        let mut theirs = recorded("memory:theirs", Status::Proposed, "everywhere", 1);
        theirs.compartment = Some(behavior::compartment_id(
            &TenantId::new(TENANT),
            &UserId::new("user:other"),
        ));
        let scopes = read(&[mine, theirs], &[], &[], 24.0, Utc::now());
        assert_eq!(scopes.len(), 2);
        let states: Vec<Standing> = scopes.iter().map(|s| s.state).collect();
        assert!(states.contains(&Standing::Waiting) && states.contains(&Standing::Proposed));
        Ok(())
    }
}

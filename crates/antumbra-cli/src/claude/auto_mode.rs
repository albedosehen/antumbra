//! Trust entries for auto mode (ADR-0021): a draft of `autoMode.environment`,
//! the prose that tells the agent's classifier what is inside the user's
//! boundary, so that pushing to their own repository or reaching their own
//! service is not mistaken for exfiltration.
//!
//! The vendor's `/auto-mode-setup` drafts these from the commands of recent
//! sessions and writes them to the user's settings. It is gated on the fetched
//! flags. This drafts from two things that are already the user's, and reads no
//! transcript: the remotes of their working trees, and what Antumbra remembers.
//! It prints and never writes, for the reason the doctor does not: the settings
//! file grants the agent its permissions.
//!
//! The rule for proposing an owner is conservative on purpose. A clone of
//! someone else's repository must not make their organization trusted, so an
//! owner is proposed only when the user demonstrably pushes there (a remote over
//! ssh) and either owns the project at hand or has more than one repository
//! under it. Everything else is listed as seen, with the reason it was left out.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::conventions::Call;

/// One repository, and what is known about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seen {
    /// `host/owner/name`, as [`antumbra_core::repo_slug_from_remote`] spells it.
    pub slug: String,
    /// The remote is over ssh: the user authenticates there, so they can push.
    pub over_ssh: bool,
    /// The session's own project, whose remotes the classifier already trusts.
    pub is_project: bool,
}

/// Whether a remote URL is ssh: `ssh://...`, or the scp form `git@host:owner/name`,
/// which is the only kind of remote with an `@` before its first colon (in a URL
/// with a scheme, what comes before the first colon is the scheme).
pub fn is_ssh(remote: &str) -> bool {
    let remote = remote.trim();
    remote.starts_with("ssh://")
        || remote
            .split_once(':')
            .is_some_and(|(before, _)| before.contains('@'))
}

/// A working tree's `origin`, as a [`Seen`]; `None` for a remote with no slug.
pub fn seen(remote: &str, is_project: bool) -> Option<Seen> {
    antumbra_core::repo_slug_from_remote(remote).map(|slug| Seen {
        slug,
        over_ssh: is_ssh(remote),
        is_project,
    })
}

/// `host/owner`, the unit a `Source control` entry trusts.
fn owner_of(slug: &str) -> Option<String> {
    let mut parts = slug.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(host), Some(owner), Some(_)) => Some(format!("{host}/{owner}")),
        _ => None,
    }
}

/// Everything known about one owner.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Owner {
    pub name: String,
    pub repositories: usize,
    pub over_ssh: usize,
    pub owns_project: bool,
    /// How many of the user's memories are anchored to a repository of theirs.
    pub remembered: usize,
}

impl Owner {
    /// The rule in the module's header.
    pub fn is_proposed(&self) -> bool {
        self.over_ssh > 0 && (self.owns_project || self.repositories > 1)
    }

    fn why_not(&self) -> String {
        if self.over_ssh == 0 && self.repositories == 0 {
            "only in memory: no working tree here pushes there".to_string()
        } else if self.over_ssh == 0 {
            "cloned over https: nothing says you push there".to_string()
        } else {
            "one repository, and not this project: it may be someone else's".to_string()
        }
    }
}

/// Group working trees and memory anchors by owner, most vouched-for first.
pub fn owners(trees: &[Seen], remembered: &[String]) -> Vec<Owner> {
    let mut by_name: BTreeMap<String, Owner> = BTreeMap::new();
    let mut counted = std::collections::BTreeSet::new();
    for tree in trees.iter().filter(|t| counted.insert(t.slug.clone())) {
        let Some(name) = owner_of(&tree.slug) else {
            continue;
        };
        let owner = by_name.entry(name.clone()).or_insert_with(|| Owner {
            name,
            ..Owner::default()
        });
        owner.repositories += 1;
        owner.over_ssh += usize::from(tree.over_ssh);
        owner.owns_project |= tree.is_project;
    }
    for name in remembered.iter().filter_map(|slug| owner_of(slug)) {
        by_name
            .entry(name.clone())
            .or_insert_with(|| Owner {
                name,
                ..Owner::default()
            })
            .remembered += 1;
    }
    let mut all: Vec<Owner> = by_name.into_values().collect();
    all.sort_by(|a, b| {
        (b.is_proposed(), b.repositories, b.remembered)
            .cmp(&(a.is_proposed(), a.repositories, a.remembered))
            .then_with(|| a.name.cmp(&b.name))
    });
    all
}

/// The `Source control` entry, or `None` when no owner earns a place in it.
pub fn source_control_entry(owners: &[Owner]) -> Option<String> {
    let proposed: Vec<String> = owners
        .iter()
        .filter(|o| o.is_proposed())
        .map(|o| format!("{} and all repos under it", o.name))
        .collect();
    (!proposed.is_empty()).then(|| format!("Source control: {}", proposed.join("; ")))
}

/// A slot of `autoMode.environment` that only prose can fill, and what to ask
/// memory in order to fill it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub name: &'static str,
    pub ask: &'static str,
}

/// The slots worth asking memory about. `Source control` is not among them: it
/// is drafted from remotes, which do not need interpreting.
pub fn slots() -> Vec<Slot> {
    vec![
        Slot {
            name: "Key internal services",
            ask: "services I run and the hosts they run on: CI, registries, databases, dashboards",
        },
        Slot {
            name: "Trusted internal domains",
            ask: "internal hostnames, IP addresses and domains of my own machines and network",
        },
        Slot {
            name: "CI/CD deploy targets",
            ask: "where and how I deploy: hosts, clusters, namespaces, git-ops repositories",
        },
        Slot {
            name: "Sensitive remote targets",
            ask: "which hosts, clusters or namespaces are production, and which are test nodes",
        },
        Slot {
            name: "Internal package registry",
            ask: "container registries and package registries I publish to or install from",
        },
        Slot {
            name: "Secrets management",
            ask: "where secrets and tokens are kept and how they reach a deployment",
        },
    ]
}

/// One memory offered for a slot.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub content: String,
    /// How close the surface said it was to the slot's question. Recall answers
    /// whether or not anything is relevant, so without this every slot is
    /// offered the same memories and nothing says which of them fit.
    pub similarity: Option<f32>,
}

fn memories_in(answer: &Value) -> Vec<&Value> {
    answer
        .get("memories")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .collect()
}

/// The repositories the user's memories are anchored to.
pub fn remembered_repositories(call: Call<'_>) -> anyhow::Result<Vec<String>> {
    let listed = call("list_memories", json!({}))?;
    Ok(memories_in(&listed)
        .into_iter()
        .filter_map(|m| m.pointer("/provenance/repo")?.as_str().map(str::to_string))
        .collect())
}

/// What memory offers for each slot, best first. A slot with nothing is kept,
/// so the report can say it asked.
pub fn candidates(
    call: Call<'_>,
    slots: &[Slot],
    each: u32,
) -> anyhow::Result<Vec<(Slot, Vec<Candidate>)>> {
    slots
        .iter()
        .map(|slot| {
            let answer = call(
                "recall_memories",
                json!({ "query": slot.ask, "top_k": each }),
            )?;
            let found = memories_in(&answer)
                .into_iter()
                .filter_map(|m| {
                    Some(Candidate {
                        id: m.get("id")?.as_str()?.to_string(),
                        content: m.get("content")?.as_str()?.to_string(),
                        // Narrowing on purpose: the surface speaks f32, JSON f64.
                        similarity: m
                            .get("similarity")
                            .and_then(Value::as_f64)
                            .map(|s| s as f32),
                    })
                })
                .collect();
            Ok((*slot, found))
        })
        .collect()
}

/// What memory had to say, or why it was not asked.
// Not `Eq`: a candidate carries a similarity, and a float has no total equality.
#[derive(Debug, Clone, PartialEq)]
pub enum Memory {
    Asked(Vec<(Slot, Vec<Candidate>)>),
    NotAsked(String),
}

fn one_line(content: &str) -> String {
    let flat = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = flat.chars().take(200).collect();
    if cut.len() < flat.len() {
        format!("{cut}...")
    } else {
        cut
    }
}

/// The draft as text: the block to paste, then what was left out and why, then
/// what memory offers for the slots only prose can fill.
pub fn render(owners: &[Owner], memory: &Memory) -> String {
    let mut entries = vec!["\"$defaults\"".to_string()];
    entries.extend(
        source_control_entry(owners)
            .into_iter()
            .map(|entry| Value::String(entry).to_string()),
    );
    let mut out = vec![
        "A draft. Nothing was written. The classifier reads autoMode from your own settings"
            .to_string(),
        "file (~/.claude/settings.json) and never from a project's. Remove any owner that is"
            .to_string(),
        "not yours before you paste this:".to_string(),
        String::new(),
        "  \"autoMode\": {".to_string(),
        "    \"environment\": [".to_string(),
        format!("      {}", entries.join(",\n      ")),
        "    ]".to_string(),
        "  }".to_string(),
    ];
    let left_out: Vec<&Owner> = owners.iter().filter(|o| !o.is_proposed()).collect();
    if !left_out.is_empty() {
        out.push(String::new());
        out.push("Seen, and not proposed:".to_string());
        out.extend(
            left_out
                .iter()
                .map(|o| format!("  {}: {}", o.name, o.why_not())),
        );
    }
    out.push(String::new());
    match memory {
        Memory::NotAsked(why) => out.push(format!("Memory was not asked: {why}")),
        Memory::Asked(slots) => {
            // Recall always answers, relevant or not, so one memory turns up under
            // many slots. Each is shown once, with the slots it was offered for.
            let mut offered: Vec<(&Candidate, Vec<&str>)> = Vec::new();
            for (slot, found) in slots {
                for candidate in found {
                    match offered.iter_mut().find(|(seen, _)| seen.id == candidate.id) {
                        Some((_, names)) => names.push(slot.name),
                        None => offered.push((candidate, vec![slot.name])),
                    }
                }
            }
            if offered.is_empty() {
                out.push("Memory was asked about every slot and remembers nothing.".to_string());
            } else {
                out.push(
                    "From memory, for the slots only prose can fill. Candidates, not entries: recall"
                        .to_string(),
                );
                out.push(
                    "always answers, so judge each one. Turn the ones that fit into a line such as"
                        .to_string(),
                );
                out.push(
                    "\"Sensitive remote targets: <host> is production; <host> is a test node\"."
                        .to_string(),
                );
            }
            // Closest first, and say how close: recall answers whatever it is
            // asked, so the number is what separates a memory that fits the
            // slot from one that merely came back.
            offered.sort_by(|a, b| {
                b.0.similarity
                    .unwrap_or(f32::MIN)
                    .total_cmp(&a.0.similarity.unwrap_or(f32::MIN))
                    .then_with(|| a.0.id.cmp(&b.0.id))
            });
            for (candidate, names) in offered {
                let closeness = match candidate.similarity {
                    Some(s) => format!("{s:.2}"),
                    None => "   ?".to_string(),
                };
                out.push(format!(
                    "  {closeness}  {}  {}",
                    candidate.id,
                    one_line(&candidate.content)
                ));
                out.push(format!("      offered for: {}", names.join(", ")));
            }
        }
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(remote: &str, is_project: bool) -> anyhow::Result<Seen> {
        seen(remote, is_project).ok_or_else(|| anyhow::anyhow!("no slug in {remote}"))
    }

    #[test]
    fn ssh_is_told_from_https_in_every_spelling() {
        let cases = [
            ("git@github.com:acme/app.git", true),
            ("ssh://git@github.com/acme/app.git", true),
            ("deploy@10.0.0.5:srv/app.git", true),
            ("https://github.com/acme/app.git", false),
            ("https://user@github.com/acme/app.git", false),
            ("http://gitea.local:3000/acme/app.git", false),
            ("/srv/git/app.git", false),
        ];
        for (remote, ssh) in cases {
            assert_eq!(is_ssh(remote), ssh, "{remote}");
        }
    }

    #[test]
    fn an_owner_is_proposed_only_when_the_user_pushes_there_and_it_is_plainly_theirs(
    ) -> anyhow::Result<()> {
        let trees = [
            tree("git@github.com:mine/project.git", true)?,
            tree("git@github.com:mine/other.git", false)?,
            tree("git@github.com:work/one.git", false)?,
            tree("git@github.com:work/two.git", false)?,
            // Someone else's, cloned to read, over ssh as a habit.
            tree("git@github.com:upstream/library.git", false)?,
            // Someone else's, cloned over https, twice.
            tree("https://github.com/vendor/a.git", false)?,
            tree("https://github.com/vendor/b.git", false)?,
        ];
        let all = owners(&trees, &[]);
        let proposed: Vec<&str> = all
            .iter()
            .filter(|o| o.is_proposed())
            .map(|o| o.name.as_str())
            .collect();
        assert_eq!(proposed, ["github.com/mine", "github.com/work"]);
        assert_eq!(
            source_control_entry(&all),
            Some(
                "Source control: github.com/mine and all repos under it; \
                 github.com/work and all repos under it"
                    .to_string()
            )
        );
        Ok(())
    }

    #[test]
    fn the_project_alone_is_enough_and_a_lone_clone_is_not() -> anyhow::Result<()> {
        let mine = owners(&[tree("git@github.com:solo/project.git", true)?], &[]);
        assert!(mine.first().is_some_and(Owner::is_proposed));
        let theirs = owners(&[tree("git@github.com:solo/project.git", false)?], &[]);
        assert!(theirs.first().is_some_and(|o| !o.is_proposed()));
        assert_eq!(source_control_entry(&theirs), None);
        Ok(())
    }

    #[test]
    fn one_repository_seen_twice_counts_once() -> anyhow::Result<()> {
        let twice = [
            tree("git@github.com:acme/app.git", false)?,
            tree("ssh://git@github.com/acme/app", false)?,
        ];
        let all = owners(&twice, &[]);
        assert_eq!(all.first().map(|o| o.repositories), Some(1));
        assert!(all.first().is_some_and(|o| !o.is_proposed()));
        Ok(())
    }

    #[test]
    fn memory_vouches_and_never_proposes() -> anyhow::Result<()> {
        let remembered = vec![
            "github.com/ghost/a".to_string(),
            "github.com/ghost/b".to_string(),
            "github.com/mine/project".to_string(),
        ];
        let all = owners(
            &[tree("git@github.com:mine/project.git", true)?],
            &remembered,
        );
        let ghost = all.iter().find(|o| o.name == "github.com/ghost");
        assert_eq!(
            ghost.map(|o| (o.remembered, o.is_proposed())),
            Some((2, false))
        );
        let mine = all.iter().find(|o| o.name == "github.com/mine");
        assert_eq!(
            mine.map(|o| (o.remembered, o.is_proposed())),
            Some((1, true))
        );
        Ok(())
    }

    #[test]
    fn the_report_gives_a_reason_for_everything_it_leaves_out() -> anyhow::Result<()> {
        let trees = [
            tree("git@github.com:mine/project.git", true)?,
            tree("https://github.com/vendor/a.git", false)?,
            tree("git@github.com:upstream/library.git", false)?,
        ];
        let all = owners(&trees, &["github.com/ghost/a".to_string()]);
        let text = render(&all, &Memory::NotAsked("no token".to_string()));
        assert!(text.contains("\"$defaults\""), "{text}");
        assert!(
            text.contains("\"Source control: github.com/mine and all repos under it\""),
            "{text}"
        );
        assert!(
            text.contains("github.com/vendor: cloned over https"),
            "{text}"
        );
        assert!(
            text.contains("github.com/upstream: one repository"),
            "{text}"
        );
        assert!(text.contains("github.com/ghost: only in memory"), "{text}");
        assert!(text.contains("Memory was not asked: no token"), "{text}");
        assert!(text.contains("Nothing was written"), "{text}");
        Ok(())
    }

    #[test]
    fn with_nothing_to_propose_the_draft_is_the_defaults_alone() {
        let text = render(&[], &Memory::NotAsked("x".to_string()));
        assert!(text.contains("      \"$defaults\"\n    ]"), "{text}");
        assert!(!text.contains("Source control"), "{text}");
    }

    /// The whole reason the surface reports a similarity: the same memory comes
    /// back for every slot, and only the number says which slot it belongs to.
    #[test]
    fn the_closest_candidate_is_offered_first_and_its_closeness_is_shown() -> anyhow::Result<()> {
        let candidate = |id: &str, content: &str, similarity: f32| Candidate {
            id: id.to_string(),
            content: content.to_string(),
            similarity: Some(similarity),
        };
        let far = candidate("memory:far", "the kettle is in the kitchen", 0.11);
        let near = candidate("memory:near", "shaman is the production cluster", 0.88);
        let unscored = Candidate {
            id: "memory:old".to_string(),
            content: "from a surface that does not score".to_string(),
            similarity: None,
        };
        let Some(slot) = slots().first().copied() else {
            anyhow::bail!("there are no slots");
        };
        let text = render(
            &[],
            &Memory::Asked(vec![(
                slot,
                vec![far.clone(), unscored.clone(), near.clone()],
            )]),
        );
        let at = |needle: &str| {
            text.find(needle)
                .ok_or_else(|| anyhow::anyhow!("`{needle}` is not in:\n{text}"))
        };
        // Closest first, then the far one, and one with no score last rather
        // than pretending it ranked.
        let order = [at("memory:near")?, at("memory:far")?, at("memory:old")?];
        assert!(order.windows(2).all(|p| p[0] < p[1]), "{text}");
        assert!(text.contains("0.88"), "{text}");
        assert!(text.contains("0.11"), "{text}");
        Ok(())
    }

    #[test]
    fn memory_is_asked_once_per_slot_and_an_empty_slot_is_still_shown() -> anyhow::Result<()> {
        let asked = std::cell::RefCell::new(Vec::new());
        let call = |tool: &str, arguments: Value| -> anyhow::Result<Value> {
            asked
                .borrow_mut()
                .push((tool.to_string(), arguments.clone()));
            let about_deploys = arguments
                .get("query")
                .and_then(Value::as_str)
                .is_some_and(|q| q.contains("deploy"));
            Ok(if about_deploys {
                json!({ "memories": [{ "id": "memory:1", "content": "shaman is production;\n  kuskokwim is a test node" }] })
            } else {
                json!({ "memories": [] })
            })
        };
        let found = candidates(&call, &slots(), 3)?;
        assert_eq!(found.len(), slots().len());
        assert_eq!(asked.borrow().len(), slots().len());
        assert!(asked
            .borrow()
            .iter()
            .all(|(tool, a)| tool == "recall_memories" && a.get("top_k") == Some(&json!(3))));

        let text = render(&[], &Memory::Asked(found));
        assert!(
            text.contains("memory:1  shaman is production; kuskokwim is a test node"),
            "{text}"
        );
        assert!(text.contains("Candidates, not entries"), "{text}");
        Ok(())
    }

    #[test]
    fn a_memory_offered_for_many_slots_is_shown_once_with_all_of_them() {
        let everything = Candidate {
            id: "memory:1".to_string(),
            content: "shaman is production".to_string(),
            similarity: Some(0.42),
        };
        let found: Vec<(Slot, Vec<Candidate>)> = slots()
            .into_iter()
            .map(|slot| (slot, vec![everything.clone()]))
            .collect();
        let text = render(&[], &Memory::Asked(found));
        assert_eq!(text.matches("memory:1").count(), 1, "{text}");
        let names: Vec<&str> = slots().iter().map(|s| s.name).collect();
        assert!(
            text.contains(&format!("offered for: {}", names.join(", "))),
            "{text}"
        );

        let nothing: Vec<(Slot, Vec<Candidate>)> =
            slots().into_iter().map(|slot| (slot, Vec::new())).collect();
        let text = render(&[], &Memory::Asked(nothing));
        assert!(text.contains("remembers nothing"), "{text}");
    }

    #[test]
    fn remembered_repositories_come_from_the_anchors() -> anyhow::Result<()> {
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            Ok(json!({ "memories": [
                { "id": "a", "provenance": { "repo": "github.com/mine/project", "commit": "abc1234" } },
                { "id": "b" },
            ] }))
        };
        assert_eq!(remembered_repositories(&call)?, ["github.com/mine/project"]);
        Ok(())
    }
}

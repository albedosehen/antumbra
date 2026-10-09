//! The rules as memories: each entry of the sovereign-mode matrix becomes a
//! `world` memory in a `claude-code` compartment, so an agent can recall why a
//! feature is missing and what stands in for it.
//!
//! The session-start block ([`super::brief`]) is what tells the agent; these are
//! what it can look up. They go in through the same tool surface an agent writes
//! through, so they are embedded, owned by the user, and private until the user
//! shares the compartment.
//!
//! Two properties matter more than the rest:
//!
//! - **Volatile.** The list changes with the agent's releases. A volatile memory
//!   stays in the store and never graduates, so no expert is ever trained on a
//!   vendor's release notes.
//! - **Idempotent.** The tool surface gives every write a fresh id, so running
//!   this twice would double the compartment. [`plan`] decides from what is
//!   already there: a rule that is present is kept, one whose text changed is
//!   stored again and the old text penalized once, and one that left the matrix
//!   is penalized once. Nothing is deleted.

use serde_json::{json, Value};

use super::{Class, Rule, SOURCE, VERIFIED_AGAINST, VERIFIED_ON};

/// The compartment's display name.
pub const COMPARTMENT: &str = "claude-code";

/// What opens every memory this module writes. It is how a later run knows a
/// memory is one of these and which rule it states.
const KEY_PREFIX: &str = antumbra_core::keyed::CLAUDE_CODE_RULE;

/// The confidence these are stored at. One penalty takes a memory to three
/// quarters of it, so anything clearly below has been judged, by an earlier run
/// or by the user, and is left alone.
const STORED_CONFIDENCE: f32 = 0.9;
const JUDGED_BELOW: f32 = STORED_CONFIDENCE - 0.05;

/// One rule, as it is remembered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Convention {
    pub key: String,
    pub content: String,
    pub evidence: Vec<String>,
}

fn evidence() -> Vec<String> {
    vec![
        format!("claude-code:{VERIFIED_AGAINST}"),
        format!("verified:{VERIFIED_ON}"),
        SOURCE.to_string(),
        "sovereign-mode".to_string(),
    ]
}

fn convention_of(rule: &Rule) -> Convention {
    let class = match rule.class {
        Class::Restored => "Antumbra restores it",
        Class::Setting { required: true } => "A local setting brings it back, and it is required",
        Class::Setting { required: false } => "A local setting brings it back",
        Class::AcceptedLoss => "It stays lost",
    };
    let key = antumbra_core::keyed::key(KEY_PREFIX, rule.id);
    Convention {
        content: format!(
            "{key} When Claude Code runs with feature-flag fetching off (DISABLE_TELEMETRY and \
             its siblings, or a third-party provider): {}. {class}: {}.",
            rule.lost, rule.response
        ),
        key,
        evidence: evidence(),
    }
}

/// Every rule, and one memory saying what release they were checked against.
pub fn conventions(rules: &[Rule]) -> Vec<Convention> {
    let key = antumbra_core::keyed::key(KEY_PREFIX, "verified");
    let checked = Convention {
        content: format!(
            "{key} The claude-code rules in this compartment were checked against Claude Code \
             {VERIFIED_AGAINST} on {VERIFIED_ON}. The list changes between releases; \
             `antumbra claude doctor` says when the installed release differs."
        ),
        key,
        evidence: evidence(),
    };
    rules
        .iter()
        .map(convention_of)
        .chain(std::iter::once(checked))
        .collect()
}

/// A memory that is already there, as the tool surface shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Existing {
    pub id: String,
    pub content: String,
    pub confidence: f32,
}

/// One thing to do.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    Keep { key: String },
    Store(Convention),
    Retire { id: String, content: String },
}

/// Decide what to do from what is there. A memory whose text is current is kept
/// whatever its confidence: if the user penalized it, that was their judgment,
/// and storing it again would be arguing with them.
pub fn plan(existing: &[Existing], wanted: &[Convention]) -> Vec<Step> {
    let ours: Vec<&Existing> = existing
        .iter()
        .filter(|m| m.content.starts_with(KEY_PREFIX))
        .collect();
    let is_current = |content: &str| wanted.iter().any(|c| c.content == content);
    let settle = wanted.iter().map(|convention| {
        if ours.iter().any(|m| m.content == convention.content) {
            Step::Keep {
                key: convention.key.clone(),
            }
        } else {
            Step::Store(convention.clone())
        }
    });
    let retire = ours
        .iter()
        .filter(|m| !is_current(&m.content) && m.confidence >= JUDGED_BELOW)
        .map(|m| Step::Retire {
            id: m.id.clone(),
            content: m.content.clone(),
        });
    settle.chain(retire).collect()
}

/// A tool call against the surface: the tool's name and arguments in, its value
/// out. The command supplies HTTP; the tests supply a table.
pub type Call<'a> = &'a dyn Fn(&str, Value) -> anyhow::Result<Value>;

fn existing_memories(call: Call<'_>) -> anyhow::Result<Vec<Existing>> {
    let listed = call("list_memories", json!({ "network": "world" }))?;
    let memories = listed
        .get("memories")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("list_memories answered without a `memories` list"))?;
    Ok(memories
        .iter()
        .filter_map(|m| {
            Some(Existing {
                id: m.get("id")?.as_str()?.to_string(),
                content: m.get("content")?.as_str()?.to_string(),
                // Narrowing on purpose: the surface speaks f32 and JSON carries f64.
                confidence: m.get("confidence")?.as_f64()? as f32,
            })
        })
        .collect())
}

pub(super) fn compartment_id(call: Call<'_>, create: bool) -> anyhow::Result<Option<String>> {
    let listed = call("list_compartments", json!({}))?;
    let found = listed
        .get("compartments")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|c| c.get("name").and_then(Value::as_str) == Some(COMPARTMENT))
        .and_then(|c| c.get("id").and_then(Value::as_str))
        .map(str::to_string);
    if found.is_some() || !create {
        return Ok(found);
    }
    let made = call("create_compartment", json!({ "name": COMPARTMENT }))?;
    made.get("id")
        .and_then(Value::as_str)
        .map(|id| Some(id.to_string()))
        .ok_or_else(|| anyhow::anyhow!("create_compartment answered without an `id`"))
}

fn short(content: &str) -> String {
    let line: String = content.chars().take(72).collect();
    if line.len() < content.len() {
        format!("{line}...")
    } else {
        line
    }
}

/// Bring the compartment up to date and say what was done, one line each. With
/// `dry_run` the surface is only read.
pub fn remember(call: Call<'_>, rules: &[Rule], dry_run: bool) -> anyhow::Result<Vec<String>> {
    let steps = plan(&existing_memories(call)?, &conventions(rules));
    let writes = steps.iter().any(|s| matches!(s, Step::Store(_)));
    let compartment = compartment_id(call, writes && !dry_run)?;
    let mut said = Vec::new();
    for step in &steps {
        said.push(match step {
            Step::Keep { key } => format!("kept     {key}"),
            Step::Store(convention) => {
                if !dry_run {
                    let Some(compartment) = &compartment else {
                        anyhow::bail!("no `{COMPARTMENT}` compartment to store into");
                    };
                    call(
                        "store_memory",
                        json!({
                            "content": convention.content,
                            "network": "world",
                            "confidence": STORED_CONFIDENCE,
                            "evidence": convention.evidence,
                            "volatile": true,
                            "compartment": compartment,
                        }),
                    )?;
                }
                format!("stored   {}", convention.key)
            }
            Step::Retire { id, content } => {
                if !dry_run {
                    call("penalize_memory", json!({ "memory_id": id }))?;
                }
                format!("retired  {id}: {}", short(content))
            }
        });
    }
    Ok(said)
}

#[cfg(test)]
mod tests {
    use super::super::rules;
    use super::*;
    use std::cell::RefCell;

    /// A tool surface small enough to read: what it holds, and what it was asked.
    #[derive(Default)]
    struct Surface {
        compartments: Vec<(String, String)>,
        memories: Vec<(Existing, Value)>,
        calls: Vec<String>,
    }

    fn answer(surface: &RefCell<Surface>, tool: &str, arguments: Value) -> anyhow::Result<Value> {
        let mut s = surface.borrow_mut();
        s.calls.push(tool.to_string());
        match tool {
            "list_compartments" => Ok(json!({
                "compartments": s.compartments.iter()
                    .map(|(id, name)| json!({ "id": id, "name": name, "origin": "user" }))
                    .collect::<Vec<_>>()
            })),
            "create_compartment" => {
                let id = format!("comp:{}", s.compartments.len());
                let name = arguments
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("create_compartment without a name"))?;
                s.compartments.push((id.clone(), name.to_string()));
                Ok(json!({ "id": id, "name": name, "origin": "user" }))
            }
            "list_memories" => Ok(json!({
                "memories": s.memories.iter()
                    .map(|(m, _)| json!({ "id": m.id, "content": m.content, "confidence": m.confidence }))
                    .collect::<Vec<_>>()
            })),
            "store_memory" => {
                let id = format!("memory:{}", s.memories.len());
                let content = arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("store_memory without content"))?
                    .to_string();
                let stored = Existing {
                    id: id.clone(),
                    content,
                    confidence: STORED_CONFIDENCE,
                };
                s.memories.push((stored, arguments));
                Ok(json!({ "id": id }))
            }
            "penalize_memory" => {
                let id = arguments.get("memory_id").and_then(Value::as_str);
                let hit = s
                    .memories
                    .iter_mut()
                    .find(|(m, _)| Some(m.id.as_str()) == id);
                match hit {
                    Some((m, _)) => {
                        m.confidence *= 0.75;
                        Ok(json!({ "found": true }))
                    }
                    None => Ok(json!({ "found": false })),
                }
            }
            other => anyhow::bail!("the test surface has no tool `{other}`"),
        }
    }

    fn run(
        surface: &RefCell<Surface>,
        rules: &[Rule],
        dry_run: bool,
    ) -> anyhow::Result<Vec<String>> {
        remember(
            &|tool, arguments| answer(surface, tool, arguments),
            rules,
            dry_run,
        )
    }

    fn count(said: &[String], word: &str) -> usize {
        said.iter().filter(|line| line.starts_with(word)).count()
    }

    #[test]
    fn a_first_run_stores_every_rule_volatile_in_its_own_compartment() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        let said = run(&surface, &rules(), false)?;
        let s = surface.borrow();
        assert_eq!(count(&said, "stored"), rules().len() + 1, "{said:?}");
        assert_eq!(
            s.compartments,
            vec![("comp:0".to_string(), COMPARTMENT.to_string())]
        );
        for (memory, arguments) in &s.memories {
            assert_eq!(
                arguments.get("volatile"),
                Some(&json!(true)),
                "{}",
                memory.content
            );
            assert_eq!(arguments.get("compartment"), Some(&json!("comp:0")));
            assert_eq!(arguments.get("network"), Some(&json!("world")));
            let evidence = arguments.get("evidence").and_then(Value::as_array);
            assert!(
                evidence
                    .is_some_and(|e| e.contains(&json!(format!("claude-code:{VERIFIED_AGAINST}")))),
                "{arguments}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_second_run_changes_nothing() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        run(&surface, &rules(), false)?;
        let before = surface.borrow().memories.len();
        surface.borrow_mut().calls.clear();
        let said = run(&surface, &rules(), false)?;
        let s = surface.borrow();
        assert_eq!(count(&said, "kept"), said.len(), "{said:?}");
        assert_eq!(s.memories.len(), before);
        assert_eq!(s.compartments.len(), 1);
        assert!(
            !s.calls.iter().any(|c| c == "store_memory"
                || c == "penalize_memory"
                || c == "create_compartment"),
            "{:?}",
            s.calls
        );
        Ok(())
    }

    #[test]
    fn a_rule_whose_text_changed_is_stored_again_and_the_old_text_retired_once(
    ) -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        let mut changed = rules();
        let Some(first) = changed.first_mut() else {
            anyhow::bail!("the matrix is empty");
        };
        first.response = "an earlier answer, since replaced";
        run(&surface, &changed, false)?;

        let said = run(&surface, &rules(), false)?;
        assert_eq!(count(&said, "stored"), 1, "{said:?}");
        assert_eq!(count(&said, "retired"), 1, "{said:?}");
        let retired = surface
            .borrow()
            .memories
            .iter()
            .filter(|(m, _)| m.confidence < STORED_CONFIDENCE)
            .map(|(m, _)| m.content.clone())
            .collect::<Vec<_>>();
        assert_eq!(retired.len(), 1);
        assert!(
            retired.iter().all(|c| c.contains("an earlier answer")),
            "{retired:?}"
        );

        let again = run(&surface, &rules(), false)?;
        assert_eq!(
            count(&again, "kept"),
            again.len(),
            "a retired memory is not retired twice: {again:?}"
        );
        Ok(())
    }

    #[test]
    fn a_rule_that_left_the_matrix_is_retired_and_nothing_is_deleted() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        run(&surface, &rules(), false)?;
        let held = surface.borrow().memories.len();
        let fewer: Vec<Rule> = rules().into_iter().skip(1).collect();
        let said = run(&surface, &fewer, false)?;
        assert_eq!(count(&said, "retired"), 1, "{said:?}");
        assert_eq!(count(&said, "stored"), 0, "{said:?}");
        assert_eq!(surface.borrow().memories.len(), held);
        Ok(())
    }

    #[test]
    fn a_current_rule_the_user_penalized_is_not_stored_again() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        run(&surface, &rules(), false)?;
        for (memory, _) in surface.borrow_mut().memories.iter_mut().take(1) {
            memory.confidence *= 0.75;
        }
        let said = run(&surface, &rules(), false)?;
        assert_eq!(count(&said, "kept"), said.len(), "{said:?}");
        Ok(())
    }

    #[test]
    fn memories_that_are_not_ours_are_never_touched() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        let theirs = Existing {
            id: "memory:theirs".to_string(),
            content: "The deploy key lives in the vault, not in the repository.".to_string(),
            confidence: 0.6,
        };
        surface
            .borrow_mut()
            .memories
            .push((theirs.clone(), json!({})));
        let said = run(&surface, &rules(), false)?;
        assert_eq!(count(&said, "retired"), 0, "{said:?}");
        let s = surface.borrow();
        assert_eq!(s.memories.first().map(|(m, _)| m), Some(&theirs));
        Ok(())
    }

    #[test]
    fn an_existing_compartment_is_reused() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        surface
            .borrow_mut()
            .compartments
            .push(("comp:mine".to_string(), COMPARTMENT.to_string()));
        run(&surface, &rules(), false)?;
        let s = surface.borrow();
        assert_eq!(s.compartments.len(), 1);
        assert!(s
            .memories
            .iter()
            .all(|(_, a)| a.get("compartment") == Some(&json!("comp:mine"))));
        Ok(())
    }

    #[test]
    fn a_dry_run_only_reads() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        let said = run(&surface, &rules(), true)?;
        let s = surface.borrow();
        assert_eq!(count(&said, "stored"), rules().len() + 1, "{said:?}");
        assert!(s.memories.is_empty() && s.compartments.is_empty());
        assert!(
            s.calls
                .iter()
                .all(|c| c == "list_memories" || c == "list_compartments"),
            "{:?}",
            s.calls
        );
        Ok(())
    }

    #[test]
    fn every_memory_opens_with_its_key_and_no_two_share_one() {
        let all = conventions(&rules());
        assert!(all.iter().all(|c| c.content.starts_with(&c.key)));
        let mut keys: Vec<&str> = all.iter().map(|c| c.key.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), all.len());
    }

    #[test]
    fn a_surface_that_answers_in_another_shape_is_an_error_and_not_an_empty_plan() {
        let broken = |_: &str, _: Value| Ok(json!({ "unexpected": true }));
        assert!(remember(&broken, &rules(), true).is_err());
    }
}

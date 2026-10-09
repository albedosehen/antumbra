//! What the Sovereign page shows: the rules about a coding agent
//! running with its feature flags off, and which skills are used.
//!
//! Nothing is queried for it. Both are keyed memories (see
//! [`antumbra_core::keyed`]) among the ones the console already loads and
//! already watches, so the page is a pure reading of `App::memories`, made once
//! per reload and not per frame.
//!
//! The console is an operator's view and reads every workspace, so a row says
//! which workspace it is about. Read-only: the page changes nothing.
//!
//! A forgotten memory never arrives here: the console's loader
//! (`memory::all_unscoped_lite`) drops tombstones, and one place deciding that
//! is better than two disagreeing.

use std::collections::BTreeMap;

use antumbra_core::keyed::{name_in, CLAUDE_CODE_RULE, SKILL_USE};
use antumbra_core::Memory;
use chrono::{DateTime, Utc};

/// A rule is written at this confidence and one penalty takes it to three
/// quarters of it, so anything clearly below has been retired (or judged wrong
/// by its owner). The same line the CLI draws when it decides what to rewrite.
const RETIRED_BELOW: f32 = 0.85;

/// The key of the memory that says which release the rules were checked against.
const CHECKED: &str = "verified";

/// One rule, across the workspaces that hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleRow {
    pub rule: String,
    /// Workspaces holding it at full standing.
    pub current_in: usize,
    /// Workspaces holding only a retired text of it.
    pub retired_in: usize,
}

/// One skill's use in one workspace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRow {
    pub skill: String,
    pub workspace: String,
    pub uses: u64,
    pub last_used: DateTime<Utc>,
}

/// Everything the page shows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct View {
    /// The text of the newest "checked against" memory, without its key.
    pub checked: Option<String>,
    pub rules: Vec<RuleRow>,
    /// Stalest first: the question the page answers is what is no longer used.
    pub skills: Vec<SkillRow>,
}

fn after_key(content: &str) -> &str {
    content
        .split_once(']')
        .map_or(content, |(_, rest)| rest)
        .trim()
}

impl View {
    pub fn from_memories(memories: &[Memory]) -> Self {
        let mut checked: Option<&Memory> = None;
        // rule -> workspace -> whether any text of it there is current.
        let mut rules: BTreeMap<&str, BTreeMap<&str, bool>> = BTreeMap::new();
        let mut skills: BTreeMap<(&str, &str), SkillRow> = BTreeMap::new();

        for memory in memories {
            let workspace = memory.tenant.as_str();
            let current = memory.confidence >= RETIRED_BELOW;
            if let Some(rule) = name_in(&memory.content, CLAUDE_CODE_RULE) {
                if rule == CHECKED {
                    if current && checked.is_none_or(|seen| memory.updated_at > seen.updated_at) {
                        checked = Some(memory);
                    }
                    continue;
                }
                *rules
                    .entry(rule)
                    .or_default()
                    .entry(workspace)
                    .or_insert(false) |= current;
            } else if let Some(skill) = name_in(&memory.content, SKILL_USE) {
                // A counter is made by its first use and reinforced by each later
                // one. Two counters for one skill (recall once missed the first)
                // are one count, and the later use is the last.
                let uses = 1 + u64::from(memory.reinforcement);
                skills
                    .entry((workspace, skill))
                    .and_modify(|row| {
                        row.uses += uses;
                        row.last_used = row.last_used.max(memory.updated_at);
                    })
                    .or_insert_with(|| SkillRow {
                        skill: skill.to_string(),
                        workspace: workspace.to_string(),
                        uses,
                        last_used: memory.updated_at,
                    });
            }
        }

        let rules = rules
            .into_iter()
            .map(|(rule, held)| {
                let current_in = held.values().filter(|current| **current).count();
                RuleRow {
                    rule: rule.to_string(),
                    current_in,
                    retired_in: held.len() - current_in,
                }
            })
            .collect();
        let mut skills: Vec<SkillRow> = skills.into_values().collect();
        skills.sort_by(|a, b| {
            a.last_used
                .cmp(&b.last_used)
                .then_with(|| a.skill.cmp(&b.skill))
                .then_with(|| a.workspace.cmp(&b.workspace))
        });
        View {
            checked: checked.map(|m| after_key(&m.content).to_string()),
            rules,
            skills,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::MemoryNetwork;
    use chrono::TimeZone;

    fn at(day: u32) -> anyhow::Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(2026, 9, day, 12, 0, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("2026-09-{day} is not a date"))
    }

    fn memory(
        workspace: &str,
        content: &str,
        confidence: f32,
        reinforcement: u32,
        day: u32,
    ) -> anyhow::Result<Memory> {
        let when = at(day)?;
        let mut m = Memory::new(
            format!("memory:{workspace}-{day}-{reinforcement}-{}", content.len()),
            workspace,
            MemoryNetwork::World,
            content,
            confidence,
            when,
        );
        m.reinforcement = reinforcement;
        m.updated_at = when;
        Ok(m)
    }

    #[test]
    fn rules_are_counted_by_the_workspaces_that_hold_them_current_or_only_retired(
    ) -> anyhow::Result<()> {
        let memories = [
            memory(
                "ws:a",
                "[claude-code:agents-md] the current text",
                0.9,
                0,
                1,
            )?,
            // An earlier text, penalized once: ws:a still holds the rule current.
            memory(
                "ws:a",
                "[claude-code:agents-md] an earlier text",
                0.675,
                0,
                1,
            )?,
            memory(
                "ws:b",
                "[claude-code:agents-md] an earlier text",
                0.675,
                0,
                1,
            )?,
            memory("ws:a", "[claude-code:advisor] lost", 0.9, 0, 1)?,
        ];
        let view = View::from_memories(&memories);
        assert_eq!(
            view.rules,
            vec![
                RuleRow {
                    rule: "advisor".into(),
                    current_in: 1,
                    retired_in: 0
                },
                RuleRow {
                    rule: "agents-md".into(),
                    current_in: 1,
                    retired_in: 1
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn the_newest_current_check_is_the_one_shown_and_is_not_a_rule() -> anyhow::Result<()> {
        let memories = [
            memory(
                "ws:a",
                "[claude-code:verified] checked against 2.1.278",
                0.675,
                0,
                9,
            )?,
            memory(
                "ws:a",
                "[claude-code:verified] checked against 2.1.300",
                0.9,
                0,
                5,
            )?,
            memory(
                "ws:b",
                "[claude-code:verified] checked against 2.1.290",
                0.9,
                0,
                3,
            )?,
        ];
        let view = View::from_memories(&memories);
        assert_eq!(view.checked.as_deref(), Some("checked against 2.1.300"));
        assert!(view.rules.is_empty());
        Ok(())
    }

    #[test]
    fn a_skill_is_counted_per_workspace_and_the_stalest_comes_first() -> anyhow::Result<()> {
        let memories = [
            memory("ws:a", "[skill-use:deploy] deploy", 0.9, 4, 19)?,
            memory("ws:a", "[skill-use:review] review", 0.9, 0, 2)?,
            memory("ws:b", "[skill-use:deploy] deploy", 0.9, 1, 10)?,
        ];
        let view = View::from_memories(&memories);
        let shown: Vec<(&str, &str, u64)> = view
            .skills
            .iter()
            .map(|row| (row.skill.as_str(), row.workspace.as_str(), row.uses))
            .collect();
        assert_eq!(
            shown,
            [
                ("review", "ws:a", 1),
                ("deploy", "ws:b", 2),
                ("deploy", "ws:a", 5)
            ]
        );
        Ok(())
    }

    #[test]
    fn two_counters_for_one_skill_are_one_count_and_the_later_use_is_the_last() -> anyhow::Result<()>
    {
        let early = memory("ws:a", "[skill-use:deploy] deploy", 0.9, 4, 1)?;
        let late = memory("ws:a", "[skill-use:deploy] deploy", 0.9, 0, 19)?;
        for memories in [[early.clone(), late.clone()], [late, early]] {
            let view = View::from_memories(&memories);
            assert_eq!(view.skills.len(), 1);
            assert_eq!(view.skills.first().map(|row| row.uses), Some(6));
            assert_eq!(view.skills.first().map(|row| row.last_used), Some(at(19)?));
        }
        Ok(())
    }

    #[test]
    fn what_is_not_keyed_at_the_front_is_not_shown() -> anyhow::Result<()> {
        let memories = [
            memory("ws:a", "In acme-api use deno install, not npm", 0.9, 7, 1)?,
            memory("ws:a", "I read [skill-use:deploy] in a log", 0.9, 7, 1)?,
        ];
        assert_eq!(View::from_memories(&memories), View::default());
        assert_eq!(View::from_memories(&[]), View::default());
        Ok(())
    }
}

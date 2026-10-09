//! Skill usage: which skills are used, how often, and when last, so
//! that the ones nobody reaches for can be found. The agent's own `/skill-doctor`
//! goes with the feature flags.
//!
//! A skill is used in two ways, and a hook on one of them misses the other
//! (measured): the agent calls the `Skill` tool, which a `PostToolUse` hook
//! sees, or the user types `/name`, which never touches the tool and arrives as
//! a `UserPromptExpansion`. [`skill_in`] reads both.
//!
//! The count is kept where everything else about the user is kept. Each skill
//! has one memory in the `claude-code` compartment, opened by a key. Using the
//! skill reinforces it, so `reinforcement` is the count and `updated_at` is the
//! last use. No table, no tool, and no file on one machine that another cannot
//! see. The memories are volatile, so a much-reinforced counter is never taken
//! for knowledge worth training an expert on.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use super::conventions::{compartment_id, Call};

const KEY_PREFIX: &str = antumbra_core::keyed::SKILL_USE;

/// A name worth keeping: what a skill can be called, and nothing a hook's input
/// could smuggle into a memory.
fn is_a_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.' | '/'))
}

/// The skill a hook's input says was used, if it says so.
pub fn skill_in(hook: &Value) -> Option<String> {
    let text = |pointer: &str| hook.pointer(pointer).and_then(Value::as_str);
    let name = match text("/hook_event_name")? {
        "PostToolUse" if text("/tool_name") == Some("Skill") => text("/tool_input/skill"),
        "UserPromptExpansion" if text("/expansion_type") == Some("slash_command") => {
            text("/command_name")
        }
        _ => None,
    }?;
    let name = name.trim().trim_start_matches('/');
    is_a_name(name).then(|| name.to_string())
}

/// The whole text of a skill's counter. The name comes twice on purpose: once in
/// the key a later run matches on, once bare so that recall finds it by name.
pub fn counter_text(name: &str) -> String {
    format!("{} {name}", antumbra_core::keyed::key(KEY_PREFIX, name))
}

/// What recording a use did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recorded {
    First,
    Again,
}

/// Count one use: reinforce the skill's counter, or make it. The counter is
/// looked for by recall and matched exactly. If recall ever misses one that is
/// there, a second is made, and the report adds them up: a count split over two
/// memories is still the count.
pub fn record(call: Call<'_>, name: &str) -> anyhow::Result<Recorded> {
    anyhow::ensure!(is_a_name(name), "`{name}` is not a skill's name");
    let wanted = counter_text(name);
    let found = call(
        "recall_memories",
        json!({ "query": wanted, "top_k": 10, "network": "world" }),
    )?;
    let existing = found
        .get("memories")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|m| m.get("content").and_then(Value::as_str) == Some(wanted.as_str()))
        .and_then(|m| m.get("id").and_then(Value::as_str));
    if let Some(id) = existing {
        call("reinforce_memory", json!({ "memory_id": id }))?;
        return Ok(Recorded::Again);
    }
    let Some(compartment) = compartment_id(call, true)? else {
        anyhow::bail!("no compartment to keep the counter in");
    };
    call(
        "store_memory",
        json!({
            "content": wanted,
            "network": "world",
            "confidence": 0.9,
            "evidence": ["sovereign-mode", "skill-use"],
            "volatile": true,
            "compartment": compartment,
        }),
    )?;
    Ok(Recorded::First)
}

/// How much one skill has been used.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Usage {
    pub uses: u64,
    /// RFC 3339, when the surface says; an older surface does not.
    pub last: Option<String>,
}

/// Every skill's usage, from the counters among the user's memories.
pub fn usage(call: Call<'_>) -> anyhow::Result<BTreeMap<String, Usage>> {
    let listed = call("list_memories", json!({ "network": "world" }))?;
    let memories = listed
        .get("memories")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("list_memories answered without a `memories` list"))?;
    let mut all: BTreeMap<String, Usage> = BTreeMap::new();
    for memory in memories {
        let Some(name) = memory
            .get("content")
            .and_then(Value::as_str)
            .and_then(|content| antumbra_core::keyed::name_in(content, KEY_PREFIX))
        else {
            continue;
        };
        let entry = all.entry(name.to_string()).or_default();
        // A counter is made by its first use and reinforced by every later one.
        entry.uses += 1 + memory
            .get("reinforcement")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let last = memory.get("updated_at").and_then(Value::as_str);
        // RFC 3339 in one zone sorts as text, and the surface writes UTC.
        if last > entry.last.as_deref() {
            entry.last = last.map(str::to_string);
        }
    }
    Ok(all)
}

/// The `name:` of a skill's front matter, or `None` when it has none.
pub fn declared_name(skill_md: &str) -> Option<String> {
    let mut lines = skill_md.lines();
    (lines.next()?.trim() == "---").then_some(())?;
    lines
        .take_while(|line| line.trim() != "---")
        .find_map(|line| line.strip_prefix("name:"))
        .map(|name| name.trim().trim_matches(['"', '\'']).to_string())
        .filter(|name| !name.is_empty())
}

/// The report: installed skills with their usage, never-used ones first, then
/// what was used and is not installed here. `stale_before` is an RFC 3339
/// instant; a skill last used before it is called out.
pub fn render(
    installed: &[String],
    used: &BTreeMap<String, Usage>,
    stale_before: Option<&str>,
) -> String {
    let nothing = Usage::default();
    let mut rows: Vec<(&String, &Usage)> = installed
        .iter()
        .map(|name| (name, used.get(name).unwrap_or(&nothing)))
        .collect();
    rows.sort_by(|a, b| a.1.uses.cmp(&b.1.uses).then_with(|| a.0.cmp(b.0)));
    let mut out = Vec::new();
    if rows.is_empty() {
        out.push("No skills are installed here.".to_string());
    }
    for (name, usage) in rows {
        let last = usage.last.as_deref();
        let note = match (usage.uses, last, stale_before) {
            (0, _, _) => "never used".to_string(),
            (uses, Some(last), Some(cutoff)) if last < cutoff => {
                format!("{uses} use(s), none since {}", day(last))
            }
            (uses, Some(last), _) => format!("{uses} use(s), last on {}", day(last)),
            (uses, None, _) => format!("{uses} use(s); this surface does not say when"),
        };
        out.push(format!("  {name}: {note}"));
    }
    let elsewhere: Vec<String> = used
        .iter()
        .filter(|(name, _)| !installed.contains(name))
        .map(|(name, usage)| format!("  {name}: {} use(s)", usage.uses))
        .collect();
    if !elsewhere.is_empty() {
        out.push(String::new());
        out.push(
            "Used, and not installed here (a plugin's, a built-in, or another machine's):"
                .to_string(),
        );
        out.extend(elsewhere);
    }
    out.join("\n")
}

fn day(rfc3339: &str) -> &str {
    rfc3339.split('T').next().unwrap_or(rfc3339)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn both_ways_of_using_a_skill_are_read_and_nothing_else_is() {
        let tool = json!({ "hook_event_name": "PostToolUse", "tool_name": "Skill", "tool_input": { "skill": "elegant-design" } });
        let typed = json!({ "hook_event_name": "UserPromptExpansion", "expansion_type": "slash_command", "command_name": "elegant-design", "prompt": "/elegant-design now" });
        assert_eq!(skill_in(&tool), Some("elegant-design".to_string()));
        assert_eq!(skill_in(&typed), Some("elegant-design".to_string()));

        let not_uses = [
            // Asked for, and perhaps denied: not a use.
            json!({ "hook_event_name": "PreToolUse", "tool_name": "Skill", "tool_input": { "skill": "x" } }),
            json!({ "hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": { "skill": "x" } }),
            json!({ "hook_event_name": "UserPromptExpansion", "expansion_type": "mcp_prompt", "command_name": "x" }),
            json!({ "hook_event_name": "UserPromptSubmit", "prompt": "/x" }),
            json!({}),
        ];
        for input in not_uses {
            assert_eq!(skill_in(&input), None, "{input}");
        }
    }

    #[test]
    fn a_name_is_kept_only_if_it_could_be_one() {
        let named = |name: &str| {
            skill_in(
                &json!({ "hook_event_name": "PostToolUse", "tool_name": "Skill", "tool_input": { "skill": name } }),
            )
        };
        assert_eq!(
            named("/typed-with-its-slash"),
            Some("typed-with-its-slash".to_string())
        );
        assert_eq!(named("plugin:skill"), Some("plugin:skill".to_string()));
        assert_eq!(
            named("apps/web:deploy"),
            Some("apps/web:deploy".to_string())
        );
        assert_eq!(named(""), None);
        assert_eq!(named("two words"), None);
        assert_eq!(named("line\nbreak"), None);
        assert_eq!(named("] forged key [skill-use:other"), None);
        assert_eq!(named(&"x".repeat(129)), None);
    }

    /// A surface that keeps memories the way the real one does, as far as this
    /// module can tell: recall finds by content, reinforce counts.
    #[derive(Default)]
    struct Surface {
        memories: Vec<(String, String, u64, bool)>,
        compartments: usize,
        recall_is_blind: bool,
    }

    fn answer(surface: &RefCell<Surface>, tool: &str, arguments: Value) -> anyhow::Result<Value> {
        let mut s = surface.borrow_mut();
        let text = |key: &str| {
            arguments
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        match tool {
            "recall_memories" | "list_memories" => {
                let blind = tool == "recall_memories" && s.recall_is_blind;
                let memories: Vec<Value> = s
                    .memories
                    .iter()
                    .filter(|_| !blind)
                    .map(|(id, content, reinforcement, _)| json!({ "id": id, "content": content, "reinforcement": reinforcement, "updated_at": format!("2026-09-{:02}T10:00:00+00:00", 1 + reinforcement) }))
                    .collect();
                Ok(json!({ "memories": memories }))
            }
            "reinforce_memory" => {
                let id = text("memory_id");
                for memory in s.memories.iter_mut().filter(|m| Some(&m.0) == id.as_ref()) {
                    memory.2 += 1;
                }
                Ok(json!({ "found": true }))
            }
            "list_compartments" => Ok(
                json!({ "compartments": (0..s.compartments).map(|n| json!({ "id": format!("comp:{n}"), "name": "claude-code" })).collect::<Vec<_>>() }),
            ),
            "create_compartment" => {
                s.compartments += 1;
                Ok(json!({ "id": "comp:0" }))
            }
            "store_memory" => {
                let id = format!("memory:{}", s.memories.len());
                let content = text("content").ok_or_else(|| anyhow::anyhow!("no content"))?;
                let volatile = arguments.get("volatile") == Some(&json!(true));
                anyhow::ensure!(
                    arguments.get("compartment") == Some(&json!("comp:0")),
                    "stored outside the compartment"
                );
                s.memories.push((id.clone(), content, 0, volatile));
                Ok(json!({ "id": id }))
            }
            other => anyhow::bail!("the test surface has no tool `{other}`"),
        }
    }

    fn used(surface: &RefCell<Surface>, name: &str) -> anyhow::Result<Recorded> {
        record(&|tool, arguments| answer(surface, tool, arguments), name)
    }

    #[test]
    fn a_first_use_makes_one_volatile_counter_and_later_uses_count_on_it() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        assert_eq!(used(&surface, "elegant-design")?, Recorded::First);
        assert_eq!(used(&surface, "elegant-design")?, Recorded::Again);
        assert_eq!(used(&surface, "elegant-design")?, Recorded::Again);
        assert_eq!(used(&surface, "other")?, Recorded::First);
        {
            let s = surface.borrow();
            assert_eq!(s.memories.len(), 2);
            assert_eq!(s.compartments, 1);
            assert!(s.memories.iter().all(|m| m.3), "every counter is volatile");
        }
        let all = usage(&|tool, arguments| answer(&surface, tool, arguments))?;
        assert_eq!(all.get("elegant-design").map(|u| u.uses), Some(3));
        assert_eq!(all.get("other").map(|u| u.uses), Some(1));
        Ok(())
    }

    #[test]
    fn a_counter_recall_missed_is_made_again_and_the_count_survives() -> anyhow::Result<()> {
        let surface = RefCell::new(Surface::default());
        used(&surface, "review")?;
        used(&surface, "review")?;
        surface.borrow_mut().recall_is_blind = true;
        assert_eq!(used(&surface, "review")?, Recorded::First);
        assert_eq!(surface.borrow().memories.len(), 2);
        let all = usage(&|tool, arguments| answer(&surface, tool, arguments))?;
        let review = all.get("review").cloned().unwrap_or_default();
        assert_eq!(review.uses, 3);
        assert_eq!(
            review.last.as_deref(),
            Some("2026-09-02T10:00:00+00:00"),
            "the later of the two"
        );
        Ok(())
    }

    #[test]
    fn a_name_that_could_not_be_one_is_refused_before_anything_is_asked() {
        let asked = RefCell::new(0);
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            *asked.borrow_mut() += 1;
            Ok(json!({}))
        };
        assert!(record(&call, "two words").is_err());
        assert_eq!(*asked.borrow(), 0);
    }

    #[test]
    fn memories_that_are_not_counters_are_not_counted() -> anyhow::Result<()> {
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            Ok(json!({ "memories": [
                { "id": "a", "content": "[claude-code:agents-md] a rule", "reinforcement": 9 },
                { "id": "b", "content": "I used the skill-use counter today", "reinforcement": 4 },
                { "id": "c", "content": "[skill-use:deploy] deploy", "reinforcement": 2 },
            ] }))
        };
        let all = usage(&call)?;
        assert_eq!(all.keys().collect::<Vec<_>>(), ["deploy"]);
        assert_eq!(
            all.get("deploy"),
            Some(&Usage {
                uses: 3,
                last: None
            })
        );
        Ok(())
    }

    #[test]
    fn of_two_counters_for_one_skill_the_later_use_is_the_last_whichever_is_listed_first(
    ) -> anyhow::Result<()> {
        let early = json!({ "id": "a", "content": "[skill-use:deploy] deploy", "reinforcement": 4, "updated_at": "2026-03-01T10:00:00+00:00" });
        let late = json!({ "id": "b", "content": "[skill-use:deploy] deploy", "reinforcement": 0, "updated_at": "2026-09-19T10:00:00+00:00" });
        for listed in [[&early, &late], [&late, &early]] {
            let call =
                |_: &str, _: Value| -> anyhow::Result<Value> { Ok(json!({ "memories": listed })) };
            let all = usage(&call)?;
            assert_eq!(
                all.get("deploy"),
                Some(&Usage {
                    uses: 6,
                    last: Some("2026-09-19T10:00:00+00:00".to_string())
                }),
                "{listed:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_skill_is_known_by_the_name_it_declares() {
        assert_eq!(
            declared_name("---\nname: elegant-design\ndescription: x\n---\nbody"),
            Some("elegant-design".to_string())
        );
        assert_eq!(
            declared_name("---\ndescription: x\nname: \"quoted\"\n---\n"),
            Some("quoted".to_string())
        );
        assert_eq!(
            declared_name("---\ndescription: x\n---\nname: in-the-body"),
            None
        );
        assert_eq!(declared_name("no front matter\nname: x"), None);
        assert_eq!(declared_name(""), None);
    }

    #[test]
    fn the_report_puts_the_unused_first_and_says_what_it_cannot_know() -> anyhow::Result<()> {
        let installed = vec![
            "busy".to_string(),
            "idle".to_string(),
            "stale".to_string(),
            "undated".to_string(),
        ];
        let used: BTreeMap<String, Usage> = [
            ("busy", 12, Some("2026-09-19T08:00:00+00:00")),
            ("stale", 3, Some("2026-05-01T08:00:00+00:00")),
            ("undated", 2, None),
            ("a-plugins:skill", 5, Some("2026-09-18T08:00:00+00:00")),
        ]
        .into_iter()
        .map(|(name, uses, last)| {
            (
                name.to_string(),
                Usage {
                    uses,
                    last: last.map(str::to_string),
                },
            )
        })
        .collect();
        let text = render(&installed, &used, Some("2026-08-20T00:00:00+00:00"));
        // A line that is missing must fail the test, not sort before the rest.
        let at = |needle: &str| {
            text.find(needle)
                .ok_or_else(|| anyhow::anyhow!("`{needle}` is not in:\n{text}"))
        };
        let order = [
            at("idle: never used")?,
            at("undated: 2 use(s); this surface does not say when")?,
            at("stale: 3 use(s), none since 2026-05-01")?,
            at("busy: 12 use(s), last on 2026-09-19")?,
            at("not installed here")?,
            at("a-plugins:skill: 5 use(s)")?,
        ];
        assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{text}");
        assert!(render(&[], &BTreeMap::new(), None).contains("No skills are installed here"));
        Ok(())
    }
}

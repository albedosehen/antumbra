//! Tool profiles: which of the server's tools a session advertises and serves.
//!
//! An agent that connects to several MCP servers pays for every tool
//! description on every turn, and picks among every tool on every call. Most
//! of this server's tools are operator actions (sharing a compartment,
//! revoking one, forgetting, listing the population) that a coding agent never
//! needs and that should not be one call away from it. A profile names the
//! tools a session gets; the rest are neither listed by `tools/list` nor
//! callable, over JSON-RPC or the REST shim, so the cut is at the
//! advertisement, not at a permission prompt that still ships the text.
//!
//! `agent` is the developer-agent profile ([`AGENT_PROFILE`]): the memory loop
//! with its feedback signal, anchored documents in and out, route/answer, and
//! the handoffs a session leaves for another of the user's machines, the
//! machines themselves, and the behaviors the user's own expert
//! learns.
//! A spec can also be an explicit comma-separated list, and `agent,population`
//! extends the named profile. Unknown names are refused with the full list, so
//! a typo cannot silently hide a tool.

use std::collections::BTreeSet;
use std::fmt;

/// The developer-agent profile: what a coding session needs and nothing an
/// operator would rather keep behind the CLI or console.
pub const AGENT_PROFILE: &[&str] = &[
    "recall_memories",
    "store_memory",
    "reinforce_memory",
    "penalize_memory",
    "recall_documents",
    "ingest_document",
    "route",
    "answer",
    "leave_handoff",
    "handoffs",
    "complete_handoff",
    "register_device",
    "devices",
    "record_behavior",
    "list_behaviors",
    "accept_behavior",
    "retire_behavior",
    "behavior_candidates",
    // An agent publishes the pages it writes and finds them again; who may see
    // one (set_relic_visibility) is the user's decision, made in the dashboard.
    "publish_relic",
    "list_relics",
    "get_relic",
];

/// The tools a session advertises and serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProfile {
    allowed: BTreeSet<String>,
}

/// Why a `--tools` spec could not become a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// The spec named a tool this server does not have.
    Unknown { name: String, known: Vec<String> },
    /// The spec named nothing.
    Empty,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Unknown { name, known } => write!(
                f,
                "--tools names `{name}`, which is not a tool of this server; the tools are: {}",
                known.join(", ")
            ),
            ProfileError::Empty => write!(
                f,
                "--tools names no tools; use `all`, `agent`, or a comma-separated list"
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

impl ToolProfile {
    /// Parse the operator's spec against the server's full tool list: `all`
    /// (no profile: everything, the default), `agent` (the developer-agent
    /// set), or a comma-separated list of tool names, in which `agent` expands
    /// to the named profile. Case-insensitive on the keywords, exact on tool
    /// names.
    pub fn parse(spec: &str, known: &[String]) -> Result<Option<Self>, ProfileError> {
        let tokens: Vec<&str> = spec
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .collect();
        if tokens.is_empty() {
            return Err(ProfileError::Empty);
        }
        if tokens.len() == 1 && tokens[0].eq_ignore_ascii_case("all") {
            return Ok(None);
        }
        let mut allowed = BTreeSet::new();
        for token in tokens {
            if token.eq_ignore_ascii_case("agent") {
                allowed.extend(AGENT_PROFILE.iter().map(|s| s.to_string()));
            } else {
                allowed.insert(token.to_string());
            }
        }
        for name in &allowed {
            if !known.iter().any(|k| k == name) {
                return Err(ProfileError::Unknown {
                    name: name.clone(),
                    known: known.to_vec(),
                });
            }
        }
        Ok(Some(Self { allowed }))
    }

    /// What one connection asked to be given, read as [`ToolProfile::parse`]
    /// reads a spec but forgivingly: a name this server does not have is
    /// passed over, since a client may be newer or older than its server and
    /// its request costs it nothing but that tool. `None` for `all`, for
    /// nothing, and for a spec that names no tool this server has.
    pub fn asked(spec: &str, known: &[String]) -> Option<Self> {
        let mut allowed = BTreeSet::new();
        for token in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            if token.eq_ignore_ascii_case("all") {
                return None;
            }
            if token.eq_ignore_ascii_case("agent") {
                allowed.extend(AGENT_PROFILE.iter().map(|s| s.to_string()));
            } else if known.iter().any(|k| k == token) {
                allowed.insert(token.to_string());
            }
        }
        allowed.retain(|name| known.iter().any(|k| k == name));
        (!allowed.is_empty()).then_some(Self { allowed })
    }

    /// This profile cut down to what `asked` also holds. A connection can
    /// ask for less than its server serves and never for more.
    #[must_use]
    pub fn within(&self, asked: &ToolProfile) -> Self {
        Self {
            allowed: self.allowed.intersection(&asked.allowed).cloned().collect(),
        }
    }

    /// Whether `name` is advertised and callable under this profile.
    pub fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }

    /// The advertised tool names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.allowed.iter().map(String::as_str)
    }

    /// Keep only the tools the profile allows, in their original order.
    pub fn filter<T>(&self, tools: Vec<T>, name_of: impl Fn(&T) -> &str) -> Vec<T> {
        tools
            .into_iter()
            .filter(|t| self.allows(name_of(t)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known() -> Vec<String> {
        [
            "store_memory",
            "recall_memories",
            "reinforce_memory",
            "penalize_memory",
            "recall_documents",
            "ingest_document",
            "route",
            "answer",
            "leave_handoff",
            "handoffs",
            "complete_handoff",
            "register_device",
            "devices",
            "record_behavior",
            "list_behaviors",
            "accept_behavior",
            "retire_behavior",
            "behavior_candidates",
            "publish_relic",
            "list_relics",
            "get_relic",
            "set_relic_visibility",
            "population",
            "share_compartment",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// The profile `spec` names, which has to be one (not `all`).
    fn profile(spec: &str) -> anyhow::Result<ToolProfile> {
        let Some(profile) = ToolProfile::parse(spec, &known())? else {
            anyhow::bail!("{spec:?} parsed as no profile at all");
        };
        Ok(profile)
    }

    #[test]
    fn all_means_no_profile_and_agent_means_the_named_set() -> anyhow::Result<()> {
        assert_eq!(ToolProfile::parse("all", &known())?, None);
        assert_eq!(ToolProfile::parse(" ALL ", &known())?, None);
        let agent = profile("agent")?;
        let names: Vec<&str> = agent.names().collect();
        assert_eq!(names.len(), AGENT_PROFILE.len());
        assert!(agent.allows("recall_memories") && agent.allows("answer"));
        assert!(!agent.allows("share_compartment") && !agent.allows("population"));
        // An agent publishes relics; who may see one is the user's call.
        assert!(agent.allows("publish_relic") && !agent.allows("set_relic_visibility"));
        Ok(())
    }

    #[test]
    fn a_list_is_exact_and_agent_extends_within_it() -> anyhow::Result<()> {
        let p = profile("recall_memories, store_memory")?;
        assert_eq!(
            p.names().collect::<Vec<_>>(),
            vec!["recall_memories", "store_memory"]
        );
        let p = profile("agent,population")?;
        assert!(p.allows("population") && p.allows("route"));
        assert!(!p.allows("share_compartment"));
        Ok(())
    }

    #[test]
    fn unknown_or_empty_specs_are_refused_with_the_full_list() -> anyhow::Result<()> {
        let Err(err) = ToolProfile::parse("agent,recal_memories", &known()) else {
            anyhow::bail!("a misspelled tool name was accepted");
        };
        assert!(matches!(&err, ProfileError::Unknown { name, .. } if name == "recal_memories"));
        assert!(err.to_string().contains("share_compartment"), "{err}");
        let Err(err) = ToolProfile::parse(" , ", &known()) else {
            anyhow::bail!("an empty spec was accepted");
        };
        assert_eq!(err, ProfileError::Empty);
        Ok(())
    }

    #[test]
    fn filter_keeps_order_and_drops_the_rest() -> anyhow::Result<()> {
        let p = profile("route,store_memory")?;
        let kept = p.filter(vec!["store_memory", "population", "route"], |t| t);
        assert_eq!(kept, vec!["store_memory", "route"]);
        Ok(())
    }

    /// What a connection asks is read forgivingly, and only ever narrows.
    #[test]
    fn a_connection_asks_for_less_and_never_for_more() -> anyhow::Result<()> {
        let known = known();
        assert!(ToolProfile::asked("all", &known).is_none());
        assert!(ToolProfile::asked("", &known).is_none());
        assert!(ToolProfile::asked("a_tool_of_some_other_version", &known).is_none());
        let asked = ToolProfile::asked("store_memory, a_tool_of_some_other_version ,route", &known)
            .ok_or_else(|| anyhow::anyhow!("nothing asked"))?;
        assert_eq!(asked.names().collect::<Vec<_>>(), ["route", "store_memory"]);
        // Within a server that serves only one of them.
        let serves = profile("store_memory,population")?;
        assert_eq!(
            serves.within(&asked).names().collect::<Vec<_>>(),
            ["store_memory"]
        );
        Ok(())
    }
}

//! Choosing which of your own memories go to a workspace you joined
//! ([`antumbra_core::elevation`]).
//!
//! Its own router, joined to the others in `engine.rs`. `bring_candidates`
//! is what a page narrows a large store with; `bring_choose` records a
//! decision for some memories, or for every memory a filter matches, so that
//! choosing a thousand is one call and not a thousand rows fetched to a
//! browser. Nothing here sends anything anywhere: a decision is a row on this
//! server, the person's own.

use std::collections::{BTreeMap, HashMap};

use super::*;

use antumbra_core::elevation::{self, Choice, Held};
use antumbra_core::provenance::GitProvenance;
use antumbra_store::repo::elevation as store_elevation;

/// The longest a memory's text is sent to a page that is choosing among many.
const SHOWN_CHARS: usize = 400;
/// The most rows one page of candidates holds.
const MAX_ROWS: u32 = 200;
/// The most values one facet lists, the commonest first.
const FACET_VALUES: usize = 40;

#[derive(Deserialize, schemars::JsonSchema, Default, Clone)]
pub(super) struct BringFilter {
    /// `world`, `bank` or `opinion`.
    #[serde(default)]
    pub(super) network: Option<String>,
    /// A compartment of yours, by name.
    #[serde(default)]
    pub(super) compartment: Option<String>,
    /// The repository a memory is anchored to, as `host/org/name`; `none` for
    /// memories with no anchor.
    #[serde(default)]
    pub(super) repo: Option<String>,
    /// The machine a memory was written on.
    #[serde(default)]
    pub(super) host: Option<String>,
    /// The month a memory was written, as `2026-11`.
    #[serde(default)]
    pub(super) month: Option<String>,
    /// Text a memory holds, in any case.
    #[serde(default)]
    pub(super) contains: Option<String>,
    /// Only memories that have been reinforced since they were written.
    #[serde(default)]
    pub(super) used: bool,
    /// What was decided: `undecided` (the default), `chosen`, `declined`,
    /// `never`, `brought`, or `any`.
    #[serde(default)]
    pub(super) state: Option<String>,
    /// Include what is held back: the hooks' own memories, and text that
    /// looks like it holds a key, a token or a password.
    #[serde(default)]
    pub(super) include_held: bool,
    /// Only undecided memories that lie nearer what you shared before than
    /// what you did not, the nearest first. Nothing is suggested until you
    /// have shared a few; what is held back is never suggested.
    #[serde(default)]
    pub(super) suggested: bool,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct BringCandidatesParams {
    #[serde(default)]
    pub(super) filter: Option<BringFilter>,
    /// Rows to return (up to 200), the most recently written first.
    #[serde(default)]
    pub(super) limit: Option<u32>,
    #[serde(default)]
    pub(super) offset: Option<u32>,
    /// Send each memory whole, with its confidence and its git anchor: what
    /// copying it elsewhere needs. Left out, a memory's text is cut to what a
    /// page choosing among many shows.
    #[serde(default)]
    pub(super) full: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct CandidateView {
    pub(super) id: String,
    pub(super) content: String,
    pub(super) truncated: bool,
    pub(super) network: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) compartment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) host: Option<String>,
    pub(super) created_at: String,
    pub(super) reinforcement: u32,
    /// What was decided, when anything was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) state: Option<String>,
    /// Why it is held back, when it is: `housekeeping` or `looks_secret`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) held: Option<String>,
    /// With `suggested`: how much nearer what you shared than what you did
    /// not it lies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) leaning: Option<f32>,
    /// With `full`: how sure its writer was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) confidence: Option<f32>,
    /// With `full`: whether it changes over time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) volatile: Option<bool>,
    /// With `full`: where it was learned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) provenance: Option<ProvenanceView>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BringCandidatesOut {
    /// How many memories the filter matches.
    pub(super) total: usize,
    pub(super) memories: Vec<CandidateView>,
    /// For each way of narrowing, its values among your memories in this
    /// state, each with how many memories it has, the commonest first.
    pub(super) facets: BTreeMap<String, Vec<(String, usize)>>,
    /// How many of your memories are in each state, whatever the filter.
    pub(super) states: BTreeMap<String, usize>,
    /// How many undecided memories are held back.
    pub(super) held_back: usize,
    /// With `suggested`, when nothing can be suggested yet: how many more
    /// memories must be shared first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) suggests_after: Option<usize>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct BringChooseParams {
    /// `chosen`, `declined`, `never`, `brought`, or `clear` to undo a decision.
    pub(super) choice: String,
    /// The memories to decide. Left out, every memory `filter` matches.
    #[serde(default)]
    pub(super) ids: Option<Vec<String>>,
    #[serde(default)]
    pub(super) filter: Option<BringFilter>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct BringChosenOut {
    /// How many memories the decision was recorded for.
    pub(super) changed: usize,
}

/// One of the caller's own memories, with what choosing among them reads.
struct Mine {
    memory: Memory,
    compartment: Option<String>,
    repo: Option<String>,
    month: String,
    held: Option<Held>,
    choice: Option<Choice>,
}

/// The state a filter asks for: a decision, none, or any.
enum Wanted {
    Undecided,
    Any,
    Decided(Choice),
}

fn wanted(filter: &BringFilter) -> Result<Wanted, ErrorData> {
    match filter
        .state
        .as_deref()
        .map(str::trim)
        .unwrap_or("undecided")
    {
        "" | "undecided" => Ok(Wanted::Undecided),
        "any" => Ok(Wanted::Any),
        other => other
            .parse()
            .map(Wanted::Decided)
            .map_err(|e: antumbra_core::AntumbraError| {
                ErrorData::invalid_params(e.to_string(), None)
            }),
    }
}

impl Mine {
    /// Whether the filter's state and its holding-back admit this memory:
    /// what the facets are counted over.
    fn in_state(&self, state: &Wanted, include_held: bool) -> bool {
        let stands = match state {
            Wanted::Undecided => self.choice.is_none(),
            Wanted::Any => true,
            Wanted::Decided(choice) => self.choice == Some(*choice),
        };
        // What a person decided themselves is shown whatever it looks like.
        stands && (include_held || self.held.is_none() || self.choice.is_some())
    }

    fn matches(&self, filter: &BringFilter) -> bool {
        let is = |wanted: &Option<String>, held: Option<&str>| match wanted.as_deref() {
            None | Some("") => true,
            Some("none") => held.is_none(),
            Some(value) => held.is_some_and(|h| h.eq_ignore_ascii_case(value)),
        };
        is(&filter.network, Some(self.memory.network.as_str()))
            && is(&filter.compartment, self.compartment.as_deref())
            && is(&filter.repo, self.repo.as_deref())
            && is(&filter.host, self.memory.author_host.as_deref())
            && is(&filter.month, Some(self.month.as_str()))
            && (!filter.used || self.memory.reinforcement > 0)
            && filter
                .contains
                .as_deref()
                .map(str::trim)
                .is_none_or(|text| {
                    text.is_empty()
                        || self
                            .memory
                            .content
                            .to_lowercase()
                            .contains(&text.to_lowercase())
                })
    }
}

impl McpServer {
    /// The caller's own memories (those in a compartment they own, and those
    /// they wrote into none), newest first, each with what was decided.
    async fn mine(&self) -> Result<Vec<Mine>, ErrorData> {
        let owned: HashMap<String, String> =
            compartment::list_owned(&self.store, &self.tenant, &self.user)
                .await
                .map_err(err)?
                .into_iter()
                .map(|c| (c.id.as_str().to_string(), c.name))
                .collect();
        let decided = store_elevation::of(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        // Without their vectors: every memory is read, and none is searched.
        let mut mine: Vec<Mine> = memory::list_lite(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .into_iter()
            .filter_map(|memory| {
                let compartment = match &memory.compartment {
                    Some(id) => Some(owned.get(id.as_str())?.clone()),
                    None => {
                        if memory.author.as_ref() != Some(&self.user) {
                            return None;
                        }
                        None
                    }
                };
                Some(Mine {
                    repo: GitProvenance::from_evidence(&memory.evidence).map(|p| p.repo),
                    month: memory.created_at.format("%Y-%m").to_string(),
                    held: elevation::held(&memory.content, compartment.as_deref()),
                    choice: decided.get(memory.id.as_str()).map(|(choice, _)| *choice),
                    compartment,
                    memory,
                })
            })
            .collect();
        mine.sort_by_key(|m| std::cmp::Reverse(m.memory.created_at));
        Ok(mine)
    }
}

/// What is suggested among a person's memories: each suggested memory's id
/// with its leaning, or how many more must be shared before anything is.
enum Suggested {
    After(usize),
    These(HashMap<String, f32>),
}

impl McpServer {
    /// Learn from what the caller decided and say which undecided memories
    /// lie with what they shared. This is the one read here that takes the
    /// memories' vectors, so it is made only when suggestions are asked for.
    async fn suggested(&self, mine: &[Mine]) -> Result<Suggested, ErrorData> {
        let shared = mine
            .iter()
            .filter(|m| matches!(m.choice, Some(Choice::Chosen | Choice::Brought)))
            .count();
        if shared < elevation::LEARNS_FROM {
            return Ok(Suggested::After(elevation::LEARNS_FROM - shared));
        }
        let vectors: HashMap<String, Vec<f32>> = memory::list(&self.store, &self.tenant)
            .await
            .map_err(err)?
            .into_iter()
            .filter_map(|m| Some((m.id.as_str().to_string(), m.embedding?)))
            .collect();
        let of = |want: &dyn Fn(&Mine) -> bool| -> Vec<&[f32]> {
            mine.iter()
                .filter(|m| want(m))
                .filter_map(|m| vectors.get(m.memory.id.as_str()).map(Vec::as_slice))
                .collect()
        };
        let open = |m: &Mine| m.choice.is_none() && m.held.is_none();
        let Some(taste) = elevation::Taste::learn(
            &of(&|m| matches!(m.choice, Some(Choice::Chosen | Choice::Brought))),
            &of(&|m| matches!(m.choice, Some(Choice::Declined | Choice::Never))),
            &of(&open),
        ) else {
            // Shared, but without vectors to learn from.
            return Ok(Suggested::These(HashMap::new()));
        };
        Ok(Suggested::These(
            mine.iter()
                .filter(|m| open(m))
                .filter_map(|m| {
                    let id = m.memory.id.as_str();
                    let leaning = taste.leaning(vectors.get(id)?);
                    (leaning > elevation::SUGGEST_ABOVE).then(|| (id.to_string(), leaning))
                })
                .collect(),
        ))
    }
}

/// A facet's values, the commonest first, then by name.
fn commonest(counts: HashMap<String, usize>) -> Vec<(String, usize)> {
    let mut values: Vec<(String, usize)> = counts.into_iter().collect();
    values.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    values.truncate(FACET_VALUES);
    values
}

#[tool_router(router = bring_router, vis = "pub(super)")]
impl McpServer {
    /// Your own memories, narrowed, for choosing which to bring to a workspace.
    #[tool(
        description = "List your own memories for choosing which to bring to a workspace you joined: narrowed by `filter` (network, compartment, repo, host, month, contains, used, state, include_held, suggested), newest first, a page at a time, with how many match, the values each way of narrowing has, and how many are in each state. The hooks' own memories and text that looks like a secret are held back unless `include_held`. With `suggested`, only undecided memories that lie nearer what you shared before than what you did not, nearest first. Read-only; nothing is sent anywhere."
    )]
    pub(super) async fn bring_candidates(
        &self,
        Parameters(p): Parameters<BringCandidatesParams>,
    ) -> Result<Json<BringCandidatesOut>, ErrorData> {
        let filter = p.filter.unwrap_or_default();
        let state = wanted(&filter)?;
        let mine = self.mine().await?;

        let mut states: BTreeMap<String, usize> = BTreeMap::new();
        for m in &mine {
            let name = m.choice.map_or("undecided", Choice::as_str);
            *states.entry(name.to_string()).or_default() += 1;
        }
        let held_back = mine
            .iter()
            .filter(|m| m.choice.is_none() && m.held.is_some())
            .count();

        let (suggested, suggests_after) = match filter.suggested {
            false => (None, None),
            true => match self.suggested(&mine).await? {
                Suggested::After(more) => (Some(HashMap::new()), Some(more)),
                Suggested::These(these) => (Some(these), None),
            },
        };
        let leaning = |m: &Mine| suggested.as_ref()?.get(m.memory.id.as_str()).copied();
        let mut in_state: Vec<&Mine> = mine
            .iter()
            .filter(|m| m.in_state(&state, filter.include_held))
            .filter(|m| suggested.is_none() || leaning(m).is_some())
            .collect();
        if suggested.is_some() {
            // The nearest what was shared first.
            in_state.sort_by(|a, b| {
                leaning(b)
                    .partial_cmp(&leaning(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let mut counts: BTreeMap<&str, HashMap<String, usize>> = BTreeMap::new();
        for m in &in_state {
            let mut count = |facet: &'static str, value: Option<&str>| {
                *counts
                    .entry(facet)
                    .or_default()
                    .entry(value.unwrap_or("none").to_string())
                    .or_default() += 1;
            };
            count("network", Some(m.memory.network.as_str()));
            count("compartment", m.compartment.as_deref());
            count("repo", m.repo.as_deref());
            count("host", m.memory.author_host.as_deref());
            count("month", Some(m.month.as_str()));
        }
        let matching: Vec<&&Mine> = in_state.iter().filter(|m| m.matches(&filter)).collect();
        let (limit, offset) = (
            p.limit.unwrap_or(50).clamp(1, MAX_ROWS) as usize,
            p.offset.unwrap_or(0) as usize,
        );
        let memories = matching
            .iter()
            .skip(offset)
            .take(limit)
            .map(|m| {
                let most = if p.full { usize::MAX } else { SHOWN_CHARS };
                let shown: String = m.memory.content.chars().take(most).collect();
                CandidateView {
                    id: m.memory.id.as_str().to_string(),
                    truncated: shown.len() < m.memory.content.len(),
                    content: shown,
                    network: m.memory.network.as_str().to_string(),
                    compartment: m.compartment.clone(),
                    repo: m.repo.clone(),
                    host: m.memory.author_host.clone(),
                    created_at: m.memory.created_at.to_rfc3339(),
                    reinforcement: m.memory.reinforcement,
                    state: m.choice.map(|c| c.as_str().to_string()),
                    held: m.held.map(|h| h.as_str().to_string()),
                    leaning: leaning(m),
                    confidence: p.full.then_some(m.memory.confidence),
                    volatile: p.full.then_some(m.memory.volatile),
                    provenance: GitProvenance::from_evidence(&m.memory.evidence)
                        .filter(|_| p.full)
                        .map(|anchor| ProvenanceView::from(&anchor)),
                }
            })
            .collect();
        Ok(Json(BringCandidatesOut {
            total: matching.len(),
            memories,
            facets: counts
                .into_iter()
                .map(|(facet, values)| (facet.to_string(), commonest(values)))
                .collect(),
            states,
            held_back,
            suggests_after,
        }))
    }

    /// Decide which of your memories go to a workspace.
    #[tool(
        description = "Record what you decide about bringing your own memories to a workspace you joined: `choice` is `chosen`, `declined`, `never`, `brought`, or `clear` to undo. For the memories in `ids`, or, with no `ids`, for every memory `filter` matches (so choosing a thousand is one call). A memory that is not yours is left out. Nothing is sent anywhere by this."
    )]
    pub(super) async fn bring_choose(
        &self,
        Parameters(p): Parameters<BringChooseParams>,
    ) -> Result<Json<BringChosenOut>, ErrorData> {
        let choice = match p.choice.trim() {
            "clear" => None,
            other => Some(
                other
                    .parse::<Choice>()
                    .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?,
            ),
        };
        let mine = self.mine().await?;
        let ids: Vec<String> = match &p.ids {
            // Only the caller's own: an id that is someone else's, or no
            // one's, is not decided.
            Some(ids) => mine
                .iter()
                .filter(|m| ids.iter().any(|id| id == m.memory.id.as_str()))
                .map(|m| m.memory.id.as_str().to_string())
                .collect(),
            None => {
                let filter = p.filter.clone().unwrap_or_default();
                let state = wanted(&filter)?;
                // Everything suggested, when that is what the filter asks.
                let suggested = match filter.suggested {
                    false => None,
                    true => match self.suggested(&mine).await? {
                        Suggested::After(_) => Some(HashMap::new()),
                        Suggested::These(these) => Some(these),
                    },
                };
                mine.iter()
                    .filter(|m| m.in_state(&state, filter.include_held) && m.matches(&filter))
                    .filter(|m| {
                        suggested
                            .as_ref()
                            .is_none_or(|s| s.contains_key(m.memory.id.as_str()))
                    })
                    .map(|m| m.memory.id.as_str().to_string())
                    .collect()
            }
        };
        match choice {
            Some(choice) => {
                store_elevation::decide(
                    &self.store,
                    &self.tenant,
                    &self.user,
                    &ids,
                    choice,
                    chrono::Utc::now(),
                )
                .await
            }
            None => store_elevation::clear(&self.store, &self.tenant, &self.user, &ids).await,
        }
        .map_err(err)?;
        Ok(Json(BringChosenOut { changed: ids.len() }))
    }
}

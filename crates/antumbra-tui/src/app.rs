//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, read through the [`Backend`] and ticked for animation.

pub use antumbra_core::console::Snapshot;
use antumbra_core::generational::GenerationHead;
use antumbra_core::{
    EvaluationRun, Expert, FailureBoundary, LearnedRouter, Memory, MemoryEdge, MemoryNetwork,
    MemoryPolicy, MemoryUse, RunId, Shadow,
};

use std::collections::{HashMap, HashSet};

use crate::backend::Backend;
use crate::command::{self, Action, Command};
use crate::events::{Event, EventKind};
use crate::theme::Theme;

pub use acting::{Notice, Pending};

/// Which list the navigation keys drive, and which detail panel is shown: one
/// per region of the cast shadow: the population (umbra), the shadows in training
/// (penumbra), or the boundaries (antumbra).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Focus {
    Experts,
    Shadows,
    Boundaries,
}

/// What the console is showing on top of the population: nothing (the live view),
/// the keybinding help, or the command palette. Input routing follows the mode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Normal,
    Help,
    Palette,
    Filter,
    Events,
    Detail,
    /// The gate (router) inspector: learned weights + self-routing health.
    Gate,
    Ask,
    /// The "connect your agent" panel: how to wire the MCP hooks + mint a token.
    Connect,
    Confirm,
    /// The workspace's memory knobs: read by anyone, turned by an operator.
    Knobs,
}

/// How the population page arranges its panels: the list in front beside the
/// detail of what is selected (`List`, the default), the graph beside all
/// three lists at once (`Dashboard`), or the graph alone, full width
/// (`Graph`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutMode {
    List,
    Dashboard,
    Graph,
    /// A full-width sortable table of the population (the data-grid view).
    Table,
}

impl LayoutMode {
    /// The lowercase name shown in the header.
    pub fn name(self) -> &'static str {
        match self {
            LayoutMode::List => "list",
            LayoutMode::Dashboard => "dashboard",
            LayoutMode::Graph => "graph",
            LayoutMode::Table => "table",
        }
    }
}

/// The column the population table sorts on (the data-grid sort key).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SortKey {
    /// Fitness, strongest first.
    Fitness,
    /// Name, A→Z.
    Name,
    /// Generation, newest first.
    Generation,
}

impl SortKey {
    pub fn name(self) -> &'static str {
        match self {
            SortKey::Fitness => "fitness",
            SortKey::Name => "name",
            SortKey::Generation => "generation",
        }
    }
}

/// A top-level page of the console, switched via the tab strip (`[`/`]`, `1`-`4`,
/// or the palette). `Population` is the live umbra/penumbra/antumbra view; the
/// others surface subsystems that the store already holds but the console has
/// not shown before.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    /// The living population: experts, shadows, boundaries, and the gate.
    Population,
    /// The penumbra memory networks (world / bank / opinion) and their edges.
    Memory,
    /// The generational loop state: grow, explore, score, decide, consolidate.
    Loop,
    /// Evaluation runs and the regression tripwire.
    Evals,
    /// A coding agent with its feature flags off: the rules, and which skills
    /// are used. Read-only.
    Sovereign,
}

impl Page {
    /// Every page, in tab order.
    pub const ALL: [Page; 5] = [
        Page::Population,
        Page::Memory,
        Page::Loop,
        Page::Evals,
        Page::Sovereign,
    ];

    /// The tab's label: what the page holds, in the dashboard's words.
    pub fn title(self) -> &'static str {
        match self {
            Page::Population => "Population",
            Page::Memory => "Memory",
            Page::Loop => "Training",
            Page::Evals => "Evaluations",
            Page::Sovereign => "Agent",
        }
    }

    /// The page a name asks for (`tui snapshot --page`): its tab's label in
    /// any case, or the name it had before the tabs were reworded. Anything
    /// else is the first page.
    pub fn from_name(name: &str) -> Page {
        let name = name.to_lowercase();
        let former = |page: Page| match page {
            Page::Loop => "loop",
            Page::Evals => "evals",
            Page::Sovereign => "sovereign",
            Page::Population | Page::Memory => "",
        };
        Page::ALL
            .into_iter()
            .find(|page| page.title().to_lowercase() == name || former(*page) == name)
            .unwrap_or(Page::Population)
    }

    /// The number key that goes to this page: `1` for the first. Asked of
    /// [`Page::ALL`], so a page added to it cannot be left without its key.
    pub fn key(self) -> Option<char> {
        u32::try_from(self.index() + 1)
            .ok()
            .and_then(|number| char::from_digit(number, 10))
    }

    /// Position in [`Page::ALL`] (the tab index).
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&p| p == self).unwrap_or(0)
    }
}

/// The command palette's transient state: the typed query and the highlighted
/// match (an index into the filtered list).
#[derive(Default)]
pub struct Palette {
    pub query: String,
    pub selected: usize,
}

/// Everything the console draws, refreshed from the store.
pub struct App {
    pub experts: Vec<Expert>,
    pub boundaries: Vec<FailureBoundary>,
    /// Recent shadows in training (the penumbra), newest first.
    pub shadows: Vec<Shadow>,
    /// Penumbra memory traces (world / bank / opinion), for the Memory page.
    pub memories: Vec<Memory>,
    /// Typed edges between memories (references / supersedes / contradicts / …).
    pub edges: Vec<MemoryEdge>,
    /// What the system has seen of each memory's use, by memory id.
    pub usage: HashMap<String, MemoryUse>,
    /// The workspace's memory knobs, which each memory's strength is read with.
    pub policy: MemoryPolicy,
    /// The knob highlighted in the knobs panel, and the values being turned
    /// there before they are applied.
    pub knob_selected: usize,
    pub knob_draft: MemoryPolicy,
    /// Index into `memories` of the highlighted trace (Memory page).
    pub selected_memory: usize,
    /// The kind of memory the Memory page lists: one network, or all three.
    pub memory_kind: Option<MemoryNetwork>,
    /// Whether the Memory page shows the graph of the kinds and their links
    /// in place of the list.
    pub memory_graph: bool,
    /// The Sovereign page's reading of `memories`, made once per reload.
    pub sovereign: crate::sovereign::View,
    /// Index into `sovereign.skills` of the highlighted row (Sovereign page).
    pub selected_skill: usize,
    /// The generational loop heads (one per run), for the Loop page.
    pub loop_heads: Vec<GenerationHead>,
    /// Index into `loop_heads` of the run the Loop page is focused on.
    pub selected_loop: usize,
    /// The runs with a graceful halt requested that they have not reached
    /// yet, by run id.
    pub halts: HashSet<String>,
    /// Recent evaluation runs (newest first), for the Evals page.
    pub evals: Vec<EvaluationRun>,
    /// Index into `evals` of the highlighted run (Evals page).
    pub selected_eval: usize,
    pub router: Option<LearnedRouter>,
    /// Index into `experts` of the highlighted node.
    pub selected: usize,
    /// Index into `boundaries` of the highlighted scope (when focused there).
    pub selected_boundary: usize,
    /// Index into `shadows` of the highlighted shadow (when focused there).
    pub selected_shadow: usize,
    /// The active top-level page (tab strip).
    pub page: Page,
    /// Which list navigation/detail targets (umbra / penumbra / antumbra).
    pub focus: Focus,
    /// How the body arranges its panels (focused / dashboard / graph / table).
    pub layout: LayoutMode,
    /// The column the population table sorts on.
    pub sort: SortKey,
    /// What's drawn on top of the live view (help / palette / nothing).
    pub mode: Mode,
    /// The command palette's query and selection (used while `mode == Palette`).
    pub palette: Palette,
    /// The focused-list filter query (used while `mode == Filter`).
    pub filter: String,
    /// The highlighted match within the filter results.
    pub filter_selected: usize,
    /// The filter's matches for the query as typed, best first: each item's
    /// place in its list and the label shown. Worked out when the query or
    /// the list changes, and only read while drawing.
    pub filter_hits: Vec<(usize, String)>,
    /// The live event stream (store changes), newest first.
    pub events: Vec<Event>,
    /// Scroll position within the events overlay.
    pub events_scroll: usize,
    /// Scroll offset within the drill-down detail overlay.
    pub detail_scroll: u16,
    /// An operator mutation awaiting confirmation (used while `mode == Confirm`).
    pub pending: Option<Pending>,
    /// What the console last did, declined to do, or could not do, for the
    /// line above the footer.
    pub notice: Option<Notice>,
    /// Whether the last read of the server failed, so what is shown may be
    /// behind.
    pub reads_failing: bool,
    /// The route-ask query text (used while `mode == Ask`).
    pub ask_query: String,
    /// The gate's routing for the last ask: `(expert name, probability)` best
    /// first, or `Some(empty)` when it escalates. `None` before the first ask.
    pub ask_result: Option<Vec<(String, f32)>>,
    /// Why the last ask was routed to no expert, in the server's words, when
    /// it was routed to none and the server said.
    pub ask_reason: Option<String>,
    /// Whether a server answers the ask, embedding the task with its own
    /// model. When false, the ask uses the demo's stand-in embedder and says
    /// so.
    pub real_embedder: bool,
    /// Per-entity fingerprints from the last reload, to diff the next one.
    pub ev_experts: HashMap<String, bool>,
    pub ev_shadows: HashMap<String, String>,
    pub ev_boundaries: HashMap<String, bool>,
    pub ev_router: bool,
    /// Whether the first reload has set the baseline (so it doesn't emit a flood).
    pub ev_baseline: bool,
    /// Index into [`crate::theme::ALL`] of the active palette.
    pub theme_idx: usize,
    /// The frame-rate cap the loop paces to (Hz), adjustable with `+`/`-`.
    pub target_fps: u32,
    /// When set, the cap follows the active monitor's refresh rate; `+`/`-` pin it.
    pub auto_fps: bool,
    /// The terminal window handle captured at launch, so the follow tracks this
    /// window between monitors (not whatever is focused). `0` = live foreground.
    pub window: usize,
    /// The measured frame rate (smoothed), shown as a live readout.
    pub fps: f64,
    /// Total elapsed animation time (ms), drives orbit/pulse/energy.
    pub clock_ms: f64,
    /// Seconds since the last reload, so the view refreshes periodically.
    pub since_reload_ms: f64,
    /// Time since the last keypress (ms); drives the idle frame-rate throttle.
    pub since_input_ms: f64,
    /// Opt in to easing the rate down after an idle spell (`--power-save`); off by
    /// default so the console renders at the full cap continuously.
    pub power_save: bool,
    /// Whether the loop is currently eased to the idle rate (no recent input).
    pub idle: bool,
    /// The resolved render capability (vector Canvas vs raster), fixed at startup.
    pub render_tier: crate::render::RenderTier,
    /// Launched with `--demo`: a throwaway seeded store, shown as a badge so the
    /// operator knows nothing persists.
    pub demo: bool,
    /// Where the console reads from (the server's address, or the demo), for
    /// the connect panel.
    pub source: String,
    /// Where the server's address and the token were found (a flag, the
    /// environment, Claude Code's settings, `antumbra setup`), for the connect
    /// panel; empty for the demo.
    pub source_from: String,
    pub token_from: String,
    /// The token may act on the substrate, and the substrate's pages are
    /// filled. Without it the console reads the workspace and acts on nothing.
    pub operator: bool,
    /// How often the view is read again ([`Backend::reload_every_ms`]).
    pub reload_every_ms: f64,
    /// Where the console was started, for each memory's scope.
    pub git: antumbra_core::GitContext,
    pub should_quit: bool,
}

impl App {
    /// The console over `backend`, with its first snapshot read. A first read
    /// that fails is the caller's to report; later ones are events
    /// ([`Self::refresh`]).
    pub async fn load(backend: &Backend) -> anyhow::Result<Self> {
        let mut app = Self {
            experts: Vec::new(),
            boundaries: Vec::new(),
            shadows: Vec::new(),
            memories: Vec::new(),
            edges: Vec::new(),
            usage: HashMap::new(),
            policy: MemoryPolicy::default(),
            knob_selected: 0,
            knob_draft: MemoryPolicy::default(),
            selected_memory: 0,
            memory_kind: None,
            memory_graph: false,
            sovereign: crate::sovereign::View::default(),
            selected_skill: 0,
            loop_heads: Vec::new(),
            selected_loop: 0,
            halts: HashSet::new(),
            evals: Vec::new(),
            selected_eval: 0,
            router: None,
            selected: 0,
            selected_boundary: 0,
            selected_shadow: 0,
            page: Page::Population,
            focus: Focus::Experts,
            layout: LayoutMode::List,
            sort: SortKey::Fitness,
            mode: Mode::Normal,
            palette: Palette::default(),
            filter: String::new(),
            filter_selected: 0,
            filter_hits: Vec::new(),
            events: Vec::new(),
            events_scroll: 0,
            detail_scroll: 0,
            pending: None,
            notice: None,
            reads_failing: false,
            ask_query: String::new(),
            ask_result: None,
            ask_reason: None,
            real_embedder: false,
            ev_experts: HashMap::new(),
            ev_shadows: HashMap::new(),
            ev_boundaries: HashMap::new(),
            ev_router: false,
            ev_baseline: false,
            theme_idx: 0,
            target_fps: 144,
            auto_fps: true,
            window: 0,
            fps: 0.0,
            clock_ms: 0.0,
            since_reload_ms: 0.0,
            since_input_ms: 0.0,
            power_save: false,
            idle: false,
            render_tier: crate::render::RenderTier::default(),
            demo: false,
            source: backend.describe(),
            source_from: String::new(),
            token_from: String::new(),
            operator: false,
            reload_every_ms: backend.reload_every_ms(),
            git: crate::here::context(&std::env::current_dir().unwrap_or_default()),
            should_quit: false,
        };
        app.real_embedder = backend.routes_for_real();
        if let Some((surface_from, token_from)) = backend.found_from() {
            app.source_from = surface_from;
            app.token_from = token_from;
        }
        app.reload(backend).await?;
        // Every list opens on its first row. The experts arrive in the
        // store's order and are sorted here, and the sort keeps the selection
        // on its expert, which at the first read is whichever came first from
        // the store; the Memory page lists kind after kind.
        app.selected = 0;
        if let Some(&first) = app.listed_memories().first() {
            app.selected_memory = first;
        }
        Ok(app)
    }

    /// True when the store holds nothing to show yet: a fresh install before any
    /// seed, loop run, or captured memory. Drives the first-run hint.
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
            && self.boundaries.is_empty()
            && self.shadows.is_empty()
            && self.memories.is_empty()
            && self.loop_heads.is_empty()
            && self.evals.is_empty()
    }

    pub async fn reload(&mut self, backend: &Backend) -> anyhow::Result<()> {
        match backend.snapshot().await? {
            Some(snapshot) => self.apply_snapshot(snapshot),
            // Nothing changed since the last read.
            None => self.since_reload_ms = 0.0,
        }
        Ok(())
    }

    /// Read the server again now, saying a failure on the main view rather
    /// than ending the console: over a network, a missed read is ordinary.
    pub async fn refresh(&mut self, backend: &Backend) {
        let read = backend.snapshot().await;
        self.read_finished(read);
    }

    /// Swap in a freshly loaded [`Snapshot`]: re-run the App-state-dependent expert
    /// sort, clamp selections to the new lengths, and diff for the event stream.
    /// Cheap and synchronous, so the run loop can call it the moment an off-thread
    /// load finishes, keeping the periodic refresh off the render thread.
    pub fn apply_snapshot(&mut self, snap: Snapshot) {
        // Every run's halt from a server that lists them, else the first
        // run's alone, which is all an older server says. Read before the
        // snapshot is taken apart.
        self.halts = snap
            .loop_heads
            .iter()
            .filter(|head| snap.halt_pending(&head.run_id))
            .map(|head| head.run_id.as_str().to_string())
            .collect();
        // What is selected, by what it is: the lists are about to be
        // replaced, and need not come back in the order they were in.
        let kept = self.kept();
        self.experts = snap.experts;
        self.sort_experts();
        self.boundaries = snap.boundaries;
        self.shadows = snap.shadows;
        self.memories = snap.memories;
        self.sovereign = crate::sovereign::View::from_memories(&self.memories);
        self.edges = snap.edges;
        self.usage = snap
            .usage
            .into_iter()
            .map(|u| (u.memory.as_str().to_string(), u))
            .collect();
        // A panel left open keeps what is being turned; a closed one follows
        // the server.
        if self.mode != Mode::Knobs {
            self.knob_draft = snap.policy.clone();
        }
        self.policy = snap.policy;
        self.loop_heads = snap.loop_heads;
        self.evals = snap.evals;
        self.router = snap.router;
        self.operator = snap.operator;
        // Each selection back on what it was on; one whose item is gone
        // keeps its row, within the list.
        self.keep(kept);
        // A list that changed under an open filter is searched again, so a
        // match never points at a row that has moved.
        if self.mode == Mode::Filter {
            self.refilter();
        }
        self.record_events();
        self.since_reload_ms = 0.0;
    }

    /// Diff this reload against the last to emit store-change events. The first
    /// reload just records the baseline (so it doesn't flood with "everything is
    /// new"); later reloads emit additions and state changes.
    pub(crate) fn record_events(&mut self) {
        let experts: HashMap<String, bool> = self
            .experts
            .iter()
            .map(|e| (e.id.as_str().to_string(), e.is_frozen()))
            .collect();
        let shadows: HashMap<String, String> = self
            .shadows
            .iter()
            .map(|s| (s.id.as_str().to_string(), s.status.as_str().to_string()))
            .collect();
        let boundaries: HashMap<String, bool> = self
            .boundaries
            .iter()
            .map(|b| (b.id.as_str().to_string(), b.is_actionable()))
            .collect();
        let router = self.router.is_some();

        let mut emit: Vec<(EventKind, String)> = Vec::new();
        if !self.ev_baseline {
            self.ev_baseline = true;
            emit.push((
                EventKind::System,
                format!(
                    "connected · {} experts · {} candidates · {} boundaries",
                    self.experts.len(),
                    self.shadows.len(),
                    self.boundaries.len()
                ),
            ));
        } else {
            for e in &self.experts {
                match self.ev_experts.get(e.id.as_str()) {
                    None => emit.push((
                        EventKind::Spawn,
                        format!("expert {} added", crate::words::expert_name(e)),
                    )),
                    Some(&frozen) if !frozen && e.is_frozen() => emit.push((
                        EventKind::Freeze,
                        format!("expert {} frozen", crate::words::expert_name(e)),
                    )),
                    Some(&frozen) if frozen && !e.is_frozen() => emit.push((
                        EventKind::Freeze,
                        format!("expert {} thawed", crate::words::expert_name(e)),
                    )),
                    _ => {}
                }
            }
            for s in &self.shadows {
                let status = s.status.as_str();
                match self.ev_shadows.get(s.id.as_str()) {
                    None => emit.push((
                        EventKind::Spawn,
                        format!("candidate {} started ({status})", s.id.as_str()),
                    )),
                    Some(prev) if prev != status => {
                        let kind = match status {
                            "graduated" => EventKind::Graduate,
                            "pruned" => EventKind::Prune,
                            _ => EventKind::System,
                        };
                        emit.push((kind, format!("candidate {} {status}", s.id.as_str())));
                    }
                    _ => {}
                }
            }
            for b in &self.boundaries {
                match self.ev_boundaries.get(b.id.as_str()) {
                    None => emit.push((
                        EventKind::Boundary,
                        format!("boundary recorded · {}", b.behavior),
                    )),
                    Some(&actionable) if !actionable && b.is_actionable() => emit.push((
                        EventKind::Boundary,
                        format!("boundary now in force · {}", b.behavior),
                    )),
                    _ => {}
                }
            }
            if router && !self.ev_router {
                emit.push((EventKind::System, "router trained".to_string()));
            }
        }

        for (kind, text) in emit {
            self.push_event(kind, text);
        }
        self.ev_experts = experts;
        self.ev_shadows = shadows;
        self.ev_boundaries = boundaries;
        self.ev_router = router;
    }

    /// Open the live event-stream overlay.
    pub fn open_events(&mut self) {
        self.mode = Mode::Events;
        self.events_scroll = 0;
    }

    /// Scroll the events overlay by `delta` (clamped).
    pub fn events_move(&mut self, delta: i32) {
        let n = self.events.len();
        if n == 0 {
            self.events_scroll = 0;
            return;
        }
        let cur = self.events_scroll.min(n - 1) as i32;
        self.events_scroll = (cur + delta).clamp(0, n as i32 - 1) as usize;
    }

    /// Open the drill-down detail overlay for the focused selection (Enter).
    /// Open the gate (router) inspector overlay.
    pub fn open_gate(&mut self) {
        self.mode = Mode::Gate;
        self.detail_scroll = 0;
    }

    /// Open the "connect your agent" panel: a guided reference for wiring the
    /// MCP lifecycle hooks and minting a hook token.
    pub fn open_connect(&mut self) {
        self.mode = Mode::Connect;
        self.detail_scroll = 0;
    }

    pub fn open_detail(&mut self) {
        // The population regions, the Evals rows, and the Loop runs have a
        // drill-down overlay; Memory shows its detail inline.
        match self.page {
            Page::Population => {}
            Page::Evals if !self.evals.is_empty() => {}
            Page::Loop if !self.loop_heads.is_empty() => {}
            _ => return,
        }
        self.mode = Mode::Detail;
        self.detail_scroll = 0;
    }

    /// Every run for the same subject as `run` (newest first), from the already-
    /// loaded set: the subject's evaluation history for the drill-down.
    pub fn eval_history<'a>(&'a self, run: &EvaluationRun) -> Vec<&'a EvaluationRun> {
        self.evals
            .iter()
            .filter(|r| r.subject_kind == run.subject_kind && r.subject_id == run.subject_id)
            .collect()
    }

    /// The loop head the Loop page is focused on (selection is clamped on
    /// reload, but `get` keeps this total in case the heads emptied mid-frame).
    pub fn selected_loop_head(&self) -> Option<&GenerationHead> {
        self.loop_heads.get(self.selected_loop)
    }

    /// The evaluation runs belonging to `head`'s run (from the loaded set):
    /// the loop run's evaluation history for the Loop drill-down, which is how
    /// the Loop page links to the evidence the Evals page tabulates.
    pub fn loop_eval_history<'a>(&'a self, head: &GenerationHead) -> Vec<&'a EvaluationRun> {
        self.evals
            .iter()
            .filter(|e| e.run_id.as_str() == head.run_id.as_str())
            .collect()
    }

    /// Scroll the detail overlay by `delta` lines (clamped at the top).
    pub fn detail_move(&mut self, delta: i32) {
        self.detail_scroll = (self.detail_scroll as i32 + delta).max(0) as u16;
    }

    /// Open the route-ask overlay (type a task, the gate routes it).
    pub fn open_ask(&mut self) {
        self.mode = Mode::Ask;
        self.ask_query.clear();
        self.ask_result = None;
        self.ask_reason = None;
    }

    /// Append a character to the ask query (invalidates the stale result).
    pub fn ask_input(&mut self, c: char) {
        self.ask_query.push(c);
        self.ask_result = None;
        self.ask_reason = None;
    }

    /// Delete the last character of the ask query (invalidates the result).
    pub fn ask_backspace(&mut self) {
        self.ask_query.pop();
        self.ask_result = None;
        self.ask_reason = None;
    }

    /// Keep what routing the task came to: each expert by the name the
    /// console shows it under, with how likely it is, and why there is none
    /// when there is none.
    pub fn set_ask_result(&mut self, routed: &crate::backend::Routed) {
        self.ask_result = Some(
            routed
                .routes
                .iter()
                .map(|(id, p)| (self.expert_called(id), *p))
                .collect(),
        );
        self.ask_reason = routed.reason.clone();
    }

    /// The name the console shows an expert under ([`crate::words::expert_name`]),
    /// or its id for one that is not in the population it reads.
    pub fn expert_called(&self, id: &antumbra_core::ExpertId) -> String {
        self.experts
            .iter()
            .find(|e| e.id.as_str() == id.as_str())
            .map_or_else(|| id.as_str().to_string(), crate::words::expert_name)
    }

    /// The standing experts that shape every answer where the console was
    /// started: the one for everywhere, then the one for the repository it is
    /// in. The server composes the same two into an answer there. A standing
    /// expert is never routed to, so the ask shows these beside what routing
    /// came to.
    pub fn standing_here(&self) -> Vec<&Expert> {
        let here = self.git.repo.as_deref();
        let of = |scope: &str| {
            self.experts
                .iter()
                .find(move |e| e.standing_scope() == Some(scope))
        };
        of(antumbra_core::behavior::EVERYWHERE)
            .into_iter()
            .chain(here.and_then(of))
            .collect()
    }

    /// How many standing experts there are, in every scope.
    pub fn standing_count(&self) -> usize {
        self.experts
            .iter()
            .filter(|e| e.standing_scope().is_some())
            .count()
    }

    /// The active palette every panel tints from.
    pub fn theme(&self) -> Theme {
        crate::theme::ALL[self.theme_idx % crate::theme::ALL.len()]
    }

    /// Advance to the next palette (wraps): night -> mist -> mono.
    pub fn cycle_theme(&mut self) {
        self.theme_idx = (self.theme_idx + 1) % crate::theme::ALL.len();
    }

    /// Raise the frame-rate cap to the next common refresh rate (pins manual).
    pub fn fps_up(&mut self) {
        self.auto_fps = false;
        self.target_fps = crate::pacing::next_preset(self.target_fps);
    }

    /// Lower the frame-rate cap to the previous common refresh rate (pins manual).
    pub fn fps_down(&mut self) {
        self.auto_fps = false;
        self.target_fps = crate::pacing::prev_preset(self.target_fps);
    }

    /// Remember the terminal window (foreground at launch) the follow tracks.
    pub fn capture_window(&mut self) {
        self.window = crate::pacing::foreground_window();
    }

    /// Pin the cap to a fixed rate (the `--fps` flag).
    pub fn pin_fps(&mut self, fps: u32) {
        self.auto_fps = false;
        self.target_fps = fps.clamp(crate::pacing::MIN_FPS, crate::pacing::MAX_FPS);
    }

    /// Hand the cap back to the active monitor's refresh rate (the `a` key).
    pub fn follow_monitor(&mut self) {
        self.auto_fps = true;
        if let Some(hz) = crate::pacing::detect_refresh(self.window) {
            self.target_fps = crate::pacing::snap_refresh(hz);
        }
    }

    /// While following, refresh the cap from the active monitor (called on an
    /// interval so dragging the terminal between monitors retargets the rate).
    pub fn poll_monitor(&mut self) {
        if self.auto_fps {
            if let Some(hz) = crate::pacing::detect_refresh(self.window) {
                self.target_fps = crate::pacing::snap_refresh(hz);
            }
        }
    }

    /// Fold this frame's duration into the smoothed FPS readout.
    pub fn record_frame(&mut self, dt_ms: f64) {
        if dt_ms <= 0.0 {
            return;
        }
        let instant = 1000.0 / dt_ms;
        // Exponential moving average so the readout is steady, not jittery.
        self.fps = if self.fps <= 0.0 {
            instant
        } else {
            self.fps * 0.9 + instant * 0.1
        };
    }

    /// The rate to display: the measured FPS once warmed, else the target (so a
    /// fresh frame, e.g. a headless snapshot, shows a stable number).
    pub fn shown_fps(&self) -> u32 {
        if self.fps >= 1.0 {
            self.fps.round() as u32
        } else {
            self.target_fps
        }
    }

    /// Toggle the keybinding help overlay, opened at its top.
    pub fn toggle_help(&mut self) {
        self.mode = match self.mode {
            Mode::Help => Mode::Normal,
            _ => Mode::Help,
        };
        self.detail_scroll = 0;
    }

    /// Whether `run` has a halt requested that it has not reached yet.
    pub fn halt_pending(&self, run: &RunId) -> bool {
        self.halts.contains(run.as_str())
    }

    /// Whether the run the Loop page is on has a halt pending.
    pub fn selected_halt_pending(&self) -> bool {
        self.selected_loop_head()
            .is_some_and(|head| self.halt_pending(&head.run_id))
    }

    /// Dismiss any overlay, returning to the live view.
    pub fn close_overlay(&mut self) {
        self.mode = Mode::Normal;
    }

    /// Open the command palette with an empty query.
    pub fn open_palette(&mut self) {
        self.mode = Mode::Palette;
        self.palette.query.clear();
        self.palette.selected = 0;
    }

    /// Append a typed character to the palette query (resets the selection).
    pub fn palette_input(&mut self, c: char) {
        self.palette.query.push(c);
        self.palette.selected = 0;
    }

    /// Delete the last character of the palette query (resets the selection).
    pub fn palette_backspace(&mut self) {
        self.palette.query.pop();
        self.palette.selected = 0;
    }

    /// Move the palette selection by `delta`, wrapping within the matches.
    pub fn palette_move(&mut self, delta: i32) {
        let n = self.palette_matches().len();
        if n == 0 {
            self.palette.selected = 0;
            return;
        }
        let cur = self.palette.selected.min(n - 1) as i32;
        self.palette.selected = (cur + delta).rem_euclid(n as i32) as usize;
    }

    /// The palette commands matching the current query, best first.
    pub fn palette_matches(&self) -> Vec<&'static Command> {
        command::matches(&self.palette.query)
    }

    /// The action of the highlighted palette match, if any.
    pub fn palette_action(&self) -> Option<Action> {
        let matches = self.palette_matches();
        let idx = self.palette.selected.min(matches.len().saturating_sub(1));
        matches.get(idx).map(|c| c.action)
    }
}

mod acting;
mod knobs;
mod selection;
#[cfg(test)]
mod tests;
mod view;

pub use knobs::knob_value;

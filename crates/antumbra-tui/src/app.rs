//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, loaded from the store and ticked for animation.

use antumbra_core::generational::{GenerationHead, LoopCommand};
use antumbra_core::{
    BoundaryId, EvaluationRun, Expert, ExpertId, FailureBoundary, LearnedRouter, Memory,
    MemoryEdge, RunId, Shadow, ShadowId, ShadowStatus,
};
use antumbra_store::repo::{
    boundary, edge, evaluation, expert, generation, loop_control, memory, router, shadow,
};
use antumbra_store::Store;

use chrono::Utc;

use std::collections::HashMap;

use crate::command::{self, Action, Command};
use crate::events::{Event, EventKind};
use crate::theme::Theme;

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
}

/// An operator mutation awaiting a yes/no confirmation.
pub enum Pending {
    /// Mark the shadow pruned (collapse it).
    PruneShadow(ShadowId),
    /// Mark the shadow graduated (promote it into the population).
    GraduateShadow(ShadowId),
    /// Freeze (`true`) or thaw (`false`) the expert.
    FreezeExpert(ExpertId, bool),
    /// Delete the boundary from the antumbra.
    DeleteBoundary(BoundaryId),
    /// Ask the generational loop to halt gracefully at the next generation.
    HaltLoop(RunId),
}

/// How the body arranges its panels: the graph beside a single focused detail
/// (`Focused`), the graph beside all three regions at once (`Dashboard`), or the
/// graph alone, full width (`Graph`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutMode {
    Focused,
    Dashboard,
    Graph,
    /// A full-width sortable table of the population (the data-grid view).
    Table,
}

impl LayoutMode {
    /// The lowercase name shown in the header.
    pub fn name(self) -> &'static str {
        match self {
            LayoutMode::Focused => "focused",
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
}

impl Page {
    /// Every page, in tab order.
    pub const ALL: [Page; 4] = [Page::Population, Page::Memory, Page::Loop, Page::Evals];

    /// The lowercase tab label.
    pub fn title(self) -> &'static str {
        match self {
            Page::Population => "population",
            Page::Memory => "memory",
            Page::Loop => "loop",
            Page::Evals => "evals",
        }
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
    /// Index into `memories` of the highlighted trace (Memory page).
    pub selected_memory: usize,
    /// The generational loop heads (one per run), for the Loop page.
    pub loop_heads: Vec<GenerationHead>,
    /// Index into `loop_heads` of the run the Loop page is focused on.
    pub selected_loop: usize,
    /// Whether a graceful halt is pending for the displayed loop run.
    pub loop_halt_pending: bool,
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
    /// The live event stream (store changes), newest first.
    pub events: Vec<Event>,
    /// Scroll position within the events overlay.
    pub events_scroll: usize,
    /// Scroll offset within the drill-down detail overlay.
    pub detail_scroll: u16,
    /// An operator mutation awaiting confirmation (used while `mode == Confirm`).
    pub pending: Option<Pending>,
    /// The route-ask query text (used while `mode == Ask`).
    pub ask_query: String,
    /// The gate's routing for the last ask: `(expert name, probability)` best
    /// first, or `Some(empty)` when it escalates. `None` before the first ask.
    pub ask_result: Option<Vec<(String, f32)>>,
    /// Whether a real embedding endpoint backs the ask (`--embed-url`). When
    /// false, the ask uses the demo embedder and warns it's only coherent on a
    /// demo store.
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
    /// The store url this console is bound to, surfaced in the connect panel so
    /// the generated MCP-server command matches what the operator is viewing.
    pub store_url: String,
    pub should_quit: bool,
}

/// A consistent read of everything the console shows, loaded in one place so the
/// run loop can run it OFF the render thread; only the cheap
/// [`App::apply_snapshot`] then touches `App`. Loading thousands of rows over a
/// `ws://` connection must never block input.
pub struct Snapshot {
    pub experts: Vec<Expert>,
    pub boundaries: Vec<FailureBoundary>,
    pub shadows: Vec<Shadow>,
    pub memories: Vec<Memory>,
    pub edges: Vec<MemoryEdge>,
    pub loop_heads: Vec<GenerationHead>,
    pub loop_halt_pending: bool,
    pub evals: Vec<EvaluationRun>,
    pub router: Option<LearnedRouter>,
}

impl Snapshot {
    /// Read the whole console view. Pure sorts that need no `App` state are done
    /// here; the expert sort (which depends on the active column) is deferred to
    /// [`App::apply_snapshot`].
    pub async fn load(store: &Store) -> anyhow::Result<Self> {
        let experts = expert::list(store).await?;
        let boundaries = boundary::list(store).await?;
        let mut shadows = shadow::list(store).await?;
        // Newest first, so the penumbra view leads with current training.
        shadows.sort_by_key(|s| std::cmp::Reverse(s.created_at));
        // Lite load (no embedding vectors): the console never shows them and a
        // whole store of embeddings stalls a ws:// read. Grouped by network then
        // strongest first, so each network leads with its anchors.
        let mut memories = memory::all_unscoped_lite(store).await?;
        memories.sort_by(|a, b| {
            a.network
                .as_str()
                .cmp(b.network.as_str())
                .then(b.confidence.total_cmp(&a.confidence))
        });
        let edges = edge::all_unscoped(store).await?;
        let loop_heads = generation::all_heads(store).await?;
        let loop_halt_pending = match loop_heads.first() {
            Some(head) => loop_control::load(store, &head.run_id).await? == LoopCommand::Halt,
            None => false,
        };
        let evals = evaluation::recent_unscoped(store).await?;
        let router = router::load(store).await?;
        Ok(Self {
            experts,
            boundaries,
            shadows,
            memories,
            edges,
            loop_heads,
            loop_halt_pending,
            evals,
            router,
        })
    }
}

impl App {
    pub async fn load(store: &Store) -> anyhow::Result<Self> {
        let mut app = Self {
            experts: Vec::new(),
            boundaries: Vec::new(),
            shadows: Vec::new(),
            memories: Vec::new(),
            edges: Vec::new(),
            selected_memory: 0,
            loop_heads: Vec::new(),
            selected_loop: 0,
            loop_halt_pending: false,
            evals: Vec::new(),
            selected_eval: 0,
            router: None,
            selected: 0,
            selected_boundary: 0,
            selected_shadow: 0,
            page: Page::Population,
            focus: Focus::Experts,
            layout: LayoutMode::Focused,
            sort: SortKey::Fitness,
            mode: Mode::Normal,
            palette: Palette::default(),
            filter: String::new(),
            filter_selected: 0,
            events: Vec::new(),
            events_scroll: 0,
            detail_scroll: 0,
            pending: None,
            ask_query: String::new(),
            ask_result: None,
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
            store_url: String::new(),
            should_quit: false,
        };
        app.reload(store).await?;
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

    pub async fn reload(&mut self, store: &Store) -> anyhow::Result<()> {
        self.apply_snapshot(Snapshot::load(store).await?);
        Ok(())
    }

    /// Swap in a freshly loaded [`Snapshot`]: re-run the App-state-dependent expert
    /// sort, clamp selections to the new lengths, and diff for the event stream.
    /// Cheap and synchronous, so the run loop can call it the moment an off-thread
    /// load finishes, keeping the periodic refresh off the render thread.
    pub fn apply_snapshot(&mut self, snap: Snapshot) {
        self.experts = snap.experts;
        self.sort_experts();
        self.boundaries = snap.boundaries;
        self.shadows = snap.shadows;
        self.memories = snap.memories;
        self.edges = snap.edges;
        self.loop_heads = snap.loop_heads;
        self.loop_halt_pending = snap.loop_halt_pending;
        self.evals = snap.evals;
        self.router = snap.router;
        if !self.experts.is_empty() && self.selected >= self.experts.len() {
            self.selected = self.experts.len() - 1;
        }
        if !self.boundaries.is_empty() && self.selected_boundary >= self.boundaries.len() {
            self.selected_boundary = self.boundaries.len() - 1;
        }
        if !self.shadows.is_empty() && self.selected_shadow >= self.shadows.len() {
            self.selected_shadow = self.shadows.len() - 1;
        }
        if !self.memories.is_empty() && self.selected_memory >= self.memories.len() {
            self.selected_memory = self.memories.len() - 1;
        }
        if !self.evals.is_empty() && self.selected_eval >= self.evals.len() {
            self.selected_eval = self.evals.len() - 1;
        }
        if !self.loop_heads.is_empty() && self.selected_loop >= self.loop_heads.len() {
            self.selected_loop = self.loop_heads.len() - 1;
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
                    "connected · {} experts · {} shadows · {} boundaries",
                    self.experts.len(),
                    self.shadows.len(),
                    self.boundaries.len()
                ),
            ));
        } else {
            for e in &self.experts {
                match self.ev_experts.get(e.id.as_str()) {
                    None => emit.push((EventKind::Spawn, format!("expert {} emerged", e.name))),
                    Some(&frozen) if !frozen && e.is_frozen() => {
                        emit.push((EventKind::Freeze, format!("expert {} frozen", e.name)))
                    }
                    Some(&frozen) if frozen && !e.is_frozen() => {
                        emit.push((EventKind::Freeze, format!("expert {} thawed", e.name)))
                    }
                    _ => {}
                }
            }
            for s in &self.shadows {
                let status = s.status.as_str();
                match self.ev_shadows.get(s.id.as_str()) {
                    None => emit.push((
                        EventKind::Spawn,
                        format!("shadow {} spawned ({status})", s.id.as_str()),
                    )),
                    Some(prev) if prev != status => {
                        let kind = match status {
                            "graduated" => EventKind::Graduate,
                            "pruned" => EventKind::Prune,
                            _ => EventKind::System,
                        };
                        emit.push((kind, format!("shadow {} {status}", s.id.as_str())));
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
                        format!("boundary now gates · {}", b.behavior),
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

    /// Record an operator-initiated event in the stream (immediate feedback).
    pub fn operator_event(&mut self, text: String) {
        self.push_event(EventKind::System, text);
    }

    /// Stage the operator action for the focused selection (`x`), opening the
    /// confirm prompt. Prune the focused shadow, or delete the focused boundary.
    pub fn request_action(&mut self) {
        // The Loop page's operator action is a graceful halt of the run.
        if self.page == Page::Loop {
            if let Some(head) = self.loop_heads.first().filter(|_| !self.loop_halt_pending) {
                self.pending = Some(Pending::HaltLoop(head.run_id.clone()));
                self.mode = Mode::Confirm;
            }
            return;
        }
        // The remaining operator actions target the population regions.
        if self.page != Page::Population {
            return;
        }
        let pending = match self.focus {
            Focus::Shadows => self
                .selected_shadow()
                .filter(|s| s.status.as_str() != "pruned")
                .map(|s| Pending::PruneShadow(s.id.clone())),
            Focus::Boundaries => self
                .selected_boundary()
                .map(|b| Pending::DeleteBoundary(b.id.clone())),
            Focus::Experts => None,
        };
        if let Some(pending) = pending {
            self.pending = Some(pending);
            self.mode = Mode::Confirm;
        }
    }

    /// Stage graduating the selected shadow (the palette's graduate command):
    /// promote it into the population, unless it already graduated.
    pub fn request_graduate(&mut self) {
        if let Some(s) = self
            .selected_shadow()
            .filter(|s| s.status.as_str() != "graduated")
        {
            self.pending = Some(Pending::GraduateShadow(s.id.clone()));
            self.mode = Mode::Confirm;
        }
    }

    /// Stage freezing the selected expert, or thawing it if already frozen (the
    /// palette's freeze command toggles on the expert's current state).
    pub fn request_freeze(&mut self) {
        if let Some(e) = self.selected_expert() {
            let freeze = !e.is_frozen();
            self.pending = Some(Pending::FreezeExpert(e.id.clone(), freeze));
            self.mode = Mode::Confirm;
        }
    }

    /// The confirmation prompt for the staged action, if any.
    pub fn pending_prompt(&self) -> Option<String> {
        self.pending.as_ref().map(|p| match p {
            Pending::PruneShadow(id) => format!("Prune shadow {} ?", id.as_str()),
            Pending::GraduateShadow(id) => format!("Graduate shadow {} ?", id.as_str()),
            Pending::FreezeExpert(id, freeze) => {
                let verb = if *freeze { "Freeze" } else { "Thaw" };
                format!("{verb} expert {} ?", id.as_str())
            }
            Pending::HaltLoop(run) => {
                format!(
                    "Halt loop {} ? (stops at the next generation)",
                    run.as_str()
                )
            }
            Pending::DeleteBoundary(id) => {
                let behavior = self
                    .boundaries
                    .iter()
                    .find(|b| b.id.as_str() == id.as_str())
                    .map_or_else(|| id.as_str().to_string(), |b| b.behavior.clone());
                format!("Delete boundary \"{behavior}\" ?")
            }
        })
    }

    /// Cancel the staged action.
    pub fn cancel_action(&mut self) {
        self.pending = None;
        self.mode = Mode::Normal;
    }

    /// Carry out the staged operator mutation against the store, then reload so
    /// the change (and the event it raises) shows immediately. Returns to the
    /// live view whether or not anything was staged.
    pub async fn apply_pending(&mut self, store: &Store) -> anyhow::Result<()> {
        if let Some(pending) = self.pending.take() {
            match pending {
                Pending::PruneShadow(id) => {
                    if let Some(s) = self.shadows.iter().find(|s| s.id.as_str() == id.as_str()) {
                        let mut pruned = s.clone();
                        pruned.status = ShadowStatus::Pruned;
                        shadow::upsert(store, &pruned).await?;
                    }
                }
                Pending::GraduateShadow(id) => {
                    if let Some(s) = self.shadows.iter().find(|s| s.id.as_str() == id.as_str()) {
                        let mut graduated = s.clone();
                        graduated.status = ShadowStatus::Graduated;
                        shadow::upsert(store, &graduated).await?;
                    }
                }
                Pending::FreezeExpert(id, freeze) => {
                    let frozen_at = if freeze { Some(Utc::now()) } else { None };
                    expert::set_frozen(store, &id, frozen_at).await?;
                }
                Pending::DeleteBoundary(id) => {
                    let behavior = self
                        .boundaries
                        .iter()
                        .find(|b| b.id.as_str() == id.as_str())
                        .map(|b| b.behavior.clone());
                    boundary::delete(store, &id).await?;
                    if let Some(behavior) = behavior {
                        self.operator_event(format!("boundary removed · {behavior}"));
                    }
                }
                Pending::HaltLoop(run) => {
                    loop_control::set(store, &run, LoopCommand::Halt).await?;
                    self.operator_event(format!("halt requested · loop {}", run.as_str()));
                }
            }
            self.reload(store).await?;
        }
        self.mode = Mode::Normal;
        Ok(())
    }

    /// Cancel a pending loop halt (clear the control back to `Run`), then reload.
    pub async fn cancel_halt(&mut self, store: &Store) -> anyhow::Result<()> {
        if let Some(run) = self.loop_heads.first().map(|h| h.run_id.clone()) {
            loop_control::clear(store, &run).await?;
            self.operator_event(format!("halt cancelled · loop {}", run.as_str()));
            self.reload(store).await?;
        }
        Ok(())
    }

    /// Record an event at the head of the stream (newest first), capped.
    fn push_event(&mut self, kind: EventKind, text: String) {
        self.events.insert(
            0,
            Event {
                at_ms: self.clock_ms,
                kind,
                text,
            },
        );
        self.events.truncate(crate::events::MAX_EVENTS);
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
    }

    /// Append a character to the ask query (invalidates the stale result).
    pub fn ask_input(&mut self, c: char) {
        self.ask_query.push(c);
        self.ask_result = None;
    }

    /// Delete the last character of the ask query (invalidates the result).
    pub fn ask_backspace(&mut self) {
        self.ask_query.pop();
        self.ask_result = None;
    }

    /// Resolve a routing distribution (over expert ids) to `(name, probability)`,
    /// looking names up in the population, and store it as the ask result.
    pub fn set_ask_result(&mut self, routed: &[(antumbra_core::ExpertId, f32)]) {
        self.ask_result = Some(
            routed
                .iter()
                .map(|(id, p)| {
                    let name = self
                        .experts
                        .iter()
                        .find(|e| e.id.as_str() == id.as_str())
                        .map_or_else(|| id.as_str().to_string(), |e| e.name.clone());
                    (name, *p)
                })
                .collect(),
        );
    }

    /// The active palette every panel tints from.
    pub fn theme(&self) -> Theme {
        crate::theme::ALL[self.theme_idx % crate::theme::ALL.len()]
    }

    /// Advance to the next palette (wraps): shadow -> ember -> mono.
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

    /// Toggle the keybinding help overlay.
    pub fn toggle_help(&mut self) {
        self.mode = match self.mode {
            Mode::Help => Mode::Normal,
            _ => Mode::Help,
        };
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

    /// Focus a specific region (the palette's focus commands).
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// The label every item of the focused list filters/jumps by.
    fn focused_labels(&self) -> Vec<String> {
        match self.focus {
            Focus::Experts => self.experts.iter().map(|e| e.name.clone()).collect(),
            Focus::Shadows => self
                .shadows
                .iter()
                .map(|s| s.id.as_str().to_string())
                .collect(),
            Focus::Boundaries => self.boundaries.iter().map(|b| b.behavior.clone()).collect(),
        }
    }

    /// Open the focused-list filter (`/`) with an empty query.
    pub fn open_filter(&mut self) {
        self.mode = Mode::Filter;
        self.filter.clear();
        self.filter_selected = 0;
    }

    /// Append a typed character to the filter query (resets the selection).
    pub fn filter_input(&mut self, c: char) {
        self.filter.push(c);
        self.filter_selected = 0;
    }

    /// Delete the last character of the filter query (resets the selection).
    pub fn filter_backspace(&mut self) {
        self.filter.pop();
        self.filter_selected = 0;
    }

    /// Move the filter selection by `delta`, wrapping within the matches.
    pub fn filter_move(&mut self, delta: i32) {
        let n = self.filtered().len();
        if n == 0 {
            self.filter_selected = 0;
            return;
        }
        let cur = self.filter_selected.min(n - 1) as i32;
        self.filter_selected = (cur + delta).rem_euclid(n as i32) as usize;
    }

    /// The focused list's items matching the filter query, best first, as
    /// `(original index, label)` pairs.
    pub fn filtered(&self) -> Vec<(usize, String)> {
        let mut scored: Vec<(usize, String, i32)> = self
            .focused_labels()
            .into_iter()
            .enumerate()
            .filter_map(|(i, label)| {
                command::fuzzy_score(&label, &self.filter).map(|score| (i, label, score))
            })
            .collect();
        scored.sort_by_key(|&(_, _, score)| score);
        scored.into_iter().map(|(i, label, _)| (i, label)).collect()
    }

    /// Jump the focused selection to the highlighted filter match and close.
    pub fn filter_apply(&mut self) {
        let matches = self.filtered();
        let idx = self.filter_selected.min(matches.len().saturating_sub(1));
        if let Some(&(target, _)) = matches.get(idx) {
            self.set_focused_selection(target);
        }
        self.mode = Mode::Normal;
        self.filter.clear();
    }

    /// Cycle the body layout: focused -> dashboard -> graph.
    pub fn cycle_layout(&mut self) {
        let next = match self.layout {
            LayoutMode::Focused => LayoutMode::Dashboard,
            LayoutMode::Dashboard => LayoutMode::Graph,
            LayoutMode::Graph => LayoutMode::Table,
            LayoutMode::Table => LayoutMode::Focused,
        };
        self.set_layout(next);
    }

    /// Cycle the population table's sort column and re-order in place.
    pub fn cycle_sort(&mut self) {
        self.sort = match self.sort {
            SortKey::Fitness => SortKey::Name,
            SortKey::Name => SortKey::Generation,
            SortKey::Generation => SortKey::Fitness,
        };
        self.sort_experts();
    }

    /// Re-order the population by the active sort key, keeping the same expert
    /// selected (selection tracks the id, not the row index).
    pub fn sort_experts(&mut self) {
        let selected_id = self
            .experts
            .get(self.selected)
            .map(|e| e.id.as_str().to_string());
        match self.sort {
            SortKey::Fitness => self.experts.sort_by(|a, b| b.fitness.total_cmp(&a.fitness)),
            SortKey::Name => self.experts.sort_by(|a, b| a.name.cmp(&b.name)),
            SortKey::Generation => self
                .experts
                .sort_by_key(|e| std::cmp::Reverse(e.generation.0)),
        }
        if let Some(id) = selected_id {
            if let Some(pos) = self.experts.iter().position(|e| e.id.as_str() == id) {
                self.selected = pos;
            }
        }
    }

    /// Set the body layout (the palette's layout commands).
    pub fn set_layout(&mut self, layout: LayoutMode) {
        self.layout = layout;
        // The table is the expert data-grid; point navigation at the population.
        if layout == LayoutMode::Table {
            self.focus = Focus::Experts;
        }
    }

    /// Jump straight to a page (the palette / tab-number keys).
    pub fn set_page(&mut self, page: Page) {
        self.page = page;
    }

    /// Step the active page by `delta` tabs, wrapping (`[` / `]`).
    pub fn cycle_page(&mut self, delta: i32) {
        let n = Page::ALL.len() as i32;
        let i = (self.page.index() as i32 + delta).rem_euclid(n);
        self.page = Page::ALL[i as usize];
    }

    /// Jump to a page by tab index (the `1`-`4` keys); ignored if out of range.
    pub fn goto_page(&mut self, idx: usize) {
        if let Some(&page) = Page::ALL.get(idx) {
            self.page = page;
        }
    }

    /// Cycle which region the keys drive and detail: umbra -> penumbra -> antumbra.
    pub fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Experts => Focus::Shadows,
            Focus::Shadows => Focus::Boundaries,
            Focus::Boundaries => Focus::Experts,
        };
    }

    pub fn tick(&mut self, dt_ms: f64) {
        self.clock_ms += dt_ms;
        self.since_reload_ms += dt_ms;
        self.since_input_ms += dt_ms;
    }

    /// Note a keypress: restore full-rate rendering on the next frame.
    pub fn note_input(&mut self) {
        self.since_input_ms = 0.0;
    }

    /// The rate to pace this frame at: the full target, unless `--power-save` is
    /// on and the view has been still (no input, no animation), in which case it
    /// eases to [`crate::pacing::IDLE_FPS`].
    pub fn frame_cap(&self, animating: bool) -> u32 {
        let still = !animating && self.since_input_ms >= crate::pacing::IDLE_AFTER_MS;
        if self.power_save && still {
            self.target_fps.min(crate::pacing::IDLE_FPS)
        } else {
            self.target_fps
        }
    }

    /// Reload from the store roughly every two seconds so the console stays live
    /// while the population is trained from another process.
    pub fn wants_reload(&self) -> bool {
        self.since_reload_ms >= 2000.0
    }

    /// Advance the highlighted item in the focused list (wraps). Routes through
    /// [`Self::focused_list`] so every page's selection follows one dispatch.
    pub fn select_next(&mut self) {
        let (len, sel) = self.focused_list();
        if len > 0 {
            self.set_focused_selection((sel + 1) % len);
        }
    }

    pub fn select_prev(&mut self) {
        let (len, sel) = self.focused_list();
        if len > 0 {
            self.set_focused_selection((sel + len - 1) % len);
        }
    }

    /// The `(len, selected)` of the list the navigation keys drive: per non-
    /// Population page, else the focused population region. The single source of
    /// truth all the selection movers (next/prev/first/last/page) dispatch on.
    fn focused_list(&self) -> (usize, usize) {
        match self.page {
            Page::Memory => return (self.memories.len(), self.selected_memory),
            Page::Evals => return (self.evals.len(), self.selected_eval),
            Page::Loop => return (self.loop_heads.len(), self.selected_loop),
            Page::Population => {}
        }
        match self.focus {
            Focus::Experts => (self.experts.len(), self.selected),
            Focus::Shadows => (self.shadows.len(), self.selected_shadow),
            Focus::Boundaries => (self.boundaries.len(), self.selected_boundary),
        }
    }

    /// Set the highlighted index of the focused list (the write counterpart to
    /// [`Self::focused_list`]).
    fn set_focused_selection(&mut self, idx: usize) {
        match self.page {
            Page::Memory => {
                self.selected_memory = idx;
                return;
            }
            Page::Evals => {
                self.selected_eval = idx;
                return;
            }
            Page::Loop => {
                self.selected_loop = idx;
                return;
            }
            Page::Population => {}
        }
        match self.focus {
            Focus::Experts => self.selected = idx,
            Focus::Shadows => self.selected_shadow = idx,
            Focus::Boundaries => self.selected_boundary = idx,
        }
    }

    /// Jump to the first item of the focused list (Home / `g`).
    pub fn select_first(&mut self) {
        if self.focused_list().0 > 0 {
            self.set_focused_selection(0);
        }
    }

    /// Jump to the last item of the focused list (End / `G`).
    pub fn select_last(&mut self) {
        let (n, _) = self.focused_list();
        if n > 0 {
            self.set_focused_selection(n - 1);
        }
    }

    /// Move the selection by a page (PageUp/PageDown), clamped to the ends.
    pub fn select_page(&mut self, delta: i32) {
        const PAGE: i32 = 10;
        let (n, cur) = self.focused_list();
        if n == 0 {
            return;
        }
        let next = (cur as i32 + delta * PAGE).clamp(0, n as i32 - 1);
        self.set_focused_selection(next as usize);
    }

    /// The Canvas marker for vector drawings, degraded to match the render tier
    /// (Braille normally, coarse Dot under `--render=ascii`).
    pub fn canvas_marker(&self) -> ratatui::symbols::Marker {
        self.render_tier.marker()
    }

    pub fn selected_expert(&self) -> Option<&Expert> {
        self.experts.get(self.selected)
    }

    pub fn selected_boundary(&self) -> Option<&FailureBoundary> {
        self.boundaries.get(self.selected_boundary)
    }

    pub fn selected_shadow(&self) -> Option<&Shadow> {
        self.shadows.get(self.selected_shadow)
    }

    pub fn selected_memory(&self) -> Option<&Memory> {
        self.memories.get(self.selected_memory)
    }

    /// The KPI strip: five `[0,1]` quality ratios across the substrate, namely mean
    /// population fitness, the frozen ratio, the shadow graduation rate, the
    /// share of boundaries actively gating, and mean memory strength. Always
    /// rendered above the body, so every page carries the same health readout.
    pub fn dashboard_metrics(&self) -> [(&'static str, f32); 5] {
        let mean = |vals: &[f32]| {
            if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f32>() / vals.len() as f32
            }
        };
        let ratio = |num: usize, den: usize| {
            if den == 0 {
                0.0
            } else {
                num as f32 / den as f32
            }
        };
        let fitness = mean(&self.experts.iter().map(|e| e.fitness).collect::<Vec<_>>());
        let frozen = ratio(
            self.experts.iter().filter(|e| e.is_frozen()).count(),
            self.experts.len(),
        );
        let graduated = ratio(
            self.shadows
                .iter()
                .filter(|s| s.status.as_str() == "graduated")
                .count(),
            self.shadows.len(),
        );
        let gating = ratio(
            self.boundaries.iter().filter(|b| b.is_actionable()).count(),
            self.boundaries.len(),
        );
        let memory = mean(
            &self
                .memories
                .iter()
                .map(|m| m.confidence)
                .collect::<Vec<_>>(),
        );
        [
            ("fitness", fitness),
            ("frozen", frozen),
            ("graduated", graduated),
            ("gating", gating),
            ("memory", memory),
        ]
    }

    /// Edges touching `id`, as `(edge, is_outgoing)`: the selected trace's links.
    pub fn memory_edges(&self, id: &str) -> Vec<(&MemoryEdge, bool)> {
        self.edges
            .iter()
            .filter_map(|e| {
                if e.from_id.as_str() == id {
                    Some((e, true))
                } else if e.to_id.as_str() == id {
                    Some((e, false))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::generational::LoopState;
    use antumbra_core::{EdgeType, EvalStatus, Generation, MemoryNetwork, RunId, SubjectKind};
    use antumbra_store::EMBED_DIM;
    use chrono::Utc;

    /// An in-memory store with one exploring shadow and one boundary, loaded into
    /// a fresh `App` (so reads go through the real `App::load`/`reload` path).
    async fn seeded() -> (Store, App) {
        let store = Store::connect_memory(EMBED_DIM).await.unwrap();
        shadow::upsert(
            &store,
            &Shadow {
                id: ShadowId::new("shadow:s1"),
                parent_expert: None,
                adapter_uri: None,
                status: ShadowStatus::Exploring,
                generation: Generation(1),
                reward_curve: vec![],
                created_at: Utc::now(),
            },
        )
        .await
        .unwrap();
        boundary::upsert(
            &store,
            &FailureBoundary {
                id: BoundaryId::new("boundary:b1"),
                behavior: "do the thing".into(),
                fail_context: serde_json::json!({}),
                near_ok_context: None,
                governing_features: Vec::new(),
                grain: None,
                context_vec: None,
                ok_context_vec: None,
                confidence: 0.5,
                generation: Generation::ZERO,
                created_at: Utc::now(),
            },
        )
        .await
        .unwrap();
        let app = App::load(&store).await.unwrap();
        (store, app)
    }

    #[tokio::test]
    async fn prune_writes_to_the_store_and_raises_an_event() {
        let (store, mut app) = seeded().await;
        app.focus = Focus::Shadows;
        app.request_action();
        assert!(matches!(app.pending, Some(Pending::PruneShadow(_))));
        app.apply_pending(&store).await.unwrap();

        let s = app
            .shadows
            .iter()
            .find(|s| s.id.as_str() == "shadow:s1")
            .unwrap();
        assert_eq!(s.status.as_str(), "pruned", "the store reflects the prune");
        assert!(
            app.events.iter().any(|e| e.text.contains("pruned")),
            "the prune surfaced as an event"
        );
        assert_eq!(app.mode, Mode::Normal);
        assert!(app.pending.is_none());
    }

    #[tokio::test]
    async fn delete_removes_the_boundary_and_raises_an_event() {
        let (store, mut app) = seeded().await;
        app.focus = Focus::Boundaries;
        app.request_action();
        app.apply_pending(&store).await.unwrap();

        assert!(app.boundaries.is_empty(), "the boundary is gone");
        assert!(
            app.events
                .iter()
                .any(|e| e.text.contains("boundary removed")),
            "the deletion surfaced as an event"
        );
    }

    #[tokio::test]
    async fn graduate_writes_to_the_store_and_raises_an_event() {
        let (store, mut app) = seeded().await;
        app.focus = Focus::Shadows;
        app.request_graduate();
        assert!(matches!(app.pending, Some(Pending::GraduateShadow(_))));
        app.apply_pending(&store).await.unwrap();

        let s = app
            .shadows
            .iter()
            .find(|s| s.id.as_str() == "shadow:s1")
            .unwrap();
        assert_eq!(
            s.status.as_str(),
            "graduated",
            "the store reflects the promotion"
        );
        assert!(
            app.events.iter().any(|e| e.text.contains("graduated")),
            "the graduation surfaced as an event"
        );
        assert_eq!(app.mode, Mode::Normal);
    }

    #[tokio::test]
    async fn request_graduate_skips_an_already_graduated_shadow() {
        let (_store, mut app) = seeded().await;
        app.shadows[0].status = ShadowStatus::Graduated;
        app.focus = Focus::Shadows;
        app.request_graduate();
        assert!(app.pending.is_none());
    }

    #[tokio::test]
    async fn freeze_toggles_the_expert_and_raises_events() {
        let (store, mut app) = seeded().await;
        let mut v = vec![0.0f32; EMBED_DIM];
        v[0] = 1.0;
        expert::insert(
            &store,
            &Expert {
                id: ExpertId::new("expert:e1"),
                name: "scout".into(),
                base_model: "base".into(),
                artifact_uri: "a.safetensors".into(),
                capability_card: serde_json::json!({}),
                capability_vec: Some(v),
                fitness: 1.0,
                frozen_at: None,
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                created_at: Utc::now(),
            },
        )
        .await
        .unwrap();
        app.reload(&store).await.unwrap();
        app.focus = Focus::Experts;
        app.selected = app
            .experts
            .iter()
            .position(|e| e.id.as_str() == "expert:e1")
            .unwrap();

        // First toggle freezes.
        app.request_freeze();
        assert!(matches!(app.pending, Some(Pending::FreezeExpert(_, true))));
        app.apply_pending(&store).await.unwrap();
        let frozen = app
            .experts
            .iter()
            .find(|e| e.id.as_str() == "expert:e1")
            .unwrap();
        assert!(frozen.is_frozen(), "the store reflects the freeze");
        assert!(app.events.iter().any(|ev| ev.text.contains("frozen")));

        // Second toggle thaws (the command reads the current state).
        app.request_freeze();
        assert!(matches!(app.pending, Some(Pending::FreezeExpert(_, false))));
        app.apply_pending(&store).await.unwrap();
        let thawed = app
            .experts
            .iter()
            .find(|e| e.id.as_str() == "expert:e1")
            .unwrap();
        assert!(!thawed.is_frozen(), "the store reflects the thaw");
        assert!(app.events.iter().any(|ev| ev.text.contains("thawed")));
    }

    fn an_expert(id: &str, name: &str, fitness: f32, gen: u32) -> Expert {
        Expert {
            id: ExpertId::new(id),
            name: name.into(),
            base_model: "base".into(),
            artifact_uri: "a".into(),
            capability_card: serde_json::json!({}),
            capability_vec: None,
            fitness,
            frozen_at: None,
            generation: Generation(gen),
            owner: None,
            compartment: None,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn sort_experts_orders_and_preserves_selection() {
        let (_store, mut app) = seeded().await;
        app.experts = vec![
            an_expert("expert:b", "beta", 0.5, 2),
            an_expert("expert:a", "alpha", 0.9, 1),
            an_expert("expert:c", "gamma", 0.7, 3),
        ];
        app.selected = 0; // beta

        app.sort = SortKey::Fitness;
        app.sort_experts();
        let names: Vec<&str> = app.experts.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["alpha", "gamma", "beta"],
            "fitness, strongest first"
        );
        assert_eq!(
            app.selected_expert().unwrap().name,
            "beta",
            "selection tracks the expert across the re-sort"
        );

        app.cycle_sort(); // Fitness → Name
        assert_eq!(app.sort, SortKey::Name);
        let names: Vec<&str> = app.experts.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["alpha", "beta", "gamma"]);

        app.cycle_sort(); // Name → Generation
        assert_eq!(
            app.experts.first().unwrap().name,
            "gamma",
            "newest gen first"
        );
    }

    #[tokio::test]
    async fn loop_and_evals_load_for_their_pages() {
        let (store, mut app) = seeded().await;
        assert!(app.loop_heads.is_empty());
        assert!(app.evals.is_empty());
        let now = Utc::now();

        let mut head = GenerationHead::new(RunId::new("run:x"), now);
        head.state = LoopState::Decide;
        generation::save_head(&store, &head).await.unwrap();
        evaluation::insert(
            &store,
            &EvaluationRun {
                run_id: RunId::new("run:e"),
                subject_kind: SubjectKind::Expert,
                subject_id: "expert:a".into(),
                corpus_task_id: "task:a".into(),
                status: EvalStatus::Failure,
                metrics: None,
                regression_fingerprint: None,
                created_at: now,
            },
        )
        .await
        .unwrap();
        app.reload(&store).await.unwrap();

        assert_eq!(app.loop_heads.len(), 1);
        assert_eq!(app.loop_heads[0].state, LoopState::Decide);
        assert_eq!(app.evals.len(), 1);
        assert_eq!(app.evals[0].status, EvalStatus::Failure);

        // The Evals page drives the run selection (and is inert for mutations).
        app.page = Page::Evals;
        app.select_next();
        assert_eq!(app.selected_eval, 0, "single run, selection holds");
        app.request_action();
        assert!(app.pending.is_none());
    }

    #[tokio::test]
    async fn loop_page_selects_runs_and_drills_into_eval_history() {
        let (store, mut app) = seeded().await;
        let now = Utc::now();
        let mut a = GenerationHead::new(RunId::new("run:a"), now);
        a.state = LoopState::Score;
        let b = GenerationHead::new(RunId::new("run:b"), now);
        generation::save_head(&store, &a).await.unwrap();
        generation::save_head(&store, &b).await.unwrap();
        // One evaluation, tied to run:b only.
        evaluation::insert(
            &store,
            &EvaluationRun {
                run_id: RunId::new("run:b"),
                subject_kind: SubjectKind::Shadow,
                subject_id: "shadow:b0".into(),
                corpus_task_id: "task:b".into(),
                status: EvalStatus::Success,
                metrics: None,
                regression_fingerprint: None,
                created_at: now,
            },
        )
        .await
        .unwrap();
        app.reload(&store).await.unwrap();
        assert_eq!(app.loop_heads.len(), 2);

        // j/k drive selected_loop and wrap (the Loop page now owns a selection).
        app.page = Page::Loop;
        assert_eq!(app.selected_loop, 0);
        app.select_next();
        assert_eq!(app.selected_loop, 1);
        app.select_next();
        assert_eq!(app.selected_loop, 0, "wraps");
        app.select_prev();
        assert_eq!(app.selected_loop, 1, "wraps back");

        // Enter drills in — the Loop page now has a detail overlay.
        app.open_detail();
        assert_eq!(app.mode, Mode::Detail);

        // The drill-down's eval history is scoped to one run (the Loop→Evals
        // link), independent of how the store ordered the heads.
        let with_eval = app
            .loop_heads
            .iter()
            .find(|h| h.run_id.as_str() == "run:b")
            .unwrap()
            .clone();
        let without = app
            .loop_heads
            .iter()
            .find(|h| h.run_id.as_str() == "run:a")
            .unwrap()
            .clone();
        let hist = app.loop_eval_history(&with_eval);
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].subject_id, "shadow:b0");
        assert!(app.loop_eval_history(&without).is_empty());

        // Reload clamps an out-of-range selection back into the heads.
        app.selected_loop = 9;
        app.reload(&store).await.unwrap();
        assert!(app.selected_loop < app.loop_heads.len());

        // With no heads, Enter is a no-op (no empty drill-down).
        app.loop_heads.clear();
        app.mode = Mode::Normal;
        app.open_detail();
        assert_eq!(app.mode, Mode::Normal);
    }

    #[tokio::test]
    async fn loop_halt_request_writes_the_control_and_cancel_clears_it() {
        let (store, mut app) = seeded().await;
        let head = GenerationHead::new(RunId::new("run:z"), Utc::now());
        generation::save_head(&store, &head).await.unwrap();
        app.reload(&store).await.unwrap();
        assert_eq!(app.loop_heads.len(), 1);
        assert!(!app.loop_halt_pending);

        // `x` on the Loop page stages a halt confirm; applying writes the control.
        app.page = Page::Loop;
        app.request_action();
        assert!(matches!(app.pending, Some(Pending::HaltLoop(_))));
        app.apply_pending(&store).await.unwrap();
        assert!(app.loop_halt_pending, "halt is now pending");
        assert_eq!(
            loop_control::load(&store, &RunId::new("run:z"))
                .await
                .unwrap(),
            LoopCommand::Halt
        );

        // Re-requesting is inert while a halt is already pending.
        app.request_action();
        assert!(app.pending.is_none());

        // Cancelling clears the control back to Run.
        app.cancel_halt(&store).await.unwrap();
        assert!(!app.loop_halt_pending);
        assert_eq!(
            loop_control::load(&store, &RunId::new("run:z"))
                .await
                .unwrap(),
            LoopCommand::Run
        );
    }

    #[tokio::test]
    async fn eval_history_groups_by_subject_and_drill_down_opens() {
        let (_store, mut app) = seeded().await;
        let now = Utc::now();
        let mk = |id: &str, subj: &str, status| EvaluationRun {
            run_id: RunId::new(id),
            subject_kind: SubjectKind::Expert,
            subject_id: subj.into(),
            corpus_task_id: "task:a".into(),
            status,
            metrics: None,
            regression_fingerprint: None,
            created_at: now,
        };
        app.evals = vec![
            mk("run:x2", "expert:x", EvalStatus::Failure),
            mk("run:x1", "expert:x", EvalStatus::Success),
            mk("run:y", "expert:y", EvalStatus::Success),
        ];
        let first = app.evals[0].clone();
        assert_eq!(app.eval_history(&first).len(), 2, "two runs for expert:x");
        assert_eq!(app.eval_history(&app.evals[2].clone()).len(), 1);

        // Enter drills into a run on the Evals page (inert when there are none).
        app.page = Page::Evals;
        app.selected_eval = 0;
        app.open_detail();
        assert_eq!(app.mode, Mode::Detail);
        app.mode = Mode::Normal;
        app.evals.clear();
        app.open_detail();
        assert_eq!(app.mode, Mode::Normal, "no drill-down with no runs");
    }

    #[tokio::test]
    async fn dashboard_metrics_summarize_the_substrate() {
        let (_store, mut app) = seeded().await;
        let m = app.dashboard_metrics();
        assert_eq!(
            m.map(|(label, _)| label),
            ["fitness", "frozen", "graduated", "gating", "memory"]
        );
        // No experts seeded → zero mean fitness; the one shadow is exploring.
        assert_eq!(m[0].1, 0.0, "no experts");
        assert_eq!(m[2].1, 0.0, "no graduated shadows");
        // Graduate the lone shadow → the graduation ratio becomes 1/1.
        app.shadows[0].status = ShadowStatus::Graduated;
        assert_eq!(app.dashboard_metrics()[2].1, 1.0);
    }

    #[tokio::test]
    async fn memories_and_edges_load_and_drive_the_memory_page() {
        let (store, mut app) = seeded().await;
        assert!(app.memories.is_empty(), "no memory traces in the base seed");
        let now = Utc::now();
        memory::upsert(
            &store,
            &Memory::new("memory:a", "ws:1", MemoryNetwork::World, "alpha", 0.5, now),
        )
        .await
        .unwrap();
        memory::upsert(
            &store,
            &Memory::new("memory:b", "ws:1", MemoryNetwork::Bank, "beta", 0.9, now),
        )
        .await
        .unwrap();
        edge::relate(
            &store,
            &MemoryEdge::new(
                "ws:1",
                "memory:a",
                "memory:b",
                EdgeType::References,
                0.5,
                now,
            ),
        )
        .await
        .unwrap();
        app.reload(&store).await.unwrap();

        assert_eq!(app.memories.len(), 2);
        assert_eq!(app.edges.len(), 1);

        // On the Memory page, navigation drives the trace selection and wraps.
        app.page = Page::Memory;
        app.selected_memory = 0;
        let id = app.selected_memory().unwrap().id.as_str().to_string();
        assert_eq!(app.memory_edges(&id).len(), 1, "the selected trace's edge");
        app.select_next();
        assert_eq!(app.selected_memory, 1);
        app.select_next();
        assert_eq!(app.selected_memory, 0, "selection wraps");

        // Operator actions and the drill-down overlay are inert on the Memory page.
        app.request_action();
        assert!(app.pending.is_none());
        app.open_detail();
        assert_eq!(app.mode, Mode::Normal);
    }

    #[tokio::test]
    async fn apply_pending_is_a_noop_with_nothing_staged() {
        let (store, mut app) = seeded().await;
        app.apply_pending(&store).await.unwrap();
        assert_eq!(app.shadows.len(), 1);
        assert_eq!(app.boundaries.len(), 1);
        assert_eq!(app.mode, Mode::Normal);
    }

    #[tokio::test]
    async fn request_action_skips_a_pruned_shadow_and_experts() {
        let (_store, mut app) = seeded().await;
        // Pruned shadow: no action is staged.
        app.shadows[0].status = ShadowStatus::Pruned;
        app.focus = Focus::Shadows;
        app.request_action();
        assert!(
            app.pending.is_none(),
            "no prune on an already-pruned shadow"
        );
        assert_eq!(app.mode, Mode::Normal);
        // Experts focus has no operator action.
        app.focus = Focus::Experts;
        app.request_action();
        assert!(app.pending.is_none());
    }
}

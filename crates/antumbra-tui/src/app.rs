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
    /// A coding agent with its feature flags off: the rules, and which skills
    /// are used (ADR-0021). Read-only.
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

    /// The lowercase tab label.
    pub fn title(self) -> &'static str {
        match self {
            Page::Population => "population",
            Page::Memory => "memory",
            Page::Loop => "loop",
            Page::Evals => "evals",
            Page::Sovereign => "sovereign",
        }
    }

    /// The tab a number key selects: `1` is the first page. Asked of
    /// [`Page::ALL`], so a page added to it cannot be left without its key.
    pub fn index_for_key(key: char) -> Option<usize> {
        let number = key.to_digit(10)? as usize;
        (1..=Page::ALL.len()).contains(&number).then(|| number - 1)
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
    /// The Sovereign page's reading of `memories`, made once per reload.
    pub sovereign: crate::sovereign::View,
    /// Index into `sovereign.skills` of the highlighted row (Sovereign page).
    pub selected_skill: usize,
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
            sovereign: crate::sovereign::View::default(),
            selected_skill: 0,
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
        self.sovereign = crate::sovereign::View::from_memories(&self.memories);
        self.selected_skill = self
            .selected_skill
            .min(self.sovereign.skills.len().saturating_sub(1));
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

}

#[cfg(test)]
mod tests;
mod view;

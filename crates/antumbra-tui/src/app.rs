//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, loaded from the store and ticked for animation.

use antumbra_core::{Expert, FailureBoundary, LearnedRouter, Shadow};
use antumbra_store::repo::{boundary, expert, router, shadow};
use antumbra_store::Store;

use std::collections::HashMap;

use crate::command::{self, Action, Command};
use crate::events::{Event, EventKind};
use crate::theme::Theme;

/// Which list the navigation keys drive, and which detail panel is shown — one
/// per region of the cast shadow: the population (umbra), the shadows in training
/// (penumbra), or the boundaries (antumbra, the keystone).
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
    Ask,
}

/// How the body arranges its panels: the graph beside a single focused detail
/// (`Focused`), the graph beside all three regions at once (`Dashboard`), or the
/// graph alone, full width (`Graph`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayoutMode {
    Focused,
    Dashboard,
    Graph,
}

impl LayoutMode {
    /// The lowercase name shown in the header.
    pub fn name(self) -> &'static str {
        match self {
            LayoutMode::Focused => "focused",
            LayoutMode::Dashboard => "dashboard",
            LayoutMode::Graph => "graph",
        }
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
    pub router: Option<LearnedRouter>,
    /// Index into `experts` of the highlighted node.
    pub selected: usize,
    /// Index into `boundaries` of the highlighted scope (when focused there).
    pub selected_boundary: usize,
    /// Index into `shadows` of the highlighted shadow (when focused there).
    pub selected_shadow: usize,
    /// Which list navigation/detail targets (umbra / penumbra / antumbra).
    pub focus: Focus,
    /// How the body arranges its panels (focused / dashboard / graph).
    pub layout: LayoutMode,
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
    /// The route-ask query text (used while `mode == Ask`).
    pub ask_query: String,
    /// The gate's routing for the last ask: `(expert name, probability)` best
    /// first, or `Some(empty)` when it escalates. `None` before the first ask.
    pub ask_result: Option<Vec<(String, f32)>>,
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
    pub should_quit: bool,
}

impl App {
    pub async fn load(store: &Store) -> anyhow::Result<Self> {
        let mut app = Self {
            experts: Vec::new(),
            boundaries: Vec::new(),
            shadows: Vec::new(),
            router: None,
            selected: 0,
            selected_boundary: 0,
            selected_shadow: 0,
            focus: Focus::Experts,
            layout: LayoutMode::Focused,
            mode: Mode::Normal,
            palette: Palette::default(),
            filter: String::new(),
            filter_selected: 0,
            events: Vec::new(),
            events_scroll: 0,
            detail_scroll: 0,
            ask_query: String::new(),
            ask_result: None,
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
            should_quit: false,
        };
        app.reload(store).await?;
        Ok(app)
    }

    pub async fn reload(&mut self, store: &Store) -> anyhow::Result<()> {
        self.experts = expert::list(store).await?;
        self.boundaries = boundary::list(store).await?;
        self.shadows = shadow::list(store).await?;
        // Newest first, so the penumbra view leads with current training.
        self.shadows
            .sort_by_key(|s| std::cmp::Reverse(s.created_at));
        self.router = router::load(store).await?;
        if !self.experts.is_empty() && self.selected >= self.experts.len() {
            self.selected = self.experts.len() - 1;
        }
        if !self.boundaries.is_empty() && self.selected_boundary >= self.boundaries.len() {
            self.selected_boundary = self.boundaries.len() - 1;
        }
        if !self.shadows.is_empty() && self.selected_shadow >= self.shadows.len() {
            self.selected_shadow = self.shadows.len() - 1;
        }
        self.record_events();
        self.since_reload_ms = 0.0;
        Ok(())
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
    pub fn open_detail(&mut self) {
        self.mode = Mode::Detail;
        self.detail_scroll = 0;
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
    /// fresh frame — e.g. a headless snapshot — shows a stable number).
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
        self.layout = match self.layout {
            LayoutMode::Focused => LayoutMode::Dashboard,
            LayoutMode::Dashboard => LayoutMode::Graph,
            LayoutMode::Graph => LayoutMode::Focused,
        };
    }

    /// Set the body layout (the palette's layout commands).
    pub fn set_layout(&mut self, layout: LayoutMode) {
        self.layout = layout;
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

    /// Advance the highlighted item in the focused list (wraps).
    pub fn select_next(&mut self) {
        match self.focus {
            Focus::Experts => {
                if !self.experts.is_empty() {
                    self.selected = (self.selected + 1) % self.experts.len();
                }
            }
            Focus::Shadows => {
                if !self.shadows.is_empty() {
                    self.selected_shadow = (self.selected_shadow + 1) % self.shadows.len();
                }
            }
            Focus::Boundaries => {
                if !self.boundaries.is_empty() {
                    self.selected_boundary = (self.selected_boundary + 1) % self.boundaries.len();
                }
            }
        }
    }

    pub fn select_prev(&mut self) {
        match self.focus {
            Focus::Experts => {
                if !self.experts.is_empty() {
                    self.selected = (self.selected + self.experts.len() - 1) % self.experts.len();
                }
            }
            Focus::Shadows => {
                if !self.shadows.is_empty() {
                    self.selected_shadow =
                        (self.selected_shadow + self.shadows.len() - 1) % self.shadows.len();
                }
            }
            Focus::Boundaries => {
                if !self.boundaries.is_empty() {
                    self.selected_boundary = (self.selected_boundary + self.boundaries.len() - 1)
                        % self.boundaries.len();
                }
            }
        }
    }

    /// The `(len, selected)` of the list the navigation keys drive.
    fn focused_list(&self) -> (usize, usize) {
        match self.focus {
            Focus::Experts => (self.experts.len(), self.selected),
            Focus::Shadows => (self.shadows.len(), self.selected_shadow),
            Focus::Boundaries => (self.boundaries.len(), self.selected_boundary),
        }
    }

    /// Set the highlighted index of the focused list.
    fn set_focused_selection(&mut self, idx: usize) {
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

    pub fn selected_expert(&self) -> Option<&Expert> {
        self.experts.get(self.selected)
    }

    pub fn selected_boundary(&self) -> Option<&FailureBoundary> {
        self.boundaries.get(self.selected_boundary)
    }

    pub fn selected_shadow(&self) -> Option<&Shadow> {
        self.shadows.get(self.selected_shadow)
    }
}

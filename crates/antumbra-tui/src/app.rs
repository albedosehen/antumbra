//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, loaded from the store and ticked for animation.

use antumbra_core::{Expert, FailureBoundary, LearnedRouter};
use antumbra_store::repo::{boundary, expert, router};
use antumbra_store::Store;

/// Which list the navigation keys drive, and which detail panel is shown: the
/// population (umbra) or the boundaries (antumbra, the keystone).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Experts,
    Boundaries,
}

/// Everything the console draws, refreshed from the store.
pub struct App {
    pub experts: Vec<Expert>,
    pub boundaries: Vec<FailureBoundary>,
    pub router: Option<LearnedRouter>,
    /// Index into `experts` of the highlighted node.
    pub selected: usize,
    /// Index into `boundaries` of the highlighted scope (when focused there).
    pub selected_boundary: usize,
    /// Whether navigation/detail targets the population or the boundaries.
    pub focus: Focus,
    /// Total elapsed animation time (ms), drives orbit/pulse/energy.
    pub clock_ms: f64,
    /// Seconds since the last reload, so the view refreshes periodically.
    pub since_reload_ms: f64,
    pub should_quit: bool,
}

impl App {
    pub async fn load(store: &Store) -> anyhow::Result<Self> {
        let mut app = Self {
            experts: Vec::new(),
            boundaries: Vec::new(),
            router: None,
            selected: 0,
            selected_boundary: 0,
            focus: Focus::Experts,
            clock_ms: 0.0,
            since_reload_ms: 0.0,
            should_quit: false,
        };
        app.reload(store).await?;
        Ok(app)
    }

    pub async fn reload(&mut self, store: &Store) -> anyhow::Result<()> {
        self.experts = expert::list(store).await?;
        self.boundaries = boundary::list(store).await?;
        self.router = router::load(store).await?;
        if !self.experts.is_empty() && self.selected >= self.experts.len() {
            self.selected = self.experts.len() - 1;
        }
        if !self.boundaries.is_empty() && self.selected_boundary >= self.boundaries.len() {
            self.selected_boundary = self.boundaries.len() - 1;
        }
        self.since_reload_ms = 0.0;
        Ok(())
    }

    /// Switch which list (population / boundaries) the keys drive and detail.
    pub fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Experts => Focus::Boundaries,
            Focus::Boundaries => Focus::Experts,
        };
    }

    pub fn tick(&mut self, dt_ms: f64) {
        self.clock_ms += dt_ms;
        self.since_reload_ms += dt_ms;
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
            Focus::Boundaries => {
                if !self.boundaries.is_empty() {
                    self.selected_boundary = (self.selected_boundary + self.boundaries.len() - 1)
                        % self.boundaries.len();
                }
            }
        }
    }

    pub fn selected_expert(&self) -> Option<&Expert> {
        self.experts.get(self.selected)
    }

    pub fn selected_boundary(&self) -> Option<&FailureBoundary> {
        self.boundaries.get(self.selected_boundary)
    }
}

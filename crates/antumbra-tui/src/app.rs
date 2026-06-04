//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, loaded from the store and ticked for animation.

use antumbra_core::{Expert, FailureBoundary, LearnedRouter};
use antumbra_store::repo::{boundary, expert, router};
use antumbra_store::Store;

/// Everything the console draws, refreshed from the store.
pub struct App {
    pub experts: Vec<Expert>,
    pub boundaries: Vec<FailureBoundary>,
    pub router: Option<LearnedRouter>,
    /// Index into `experts` of the highlighted node.
    pub selected: usize,
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
        self.since_reload_ms = 0.0;
        Ok(())
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

    pub fn select_next(&mut self) {
        if !self.experts.is_empty() {
            self.selected = (self.selected + 1) % self.experts.len();
        }
    }

    pub fn select_prev(&mut self) {
        if !self.experts.is_empty() {
            self.selected = (self.selected + self.experts.len() - 1) % self.experts.len();
        }
    }

    pub fn selected_expert(&self) -> Option<&Expert> {
        self.experts.get(self.selected)
    }
}

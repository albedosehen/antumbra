//! Operator-console state: the live population (umbra), boundaries (antumbra),
//! and the learned gate, loaded from the store and ticked for animation.

use antumbra_core::{Expert, FailureBoundary, LearnedRouter, Shadow};
use antumbra_store::repo::{boundary, expert, router, shadow};
use antumbra_store::Store;

use crate::theme::Theme;

/// Which list the navigation keys drive, and which detail panel is shown — one
/// per region of the cast shadow: the population (umbra), the shadows in training
/// (penumbra), or the boundaries (antumbra, the keystone).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Experts,
    Shadows,
    Boundaries,
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
    /// Index into [`crate::theme::ALL`] of the active palette.
    pub theme_idx: usize,
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
            shadows: Vec::new(),
            router: None,
            selected: 0,
            selected_boundary: 0,
            selected_shadow: 0,
            focus: Focus::Experts,
            theme_idx: 0,
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
        self.since_reload_ms = 0.0;
        Ok(())
    }

    /// The active palette every panel tints from.
    pub fn theme(&self) -> Theme {
        crate::theme::ALL[self.theme_idx % crate::theme::ALL.len()]
    }

    /// Advance to the next palette (wraps): shadow -> ember -> mono.
    pub fn cycle_theme(&mut self) {
        self.theme_idx = (self.theme_idx + 1) % crate::theme::ALL.len();
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

//! What the operator is looking at, and what follows from it: which region has
//! focus, the filter, the sort, the layout, the page, the selection within it,
//! and the few values derived from those.
//!
//! A child module of `app`, so these are the same inherent methods on the same
//! `App`; they moved here only because `app.rs` had grown past the size rule.

use super::*;

impl App {
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
            Page::Sovereign => return (self.sovereign.skills.len(), self.selected_skill),
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
            Page::Sovereign => {
                self.selected_skill = idx;
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

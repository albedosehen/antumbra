//! What the operator is looking at, and what follows from it: which region has
//! focus, the filter, the sort, the layout, the page, the selection within it,
//! and the few values derived from those.
//!
//! A child module of `app`, so these are the same inherent methods on the same
//! `App`; they moved here only because `app.rs` had grown past the size rule.

use std::borrow::Cow;

use super::*;

impl App {
    /// Focus a specific region (the palette's focus commands).
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// What the filter searches, in a word: the list the keys drive on this
    /// page.
    pub fn filter_subject(&self) -> &'static str {
        match self.page {
            Page::Memory => "memories",
            Page::Evals => "evaluations",
            Page::Sovereign => "skills",
            Page::Loop => "training runs",
            Page::Services => "services",
            Page::Population => match self.focus {
                Focus::Experts => "experts",
                Focus::Shadows => "candidates",
                Focus::Boundaries => "boundaries",
            },
        }
    }

    /// The label every item of the focused list filters/jumps by: the list
    /// the navigation keys drive on this page, in the same order, so a match's
    /// index is that item's place in it.
    fn focused_labels(&self) -> Vec<Cow<'_, str>> {
        // Borrowed wherever the label is a field: a store holds thousands of
        // memories, and a label is read to be scored, not kept.
        match self.page {
            Page::Memory => {
                return self
                    .listed_memories()
                    .into_iter()
                    .map(|i| Cow::Borrowed(self.memories[i].content.as_str()))
                    .collect()
            }
            Page::Evals => {
                return self
                    .evals
                    .iter()
                    .map(|e| Cow::Owned(format!("{} · {}", e.subject_id, e.corpus_task_id)))
                    .collect()
            }
            Page::Sovereign => {
                return self
                    .sovereign
                    .skills
                    .iter()
                    .map(|s| Cow::Borrowed(s.skill.as_str()))
                    .collect()
            }
            Page::Loop => {
                return self
                    .loop_heads
                    .iter()
                    .map(|h| Cow::Borrowed(h.run_id.as_str()))
                    .collect()
            }
            Page::Services => {
                return self
                    .services
                    .names
                    .iter()
                    .map(|name| Cow::Borrowed(name.as_str()))
                    .collect()
            }
            Page::Population => {}
        }
        match self.focus {
            Focus::Experts => self
                .experts
                .iter()
                .map(|e| Cow::Owned(crate::words::expert_name(e)))
                .collect(),
            Focus::Shadows => self
                .shadows
                .iter()
                .map(|s| Cow::Borrowed(s.id.as_str()))
                .collect(),
            Focus::Boundaries => self
                .boundaries
                .iter()
                .map(|b| Cow::Borrowed(b.behavior.as_str()))
                .collect(),
        }
    }

    /// Open the focused-list filter (`/`) with an empty query.
    pub fn open_filter(&mut self) {
        self.mode = Mode::Filter;
        self.filter.clear();
        self.refilter();
    }

    /// Append a typed character to the filter query (resets the selection).
    pub fn filter_input(&mut self, c: char) {
        self.filter.push(c);
        self.refilter();
    }

    /// Delete the last character of the filter query (resets the selection).
    pub fn filter_backspace(&mut self) {
        self.filter.pop();
        self.refilter();
    }

    /// The most matches the filter keeps. The overlay shows a dozen, and a
    /// longer query narrows the rest.
    pub const FILTER_KEPT: usize = 200;

    /// Work out the filter's matches for the query as it now reads, and put
    /// the highlight on the best. Done when the query or the list changes,
    /// never while drawing: scoring every memory of a large store is too much
    /// to do on each frame.
    pub(super) fn refilter(&mut self) {
        let needle = command::lowered(&self.filter);
        let labels = self.focused_labels();
        let mut scored: Vec<(usize, i32)> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, label)| command::score_chars(label, &needle).map(|score| (i, score)))
            .collect();
        // Stable, so equal matches keep the list's order.
        scored.sort_by_key(|&(_, score)| score);
        scored.truncate(Self::FILTER_KEPT);
        let hits = scored
            .into_iter()
            .map(|(i, _)| {
                // One line, of a length a row can show.
                let flat = labels[i].split_whitespace().collect::<Vec<_>>().join(" ");
                (i, antumbra_core::text::clipped(&flat, 96).into_owned())
            })
            .collect();
        self.filter_hits = hits;
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
    /// `(the item's place in its list, its label)` pairs: at most
    /// [`Self::FILTER_KEPT`] of them, as last worked out ([`Self::refilter`]).
    pub fn filtered(&self) -> &[(usize, String)] {
        &self.filter_hits
    }

    /// Jump the focused selection to the highlighted filter match and close.
    pub fn filter_apply(&mut self) {
        let idx = self
            .filter_selected
            .min(self.filter_hits.len().saturating_sub(1));
        if let Some(target) = self.filter_hits.get(idx).map(|(place, _)| *place) {
            self.set_focused_selection(target);
        }
        self.mode = Mode::Normal;
        self.filter.clear();
        self.filter_hits.clear();
    }

    /// Cycle the body layout: focused -> dashboard -> graph.
    pub fn cycle_layout(&mut self) {
        let next = match self.layout {
            LayoutMode::List => LayoutMode::Dashboard,
            LayoutMode::Dashboard => LayoutMode::Graph,
            LayoutMode::Graph => LayoutMode::Table,
            LayoutMode::Table => LayoutMode::List,
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
            // By the name shown, so the list reads in the order it is in.
            SortKey::Name => self.experts.sort_by_cached_key(crate::words::expert_name),
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
        // The table and the graph draw the experts alone, so the keys drive
        // the experts there: a list that is not drawn is not navigated.
        if matches!(layout, LayoutMode::Table | LayoutMode::Graph) {
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

    /// Read the view again every few seconds so the console stays live while
    /// the population is trained from another process.
    pub fn wants_reload(&self) -> bool {
        self.since_reload_ms >= self.reload_every_ms
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
            // The list is the kind in front; the selection's place in it is
            // where it sits among that kind, or the top when it is of another.
            Page::Memory => {
                let listed = self.listed_memories();
                let at = listed
                    .iter()
                    .position(|&i| i == self.selected_memory)
                    .unwrap_or(0);
                return (listed.len(), at);
            }
            Page::Evals => return (self.evals.len(), self.selected_eval),
            Page::Sovereign => return (self.sovereign.skills.len(), self.selected_skill),
            Page::Loop => return (self.loop_heads.len(), self.selected_loop),
            Page::Services => return (self.services.names.len(), self.selected_service),
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
                if let Some(&memory) = self.listed_memories().get(idx) {
                    self.selected_memory = memory;
                }
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
            Page::Services => {
                self.selected_service = idx;
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

    /// The memories the Memory page lists, as indices into `memories` in the
    /// order they are shown: those of the kind in front, or every one, kind
    /// after kind in the order the title names the kinds. The server sends
    /// each kind strongest first, and that order is kept within it.
    pub fn listed_memories(&self) -> Vec<usize> {
        // The graph is of links, so its keys move through the memories that
        // have one.
        if self.memory_graph {
            return self.linked_memories();
        }
        let of = |kind: MemoryNetwork| {
            self.memories
                .iter()
                .enumerate()
                .filter(move |(_, m)| m.network == kind)
                .map(|(i, _)| i)
        };
        match self.memory_kind {
            Some(kind) => of(kind).collect(),
            None => Self::MEMORY_KINDS
                .into_iter()
                .flatten()
                .flat_map(of)
                .collect(),
        }
    }

    /// How many memories there are of `kind`, or in all.
    pub fn memory_count(&self, kind: Option<MemoryNetwork>) -> usize {
        match kind {
            None => self.memories.len(),
            Some(kind) => self.memories.iter().filter(|m| m.network == kind).count(),
        }
    }

    /// The kinds the Memory page lists, in the order Tab moves through them:
    /// all, then each network.
    pub const MEMORY_KINDS: [Option<MemoryNetwork>; 4] = [
        None,
        Some(MemoryNetwork::World),
        Some(MemoryNetwork::Bank),
        Some(MemoryNetwork::Opinion),
    ];

    /// List the next kind of memory (Tab on the Memory page), wrapping. The
    /// selection stays where it is when it is of that kind, and moves to the
    /// strongest of that kind when it is not, so what is selected is always
    /// in the list.
    pub fn next_memory_kind(&mut self) {
        let at = Self::MEMORY_KINDS
            .iter()
            .position(|kind| *kind == self.memory_kind)
            .unwrap_or(0);
        self.memory_kind = Self::MEMORY_KINDS[(at + 1) % Self::MEMORY_KINDS.len()];
        let listed = self.listed_memories();
        if !listed.contains(&self.selected_memory) {
            if let Some(&first) = listed.first() {
                self.selected_memory = first;
            }
        }
    }

    /// The memories that are linked to another, as indices into `memories`:
    /// the most linked first, then in the order the server sent them. A store
    /// holds thousands of memories and a few links, so these are the ones a
    /// picture of links has anything to say about.
    pub fn linked_memories(&self) -> Vec<usize> {
        let mut links: HashMap<&str, usize> = HashMap::new();
        for e in &self.edges {
            *links.entry(e.from_id.as_str()).or_default() += 1;
            *links.entry(e.to_id.as_str()).or_default() += 1;
        }
        let mut linked: Vec<(usize, usize)> = self
            .memories
            .iter()
            .enumerate()
            .filter_map(|(i, m)| links.get(m.id.as_str()).map(|count| (i, *count)))
            .collect();
        linked.sort_by_key(|(i, count)| (std::cmp::Reverse(*count), *i));
        linked.into_iter().map(|(i, _)| i).collect()
    }

    /// Swap the Memory page's list for the selected memory's links, drawn,
    /// or back (`l` on the Memory page). The graph is of every kind, and its
    /// keys move through the memories that have a link; when the one selected
    /// has none, the selection goes to the most linked, so the graph opens on
    /// a picture and not on a sentence saying there is none.
    pub fn toggle_memory_graph(&mut self) {
        self.memory_graph = !self.memory_graph;
        if self.memory_graph {
            self.memory_kind = None;
            let linked = self.linked_memories();
            if !linked.contains(&self.selected_memory) {
                if let Some(&most) = linked.first() {
                    self.selected_memory = most;
                }
            }
        }
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

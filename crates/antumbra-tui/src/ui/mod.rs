//! Rendering: the top bar, the tabs and the page under them, then the notice
//! and the footer. The population page's graph (the experts orbiting a core
//! of three linked nodes) is hand-rolled from the animation clock so the
//! motion is fully under control. Every color tints from the active [`Theme`],
//! so cycling a theme (`t`) recolors the whole console; every word is from
//! [`crate::words`] or says what its panel holds in the same plain terms.
//!
//! Split across submodules: [`graph`] (the canvas), [`chrome`] (header/footer),
//! [`panels`] (the region detail panels), and [`overlays`] (help / palette /
//! filter modals). The small shared widget helpers live here.

mod chrome;
mod detail;
mod evals;
mod graph;
mod heatmap;
mod knobs;
mod lists;
mod loops;
mod memory;
mod netdraw;
mod overlays;
mod panels;
mod services;
mod sovereign;
mod table;

use std::f64::consts::TAU;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType};
use ratatui::Frame;

use antumbra_core::behavior;
use antumbra_core::{FailureBoundary, Shadow};

use crate::app::{App, Focus, LayoutMode, Mode, Page};
use crate::overlay;
use crate::theme::{Theme, Tone};
use crate::words;

pub use overlays::overlay_area;

/// The frame's rows, top to bottom: the top bar, the tabs, the sentence that
/// says what the page is, the page's body, then the two rows that are never
/// covered or dimmed, what the console last said and the keys of the view in
/// front.
pub struct FrameRows {
    pub header: Rect,
    pub tabs: Rect,
    pub caption: Rect,
    pub body: Rect,
    pub notice: Rect,
    pub footer: Rect,
}

impl FrameRows {
    /// Everything above the notice and the footer: where an overlay is drawn,
    /// and what is dimmed behind it.
    pub fn stage(&self) -> Rect {
        Rect {
            x: self.header.x,
            y: self.header.y,
            width: self.header.width,
            height: self.notice.y.saturating_sub(self.header.y),
        }
    }
}

/// Split `area` into the frame's rows. The one split the renderer, the
/// overlays and the transitions all read, so none can disagree about where
/// the body is.
pub fn frame_rows(area: Rect) -> FrameRows {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);
    FrameRows {
        header: rows[0],
        tabs: rows[1],
        caption: rows[2],
        body: rows[3],
        notice: rows[4],
        footer: rows[5],
    }
}

pub fn render(f: &mut Frame, app: &App) {
    // Guard the render loop against a degenerately-small terminal: the fixed
    // chrome layout would leave no space for the body and some widgets subtract
    // past zero. Rendering nothing is safe; a panic here would crash the console.
    if f.area().width < 8 || f.area().height < 3 {
        return;
    }
    // The page first: the theme's ground behind every cell, and its ink for
    // whatever a widget draws without a colour of its own. A theme that leaves
    // the ground to the terminal paints nothing here.
    let t = app.theme();
    f.render_widget(Block::default().style(t.page()), f.area());
    let rows = frame_rows(f.area());
    chrome::header(f, app, rows.header);
    chrome::tabs(f, app, rows.tabs);
    chrome::caption(f, app, rows.caption);
    page_body(f, app, rows.body);
    // A fresh, empty store would otherwise render as inert panels; the first-run
    // hint explains how to populate it. Only in the live view (not over a modal).
    if app.mode == Mode::Normal && app.is_empty() {
        overlays::first_run_hint(f, app);
    }
    if app.mode != Mode::Normal {
        overlay::dim_backdrop(f, &t, rows.stage());
        match app.mode {
            Mode::Help => overlays::help_overlay(f, app),
            Mode::Palette => overlays::palette_overlay(f, app),
            Mode::Filter => overlays::filter_overlay(f, app),
            Mode::Events => overlays::events_overlay(f, app),
            Mode::Detail => overlays::detail_overlay(f, app),
            Mode::Gate => overlays::gate_overlay(f, app),
            Mode::Connect => overlays::connect_overlay(f, app),
            Mode::Ask => overlays::ask_overlay(f, app),
            Mode::Confirm => overlays::confirm_overlay(f, app),
            Mode::Knobs => knobs::knobs_overlay(f, app),
            Mode::Normal => {}
        }
    }
    // Last, and never dimmed: what the console last did or could not do, and
    // the keys of whatever is in front, the page or the overlay over it.
    chrome::notice(f, app, rows.notice);
    chrome::footer(f, app, rows.footer);
}

/// Route the body to the active page.
fn page_body(f: &mut Frame, app: &App, area: Rect) {
    match app.page {
        Page::Population => body(f, app, area),
        Page::Memory => memory::page(f, app, area),
        Page::Services => services::page(f, app, area),
        Page::Loop => loops::page(f, app, area),
        Page::Evals => evals::page(f, app, area),
        Page::Sovereign => sovereign::page(f, app, area),
    }
}

/// A page that is a list beside the detail of what is selected: the list's
/// column and the detail's. The one split the pages and the transitions read.
pub fn list_columns(body: Rect) -> [Rect; 2] {
    let cols =
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).split(body);
    [cols[0], cols[1]]
}

/// The population page's body, arranged per the active layout: the list in
/// front beside the detail of what is selected, the graph beside all three
/// lists, the graph alone, or the experts as a full-width table.
fn body(f: &mut Frame, app: &App, area: Rect) {
    match app.layout {
        LayoutMode::Table => table::experts_table(f, app, area),
        LayoutMode::Graph => graph::graph(f, app, area),
        LayoutMode::List => {
            let [list, detail] = list_columns(area);
            lists::population(f, app, list);
            match app.focus {
                Focus::Experts => panels::detail(f, app, detail),
                Focus::Shadows => panels::candidate_side(f, app, detail),
                Focus::Boundaries => panels::boundary_side(f, app, detail),
            }
        }
        LayoutMode::Dashboard => {
            // Graph + the three regions up top, a full-width reward heatmap below.
            let stack = Layout::vertical([Constraint::Min(0), Constraint::Length(9)]).split(area);
            let cols = Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(stack[0]);
            graph::graph(f, app, cols[0]);
            panels::dashboard(f, app, cols[1]);
            heatmap::reward_heatmap(f, app, stack[1]);
        }
    }
}

/// `[0,1]` pulse from the animation clock.
fn pulse(clock_ms: f64, period_ms: f64, phase: f64) -> f64 {
    0.5 + 0.5 * ((clock_ms / period_ms * TAU) + phase).sin()
}

/// The visible window `[offset, offset + rows)` of a `len`-item list that keeps
/// `selected` on screen, so a `Table` builds only the rows it actually shows even
/// when the data is store-sized. Returns `(offset, in-window index of selected)`.
pub(super) fn visible_window(len: usize, selected: usize, rows: usize) -> (usize, usize) {
    if rows == 0 || len <= rows {
        return (0, selected.min(len.saturating_sub(1)));
    }
    let sel = selected.min(len - 1);
    let offset = sel.saturating_sub(rows / 2).min(len - rows);
    (offset, sel - offset)
}

/// A panel's title: its name, in the secondary ink, set off from the border.
fn title<'a>(t: &Theme, name: &str) -> Line<'a> {
    Line::from(Span::styled(
        format!(" {name} "),
        Style::default().fg(t.ink2),
    ))
}

/// What an expert's state is, to the eye and in a word: frozen is settled,
/// sealed for good; one that is not is still open to training.
fn expert_state(e: &antumbra_core::Expert) -> (Tone, &'static str) {
    if e.is_frozen() {
        (Tone::Settled, "frozen")
    } else {
        (Tone::Open, "open to training")
    }
}

/// What an empty list says. A list the server sends only to an operator is
/// empty for any other token whatever the server holds, so there it says an
/// operator's token is needed ([`words::OPERATOR_ONLY`]); else `why` it is
/// empty and what fills it.
fn nothing_here<'a>(app: &App, t: &Theme, why: &'a str) -> Vec<Line<'a>> {
    if app.operator || app.demo {
        vec![Line::from(Span::styled(why, Style::default().fg(t.ink3)))]
    } else {
        vec![Line::from(Span::styled(
            words::OPERATOR_ONLY,
            Style::default().fg(t.ink2),
        ))]
    }
}

/// A rounded panel block, its border in the line colour and its title in the
/// secondary ink.
fn panel<'a>(t: &Theme, title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.line))
        .title(title.into())
}

/// A panel whose border and title take the accent when it holds focus, so the
/// one panel the keys drive is the one that is lit.
fn panel_focused<'a>(t: &Theme, title: &str, focused: bool) -> Block<'a> {
    let (border, name, marker) = if focused {
        (t.accent, t.accent, "◆ ")
    } else {
        (t.line, t.ink2, "")
    };
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Line::from(format!(" {marker}{title} ")).style(Style::default().fg(name)))
}

/// A panel's heading, the selected item's name: bold, at full strength. The
/// accent is kept for the selection and for what is in force.
fn heading<'a>(t: &Theme, text: String) -> Line<'a> {
    Line::from(Span::styled(
        text,
        Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
    ))
}

/// A `label  value` row: the label in the secondary ink, the value at full
/// strength.
fn kv<'a>(t: &Theme, k: &str, v: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{k:<11}"), Style::default().fg(t.ink2)),
        Span::styled(v.to_string(), Style::default().fg(t.ink)),
    ])
}

/// A `label  ● word` row for a state: its shape and its word, in the state's
/// colour.
fn state_row<'a>(t: &Theme, k: &str, tone: Tone, word: &str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{k:<11}"), Style::default().fg(t.ink2)),
        Span::styled(
            format!("{} {word}", tone.glyph()),
            Style::default().fg(t.tone(tone)),
        ),
    ])
}

/// A horizontal gauge for a `[0,1]` value: `width` cells filled to eighth-of-a-
/// cell resolution in `color`, the rest of its track in the line colour.
fn gauge_spans<'a>(t: &Theme, value: f32, width: u16, color: Color) -> Vec<Span<'a>> {
    const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let v = (value.clamp(0.0, 1.0) as f64) * width as f64;
    let mut full = v.floor() as usize;
    let mut rem = ((v - full as f64) * 8.0).round() as usize;
    if rem == 8 {
        full += 1;
        rem = 0;
    }
    let mut filled = "█".repeat(full);
    let cells = full + usize::from(rem > 0);
    if rem > 0 {
        filled.push_str(EIGHTHS[rem]);
    }
    let track = "─".repeat((width as usize).saturating_sub(cells));
    vec![
        Span::styled(filled, Style::default().fg(color)),
        Span::styled(track, Style::default().fg(t.line)),
    ]
}

/// A `key  ▆▆▆▍──  0.62` row: a labeled gauge with its numeric value.
fn gauge_row<'a>(t: &Theme, k: &'a str, value: f32, color: Color) -> Line<'a> {
    let mut spans = vec![Span::styled(
        format!("{k:<11}"),
        Style::default().fg(t.ink2),
    )];
    spans.extend(gauge_spans(t, value, 12, color));
    spans.push(Span::styled(
        format!(" {value:.2}"),
        Style::default().fg(t.ink),
    ));
    Line::from(spans)
}

/// A reward curve as block bars, each bar colored along the fitness gradient by
/// its own value, so a rising (or collapsing) trajectory reads at a glance.
fn sparkline_row<'a>(t: &Theme, k: &'a str, curve: &[f32]) -> Line<'a> {
    const BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let mut spans = vec![Span::styled(
        format!("{k:<11}"),
        Style::default().fg(t.ink2),
    )];
    for &v in curve {
        let level = (v.clamp(0.0, 1.0) * 7.0).round() as usize;
        spans.push(Span::styled(
            BARS[level.min(7)].to_string(),
            Style::default().fg(t.fitness(v, 1.0)),
        ));
    }
    Line::from(spans)
}

/// What a shadow's status is to the eye: graduated is settled, pruned is
/// ended, and any other is still under way.
fn shadow_tone(status: &str) -> Tone {
    match status {
        "graduated" => Tone::Settled,
        "pruned" => Tone::Ended,
        _ => Tone::Open,
    }
}

/// A candidate's state as a list says it: its status, or `replaced` for a
/// standing expert's run that a later run took the place of. Every one of an
/// expert's runs graduated, so the status alone reads the same on each, and
/// only the newest is the adapter the expert answers with.
fn candidate_state(app: &App, s: &Shadow) -> (Tone, &'static str) {
    if app.replaced_by(s).is_some() {
        (Tone::Ended, "replaced")
    } else {
        (shadow_tone(s.status.as_str()), s.status.as_str())
    }
}

/// How long ago `at` was, as the coarsest unit that fits: `3d ago`, `5h ago`,
/// `now`.
fn since(at: chrono::DateTime<chrono::Utc>) -> String {
    elapsed(at).map_or_else(|| "now".into(), |age| format!("{age} ago"))
}

/// The same without `ago`, for a column whose heading says it is an age:
/// `3d`, `5h`, `now`.
fn age(at: chrono::DateTime<chrono::Utc>) -> String {
    elapsed(at).unwrap_or_else(|| "now".into())
}

/// The time since `at` as the coarsest unit that fits, `3d`, `5h` or `7m`;
/// `None` for under a minute.
fn elapsed(at: chrono::DateTime<chrono::Utc>) -> Option<String> {
    let d = chrono::Utc::now() - at;
    let (days, hours, minutes) = (d.num_days(), d.num_hours(), d.num_minutes());
    if days > 0 {
        Some(format!("{days}d"))
    } else if hours > 0 {
        Some(format!("{hours}h"))
    } else if minutes > 0 {
        Some(format!("{minutes}m"))
    } else {
        None
    }
}

/// A boundary in force (actionable: it gates routing) is settled; one that is
/// only recorded is open.
fn boundary_tone(b: &FailureBoundary) -> Tone {
    if b.is_actionable() {
        Tone::Settled
    } else {
        Tone::Open
    }
}

/// Clip `s` to `max` columns, ending in an ellipsis when cut.
fn clip(s: &str, max: usize) -> String {
    antumbra_core::text::clipped(s, max).into_owned()
}

/// A candidate's name: a standing expert's training by the scope it trained,
/// any other (the training loop's) by its id and generation. A standing
/// expert's trainings are all generation zero, and their ids run to some
/// sixty characters, the scope near the end where a list cuts it off.
fn shadow_name(s: &Shadow) -> String {
    match behavior::shadow_scope(&s.id) {
        Some(scope) => format!("{scope} · behaviors"),
        None => format!("{} · g{}", s.id.as_str(), s.generation.0),
    }
}

/// A reward curve from its first reading to its last, `0.42 → 0.91`, or the
/// one reading a curve of one has. A standing expert's training reads the
/// base model's held-out pass rate and then the trained expert's.
fn reward_span(curve: &[f32]) -> Option<String> {
    match curve {
        [] => None,
        [only] => Some(format!("{only:.2}")),
        [first, .., last] => Some(format!("{first:.2} → {last:.2}")),
    }
}

/// A scope as a list shows it: a repository's, such as
/// `github.com/acme/orders`, by the repository's name, the last part of its
/// path; any other, such as `everywhere`, as it is. A repository's path fills
/// a list's row before the row says anything else; the panel below the list
/// gives it whole.
fn short_scope(scope: &str) -> &str {
    scope
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or(scope)
}

/// A list row for something with a state: its shape in the state's colour,
/// then its text at a strength that follows the state. A list reads by shape
/// first, and a long one is not painted by the states in it.
fn state_line<'a>(t: &Theme, tone: Tone, text: String) -> Line<'a> {
    let strength = match tone {
        Tone::Settled | Tone::Failed => t.ink,
        Tone::Open => t.ink2,
        Tone::Ended => t.ink3,
    };
    Line::from(vec![
        Span::styled(
            format!("{} ", tone.glyph()),
            Style::default().fg(t.tone(tone)),
        ),
        Span::styled(text, Style::default().fg(strength)),
    ])
}

/// A shadow's list row: its state's shape (graduated ●, replaced or pruned ×,
/// under way ○), what it trained, when, and where its reward went. The shape
/// carries the state, which the training panel spells out; the list is
/// narrow, and the trained reading at the row's end is what it is read for. A
/// standing expert's training is named by its scope, shortened, and told from
/// the expert's other trainings by when it was; the loop's by its id and
/// generation.
fn shadow_row<'a>(t: &Theme, app: &App, s: &Shadow) -> Line<'a> {
    let mut row = match behavior::shadow_scope(&s.id) {
        Some(scope) => format!("{} · {}", short_scope(scope), since(s.created_at)),
        None => shadow_name(s),
    };
    if let Some(r) = reward_span(&s.reward_curve) {
        row.push_str(" · ");
        row.push_str(&r);
    }
    state_line(t, candidate_state(app, s).0, row)
}

/// A boundary's list row: in force ● or open ○, the behavior, then the
/// scope its context names, shortened, when it names one (a training's
/// does). The behavior leads: a standing expert misses several behaviors in
/// one scope, and with the scope first a narrow list shows the same words on
/// each of their rows.
fn boundary_row<'a>(t: &Theme, b: &FailureBoundary) -> Line<'a> {
    let row = match b.fail_context.get("scope").and_then(|s| s.as_str()) {
        Some(scope) => format!("{} · {}", b.behavior, short_scope(scope)),
        None => b.behavior.clone(),
    };
    state_line(t, boundary_tone(b), row)
}

/// A boundary context's fields as `(key, value)` rows, a string value as it
/// is and any other as JSON: the scope, the task and the answer first, as a
/// training records them, then the rest in order. A context that is not an
/// object is one row.
fn context_fields(context: &serde_json::Value) -> Vec<(String, String)> {
    const FIRST: [&str; 3] = ["scope", "task", "answer"];
    let Some(map) = context.as_object() else {
        return match context {
            serde_json::Value::Null => Vec::new(),
            other => vec![("context".into(), other.to_string())],
        };
    };
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort_by_key(|k| FIRST.iter().position(|f| f == k).unwrap_or(FIRST.len()));
    keys.into_iter()
        .map(|k| {
            let v = &map[k];
            (
                k.clone(),
                v.as_str().map_or_else(|| v.to_string(), str::to_string),
            )
        })
        .collect()
}

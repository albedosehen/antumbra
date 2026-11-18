//! The lists a selection moves in: the population page's three (experts,
//! candidates, boundaries), each a table, under a title that names all three
//! and marks the one in front; and the same title for any list that is one of
//! several ([`segments`]).
//!
//! The dashboard shows the same experts as a table, and the columns here are
//! its columns. A narrow terminal drops the least useful columns before it
//! cuts a name.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::Frame;

use antumbra_core::behavior;

use crate::app::{App, Focus};
use crate::theme::{Theme, Tone};

use super::{
    boundary_tone, gauge_spans, nothing_here, reward_span, shadow_name, shadow_tone, short_scope,
    visible_window,
};

/// A panel whose title is several lists side by side, each with its count,
/// the one in front marked as the dashboard marks a chosen segment. It shows
/// that there is more than one list here, which one is in front, and, with
/// the footer's Tab, how to get to the next.
pub(super) fn segments<'a>(t: &Theme, lists: &[(&str, usize)], front: usize) -> Block<'a> {
    let mut spans = vec![Span::raw(" ")];
    for (i, (name, count)) in lists.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("│", Style::default().fg(t.line)));
        }
        let style = if i == front {
            t.selected().fg(t.ink)
        } else {
            Style::default().fg(t.ink2)
        };
        spans.push(Span::styled(format!(" {name} {count} "), style));
    }
    spans.push(Span::raw(" "));
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.accent))
        .title(Line::from(spans))
}

/// A table's heading row: its column names in capitals, in the secondary ink,
/// as the dashboard sets a table's.
pub(super) fn heading<'a>(t: &Theme, names: &[&'a str]) -> Row<'a> {
    Row::new(names.iter().map(|name| Cell::from(*name)))
        .style(Style::default().fg(t.ink2).add_modifier(Modifier::BOLD))
}

/// A table's contents: its heading row, the rows in view, and each column's
/// width.
pub(super) struct Grid<'a> {
    pub header: Row<'a>,
    pub rows: Vec<Row<'a>>,
    pub widths: &'a [Constraint],
}

/// Draw `grid` as a table in `block`, the selected row marked. Only the rows
/// that fit are built by the caller ([`window`]).
pub(super) fn table<'a>(
    f: &mut Frame,
    t: &Theme,
    area: Rect,
    block: Block<'a>,
    grid: Grid<'a>,
    selected: Option<usize>,
) {
    let table = Table::new(grid.rows, grid.widths)
        .header(grid.header)
        .block(block)
        .row_highlight_style(t.selected())
        .highlight_symbol("▸ ");
    let mut state = TableState::default();
    state.select(selected);
    f.render_stateful_widget(table, area, &mut state);
}

/// The rows a table of `len` items shows in `area`, as `(first, last, the
/// selected row's place among them)`: the window that keeps the selection in
/// view.
pub(super) fn window(area: Rect, len: usize, selected: usize) -> (usize, usize, usize) {
    // The borders and the heading row take three.
    let visible = usize::from(area.height).saturating_sub(3);
    let (offset, at) = visible_window(len, selected, visible);
    (offset, (offset + visible).min(len), at)
}

/// The population page's list in front, under a title naming all three.
pub(super) fn population(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let lists = [
        ("Experts", app.experts.len()),
        ("Candidates", app.shadows.len()),
        ("Boundaries", app.boundaries.len()),
    ];
    let front = match app.focus {
        Focus::Experts => 0,
        Focus::Shadows => 1,
        Focus::Boundaries => 2,
    };
    let block = segments(&t, &lists, front);
    match app.focus {
        Focus::Experts => experts(f, app, &t, area, block),
        Focus::Shadows => candidates(f, app, &t, area, block),
        Focus::Boundaries => boundaries(f, app, &t, area, block),
    }
}

/// An empty list: why it is empty and what fills it, or that the token may
/// not read it.
fn empty(f: &mut Frame, app: &App, t: &Theme, area: Rect, block: Block, why: &str) {
    f.render_widget(
        Paragraph::new(nothing_here(app, t, why))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The experts, in the order they are sorted: name, generation, fitness as a
/// meter, state, and whether each is shared or one user's.
fn experts(f: &mut Frame, app: &App, t: &Theme, area: Rect, block: Block) {
    if app.experts.is_empty() {
        // Every token reads its workspace's experts, so none listed is none.
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "No experts yet: `antumbra train` or `antumbra teach` trains one from verified work.",
                Style::default().fg(t.ink3),
            )))
            .block(block)
            .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    // The generation and the visibility are the first to go when it is narrow.
    let wide = area.width >= 64;
    let (first, last, at) = window(area, app.experts.len(), app.selected);
    let rows: Vec<Row> = app.experts[first..last]
        .iter()
        .map(|e| {
            let mut fit = gauge_spans(t, e.fitness, 8, t.accent);
            fit.push(Span::styled(
                format!(" {:.2}", e.fitness),
                Style::default().fg(t.ink),
            ));
            let (tone, state) = if e.is_frozen() {
                (Tone::Settled, "frozen")
            } else {
                (Tone::Open, "open")
            };
            let mut cells = vec![Cell::from(Span::styled(
                e.name.clone(),
                Style::default().fg(t.ink),
            ))];
            if wide {
                cells.push(Cell::from(Span::styled(
                    e.generation.0.to_string(),
                    Style::default().fg(t.ink2),
                )));
            }
            cells.push(Cell::from(Line::from(fit)));
            cells.push(Cell::from(Span::styled(
                format!("{} {state}", tone.glyph()),
                Style::default().fg(t.tone(tone)),
            )));
            if wide {
                cells.push(Cell::from(Span::styled(
                    if e.is_private() { "private" } else { "shared" },
                    Style::default().fg(t.ink2),
                )));
            }
            Row::new(cells)
        })
        .collect();
    let (names, widths): (&[&str], &[Constraint]) = if wide {
        (
            &["EXPERT", "GEN", "FITNESS", "STATE", "VISIBLE TO"],
            &[
                Constraint::Min(14),
                Constraint::Length(3),
                Constraint::Length(13),
                Constraint::Length(8),
                Constraint::Length(10),
            ],
        )
    } else {
        (
            &["EXPERT", "FITNESS", "STATE"],
            &[
                Constraint::Min(12),
                Constraint::Length(13),
                Constraint::Length(8),
            ],
        )
    };
    let grid = Grid {
        header: heading(t, names),
        rows,
        widths,
    };
    table(f, t, area, block, grid, Some(at));
}

/// The candidates, newest first: what each trained, where its reward went
/// from and to, and its status.
fn candidates(f: &mut Frame, app: &App, t: &Theme, area: Rect, block: Block) {
    if app.shadows.is_empty() {
        let why = "No candidates yet: one appears while an expert is being trained.";
        empty(f, app, t, area, block, why);
        return;
    }
    let (first, last, at) = window(area, app.shadows.len(), app.selected_shadow);
    let rows: Vec<Row> = app.shadows[first..last]
        .iter()
        .map(|s| {
            let tone = shadow_tone(s.status.as_str());
            // A standing expert's training reads by the scope it trained,
            // shortened; the loop's by its id and generation.
            let name = match behavior::shadow_scope(&s.id) {
                Some(scope) => short_scope(scope).to_string(),
                None => shadow_name(s),
            };
            // A candidate that ended reads fainter than one that counts.
            let ink = if tone == Tone::Ended { t.ink3 } else { t.ink };
            Row::new(vec![
                Cell::from(Span::styled(name, Style::default().fg(ink))),
                Cell::from(Span::styled(
                    reward_span(&s.reward_curve).unwrap_or_else(|| "none yet".into()),
                    Style::default().fg(t.ink2),
                )),
                Cell::from(Span::styled(
                    format!("{} {}", tone.glyph(), s.status.as_str()),
                    Style::default().fg(t.tone(tone)),
                )),
            ])
        })
        .collect();
    let widths = [
        Constraint::Min(14),
        Constraint::Length(11),
        Constraint::Length(11),
    ];
    let grid = Grid {
        header: heading(t, &["CANDIDATE", "REWARD", "STATUS"]),
        rows,
        widths: &widths,
    };
    table(f, t, area, block, grid, Some(at));
}

/// The boundaries: the behavior, where it applies when its context says, and
/// whether it is in force.
fn boundaries(f: &mut Frame, app: &App, t: &Theme, area: Rect, block: Block) {
    if app.boundaries.is_empty() {
        let why =
            "No boundaries yet: one is recorded when an expert is found to be wrong somewhere.";
        empty(f, app, t, area, block, why);
        return;
    }
    let wide = area.width >= 64;
    let (first, last, at) = window(area, app.boundaries.len(), app.selected_boundary);
    let rows: Vec<Row> = app.boundaries[first..last]
        .iter()
        .map(|b| {
            let tone = boundary_tone(b);
            let state = if b.is_actionable() {
                "in force"
            } else {
                "recorded"
            };
            let ink = if tone == Tone::Settled { t.ink } else { t.ink2 };
            let mut cells = vec![Cell::from(Span::styled(
                b.behavior.clone(),
                Style::default().fg(ink),
            ))];
            if wide {
                // What its context names: the scope a training gives, or the
                // feature that decides it.
                let place = b
                    .fail_context
                    .get("scope")
                    .and_then(|s| s.as_str())
                    .map(|scope| short_scope(scope).to_string())
                    .or_else(|| b.governing_features.first().cloned())
                    .unwrap_or_default();
                cells.push(Cell::from(Span::styled(place, Style::default().fg(t.ink2))));
            }
            cells.push(Cell::from(Span::styled(
                format!("{} {state}", tone.glyph()),
                Style::default().fg(t.tone(tone)),
            )));
            Row::new(cells)
        })
        .collect();
    let (names, widths): (&[&str], &[Constraint]) = if wide {
        (
            &["BOUNDARY", "WHERE", "STATUS"],
            &[
                Constraint::Min(16),
                Constraint::Length(14),
                Constraint::Length(10),
            ],
        )
    } else {
        (
            &["BOUNDARY", "STATUS"],
            &[Constraint::Min(14), Constraint::Length(10)],
        )
    };
    let grid = Grid {
        header: heading(t, names),
        rows,
        widths,
    };
    table(f, t, area, block, grid, Some(at));
}

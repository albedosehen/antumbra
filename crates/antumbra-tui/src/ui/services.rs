//! The Services page: what depends on what, drawn as a graph, beside what a
//! change to the selected service reaches.
//!
//! The graph reads left to right: a service is drawn before what it depends
//! on, and an arrowhead marks the service a line arrives at. A line's stroke
//! is the source of the claim (declared in a manifest, observed, or learned
//! and claimed), as the dashboard's tags say it. The selected service is
//! where the walk starts; what it reaches is lit, with the lines it took.

use std::collections::BTreeSet;

use antumbra_core::depgraph::{Direction, Reached, Source};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::netgraph;
use crate::services::{drawn_as, short, Services, WALKED_FROM};
use crate::theme::Theme;

use super::netdraw::{self, Chip, Edge, Node, Stroke};
use super::{clip, heading, panel, title};

/// The page: the graph beside the selected service's blast radius.
pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    if app.services.is_empty() {
        let why = "No dependencies recorded yet. `antumbra claude dependencies` records the \
                   ones a repository's manifests declare, the GitHub App records them on \
                   every merge, and an agent records one with `record_dependency`.";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(why, Style::default().fg(t.ink3))))
                .block(panel(&t, title(&t, "Services")))
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }
    let cols =
        Layout::horizontal([Constraint::Percentage(64), Constraint::Percentage(36)]).split(area);
    let reached = app.service_reach();
    graph(f, app, cols[0], &reached);
    radius(f, app, cols[1], &reached);
}

/// What the walk's direction is called, from the selected service's side.
pub(super) fn direction_words(direction: Direction) -> &'static str {
    match direction {
        Direction::Dependents => "what depends on it",
        Direction::Dependencies => "what it depends on",
    }
}

/// `1 hop`, `3 hops`.
pub(super) fn hops_words(hops: u32) -> String {
    format!("{hops} hop{}", if hops == 1 { "" } else { "s" })
}

/// The graph: every service and every pair, or, when that is more than the
/// panel holds, the selected service and what is around it.
fn graph(f: &mut Frame, app: &App, area: Rect, reached: &[Reached]) {
    let t = app.theme();
    let services = &app.services;
    let name = format!(
        "Services {} · {} dependenc{}",
        services.names.len(),
        services.pairs.len(),
        if services.pairs.len() == 1 {
            "y"
        } else {
            "ies"
        }
    );
    let block = panel(&t, title(&t, &name));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let rows = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(inner);

    let origin = app.selected_service_name().unwrap_or_default();
    let lit: BTreeSet<(&str, &str)> = reached
        .iter()
        .flat_map(|r| r.path.iter())
        .map(|hop| (hop.from.as_str(), hop.to.as_str()))
        .collect();
    let within: BTreeSet<&str> = reached.iter().map(|r| r.service.as_str()).collect();

    // Everything when it fits; else the selected service and what is within
    // reach of it either way, which is what the picture is asked for.
    let every: Vec<&str> = services.names.iter().map(String::as_str).collect();
    let whole = netdraw::fits(
        rows[0],
        &services.layout,
        &labelled(&every, origin, &within),
    );
    let (names, layout, note): (Vec<&str>, netgraph::Layout, Option<String>) = if whole {
        (every, services.layout.clone(), None)
    } else {
        let (names, layout) = around(app, origin);
        let note = format!(
            "showing {} of {} services: the selected one and what is within reach of it",
            names.len(),
            services.names.len()
        );
        (names, layout, Some(note))
    };
    let nodes = labelled(&names, origin, &within);
    if !netdraw::fits(rows[0], &layout, &nodes) {
        let why = "This panel is too small to draw the graph. The blast radius beside it                    lists what the selected service reaches.";
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(why, Style::default().fg(t.ink3))))
                .wrap(Wrap { trim: true }),
            rows[0],
        );
        return;
    }

    let at = |service: &str| names.iter().position(|n| *n == service);
    let edges: Vec<Edge> = services
        .pairs
        .iter()
        .filter_map(|pair| {
            Some(Edge {
                from: at(&pair.from)?,
                to: at(&pair.to)?,
                stroke: match drawn_as(pair) {
                    Source::Declared => Stroke::Solid,
                    Source::Observed => Stroke::Dashed,
                    Source::Learned | Source::Claimed => Stroke::Dotted,
                },
                lit: lit.contains(&(pair.from.as_str(), pair.to.as_str())),
                faint: pair.weight < WALKED_FROM,
            })
        })
        .collect();
    netdraw::draw(f, app, rows[0], &layout, &nodes, &edges);

    if let Some(note) = note {
        f.render_widget(
            Paragraph::new(Span::styled(
                format!(" {note}"),
                Style::default().fg(t.ink3),
            )),
            rows[1],
        );
    }
    let mut key = netdraw::key(&t, ["declared", "observed", "learned or claimed"]);
    key.push(Span::styled(
        "   ▸ what a service depends on",
        Style::default().fg(t.ink3),
    ));
    f.render_widget(Paragraph::new(Line::from(key)), rows[2]);
}

/// Each of `names` as a node of the picture: by its short name, and as the
/// service the walk starts from, one it reached, or neither.
fn labelled<'a>(names: &[&'a str], origin: &str, within: &BTreeSet<&str>) -> Vec<Node<'a>> {
    names
        .iter()
        .map(|n| Node {
            label: short(n),
            chip: if *n == origin {
                Chip::Origin
            } else if within.contains(n) {
                Chip::Reached
            } else {
                Chip::Plain
            },
        })
        .collect()
}

/// The selected service with every service a walk reaches from it in either
/// direction, laid out on their own: the part of a large graph a panel can
/// hold.
fn around<'a>(app: &'a App, origin: &'a str) -> (Vec<&'a str>, netgraph::Layout) {
    let services: &Services = &app.services;
    let now = chrono::Utc::now();
    let mut keep: BTreeSet<&str> = BTreeSet::new();
    keep.insert(origin);
    let near = [Direction::Dependents, Direction::Dependencies]
        .into_iter()
        .flat_map(|d| services.reach(origin, d, app.service_hops, now));
    let near: Vec<String> = near.map(|r| r.service).collect();
    for name in &services.names {
        if near.iter().any(|n| n == name) {
            keep.insert(name.as_str());
        }
    }
    // In the whole graph's reading order, so the picture does not reshuffle
    // as the selection moves through one neighborhood.
    let names: Vec<&str> = services
        .names
        .iter()
        .map(String::as_str)
        .filter(|n| keep.contains(n))
        .collect();
    let at = |service: &str| names.iter().position(|n| *n == service);
    let joins: Vec<(usize, usize)> = services
        .pairs
        .iter()
        .filter_map(|pair| Some((at(&pair.from)?, at(&pair.to)?)))
        .collect();
    let layout = netgraph::Layout::of(names.len(), &joins);
    (names, layout)
}

/// The blast radius: the selected service, which way and how far the walk
/// goes, and each service it reaches with how sure the path is and what says
/// so.
fn radius(f: &mut Frame, app: &App, area: Rect, reached: &[Reached]) {
    let t = app.theme();
    let block = panel(&t, title(&t, "Blast radius"));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(origin) = app.selected_service_name() else {
        return;
    };
    let width = usize::from(inner.width);
    let mut lines: Vec<Line> = vec![
        heading(&t, clip(short(origin), width)),
        Line::from(Span::styled(
            clip(origin, width),
            Style::default().fg(t.ink3),
        )),
        Line::from(Span::styled(
            format!(
                "{} · within {}",
                direction_words(app.service_direction),
                hops_words(app.service_hops)
            ),
            Style::default().fg(t.ink2),
        )),
        Line::from(""),
    ];
    if reached.is_empty() {
        let none = match app.service_direction {
            Direction::Dependents => format!(
                "Nothing recorded depends on {}. A change here stays here.",
                short(origin)
            ),
            Direction::Dependencies => {
                format!("{} depends on nothing recorded.", short(origin))
            }
        };
        lines.push(Line::from(Span::styled(none, Style::default().fg(t.ink3))));
    }
    // Two rows a service; what does not fit is counted, not cut silently.
    let room = usize::from(inner.height).saturating_sub(lines.len());
    let shown = if reached.len() * 2 > room {
        room.saturating_sub(1) / 2
    } else {
        reached.len()
    };
    for r in reached.iter().take(shown) {
        lines.extend(reached_rows(&t, r, origin, width));
    }
    if shown < reached.len() {
        lines.push(Line::from(Span::styled(
            format!("and {} more: enter lists them all", reached.len() - shown),
            Style::default().fg(t.ink3),
        )));
    }
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// One reached service: its name, how sure the path is and how long, then
/// what says so: the claim itself when it is one hop away, or the service it
/// was reached through.
pub(super) fn reached_rows<'a>(
    t: &Theme,
    r: &Reached,
    origin: &str,
    width: usize,
) -> [Line<'a>; 2] {
    let figures = format!("{:.2} · {}", r.weight, hops_words(r.depth));
    let name = clip(short(&r.service), width.saturating_sub(figures.len() + 2));
    let pad = width.saturating_sub(name.chars().count() + figures.chars().count());
    let first = Line::from(vec![
        Span::styled(
            name,
            Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" ".repeat(pad)),
        Span::styled(figures, Style::default().fg(t.ink2)),
    ]);
    let says = match r.path.last() {
        Some(hop) if r.depth > 1 => {
            // The end of the last hop that is not the service reached.
            let through = if hop.from == r.service {
                &hop.to
            } else {
                &hop.from
            };
            if through == origin {
                String::new()
            } else {
                format!("via {}", short(through))
            }
        }
        Some(hop) => hop
            .edges
            .first()
            .map_or_else(String::new, |edge| match &edge.detail {
                Some(detail) => format!("{} · {detail}", edge.source.as_str()),
                None => edge.source.as_str().to_string(),
            }),
        None => String::new(),
    };
    let second = Line::from(Span::styled(
        format!("  {}", clip(&says, width.saturating_sub(2))),
        Style::default().fg(t.ink3),
    ));
    [first, second]
}

//! The Memory page's graph: the selected memory and the memories linked to
//! it, drawn.
//!
//! A memory is linked to another when it references it, supersedes it,
//! follows it, was caused by it or contradicts it. A store holds thousands of
//! memories and a few links, so a picture of every memory is a picture of
//! dots; the one worth drawing is of the memory in hand: what it links to,
//! what links to it, and what those are linked to in turn. It reads left to
//! right, as the Services page does: a memory is drawn before what it links
//! to, and an arrowhead marks the memory a line arrives at.

use std::collections::{BTreeMap, BTreeSet};

use antumbra_core::{EdgeType, MemoryEdge};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::App;
use crate::netgraph;

use super::netdraw::{self, Chip, Edge, Node, Stroke};
use super::{panel, title};

/// The most of a memory's content a chip shows: enough to tell it from its
/// neighbors, the panel beside giving it whole.
const LABEL: usize = 28;
/// The most memories a picture holds: past this the lines are not followed.
const MOST: usize = 24;
/// The fewest cells a memory's chip keeps for a second ring of memories to
/// be drawn round the first: under this, names are cut to where they no
/// longer tell one memory from the next.
const READABLE: u16 = 20;

/// Each of `ids` as its chip names it: how the memory's content starts, on
/// one line. Found in one pass over the store's memories, which a frame
/// cannot afford once for every id.
fn labels(app: &App, ids: &[&str]) -> Vec<String> {
    let wanted: BTreeSet<&str> = ids.iter().copied().collect();
    let found: BTreeMap<&str, String> = app
        .memories
        .iter()
        .filter(|m| wanted.contains(m.id.as_str()))
        .map(|m| {
            // Only as much of it as a chip shows is read.
            let start: Vec<&str> = m.content.split_whitespace().take(LABEL).collect();
            (m.id.as_str(), super::clip(&start.join(" "), LABEL))
        })
        .collect();
    ids.iter()
        .map(|id| match found.get(id) {
            Some(label) => label.clone(),
            // A link can outlive a memory it names; its id is all there is.
            None => super::clip(id, LABEL),
        })
        .collect()
}

/// The ids within `hops` links of `start`, either way along a link, nearest
/// first, and the nearest `most` of them at that: `start`, then what it is
/// linked with, then what those are.
fn within<'a>(edges: &'a [MemoryEdge], start: &'a str, hops: usize, most: usize) -> Vec<&'a str> {
    let mut found: Vec<&str> = vec![start];
    let mut seen: BTreeSet<&str> = BTreeSet::from([start]);
    let mut edge = 0;
    for _ in 0..hops {
        let ring = found.len();
        for at in edge..ring {
            for e in edges {
                let other = if e.from_id.as_str() == found[at] {
                    e.to_id.as_str()
                } else if e.to_id.as_str() == found[at] {
                    e.from_id.as_str()
                } else {
                    continue;
                };
                if found.len() < most && seen.insert(other) {
                    found.push(other);
                }
            }
        }
        edge = ring;
    }
    found
}

/// The selected memory's links, drawn, under a title that says how many of
/// the store's memories have one.
pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let linked = app.linked_memories();
    let name = format!(
        "Links · {} of {} memories have one · {} link{}",
        linked.len(),
        app.memories.len(),
        app.edges.len(),
        if app.edges.len() == 1 { "" } else { "s" }
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
    let say = |f: &mut Frame, why: &str| {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                why.to_string(),
                Style::default().fg(t.ink3),
            )))
            .wrap(Wrap { trim: true }),
            rows[0],
        );
    };
    if linked.is_empty() {
        say(
            f,
            "No memory is linked to another yet. The server links a new memory to the \
             ones it restates, and an agent links two with `relate_memories`.",
        );
        return;
    }
    let Some(selected) = app.selected_memory() else {
        return;
    };
    let start = selected.id.as_str();
    if app.memory_edges(start).is_empty() {
        say(
            f,
            "This memory is not linked to another. ↑↓ moves through the ones that are.",
        );
        return;
    }

    // Two links out when the names stay readable, else one: the memory's own
    // links are the picture, and their neighbors, which say where each one
    // leads, are not worth cutting every name for.
    let mut drawn = None;
    for hops in [2, 1] {
        let ids = within(&app.edges, start, hops, MOST);
        let labels = labels(app, &ids);
        let at = |id: &str| ids.iter().position(|i| *i == id);
        // With one ring, only the memory's own links place a memory: what
        // links to it on one side, what it links to on the other. Its
        // neighbors' links to each other would add columns, and every
        // column costs each name some of its letters.
        let joins: Vec<(usize, usize)> = app
            .edges
            .iter()
            .filter(|e| hops > 1 || e.from_id.as_str() == start || e.to_id.as_str() == start)
            .filter_map(|e| Some((at(e.from_id.as_str())?, at(e.to_id.as_str())?)))
            .collect();
        let layout = netgraph::Layout::of(ids.len(), &joins);
        let nodes: Vec<Node> = labels
            .iter()
            .map(|l| Node {
                label: l.as_str(),
                chip: Chip::Plain,
            })
            .collect();
        let roomy = !netdraw::cramped(rows[0].width, &layout, &nodes, READABLE);
        if netdraw::fits(rows[0], &layout, &nodes) && (roomy || hops == 1) {
            drawn = Some((ids, labels, layout, hops));
            break;
        }
    }
    let Some((ids, labels, layout, hops)) = drawn else {
        say(
            f,
            "This panel is too small to draw this memory's links. The panel beside it \
             lists them.",
        );
        return;
    };

    let near: BTreeSet<&str> = app
        .memory_edges(start)
        .into_iter()
        .map(|(e, out)| {
            if out {
                e.to_id.as_str()
            } else {
                e.from_id.as_str()
            }
        })
        .collect();
    let nodes: Vec<Node> = ids
        .iter()
        .zip(&labels)
        .map(|(id, label)| Node {
            label: label.as_str(),
            chip: if *id == start {
                Chip::Origin
            } else if near.contains(id) {
                Chip::Reached
            } else {
                Chip::Plain
            },
        })
        .collect();
    let at = |id: &str| ids.iter().position(|i| *i == id);
    let edges: Vec<Edge> = app
        .edges
        .iter()
        .filter_map(|e| {
            let (from, to) = (e.from_id.as_str(), e.to_id.as_str());
            // One ring draws the memory's own links and no others.
            if hops == 1 && from != start && to != start {
                return None;
            }
            Some(Edge {
                from: at(from)?,
                to: at(to)?,
                stroke: match e.edge_type {
                    EdgeType::References => Stroke::Solid,
                    EdgeType::Supersedes => Stroke::Dashed,
                    EdgeType::Follows | EdgeType::Caused | EdgeType::Contradicts => Stroke::Dotted,
                },
                // The selected memory's own links, over its neighbors'.
                lit: from == start || to == start,
                faint: false,
            })
        })
        .collect();
    netdraw::draw(f, app, rows[0], &layout, &nodes, &edges);

    // Every link of the memory when they fit, and how many when not.
    let (own, shown) = (
        app.memory_edges(start).len(),
        near.iter().filter(|id| ids.contains(id)).count(),
    );
    let note = if shown < near.len() {
        format!(
            " {shown} of the {} memories it is linked with · ↑↓ the next",
            near.len()
        )
    } else {
        format!(
            " its {own} link{}{} · ↑↓ the next linked memory",
            if own == 1 { "" } else { "s" },
            // The second ring is what says where each one leads.
            if hops > 1 { " and where they lead" } else { "" }
        )
    };
    f.render_widget(
        Paragraph::new(Span::styled(note, Style::default().fg(t.ink3))),
        rows[1],
    );
    let mut key = netdraw::key(
        &t,
        ["references", "supersedes", "follows, caused, contradicts"],
    );
    key.push(Span::styled("  ▸ links to", Style::default().fg(t.ink3)));
    f.render_widget(Paragraph::new(Line::from(key)), rows[2]);
}

//! The Memory page (the penumbra): the world/bank/opinion networks as a typed
//! edge graph beside the selected trace's detail. Memories cluster by network on
//! a hand-rolled Canvas; edges color by kind (contradiction/supersession are the
//! consolidation→retire signal). This is the soft store the population graduates
//! from: the hippocampus to the umbra's neocortex.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use antumbra_core::{scope_of_evidence, EdgeType, GitProvenance, MemoryNetwork, Scope};

use crate::app::App;
use crate::theme::Theme;

use super::{gauge_row, heading, kv, panel};

/// The three networks and where their clusters anchor on the canvas.
const ANCHORS: [(MemoryNetwork, (f64, f64)); 3] = [
    (MemoryNetwork::World, (0.0, 62.0)),
    (MemoryNetwork::Bank, (-60.0, -38.0)),
    (MemoryNetwork::Opinion, (60.0, -38.0)),
];

/// Cap the nodes drawn per network on the canvas. The graph is a visualization,
/// not a list: a real store holds thousands of memories, and laying out and
/// drawing every one each frame pegs a core and starves input for no visual gain
/// (the ring just fills solid). Memories arrive sorted strongest-first per
/// network, so the cap keeps the most reinforced anchors.
const MAX_GRAPH_NODES_PER_NETWORK: usize = 48;

/// Bound the edge scan too, so a dense relation set can't stall the frame (only
/// edges joining drawn nodes paint anyway).
const MAX_GRAPH_EDGES: usize = 256;

pub(super) fn page(f: &mut Frame, app: &App, area: Rect) {
    let cols =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(area);
    graph(f, app, cols[0]);
    detail(f, app, cols[1]);
}

/// Color an edge by its kind: contradiction/supersession (the retire signal)
/// stand out; reference/follows/caused are quieter.
fn edge_color(t: &Theme, e: EdgeType) -> Color {
    match e {
        EdgeType::Contradicts => t.alert,
        EdgeType::Supersedes => t.warning,
        EdgeType::Caused => t.value,
        EdgeType::Follows => t.accent,
        EdgeType::References => t.dim,
    }
}

/// The penumbra graph: memories clustered by network, edges drawn between them.
fn graph(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(
        &t,
        Span::styled(" penumbra · memory networks ", Style::default().fg(t.ink)),
    );
    let canvas = Canvas::default()
        .block(block)
        .marker(app.canvas_marker())
        .x_bounds([-100.0, 100.0])
        .y_bounds([-100.0, 100.0])
        .paint(move |ctx| {
            // Lay each network's traces on a ring around its anchor. Only the
            // capped subset is positioned and drawn (a real store holds thousands;
            // the ring fills solid past a few dozen and drawing every one pegs the
            // render). Keep each node's global index so the selection still lands.
            let mut pos: HashMap<&str, (f64, f64)> = HashMap::new();
            let mut nodes: Vec<(usize, &_, f64, f64)> = Vec::new();
            for (net, (ax, ay)) in ANCHORS {
                let mems: Vec<(usize, &_)> = app
                    .memories
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.network == net)
                    .take(MAX_GRAPH_NODES_PER_NETWORK)
                    .collect();
                let n = mems.len();
                for (j, (gi, m)) in mems.iter().enumerate() {
                    let p = if n <= 1 {
                        (ax, ay)
                    } else {
                        let ang = (j as f64 / n as f64) * TAU - FRAC_PI_2;
                        (ax + 24.0 * ang.cos(), ay + 24.0 * ang.sin())
                    };
                    pos.insert(m.id.as_str(), p);
                    nodes.push((*gi, *m, p.0, p.1));
                }
            }
            // Edges first, under the nodes (only those joining drawn nodes).
            for e in app.edges.iter().take(MAX_GRAPH_EDGES) {
                if let (Some(&(x1, y1)), Some(&(x2, y2))) =
                    (pos.get(e.from_id.as_str()), pos.get(e.to_id.as_str()))
                {
                    ctx.draw(&CanvasLine {
                        x1,
                        y1,
                        x2,
                        y2,
                        color: edge_color(&t, e.edge_type),
                    });
                }
            }
            // Nodes: confidence tints the glyph; the selected trace is accented,
            // a consolidated one is filled. Iterate only the capped set.
            for (gi, m, x, y) in &nodes {
                let selected = *gi == app.selected_memory;
                let glyph = if selected {
                    "◉"
                } else if m.is_consolidated() {
                    "●"
                } else {
                    "○"
                };
                let col = if selected {
                    t.accent
                } else {
                    t.fitness(m.confidence, 0.9)
                };
                ctx.print(*x, *y, Span::styled(glyph, Style::default().fg(col)));
            }
            // Network labels under each cluster.
            for (net, (ax, ay)) in ANCHORS {
                ctx.print(
                    ax - 3.0,
                    ay - 30.0,
                    Span::styled(
                        net.as_str(),
                        Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
                    ),
                );
            }
        });
    f.render_widget(canvas, area);
}

/// A bold, dim section label.
fn group<'a>(t: &Theme, label: String) -> Line<'a> {
    Line::from(Span::styled(
        label,
        Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
    ))
}

/// Where a memory was learned, and how that relates to where the console was
/// started: the scope recall would tag it with (ADR-0018). `None` for a memory
/// with no git anchor.
fn anchor_lines<'a>(t: &Theme, evidence: &[String], app: &App, w: usize) -> Option<Vec<Line<'a>>> {
    let anchor = GitProvenance::from_evidence(evidence)?;
    let commit: String = anchor.commit.chars().take(7).collect();
    let room = w.saturating_sub(11);
    let mut lines = vec![kv(
        t,
        "anchor",
        &clip(&format!("{}@{commit}", anchor.repo), room),
    )];
    if let Some(branch) = &anchor.branch {
        lines.push(kv(t, "branch", &clip(branch, room)));
    }
    let scope = scope_of_evidence(evidence, &app.git);
    let color = match scope {
        Scope::InScope => t.accent,
        Scope::OtherBranch | Scope::OtherRepo => t.warning,
        Scope::Orphaned => t.alert,
        Scope::Unknown => t.dim,
    };
    lines.push(Line::from(vec![
        Span::styled(format!("{:<11}", "scope"), Style::default().fg(t.dim)),
        Span::styled(scope.as_str(), Style::default().fg(color)),
    ]));
    Some(lines)
}

/// Clip `s` to `max` columns, ending in an ellipsis when cut.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The selected trace: its fields, evidence, and the edges touching it.
fn detail(f: &mut Frame, app: &App, area: Rect) {
    let t = app.theme();
    let block = panel(&t, Span::styled(" memory ", Style::default().fg(t.ink)));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(m) = app.selected_memory() else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "penumbra is empty; no memory traces",
                Style::default().fg(t.dim),
            ))),
            inner,
        );
        return;
    };
    let w = inner.width.saturating_sub(1) as usize;

    let mut lines = vec![
        heading(&t, clip(&m.content, w)),
        Line::from(""),
        kv(&t, "network", m.network.as_str()),
        kv(&t, "status", m.status.as_str()),
        gauge_row(&t, "confidence", m.confidence, t.fitness(m.confidence, 1.0)),
        kv(&t, "reinforce", &m.reinforcement.to_string()),
        kv(&t, "volatile", if m.volatile { "yes" } else { "no" }),
    ];
    if let Some(exp) = &m.consolidated_expert {
        lines.push(kv(&t, "expert", exp.as_str()));
    }
    if let Some(author) = &m.author {
        let host = m.author_host.as_deref().unwrap_or("-");
        lines.push(kv(&t, "author", &format!("{} · {host}", author.as_str())));
    }
    if let Some(anchor) = anchor_lines(&t, &m.evidence, app, w) {
        lines.extend(anchor);
    }

    if !m.evidence.is_empty() {
        lines.push(Line::from(""));
        lines.push(group(&t, "evidence".into()));
        for ev in &m.evidence {
            lines.push(Line::from(Span::styled(
                format!("  · {}", clip(ev, w.saturating_sub(4))),
                Style::default().fg(t.value),
            )));
        }
    }

    let edges = app.memory_edges(m.id.as_str());
    lines.push(Line::from(""));
    lines.push(group(&t, format!("edges ({})", edges.len())));
    for (e, outgoing) in &edges {
        let arrow = if *outgoing { "→" } else { "←" };
        let other = if *outgoing {
            e.to_id.as_str()
        } else {
            e.from_id.as_str()
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {arrow} "), Style::default().fg(t.dim)),
            Span::styled(
                format!("{:<12}", e.edge_type.as_str()),
                Style::default().fg(edge_color(&t, e.edge_type)),
            ),
            Span::styled(
                clip(other, w.saturating_sub(17)),
                Style::default().fg(t.ink),
            ),
        ]));
    }

    f.render_widget(Paragraph::new(lines), inner);
}

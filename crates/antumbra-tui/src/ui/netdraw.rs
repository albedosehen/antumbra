//! Draws a layered graph ([`crate::netgraph::Layout`]): each node a labelled
//! chip in its layer's column, each edge a line from a chip to the next
//! column, where an arrowhead marks the chip it arrives at.
//!
//! A line runs along its node's row, turns once in the gap between two
//! columns, and runs along the row of where it is going. Every line into one
//! node turns in the same cell column, so they join there into one trunk,
//! and lines into different nodes turn in different ones, so a turn is never
//! taken for a join. Lines are box-drawing characters, drawn from what
//! meets in each cell: a terminal draws those crisply at any size, where a
//! slanted line of dots reads as noise.

use std::collections::BTreeMap;

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;
use crate::netgraph::{Layout, Slot};
use crate::theme::Theme;

use super::clip;

/// How a node's chip is drawn.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Chip {
    /// The node the picture is about: filled with the accent.
    Origin,
    /// One the walk from it reached.
    Reached,
    /// Any other.
    Plain,
}

/// How an edge's line is drawn. The order is from the surest to the least
/// sure, so where two edges share a cell the surer one's stroke is kept.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Stroke {
    Solid,
    Dashed,
    Dotted,
}

pub(super) struct Node<'a> {
    pub label: &'a str,
    pub chip: Chip,
}

pub(super) struct Edge {
    pub from: usize,
    pub to: usize,
    pub stroke: Stroke,
    /// On a path the walk took: drawn in the accent, over the others.
    pub lit: bool,
    /// Too weak for a walk to follow: drawn in the faintest ink.
    pub faint: bool,
}

/// The cells before a chip: where the lines into it end, then its arrowhead.
const LEAD: u16 = 2;
/// The fewest cells between a column's chips and the next column, for the
/// lines to turn in.
const GAP: u16 = 4;
/// The fewest cells of a name a chip shows, with its two of padding.
const SHORTEST: u16 = 6;

/// Where a slot is drawn: its row, the cell a line into it ends at, and the
/// cell a line out of it starts from. A chip sits between the two.
#[derive(Clone, Copy)]
struct Spot {
    y: u16,
    enter: u16,
    leave: u16,
}

/// A layer's column: where it starts, how wide its chips may be, and where
/// the gap after them starts and ends.
#[derive(Clone, Copy)]
struct Column {
    left: u16,
    chip: u16,
    gap: (u16, u16),
}

/// What meets in one cell: which of its four sides a line leaves by, and how
/// the lines through it are drawn.
#[derive(Clone, Copy)]
struct Cell {
    sides: u8,
    stroke: Stroke,
    lit: bool,
    faint: bool,
}

const N: u8 = 1;
const E: u8 = 2;
const S: u8 = 4;
const W: u8 = 8;

/// The widest chip of each layer: its longest name and a space each side.
fn widest(layout: &Layout, nodes: &[Node]) -> Vec<u16> {
    layout
        .layers
        .iter()
        .map(|slots| {
            slots
                .iter()
                .filter_map(|slot| match slot {
                    Slot::Node(node) => nodes.get(*node),
                    Slot::Lane(_) => None,
                })
                .map(|n| u16::try_from(n.label.chars().count() + 2).unwrap_or(u16::MAX))
                .max()
                .unwrap_or(0)
        })
        .collect()
}

/// The gap a layer needs after it: a cell for every slot of the next layer
/// to turn in, and one clear each side.
fn turns(layout: &Layout, layer: usize) -> u16 {
    let next = layout.layers.get(layer + 1).map_or(0, Vec::len);
    u16::try_from(next + 2).unwrap_or(u16::MAX).max(GAP)
}

/// Each layer's column in `width` cells, or `None` when even names cut to
/// their shortest do not fit. Names are cut before a gap is narrowed: a line
/// that has nowhere to turn is worse than a name with an ellipsis.
fn columns(width: u16, layout: &Layout, nodes: &[Node]) -> Option<Vec<Column>> {
    let last = layout.layers.len().checked_sub(1)?;
    let chips = widest(layout, nodes);
    let gaps: Vec<u16> = (0..=last)
        .map(|layer| {
            if layer == last {
                0
            } else {
                turns(layout, layer)
            }
        })
        .collect();
    let count = u16::try_from(last + 1).ok()?;
    let fixed: u16 = gaps
        .iter()
        .sum::<u16>()
        .checked_add(LEAD.checked_mul(count)?)?;
    let wanted: u16 = chips.iter().sum();
    let room = width.checked_sub(fixed)?;
    // Every name whole when there is room; else each cut by the same share.
    let chips: Vec<u16> = if wanted <= room {
        chips
    } else {
        let cut: Vec<u16> = chips
            .iter()
            .map(|chip| {
                let share = u32::from(*chip) * u32::from(room) / u32::from(wanted.max(1));
                u16::try_from(share)
                    .unwrap_or(u16::MAX)
                    .max(SHORTEST.min(*chip))
            })
            .collect();
        if cut.iter().sum::<u16>() > room {
            return None;
        }
        cut
    };
    // What is left over widens the gaps, where it straightens the picture.
    let spare = room - chips.iter().sum::<u16>();
    let each = if last == 0 { 0 } else { spare / (count - 1) };
    let mut left = 0;
    let mut out = Vec::with_capacity(last + 1);
    for layer in 0..=last {
        let gap = if layer == last { 0 } else { gaps[layer] + each };
        let start = left + LEAD + chips[layer];
        out.push(Column {
            left,
            chip: chips[layer],
            gap: (start, start + gap),
        });
        left = start + gap;
    }
    Some(out)
}

/// Whether `layout` fits `area`: a column for every layer, and a row for
/// every slot of its longest.
pub(super) fn fits(area: Rect, layout: &Layout, nodes: &[Node]) -> bool {
    let longest = layout.layers.iter().map(Vec::len).max().unwrap_or(0);
    longest <= usize::from(area.height) && columns(area.width, layout, nodes).is_some()
}

/// Each slot's row. The first layer is spread down the picture; every later
/// one puts a slot level with what points at it, where the slots above leave
/// room, so most lines run straight across.
fn rows(area: Rect, layout: &Layout, edges: &[Edge]) -> BTreeMap<(usize, usize), u16> {
    let mut rows: BTreeMap<(usize, usize), u16> = BTreeMap::new();
    let tall = area.height.max(1);
    for (layer, slots) in layout.layers.iter().enumerate() {
        let count = u16::try_from(slots.len()).unwrap_or(u16::MAX).max(1);
        // Two rows a slot where there is room, so chips do not stack solid.
        let step = if count.saturating_mul(2) - 1 <= tall {
            2
        } else {
            1
        };
        let spread = (tall / count).clamp(step, 3);
        let block = spread * (count - 1) + 1;
        let top = tall.saturating_sub(block) / 2;
        let mut wanted: Vec<u16> = (0..count).map(|place| top + spread * place).collect();
        if layer > 0 {
            for (place, slot) in slots.iter().enumerate() {
                let target = match slot {
                    Slot::Node(node) | Slot::Lane(node) => *node,
                };
                // The rows of the slots a line comes into this one from.
                let from: Vec<u16> = edges
                    .iter()
                    .filter(|edge| edge.to == target)
                    .filter_map(|edge| {
                        let path = layout.path(edge.from, edge.to);
                        let here = path.iter().position(|(l, s)| *l == layer && s == slot)?;
                        let (l, s) = path.get(here.checked_sub(1)?)?;
                        rows.get(&(*l, layout.place(*l, *s)?)).copied()
                    })
                    .collect();
                if !from.is_empty() {
                    let sum: u32 = from.iter().map(|y| u32::from(*y - area.y)).sum();
                    wanted[place] = u16::try_from(sum / from.len() as u32).unwrap_or(0);
                }
            }
            // In order and apart: pushed down from the top, and then, where
            // that ran off the bottom, back up from it.
            for place in 1..wanted.len() {
                wanted[place] = wanted[place].max(wanted[place - 1] + step);
            }
            let mut floor = tall - 1;
            for place in (0..wanted.len()).rev() {
                wanted[place] = wanted[place].min(floor);
                floor = wanted[place].saturating_sub(step);
            }
        }
        for (place, y) in wanted.into_iter().enumerate() {
            rows.insert((layer, place), area.y + y.min(tall - 1));
        }
    }
    rows
}

/// Draw `nodes` and `edges` as `layout` places them, in `area`. The caller
/// checks [`fits`] first; a picture that does not fit is not drawn.
pub(super) fn draw(
    f: &mut Frame,
    app: &App,
    area: Rect,
    layout: &Layout,
    nodes: &[Node],
    edges: &[Edge],
) {
    let t = app.theme();
    let Some(columns) = columns(area.width, layout, nodes) else {
        return;
    };
    let rows = rows(area, layout, edges);

    let mut spots: BTreeMap<(usize, usize), Spot> = BTreeMap::new();
    let mut chips: Vec<(Rect, usize)> = Vec::new();
    for (layer, slots) in layout.layers.iter().enumerate() {
        let column = columns[layer];
        let left = area.x + column.left;
        for (place, slot) in slots.iter().enumerate() {
            let Some(&y) = rows.get(&(layer, place)) else {
                continue;
            };
            let spot = match slot {
                // A lane's line runs the whole column, to the gap after it.
                Slot::Lane(_) => Spot {
                    y,
                    enter: left,
                    leave: area.x + column.gap.0,
                },
                Slot::Node(node) => {
                    let label = nodes.get(*node).map_or("", |n| n.label);
                    let wide = u16::try_from(label.chars().count() + 2).unwrap_or(u16::MAX);
                    let width = wide.min(column.chip).max(1);
                    chips.push((Rect::new(left + LEAD, y, width, 1), *node));
                    Spot {
                        y,
                        enter: left,
                        leave: left + LEAD + width,
                    }
                }
            };
            spots.insert((layer, place), spot);
        }
    }

    let mut cells: BTreeMap<(u16, u16), Cell> = BTreeMap::new();
    let mut arrives: BTreeMap<usize, bool> = BTreeMap::new();
    for edge in edges {
        let path: Vec<(usize, usize, Slot)> = layout
            .path(edge.from, edge.to)
            .into_iter()
            .filter_map(|(layer, slot)| Some((layer, layout.place(layer, slot)?, slot)))
            .collect();
        for (i, pair) in path.windows(2).enumerate() {
            let (a, b) = ((pair[0].0, pair[0].1), (pair[1].0, pair[1].1));
            let (Some(from), Some(to)) = (spots.get(&a), spots.get(&b)) else {
                continue;
            };
            // An edge that points backward, as one of a cycle does, has no
            // gap of its own to turn in: it is left to the list beside.
            if b.0 != a.0 + 1 {
                continue;
            }
            // An edge's own stroke is its first stretch's; past that it is
            // on a trunk it shares.
            let pen = Pen {
                stroke: if i == 0 { edge.stroke } else { Stroke::Solid },
                lit: edge.lit,
                faint: edge.faint,
            };
            // Every line into one slot turns in that slot's own cell column
            // of the gap, counted back from the gap's far side.
            let gap = columns[a.0].gap;
            let lanes = gap.1.saturating_sub(gap.0).saturating_sub(1).max(1);
            let turn = area.x + gap.1 - 1 - (u16::try_from(b.1).unwrap_or(0) % lanes);
            if from.y == to.y {
                along(&mut cells, area, from.leave, to.enter, to.y, pen);
            } else {
                along(&mut cells, area, from.leave, turn, from.y, pen);
                along(&mut cells, area, turn, to.enter, to.y, pen);
                let (top, bottom) = (from.y.min(to.y), from.y.max(to.y));
                for y in top..=bottom {
                    let sides = if y == top {
                        S
                    } else if y == bottom {
                        N
                    } else {
                        N | S
                    };
                    mark(&mut cells, area, (turn, y), sides, pen);
                }
            }
            // A lane carries the line across its column to the next gap.
            if matches!(pair[1].2, Slot::Lane(_)) {
                let solid = Pen {
                    stroke: Stroke::Solid,
                    ..pen
                };
                along(&mut cells, area, to.enter, to.leave, to.y, solid);
            }
        }
        *arrives.entry(edge.to).or_insert(false) |= edge.lit;
    }

    for ((x, y), cell) in &cells {
        let color = if cell.lit {
            t.accent
        } else if cell.faint {
            t.line
        } else {
            t.ink3
        };
        f.render_widget(
            Paragraph::new(Span::styled(
                glyph(cell.sides, cell.stroke),
                Style::default().fg(color).bg(t.ground),
            )),
            Rect::new(*x, *y, 1, 1),
        );
    }
    for (cells, node) in chips {
        let Some(n) = nodes.get(node) else {
            continue;
        };
        if let Some(lit) = arrives.get(&node) {
            let color = if *lit { t.accent } else { t.ink3 };
            f.render_widget(
                Paragraph::new(Span::styled("▸", Style::default().fg(color).bg(t.ground))),
                Rect::new(cells.x - 1, cells.y, 1, 1),
            );
        }
        let inner = usize::from(cells.width.saturating_sub(2));
        let text = format!(" {:<inner$} ", clip(n.label, inner));
        f.render_widget(Paragraph::new(Span::styled(text, chip(&t, n.chip))), cells);
    }
}

/// How a stretch of line is drawn.
#[derive(Clone, Copy)]
struct Pen {
    stroke: Stroke,
    lit: bool,
    faint: bool,
}

/// Note that a line leaves the cell at `at` by `sides`. Where two lines share
/// a cell the surer stroke is kept, and the cell is lit if either is.
fn mark(cells: &mut BTreeMap<(u16, u16), Cell>, area: Rect, at: (u16, u16), sides: u8, pen: Pen) {
    if at.0 < area.x || at.0 >= area.right() || at.1 < area.y || at.1 >= area.bottom() {
        return;
    }
    let cell = cells.entry(at).or_insert(Cell {
        sides: 0,
        stroke: pen.stroke,
        lit: false,
        faint: true,
    });
    cell.sides |= sides;
    cell.stroke = cell.stroke.min(pen.stroke);
    cell.lit |= pen.lit;
    cell.faint &= pen.faint;
}

/// A run of line along row `y`, from `x1` to `x2`, either way round.
fn along(cells: &mut BTreeMap<(u16, u16), Cell>, area: Rect, x1: u16, x2: u16, y: u16, pen: Pen) {
    let (a, b) = (x1.min(x2), x1.max(x2));
    for x in a..=b {
        let sides = if a == b {
            0
        } else if x == a {
            E
        } else if x == b {
            W
        } else {
            E | W
        };
        mark(cells, area, (x, y), sides, pen);
    }
}

/// The box-drawing character for a cell a line leaves by `sides`, in
/// `stroke`. A straight run is dashed or dotted; a corner or a join is
/// always drawn whole, which is the only way those characters come.
fn glyph(sides: u8, stroke: Stroke) -> &'static str {
    const UP_DOWN: u8 = N | S;
    const DOWN_RIGHT: u8 = S | E;
    const DOWN_LEFT: u8 = S | W;
    const UP_RIGHT: u8 = N | E;
    const UP_LEFT: u8 = N | W;
    const TEE_RIGHT: u8 = N | S | E;
    const TEE_LEFT: u8 = N | S | W;
    const TEE_DOWN: u8 = E | W | S;
    const TEE_UP: u8 = E | W | N;
    const CROSS: u8 = N | E | S | W;
    match sides {
        UP_DOWN | N | S => match stroke {
            Stroke::Solid => "│",
            Stroke::Dashed => "╎",
            Stroke::Dotted => "┆",
        },
        DOWN_RIGHT => "╭",
        DOWN_LEFT => "╮",
        UP_RIGHT => "╰",
        UP_LEFT => "╯",
        TEE_RIGHT => "├",
        TEE_LEFT => "┤",
        TEE_DOWN => "┬",
        TEE_UP => "┴",
        CROSS => "┼",
        // Along a row, or one end of such a run.
        _ => match stroke {
            Stroke::Solid => "─",
            Stroke::Dashed => "╌",
            Stroke::Dotted => "┄",
        },
    }
}

/// A chip's style: the origin filled with the accent, as the dashboard fills
/// what is chosen; one the walk reached on the accent's tint; any other on
/// the raised surface, in the secondary ink.
fn chip(t: &Theme, chip: Chip) -> Style {
    match chip {
        Chip::Origin => t.chip_primary(false).add_modifier(Modifier::BOLD),
        Chip::Reached => Style::default().fg(t.ink).bg(t.accent_soft),
        Chip::Plain => Style::default().fg(t.ink2).bg(t.raised),
    }
}

/// The key to the three strokes, for the row under a picture.
pub(super) fn key<'a>(t: &Theme, names: [&'a str; 3]) -> Vec<Span<'a>> {
    let mark = |m: &'static str| Span::styled(m, Style::default().fg(t.ink3));
    let word = |w: &'a str| Span::styled(w, Style::default().fg(t.ink2));
    vec![
        mark(" ── "),
        word(names[0]),
        mark("   ╌╌ "),
        word(names[1]),
        mark("   ┄┄ "),
        word(names[2]),
    ]
}

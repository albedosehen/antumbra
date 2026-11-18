//! The modal overlays drawn over the dimmed live view: the keybinding help, the
//! command palette, and the focused-list filter.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, Mode, Page};
use crate::events::EventKind;
use crate::keys::{self, Binding, Group};
use crate::overlay;
use crate::scroll;
use crate::theme::{Theme, Tone};
use crate::words;

use super::{gauge_spans, heading};

/// The modal box rectangle for the active overlay (the size source the open
/// animation also targets). `None` in the live view.
pub fn overlay_area(app: &App, frame: Rect) -> Option<Rect> {
    // Above the notice and the footer, which stay in view under an overlay.
    let frame = super::frame_rows(frame).stage();
    match app.mode {
        Mode::Help => {
            let rows = u16::try_from(help_rows(app).len()).unwrap_or(u16::MAX);
            Some(overlay::centered(frame, 76, rows.saturating_add(2)))
        }
        Mode::Palette => {
            let listed = app.palette_matches().len().max(1) as u16;
            Some(overlay::centered(frame, 62, listed + 4))
        }
        Mode::Filter => {
            let listed = (app.filtered().len() as u16 + 4).min(18);
            Some(overlay::centered(frame, 50, listed))
        }
        Mode::Events => {
            let listed = (app.events.len() as u16 + 2).clamp(6, 24);
            Some(overlay::centered(frame, 64, listed))
        }
        Mode::Detail => Some(overlay::centered(frame, 74, 30)),
        Mode::Gate => Some(overlay::centered(frame, 72, 24)),
        Mode::Connect => Some(overlay::centered(frame, 78, 27)),
        Mode::Ask => {
            // The task's two rows, what it came to, and the border.
            let rows = u16::try_from(ask_lines(app, &app.theme()).len()).unwrap_or(u16::MAX);
            Some(overlay::centered(frame, ASK_WIDTH, rows.saturating_add(4)))
        }
        // The question, what going ahead does, a gap and the two keys.
        Mode::Confirm => Some(overlay::centered(frame, 70, 6)),
        Mode::Knobs => Some(overlay::centered(
            frame,
            86,
            antumbra_core::usage::KNOBS.len() as u16 + 8,
        )),
        Mode::Normal => None,
    }
}

/// The first-run card: shown over a workspace that holds nothing yet, so a
/// fresh install says what will appear and how, in place of empty panels.
/// Drawn in the live view (not a modal), so keys still work underneath. It
/// names only what exists: the connect panel, the help, and the demo.
pub(super) fn first_run_hint(f: &mut Frame, app: &App) {
    let t = app.theme();
    let stage = super::frame_rows(f.area()).stage();
    let area = overlay::centered(stage, 68, 13);
    let inner = overlay::modal(f, &t, area, "Welcome");
    let dim = Style::default().fg(t.ink2);
    let strong = Style::default().fg(t.ink).add_modifier(Modifier::BOLD);
    let key = |chip: &'static str, does: &'static str| {
        Line::from(vec![
            Span::raw("  "),
            Span::styled(format!(" {chip} "), t.chip()),
            Span::styled(format!("  {does}"), dim),
        ])
    };
    let lines = vec![
        Line::from(Span::styled("  Nothing is here yet.", strong)),
        Line::from(""),
        Line::from(Span::styled(
            "  This console shows what your agents store, and the experts",
            dim,
        )),
        Line::from(Span::styled(
            "  trained from it. Once an agent is connected and has worked,",
            dim,
        )),
        Line::from(Span::styled(
            "  what it stored appears on the Memory page.",
            dim,
        )),
        Line::from(""),
        key("c", "how to connect an agent"),
        key("?", "every key, and what each word means"),
        Line::from(""),
        Line::from(vec![
            Span::styled("  To look around first, quit and run ", dim),
            Span::styled("antumbra tui --demo", strong),
        ]),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// The connect panel (`c`): how to connect an agent to Antumbra, which server
/// this console reads, and what its token may do, so connecting is a step in
/// the console and not a hunt through the README.
pub(super) fn connect_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "Connect an agent");
    let dim = Style::default().fg(t.ink2);
    let val = Style::default().fg(t.ink);
    let step = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let role = if app.demo {
        "This is the demo: sample data held in memory. Nothing here is saved."
    } else if app.operator {
        "Your token is an operator's: it may act here (freeze, prune, halt)."
    } else {
        "Your token is read-only: it reads your workspace and acts on nothing."
    };
    let cap = if app.auto_fps {
        "following the monitor"
    } else {
        "pinned"
    };
    let numbered = |n: &'static str, text: &'static str| {
        Line::from(vec![
            Span::styled(format!("{n}  "), step),
            Span::styled(text, dim),
        ])
    };
    let command = |text: &'static str| Line::from(Span::styled(format!("     {text}"), val));
    let mut lines = vec![
        Line::from(Span::styled(
            "Connect a coding agent (Claude Code) to Antumbra.",
            val,
        )),
        Line::from(""),
        numbered("1", "Set it up in one command:"),
        command("antumbra setup local           a server on this machine"),
        command("antumbra setup hosted <url>    a server you were given"),
        Line::from(""),
        numbered("2", "Check what is in place:"),
        command("antumbra setup check"),
        Line::from(""),
        numbered("3", "This console reads the same server, with no flags:"),
        Line::from(Span::styled(format!("     {}", app.source), val)),
    ];
    if !app.source_from.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(
                "     server from {}, token from {}",
                app.source_from, app.token_from
            ),
            dim,
        )));
    }
    lines.push(Line::from(Span::styled(format!("     {role}"), dim)));
    if !app.operator && !app.demo {
        lines.push(Line::from(Span::styled(
            "     An operator's: antumbra-mcp --mint-token --operator",
            dim,
        )));
    }
    lines.extend([
        Line::from(""),
        Line::from(Span::styled(
            "The full walkthrough is docs/agent-setup.md.",
            dim,
        )),
        Line::from(Span::styled(
            format!(
                "Showing {} memories and {} experts, drawn at {} fps{} (cap {}, {cap}).",
                app.memories.len(),
                app.experts.len(),
                app.shown_fps(),
                if app.idle { ", idle" } else { "" },
                app.target_fps,
            ),
            dim,
        )),
    ]);
    // Clamp the scroll so paging never runs off the end of the guide.
    let max = u16::try_from(lines.len().saturating_sub(usize::from(inner.height))).unwrap_or(0);
    let off = app.detail_scroll.min(max);
    f.render_widget(Paragraph::new(lines).scroll((off, 0)), inner);
}

/// The operator-action confirmation (`x`): a yes/no prompt before a store
/// mutation (prune a shadow, delete a boundary).
pub(super) fn confirm_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "Confirm");
    let prompt = app.pending_prompt().unwrap_or_default();
    // Going ahead with something that removes is the one chip in the danger
    // colour; going ahead with anything else is the accent.
    let removes = app.pending.as_ref().is_some_and(|p| p.removes());
    let lines = vec![
        Line::from(Span::styled(
            format!("  {prompt}"),
            Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
        )),
        // What going ahead does, under the question.
        Line::from(Span::styled(
            format!("  {}", app.pending_effect().unwrap_or_default()),
            Style::default().fg(t.ink2),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(" y ", t.chip_primary(removes)),
            Span::styled("  confirm     ", Style::default().fg(t.ink2)),
            Span::styled(" n ", t.chip()),
            Span::styled("  cancel", Style::default().fg(t.ink2)),
        ]),
    ];
    // Untrimmed, so the prompt keeps its indent and a chip its padding.
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// The ask's width, and the room for text inside its border and indent.
const ASK_WIDTH: u16 = 76;
const ASK_ROOM: usize = 70;

/// `text` as lines of at most `width` columns, broken between words.
fn wrapped(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// What the ask shows under the task: what routing it came to, then the
/// standing experts that shape every answer here whatever the task.
///
/// The two are different things and a person asking needs both. Routing
/// answers "which expert would be sent this task", and when the answer is
/// none it says why, in the server's words. A standing expert is never sent a
/// task, so a workspace that has only those is told "none" for every task;
/// which of them shape an answer here is the half of the answer it does have.
fn ask_lines<'a>(app: &App, t: &Theme) -> Vec<Line<'a>> {
    let dim = Style::default().fg(t.ink2);
    let mut lines: Vec<Line> = Vec::new();
    match &app.ask_result {
        None => lines.push(Line::from(Span::styled(
            "  Type a task and press Enter to see which expert it would be sent to.",
            Style::default().fg(t.ink3),
        ))),
        Some(routes) if routes.is_empty() => {
            lines.push(Line::from(Span::styled(
                "  No expert is routed to it: it goes to your agent.",
                Style::default().fg(t.ink),
            )));
            if let Some(reason) = &app.ask_reason {
                for (i, part) in wrapped(reason, ASK_ROOM - 7).into_iter().enumerate() {
                    let lead = if i == 0 { "  Why: " } else { "       " };
                    lines.push(Line::from(Span::styled(format!("{lead}{part}"), dim)));
                }
            }
        }
        Some(routes) => {
            for (name, p) in routes.iter().take(5) {
                let mut spans = vec![Span::styled(
                    format!("  {:<30}", super::clip(name, 29)),
                    Style::default().fg(t.ink),
                )];
                spans.extend(gauge_spans(t, *p, 10, t.accent));
                spans.push(Span::styled(
                    format!(" {:>3.0}%", p * 100.0),
                    Style::default().fg(t.ink),
                ));
                lines.push(Line::from(spans));
            }
        }
    }

    let standing = app.standing_count();
    if standing > 0 {
        let here = app.standing_here();
        let place = match app.git.repo.as_deref() {
            Some(repo) => super::clip(words::scope_name(repo), 26),
            None => "no repository".to_string(),
        };
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("  Standing experts, in every answer here ({place}):"),
            dim,
        )));
        for e in &here {
            // The scope whole, beside its name in brief; the one for
            // everywhere has no more to its scope than its name.
            let scope = match e.standing_scope() {
                Some(antumbra_core::behavior::EVERYWHERE) => "every repository",
                Some(scope) => scope,
                None => "",
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", Tone::Settled.glyph()),
                    Style::default().fg(t.accent),
                ),
                Span::styled(
                    format!("{:<28}", super::clip(&words::expert_name(e), 27)),
                    Style::default().fg(t.ink),
                ),
                Span::styled(
                    super::clip(scope, ASK_ROOM - 34),
                    Style::default().fg(t.ink3),
                ),
            ]));
        }
        if here.is_empty() {
            lines.push(Line::from(Span::styled(
                "  none of them is for this place",
                Style::default().fg(t.ink3),
            )));
        }
        let elsewhere = standing - here.len();
        if elsewhere > 0 {
            lines.push(Line::from(Span::styled(
                format!("  and {elsewhere} more, for other repositories"),
                Style::default().fg(t.ink3),
            )));
        }
    }
    lines
}

/// The ask (`a`): a task, routed as the server routes one, with what it came
/// to under it ([`ask_lines`]).
pub(super) fn ask_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let title = if app.real_embedder {
        "Ask · which expert takes a task?".to_string()
    } else {
        "Ask · which expert takes a task? · demo".to_string()
    };
    let inner = overlay::modal(f, &t, area, &title);
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    let mut head = vec![Line::from(vec![
        Span::styled("task › ", Style::default().fg(t.accent)),
        Span::styled(app.ask_query.clone(), Style::default().fg(t.ink)),
        Span::styled("▏", Style::default().fg(t.accent)),
    ])];
    if !app.real_embedder {
        head.push(Line::from(Span::styled(
            "  the demo matches by a stand-in; a server matches with its own model",
            Style::default().fg(t.ink2),
        )));
    }
    f.render_widget(Paragraph::new(head), rows[0]);
    f.render_widget(Paragraph::new(ask_lines(app, &t)), rows[1]);
}

/// A dim, bold section divider within a detail view.
pub(super) fn section<'a>(t: &Theme, label: &str) -> Line<'a> {
    Line::from(Span::styled(
        label.to_string(),
        Style::default().fg(t.ink2).add_modifier(Modifier::BOLD),
    ))
}

/// A JSON value pretty-printed into indented, value-colored lines.
pub(super) fn json_lines<'a>(t: &Theme, value: &serde_json::Value) -> Vec<Line<'a>> {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|_| value.to_string())
        .lines()
        .map(|s| Line::from(Span::styled(format!("  {s}"), Style::default().fg(t.ink))))
        .collect()
}

/// The gate (router) inspector (`gate` command): the learned per-dimension
/// weight profile, the gate scalars, and a self-routing health check: each
/// expert's own centroid routed through the gate should come back to itself.
pub(super) fn gate_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "Router");
    let w = inner.width as usize;

    let Some(router) = app.router.as_ref() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "The router is not trained yet.",
                    Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Until it is, a task goes to the expert nearest to it. An operator's token is needed to read it once trained.",
                    Style::default().fg(t.ink3),
                )),
            ])
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };

    let mut l = vec![
        heading(&t, format!("Trained on {} experts", router.experts.len())),
        Line::from(vec![
            Span::styled("sharpness  ", Style::default().fg(t.ink2)),
            Span::styled(
                format!("{:.2}", router.temperature),
                Style::default().fg(t.ink),
            ),
            Span::styled(
                "   lower prefers the best match more strongly",
                Style::default().fg(t.ink3),
            ),
        ]),
        Line::from(vec![
            Span::styled("hands on   ", Style::default().fg(t.ink2)),
            Span::styled(format!("{:.2}", router.floor), Style::default().fg(t.ink)),
            Span::styled(
                "   below this match, no expert takes the task",
                Style::default().fg(t.ink3),
            ),
        ]),
        Line::from(""),
        section(&t, "What it weighs · learned emphasis across the embedding"),
    ];

    // The weight profile as a shade strip across the inner width.
    if router.weights.is_empty() {
        l.push(Line::from(Span::styled(
            "  every part alike: nothing learned yet",
            Style::default().fg(t.ink3),
        )));
    } else {
        let n = router.weights.len();
        let maxw = router
            .weights
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max)
            .max(1e-6);
        let spans: Vec<Span> = (0..w.saturating_sub(2).max(1))
            .map(|c| {
                let cols = w.saturating_sub(2).max(1);
                let idx = if cols <= 1 {
                    0
                } else {
                    c * (n - 1) / (cols - 1)
                };
                let ratio = (router.weights[idx.min(n - 1)] / maxw).clamp(0.0, 1.0);
                Span::styled(
                    super::heatmap::heat_cell(ratio).to_string(),
                    Style::default().fg(t.fitness(ratio, 1.0)),
                )
            })
            .collect();
        let mut row = vec![Span::raw("  ")];
        row.extend(spans);
        l.push(Line::from(row));
    }

    l.push(Line::from(""));
    l.push(section(
        &t,
        "Self-check · each expert's own centre should route back to it",
    ));
    if router.experts.is_empty() {
        l.push(Line::from(Span::styled(
            "  no experts to route to",
            Style::default().fg(t.ink3),
        )));
    }
    for e in &router.experts {
        let routed = router.route(&e.centroid);
        let (top_id, prob) = routed
            .first()
            .map(|(id, p)| (id.as_str().to_string(), *p))
            .unwrap_or_else(|| ("none".to_string(), 0.0));
        let healthy = top_id == e.id.as_str();
        let name =
            e.id.as_str()
                .strip_prefix("expert:")
                .unwrap_or(e.id.as_str());
        let top = top_id.strip_prefix("expert:").unwrap_or(&top_id);
        // Routed back to itself is settled; routed to another, failed.
        let tone = if healthy { Tone::Settled } else { Tone::Failed };
        let (mark, color) = (tone.glyph(), t.tone(tone));
        l.push(Line::from(vec![
            Span::styled(format!("  {name:<22}→ "), Style::default().fg(t.ink2)),
            Span::styled(format!("{top:<22}"), Style::default().fg(t.ink)),
            Span::styled(
                format!("{:>3.0}%  ", prob * 100.0),
                Style::default().fg(t.ink),
            ),
            Span::styled(
                mark,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    f.render_widget(
        Paragraph::new(l)
            .scroll((app.detail_scroll, 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

/// The drill-down detail of the focused selection (Enter): everything the summary
/// panels omit: full capability card, reward curve, and contrastive contexts.
pub(super) fn detail_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let (title, lines) = match app.page {
        Page::Evals => super::detail::eval_detail(app, &t),
        Page::Loop => super::detail::loop_detail(app, &t),
        _ => match app.focus {
            Focus::Experts => super::detail::expert_detail(app, &t),
            Focus::Shadows => super::detail::shadow_detail(app, &t),
            Focus::Boundaries => super::detail::boundary_detail(app, &t),
        },
    };
    let inner = overlay::modal(f, &t, area, &title);
    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.detail_scroll, 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

/// The theme color an event's glyph renders in, by kind: the accent for what
/// joined the population, strengths of the ink for the rest. The glyph is
/// what tells the kinds apart.
fn event_color(t: &Theme, kind: EventKind) -> Color {
    match kind {
        EventKind::Graduate => t.accent,
        EventKind::Spawn | EventKind::Freeze | EventKind::Boundary => t.ink2,
        EventKind::Prune | EventKind::System => t.ink3,
        EventKind::Problem => t.danger,
    }
}

/// The live event stream (`e`): store changes newest-first, scrollable, the loop
/// keeps reloading so it fills while it's open.
pub(super) fn events_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, &format!("History · {}", app.events.len()));
    if app.events.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Nothing has changed since the console opened.",
                Style::default().fg(t.ink3),
            )),
            inner,
        );
        return;
    }
    let now = app.clock_ms;
    let lines: Vec<Line> = app
        .events
        .iter()
        .map(|e| {
            let color = event_color(&t, e.kind);
            let ago = (((now - e.at_ms) / 1000.0).max(0.0)) as u64;
            Line::from(vec![
                Span::styled(format!("{} ", e.kind.glyph()), Style::default().fg(color)),
                Span::styled(e.text.clone(), Style::default().fg(t.ink2)),
                Span::styled(format!("  {ago}s ago"), Style::default().fg(t.ink3)),
            ])
        })
        .collect();
    scroll::list(
        f,
        &t,
        inner,
        Block::default(),
        lines,
        Some(app.events_scroll),
    );
}

/// One row of the help: a group's heading, a binding, or the gap between two
/// groups.
pub(super) enum HelpRow {
    Heading(&'static str),
    Key {
        /// The keys, the footer's chip first and then any others.
        keys: String,
        does: &'static str,
    },
    Gap,
}

/// The help's rows for the page the console is on: every binding of the main
/// view there ([`keys::page_bindings`]), grouped. The keys take effect from
/// the same list, so the help cannot name a key that does something else, or
/// leave one out.
pub(super) fn help_rows(app: &App) -> Vec<HelpRow> {
    let bindings = keys::page_bindings(app);
    let mut rows = Vec::new();
    for group in Group::ALL {
        let listed: Vec<&Binding> = bindings.iter().filter(|b| b.group == group).collect();
        if listed.is_empty() {
            continue;
        }
        if !rows.is_empty() {
            rows.push(HelpRow::Gap);
        }
        rows.push(HelpRow::Heading(group.title()));
        for b in listed {
            let keys = if b.also.is_empty() {
                b.chip.to_string()
            } else {
                format!("{}  {}", b.chip, b.also)
            };
            rows.push(HelpRow::Key { keys, does: b.does });
        }
    }
    rows.push(HelpRow::Gap);
    rows.push(HelpRow::Heading("in a panel"));
    rows.push(HelpRow::Key {
        keys: "esc".into(),
        does: "close it; the footer shows its other keys",
    });
    // Then what the words mean: the same list for every page, so a word met
    // anywhere is found here.
    rows.push(HelpRow::Gap);
    rows.push(HelpRow::Heading("words"));
    for word in words::GLOSSARY {
        for (i, line) in word.means.iter().enumerate() {
            rows.push(HelpRow::Key {
                keys: if i == 0 {
                    word.term.to_string()
                } else {
                    String::new()
                },
                does: line,
            });
        }
    }
    rows
}

/// The key reference (`?`), a centered modal over the live view, for the page
/// it was opened on. It scrolls where it does not fit.
pub(super) fn help_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, &format!("Help · {}", app.page.title()));
    let lines: Vec<Line> = help_rows(app)
        .into_iter()
        .map(|row| match row {
            HelpRow::Heading(title) => Line::from(Span::styled(
                title,
                Style::default().fg(t.ink2).add_modifier(Modifier::BOLD),
            )),
            HelpRow::Key { keys, does } => Line::from(vec![
                Span::styled(
                    format!("  {keys:<15}"),
                    Style::default().fg(t.ink).add_modifier(Modifier::BOLD),
                ),
                Span::styled(does, Style::default().fg(t.ink2)),
            ]),
            HelpRow::Gap => Line::from(""),
        })
        .collect();
    // Clamp the scroll so paging never runs off the end.
    let max = u16::try_from(lines.len().saturating_sub(usize::from(inner.height))).unwrap_or(0);
    f.render_widget(
        Paragraph::new(lines).scroll((app.detail_scroll.min(max), 0)),
        inner,
    );
}

/// The command palette: a query line over a fuzzy-ranked command list (`:` opens).
pub(super) fn palette_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let matches = app.palette_matches();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "Commands");
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    // Query line with a block cursor.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", Style::default().fg(t.accent)),
            Span::styled(app.palette.query.clone(), Style::default().fg(t.ink)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    // Ranked matches as a list, the selection full-row highlighted.
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  No command matches.",
                Style::default().fg(t.ink3),
            )),
            rows[1],
        );
    } else {
        // Each command with the key that does the same from the main view at
        // its right, so the palette teaches the keys as it is used.
        let room = usize::from(rows[1].width).saturating_sub(6);
        let lines: Vec<Line> = matches
            .iter()
            .map(|c| {
                Line::from(vec![
                    Span::styled(format!("{:<room$}", c.label), Style::default().fg(t.ink2)),
                    Span::styled(c.key, Style::default().fg(t.ink3)),
                ])
            })
            .collect();
        scroll::list(
            f,
            &t,
            rows[1],
            Block::default(),
            lines,
            Some(app.palette.selected),
        );
    }
}

/// The focused-list filter (`/`): a query over the focused region's items,
/// fuzzy-ranked; picking one jumps the selection to it.
pub(super) fn filter_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, &format!("find in {}", app.filter_subject()));
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("/ ", Style::default().fg(t.accent)),
            Span::styled(app.filter.clone(), Style::default().fg(t.ink)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    let matches = app.filtered();
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  Nothing matches.",
                Style::default().fg(t.ink3),
            )),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|(_, label)| Line::from(Span::styled(label.clone(), Style::default().fg(t.ink2))))
            .collect();
        scroll::list(
            f,
            &t,
            rows[1],
            Block::default(),
            lines,
            Some(app.filter_selected),
        );
    }
}

//! The modal overlays drawn over the dimmed live view: the keybinding help, the
//! command palette, and the focused-list filter.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, Mode, Page};
use crate::events::EventKind;
use crate::overlay;
use crate::scroll;
use crate::theme::Theme;

use super::{gauge_spans, heading, kv};

/// The modal box rectangle for the active overlay (the size source the open
/// animation also targets). `None` in the live view.
pub fn overlay_area(app: &App, frame: Rect) -> Option<Rect> {
    match app.mode {
        Mode::Help => Some(overlay::centered(frame, 56, 28)),
        Mode::Palette => {
            let listed = app.palette_matches().len().max(1) as u16;
            Some(overlay::centered(frame, 56, listed + 4))
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
            let rows = app
                .ask_result
                .as_ref()
                .map_or(1, |r| r.len().clamp(1, 5) as u16);
            Some(overlay::centered(frame, 60, rows + 5))
        }
        Mode::Confirm => Some(overlay::centered(frame, 58, 6)),
        Mode::Normal => None,
    }
}

/// The first-run hint: a centered card shown over an empty store so a fresh
/// install explains how to populate itself instead of looking inert. Drawn in
/// the live view (not a modal), so keys still work underneath.
pub(super) fn first_run_hint(f: &mut Frame, app: &App) {
    let t = app.theme();
    let area = overlay::centered(f.area(), 64, 11);
    let inner = overlay::modal(f, &t, area, "antumbra");
    let dim = Style::default().fg(t.dim);
    let key = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let chip = Style::default().fg(Color::Black).bg(t.ink);
    let lines = vec![
        Line::from(Span::styled(
            "  The store is empty: no experts, memory, or loop runs yet.",
            Style::default().fg(t.value),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Populate it from another terminal, then press r to reload:",
            dim,
        )),
        Line::from(vec![
            Span::styled("    antumbra seed", key),
            Span::styled("                 seed demo specialists", dim),
        ]),
        Line::from(vec![
            Span::styled("    antumbra loop --generations 3", key),
            Span::styled("  grow + boundaries", dim),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Or relaunch with ", dim),
            Span::styled("antumbra-tui --demo", key),
            Span::styled(" for a throwaway population.", dim),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" ? ", chip),
            Span::styled(" keys    ", dim),
            Span::styled(" : ", chip),
            Span::styled(" commands    ", dim),
            Span::styled(" q ", chip),
            Span::styled(" quit", dim),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

/// The "connect your agent" panel (`c`): a guided, copyable reference for wiring
/// the MCP lifecycle hooks and minting a hook token, so connecting an agent is an
/// in-app step instead of a hunt through the README.
pub(super) fn connect_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "connect · wire your agent");
    let dim = Style::default().fg(t.dim);
    let val = Style::default().fg(t.value);
    let cmd = Style::default().fg(t.accent);
    let step = Style::default().fg(t.accent).add_modifier(Modifier::BOLD);
    let url = if app.store_url.is_empty() {
        "<your --url>"
    } else {
        app.store_url.as_str()
    };
    let line = |spans: Vec<Span<'static>>| Line::from(spans);
    let lines = vec![
        Line::from(Span::styled(
            "Plug a coding agent (Claude Code / Cursor) into Antumbra over MCP.",
            val,
        )),
        Line::from(""),
        line(vec![
            Span::styled("1  ", step),
            Span::styled("Serve this store over MCP (HTTP):", dim),
        ]),
        Line::from(Span::styled(
            "     antumbra-mcp --http 127.0.0.1:8081 \\".to_string(),
            cmd,
        )),
        Line::from(Span::styled(format!("       --url {url}"), cmd)),
        Line::from(""),
        line(vec![
            Span::styled("2  ", step),
            Span::styled("Mint a long-lived hook token:", dim),
        ]),
        Line::from(Span::styled(
            "     antumbra-mcp --mint-token --tenant <ws> --user <you> \\".to_string(),
            cmd,
        )),
        Line::from(Span::styled(
            "       --jwt-secret <secret> --token-ttl-days 365".to_string(),
            cmd,
        )),
        Line::from(""),
        line(vec![
            Span::styled("3  ", step),
            Span::styled("Set the hook environment:", dim),
        ]),
        Line::from(Span::styled(
            "     ANTUMBRA_URL=http://127.0.0.1:8081   ANTUMBRA_WORKSPACE_ID=<ws>".to_string(),
            val,
        )),
        Line::from(Span::styled(
            "     ANTUMBRA_TOKEN=<step 2>   ANTUMBRA_HOST_ID=<this device>".to_string(),
            val,
        )),
        Line::from(""),
        line(vec![
            Span::styled("4  ", step),
            Span::styled("Add the hooks to your agent's settings.json:", dim),
        ]),
        Line::from(Span::styled(
            "     macOS / Linux   bash ./scripts/hooks/<name>.sh".to_string(),
            cmd,
        )),
        Line::from(Span::styled(
            "     Windows         pwsh -File ./scripts/hooks/<name>.ps1".to_string(),
            cmd,
        )),
        Line::from(Span::styled(
            "     SessionStart=bootstrap  Stop=capture  PreToolUse=strip".to_string(),
            dim,
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Full walkthrough: scripts/hooks/README.md".to_string(),
            dim,
        )),
        Line::from(Span::styled(
            format!(
                "store: {} memories, {} experts          q/esc to close",
                app.memories.len(),
                app.experts.len()
            ),
            dim,
        )),
    ];
    // Clamp the scroll so paging never runs off the end of the guide.
    let max = lines.len().saturating_sub(inner.height as usize) as u16;
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
    let inner = overlay::modal(f, &t, area, "confirm");
    let prompt = app.pending_prompt().unwrap_or_default();
    let lines = vec![
        Line::from(Span::styled(
            format!("  {prompt}"),
            Style::default().fg(t.alert),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(" y ", Style::default().fg(Color::Black).bg(t.alert)),
            Span::styled("  confirm     ", Style::default().fg(t.ink)),
            Span::styled(" n ", Style::default().fg(Color::Black).bg(t.ink)),
            Span::styled("  cancel", Style::default().fg(t.ink)),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

/// The route-ask (`ask` command): a task query embedded and run through the
/// learned gate, the resulting routing distribution shown as probability gauges.
pub(super) fn ask_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let title = if app.real_embedder {
        "ask the gate".to_string()
    } else {
        "ask the gate · demo embedder".to_string()
    };
    let inner = overlay::modal(f, &t, area, &title);
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    let mut head = vec![Line::from(vec![
        Span::styled("ask › ", Style::default().fg(t.accent)),
        Span::styled(app.ask_query.clone(), Style::default().fg(t.text)),
        Span::styled("▏", Style::default().fg(t.accent)),
    ])];
    if !app.real_embedder {
        head.push(Line::from(Span::styled(
            "  byte-histogram · pass --embed-url to match your model",
            Style::default().fg(t.warning),
        )));
    }
    f.render_widget(Paragraph::new(head), rows[0]);

    let lines: Vec<Line> = match &app.ask_result {
        None => vec![Line::from(Span::styled(
            "  type a task, Enter to route it through the gate",
            Style::default().fg(t.dim),
        ))],
        Some(r) if r.is_empty() => vec![Line::from(Span::styled(
            "  escalates · out of distribution",
            Style::default().fg(t.warning),
        ))],
        Some(r) => r
            .iter()
            .take(5)
            .map(|(name, p)| {
                let mut spans = vec![Span::styled(
                    format!("  {name:<20}"),
                    Style::default().fg(t.ink),
                )];
                spans.extend(gauge_spans(&t, *p, 10, t.accent));
                spans.push(Span::styled(
                    format!(" {:>3.0}%", p * 100.0),
                    Style::default().fg(t.value),
                ));
                Line::from(spans)
            })
            .collect(),
    };
    f.render_widget(Paragraph::new(lines), rows[1]);
}

/// A dim, bold section divider within a detail view.
pub(super) fn section<'a>(t: &Theme, label: &str) -> Line<'a> {
    Line::from(Span::styled(
        label.to_string(),
        Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
    ))
}

/// A JSON value pretty-printed into indented, value-colored lines.
pub(super) fn json_lines<'a>(t: &Theme, value: &serde_json::Value) -> Vec<Line<'a>> {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|_| value.to_string())
        .lines()
        .map(|s| Line::from(Span::styled(format!("  {s}"), Style::default().fg(t.value))))
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
    let inner = overlay::modal(f, &t, area, "gate · router");
    let w = inner.width as usize;

    let Some(router) = app.router.as_ref() else {
        f.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "gate untrained",
                    Style::default().fg(t.warning).add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "routing falls back to heuristic KNN; no learned metric yet",
                    Style::default().fg(t.dim),
                )),
            ])
            .wrap(Wrap { trim: true }),
            inner,
        );
        return;
    };

    let mut l = vec![
        heading(
            &t,
            format!("learned gate · {} experts", router.experts.len()),
        ),
        Line::from(vec![
            Span::styled("temperature ", Style::default().fg(t.dim)),
            Span::styled(
                format!("{:.2}", router.temperature),
                Style::default().fg(t.value),
            ),
        ]),
        kv(&t, "floor", &format!("{:.2}", router.floor)),
        Line::from(""),
        section(&t, "metric weights  (learned per-dimension emphasis)"),
    ];

    // The weight profile as a shade strip across the inner width.
    if router.weights.is_empty() {
        l.push(Line::from(Span::styled(
            "  (uniform, no reweighting)",
            Style::default().fg(t.dim),
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
        "self-routing  (each centroid routed → should pick itself)",
    ));
    if router.experts.is_empty() {
        l.push(Line::from(Span::styled(
            "  (no experts in the gate)",
            Style::default().fg(t.dim),
        )));
    }
    for e in &router.experts {
        let routed = router.route(&e.centroid);
        let (top_id, prob) = routed
            .first()
            .map(|(id, p)| (id.as_str().to_string(), *p))
            .unwrap_or_else(|| ("n/a".to_string(), 0.0));
        let healthy = top_id == e.id.as_str();
        let name =
            e.id.as_str()
                .strip_prefix("expert:")
                .unwrap_or(e.id.as_str());
        let top = top_id.strip_prefix("expert:").unwrap_or(&top_id);
        let (mark, color) = if healthy {
            ("✓", t.success)
        } else {
            ("✗", t.alert)
        };
        l.push(Line::from(vec![
            Span::styled(format!("  {name:<22}→ "), Style::default().fg(t.dim)),
            Span::styled(format!("{top:<22}"), Style::default().fg(t.value)),
            Span::styled(
                format!("{:>3.0}%  ", prob * 100.0),
                Style::default().fg(t.value),
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

/// The theme color an event renders in, by kind.
fn event_color(t: &Theme, kind: EventKind) -> Color {
    match kind {
        EventKind::Spawn => t.accent,
        EventKind::Graduate => t.success,
        EventKind::Prune => t.alert,
        EventKind::Freeze => t.value,
        EventKind::Boundary => t.warning,
        EventKind::System => t.dim,
    }
}

/// The live event stream (`e`): store changes newest-first, scrollable, the loop
/// keeps reloading so it fills while it's open.
pub(super) fn events_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, &format!("events · {}", app.events.len()));
    if app.events.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  (no changes yet; training elsewhere will show here)",
                Style::default().fg(t.dim),
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
                Span::styled(e.text.clone(), Style::default().fg(t.ink)),
                Span::styled(format!("  {ago}s ago"), Style::default().fg(t.dim)),
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

/// The keybinding reference, a centered modal over the live view (`?` toggles).
pub(super) fn help_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "help");

    let group = |label: &str| {
        Line::from(Span::styled(
            label.to_string(),
            Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
        ))
    };
    let bind = |keys: &str, desc: &str| {
        Line::from(vec![
            Span::styled(
                format!("  {keys:<8}"),
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(desc.to_string(), Style::default().fg(t.ink)),
        ])
    };
    let lines = vec![
        group("navigate"),
        bind("↑↓ jk", "select in the focused list"),
        bind("enter", "drill into the selected item"),
        bind("g G", "jump to first / last  (home / end)"),
        bind("pgup/dn", "move by a page"),
        bind("tab", "switch focus: umbra / penumbra / antumbra"),
        Line::from(""),
        group("view"),
        bind("[ ] 1-4", "switch page: population/memory/loop/evals"),
        bind("l", "cycle layout: focused/dashboard/graph/table"),
        bind("s", "sort the population table (fitness/name/gen)"),
        bind("t", "cycle theme: shadow / ember / mono"),
        Line::from(""),
        group("frame rate"),
        bind("+ -", "pin the cap to a refresh rate"),
        bind("a", "follow the active monitor"),
        Line::from(""),
        group("command"),
        bind("/", "filter the focused list, jump to a match"),
        bind(":", "command palette (ask · route a task here)"),
        bind("e", "live event stream of store changes"),
        bind("c", "connect your agent: hooks + token"),
        bind("x", "prune/graduate · freeze · delete"),
        bind("r", "reload from the store"),
        bind("? esc", "close this help"),
        bind("q", "quit"),
    ];
    f.render_widget(Paragraph::new(lines), inner);
}

/// The command palette: a query line over a fuzzy-ranked command list (`:` opens).
pub(super) fn palette_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let matches = app.palette_matches();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "command");
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    // Query line with a block cursor.
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", Style::default().fg(t.accent)),
            Span::styled(app.palette.query.clone(), Style::default().fg(t.text)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    // Ranked matches as a list, the selection full-row highlighted.
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "  (no matching command)",
                Style::default().fg(t.dim),
            )),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|c| Line::from(Span::styled(c.label, Style::default().fg(t.ink))))
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
    let region = match app.focus {
        Focus::Experts => "umbra",
        Focus::Shadows => "penumbra",
        Focus::Boundaries => "antumbra",
    };
    let inner = overlay::modal(f, &t, area, &format!("filter {region}"));
    let rows = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(inner);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("/ ", Style::default().fg(t.accent)),
            Span::styled(app.filter.clone(), Style::default().fg(t.text)),
            Span::styled("▏", Style::default().fg(t.accent)),
        ])),
        rows[0],
    );

    let matches = app.filtered();
    if matches.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled("  (no match)", Style::default().fg(t.dim))),
            rows[1],
        );
    } else {
        let lines: Vec<Line> = matches
            .iter()
            .map(|(_, label)| Line::from(Span::styled(label.clone(), Style::default().fg(t.ink))))
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

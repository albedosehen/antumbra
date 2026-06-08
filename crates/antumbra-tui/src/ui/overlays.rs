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

use super::evals::{status_color, status_label};
use super::{gauge_row, gauge_spans, heading, kv, shadow_color, sparkline_row};

/// The modal box rectangle for the active overlay (the size source the open
/// animation also targets). `None` in the live view.
pub fn overlay_area(app: &App, frame: Rect) -> Option<Rect> {
    match app.mode {
        Mode::Help => Some(overlay::centered(frame, 56, 27)),
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
fn section<'a>(t: &Theme, label: &str) -> Line<'a> {
    Line::from(Span::styled(
        label.to_string(),
        Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
    ))
}

/// A JSON value pretty-printed into indented, value-coloured lines.
fn json_lines<'a>(t: &Theme, value: &serde_json::Value) -> Vec<Line<'a>> {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|_| value.to_string())
        .lines()
        .map(|s| Line::from(Span::styled(format!("  {s}"), Style::default().fg(t.value))))
        .collect()
}

/// The drill-down detail of the focused selection (Enter): everything the summary
/// panels omit — full capability card, reward curve, and contrastive contexts.
pub(super) fn detail_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let (title, lines) = if app.page == Page::Evals {
        eval_detail(app, &t)
    } else {
        match app.focus {
            Focus::Experts => expert_detail(app, &t),
            Focus::Shadows => shadow_detail(app, &t),
            Focus::Boundaries => boundary_detail(app, &t),
        }
    };
    let inner = overlay::modal(f, &t, area, &title);
    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.detail_scroll, 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

/// The evaluation drill-down: the run's fields, the regression comparison
/// against the previous run for the same subject (the ADR-0001 no-forgetting
/// tripwire), and the subject's run history — all from the loaded set.
fn eval_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
    let Some(run) = app.evals.get(app.selected_eval) else {
        return (
            "evaluation".into(),
            vec![Line::from(Span::styled(
                "(no run selected)",
                Style::default().fg(t.dim),
            ))],
        );
    };
    let group = |label: String| {
        Line::from(Span::styled(
            label,
            Style::default().fg(t.dim).add_modifier(Modifier::BOLD),
        ))
    };
    let short = |fp: Option<&str>| {
        fp.map(|s| s.chars().take(12).collect::<String>())
            .unwrap_or_else(|| "—".to_string())
    };

    let mut l = vec![heading(t, run.subject_id.clone())];
    l.push(kv(t, "kind", run.subject_kind.as_str()));
    l.push(kv(t, "task", &run.corpus_task_id));
    l.push(Line::from(vec![
        Span::styled(format!("{:<11}", "status"), Style::default().fg(t.dim)),
        Span::styled(
            status_label(run.status),
            Style::default()
                .fg(status_color(t, run.status))
                .add_modifier(Modifier::BOLD),
        ),
    ]));
    l.push(Line::from(vec![
        Span::styled("fingerprint ", Style::default().fg(t.dim)),
        Span::styled(
            run.regression_fingerprint
                .as_deref()
                .unwrap_or("—")
                .to_string(),
            Style::default().fg(t.value),
        ),
    ]));
    if let Some(m) = &run.metrics {
        l.push(kv(t, "metrics", &m.to_string()));
    }

    // Compare this run's fingerprint to the previous run for the same subject.
    let history = app.eval_history(run);
    let prev = history
        .iter()
        .position(|r| r.run_id.as_str() == run.run_id.as_str())
        .and_then(|i| history.get(i + 1))
        .copied();
    l.push(Line::from(""));
    l.push(group(
        "regression  (vs previous run for this subject)".into(),
    ));
    match prev {
        None => l.push(Line::from(Span::styled(
            "  no prior run to compare",
            Style::default().fg(t.dim),
        ))),
        Some(p) => {
            let (verdict, color) = match (&run.regression_fingerprint, &p.regression_fingerprint) {
                (Some(_), Some(_)) if run.fingerprint_matches(p) => ("stable", t.success),
                (Some(_), Some(_)) => ("DRIFTED — no-forgetting tripwire", t.alert),
                _ => ("n/a — no fingerprint", t.dim),
            };
            l.push(Line::from(vec![
                Span::styled("  this      ", Style::default().fg(t.dim)),
                Span::styled(
                    short(run.regression_fingerprint.as_deref()),
                    Style::default().fg(t.value),
                ),
            ]));
            l.push(Line::from(vec![
                Span::styled("  previous  ", Style::default().fg(t.dim)),
                Span::styled(
                    short(p.regression_fingerprint.as_deref()),
                    Style::default().fg(t.value),
                ),
                Span::styled(
                    format!("   → {verdict}"),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]));
        }
    }

    // The subject's run history (newest first).
    l.push(Line::from(""));
    l.push(group(format!("history  ·  {} runs", history.len())));
    for r in &history {
        let marker = if r.run_id.as_str() == run.run_id.as_str() {
            "▸"
        } else {
            " "
        };
        l.push(Line::from(vec![
            Span::styled(format!(" {marker} "), Style::default().fg(t.accent)),
            Span::styled(
                format!("{:<8}", status_label(r.status)),
                Style::default().fg(status_color(t, r.status)),
            ),
            Span::styled(
                format!("{:<14}", short(r.regression_fingerprint.as_deref())),
                Style::default().fg(t.value),
            ),
            Span::styled(r.corpus_task_id.clone(), Style::default().fg(t.dim)),
        ]));
    }

    (format!("evaluation · {}", run.subject_id), l)
}

fn expert_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
    let Some(e) = app.selected_expert() else {
        return (
            "expert".into(),
            vec![Line::from(Span::styled(
                "(no expert selected)",
                Style::default().fg(t.dim),
            ))],
        );
    };
    let mut l = vec![heading(t, e.name.clone())];
    l.push(gauge_row(
        t,
        "fitness",
        e.fitness,
        t.fitness(e.fitness, 1.0),
    ));
    l.push(kv(t, "frozen", if e.is_frozen() { "yes" } else { "no" }));
    l.push(kv(t, "generation", &e.generation.0.to_string()));
    l.push(kv(t, "base", &e.base_model));
    l.push(kv(t, "artifact", &e.artifact_uri));
    if e.owner.is_some() || e.compartment.is_some() {
        l.push(kv(t, "owner", &format!("{:?}", e.owner)));
        l.push(kv(t, "compartment", &format!("{:?}", e.compartment)));
    }
    l.push(kv(t, "created", &e.created_at.to_rfc3339()));
    l.push(Line::from(""));
    l.push(section(t, "capability card"));
    if let Some(desc) = e
        .capability_card
        .get("description")
        .and_then(|v| v.as_str())
    {
        l.push(Line::from(Span::styled(
            format!("  {desc}"),
            Style::default().fg(t.ink),
        )));
    }
    if let Some(ex) = e
        .capability_card
        .get("exemplars")
        .and_then(|v| v.as_array())
    {
        l.push(Line::from(Span::styled(
            format!("  {} exemplars", ex.len()),
            Style::default().fg(t.dim),
        )));
        for (i, x) in ex.iter().enumerate() {
            let s = x.as_str().map_or_else(|| x.to_string(), str::to_string);
            l.push(Line::from(Span::styled(
                format!("    {}. {s}", i + 1),
                Style::default().fg(t.value),
            )));
        }
    }

    // Route preview: probe the learned gate with this expert's own capability
    // vector to show how tasks in its specialty would be routed (no embedder
    // needed — the vector is already learned).
    if let (Some(router), Some(vec)) = (&app.router, &e.capability_vec) {
        l.push(Line::from(""));
        l.push(section(t, "gate routing · this specialty"));
        let routed = router.route(vec);
        if routed.is_empty() {
            l.push(Line::from(Span::styled(
                "  escalates · out of distribution",
                Style::default().fg(t.warning),
            )));
        } else {
            for (id, p) in routed.iter().take(5) {
                let is_self = id.as_str() == e.id.as_str();
                let name = app
                    .experts
                    .iter()
                    .find(|x| x.id.as_str() == id.as_str())
                    .map_or_else(|| id.as_str().to_string(), |x| x.name.clone());
                let label = Style::default().fg(if is_self { t.accent } else { t.ink });
                let mut spans = vec![Span::styled(format!("  {name:<20}"), label)];
                spans.extend(gauge_spans(
                    t,
                    *p,
                    10,
                    if is_self { t.accent } else { t.value },
                ));
                spans.push(Span::styled(
                    format!(" {:>3.0}%", p * 100.0),
                    Style::default().fg(t.value),
                ));
                l.push(Line::from(spans));
            }
        }
    }
    (format!("expert · {}", e.name), l)
}

fn shadow_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
    let Some(s) = app.selected_shadow() else {
        return (
            "shadow".into(),
            vec![Line::from(Span::styled(
                "(no shadow selected)",
                Style::default().fg(t.dim),
            ))],
        );
    };
    let mut l = vec![heading(t, s.id.as_str().to_string())];
    l.push(Line::from(vec![
        Span::styled(format!("{:<11}", "status"), Style::default().fg(t.dim)),
        Span::styled(
            s.status.as_str().to_string(),
            Style::default().fg(shadow_color(t, s.status.as_str())),
        ),
    ]));
    l.push(kv(t, "generation", &s.generation.0.to_string()));
    l.push(kv(
        t,
        "parent",
        s.parent_expert.as_ref().map_or("-", |p| p.as_str()),
    ));
    l.push(kv(t, "adapter", s.adapter_uri.as_deref().unwrap_or("-")));
    let final_reward = s.reward_curve.last().copied().unwrap_or(0.0);
    l.push(gauge_row(
        t,
        "final reward",
        final_reward,
        t.fitness(final_reward, 1.0),
    ));
    if !s.reward_curve.is_empty() {
        l.push(sparkline_row(t, "reward", &s.reward_curve));
        let vals = s
            .reward_curve
            .iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join("  ");
        l.push(Line::from(Span::styled(
            format!("  {vals}"),
            Style::default().fg(t.value),
        )));
    }
    l.push(kv(t, "created", &s.created_at.to_rfc3339()));
    (format!("shadow · {}", s.id.as_str()), l)
}

fn boundary_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
    let Some(b) = app.selected_boundary() else {
        return (
            "boundary".into(),
            vec![Line::from(Span::styled(
                "(no boundary selected)",
                Style::default().fg(t.dim),
            ))],
        );
    };
    let mut l = vec![heading(t, b.behavior.clone())];
    let (status, sc) = if b.is_actionable() {
        ("actionable · gates routing", t.alert)
    } else {
        ("open · recorded, inert", t.dim)
    };
    l.push(Line::from(vec![
        Span::styled(format!("{:<11}", "status"), Style::default().fg(t.dim)),
        Span::styled(status.to_string(), Style::default().fg(sc)),
    ]));
    l.push(gauge_row(t, "confidence", b.confidence, t.accent));
    l.push(kv(
        t,
        "grain",
        &b.grain.map_or_else(|| "-".into(), |g| format!("{g:?}")),
    ));
    l.push(kv(t, "generation", &b.generation.0.to_string()));
    let feat = if b.governing_features.is_empty() {
        "-".to_string()
    } else {
        b.governing_features.join(", ")
    };
    l.push(kv(t, "features", &feat));
    l.push(kv(t, "created", &b.created_at.to_rfc3339()));
    l.push(Line::from(""));
    l.push(section(t, "fail context (C)"));
    l.extend(json_lines(t, &b.fail_context));
    if let Some(ok) = &b.near_ok_context {
        l.push(Line::from(""));
        l.push(section(t, "acceptable context (C')"));
        l.extend(json_lines(t, ok));
    }
    (format!("boundary · {}", b.behavior), l)
}

/// The theme colour an event renders in, by kind.
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
                "  (no changes yet — training elsewhere will show here)",
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

/// The keybinding reference, a centred modal over the live view (`?` toggles).
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

//! The drill-down detail builders: the `(title, lines)` body each page's
//! `enter` overlay renders, dispatched by `overlays::detail_overlay`. Pure
//! over `&App`, so the snapshot goldens exercise them headlessly.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::app::App;
use crate::theme::Theme;

use super::evals::{status_color, status_label};
use super::loops::{stage_blurb, stage_label, STAGES};
use super::overlays::{json_lines, section};
use super::{gauge_row, gauge_spans, heading, kv, shadow_color, sparkline_row};
/// The evaluation drill-down: the run's fields, the regression comparison
/// against the previous run for the same subject (the no-forgetting
/// tripwire), and the subject's run history, all from the loaded set.
pub(super) fn eval_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
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
            .unwrap_or_else(|| "n/a".to_string())
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
                .unwrap_or("n/a")
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
                (Some(_), Some(_)) => ("DRIFTED: no-forgetting tripwire", t.alert),
                _ => ("n/a (no fingerprint)", t.dim),
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

/// The loop drill-down: the focused run's place in the generational cycle (a
/// done/current/pending lifecycle ladder), its halt status, and the run's
/// evaluation history -- the bridge from the Loop page to the evidence the
/// Evals page tabulates.
pub(super) fn loop_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
    use antumbra_core::generational::LoopState;
    let Some(head) = app.selected_loop_head() else {
        return (
            "generational loop".into(),
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

    let mut l = vec![heading(t, head.run_id.as_str().to_string())];
    l.push(kv(t, "generation", &head.generation.0.to_string()));
    l.push(kv(t, "state", stage_label(head.state)));

    // Lifecycle ladder: stages before the current are done, the current is lit,
    // the rest pending. Paused has no current stage (the cycle is at rest).
    l.push(Line::from(""));
    l.push(group("lifecycle".into()));
    let current_idx = STAGES.iter().position(|&s| s == head.state);
    for (i, &s) in STAGES.iter().enumerate() {
        let current = Some(i) == current_idx;
        let (glyph, color) = match current_idx {
            Some(ci) if i < ci => ("✓", t.success),
            _ if current => ("◉", t.accent),
            _ => ("·", t.dim),
        };
        let label_style = if current {
            Style::default().fg(t.text).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.dim)
        };
        let mut spans = vec![
            Span::styled(format!("  {glyph} "), Style::default().fg(color)),
            Span::styled(format!("{:<12}", stage_label(s)), label_style),
        ];
        // Only the current stage carries its blurb, so the ladder stays tight.
        if current {
            spans.push(Span::styled(stage_blurb(s), Style::default().fg(t.dim)));
        }
        l.push(Line::from(spans));
    }
    if head.state == LoopState::Paused {
        l.push(Line::from(Span::styled(
            "  ‖ paused; resumes into grow",
            Style::default().fg(t.warning),
        )));
    }

    l.push(Line::from(""));
    if app.loop_halt_pending {
        l.push(Line::from(Span::styled(
            "⚠ graceful halt requested; stops at the next generation",
            Style::default().fg(t.warning).add_modifier(Modifier::BOLD),
        )));
    } else {
        l.push(Line::from(Span::styled(
            "running · x requests a graceful halt",
            Style::default().fg(t.dim),
        )));
    }

    // The run's evaluation history (the Loop -> Evals link), newest first.
    let history = app.loop_eval_history(head);
    l.push(Line::from(""));
    l.push(group(format!(
        "evaluations  ·  {} for this run",
        history.len()
    )));
    if history.is_empty() {
        l.push(Line::from(Span::styled(
            "  none recorded yet",
            Style::default().fg(t.dim),
        )));
    } else {
        for r in &history {
            l.push(Line::from(vec![
                Span::styled(
                    format!("  {:<8}", status_label(r.status)),
                    Style::default()
                        .fg(status_color(t, r.status))
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{:<8}", r.subject_kind.as_str()),
                    Style::default().fg(t.dim),
                ),
                Span::styled(r.subject_id.clone(), Style::default().fg(t.text)),
            ]));
        }
    }

    (format!("generational loop · {}", head.run_id.as_str()), l)
}

pub(super) fn expert_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
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
    // needed, the vector is already learned).
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

pub(super) fn shadow_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
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

pub(super) fn boundary_detail(app: &App, t: &Theme) -> (String, Vec<Line<'static>>) {
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

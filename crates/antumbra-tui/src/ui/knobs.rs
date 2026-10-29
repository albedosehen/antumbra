//! The knobs panel: every memory knob of the workspace with its value, what it
//! does, and, while one is being turned, what it was.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use antumbra_core::usage::KNOBS;

use crate::app::knob_value;
use crate::app::App;
use crate::overlay;

use super::overlays::overlay_area;

pub(super) fn knobs_overlay(f: &mut Frame, app: &App) {
    let t = app.theme();
    let Some(area) = overlay_area(app, f.area()) else {
        return;
    };
    let inner = overlay::modal(f, &t, area, "memory knobs");
    let dim = Style::default().fg(t.dim);
    let mut lines = vec![
        Line::from(Span::styled(
            "  The system governs each memory by these: what it counts as used, how fast",
            dim,
        )),
        Line::from(Span::styled(
            "  an unused one fades, what it links. Changes apply to the next recall.",
            dim,
        )),
        Line::from(""),
    ];
    for (i, knob) in KNOBS.iter().enumerate() {
        let selected = i == app.knob_selected;
        let now = knob_value(knob, &app.knob_draft);
        let was = knob_value(knob, &app.policy);
        let marker = if selected { "› " } else { "  " };
        let key_style = if selected {
            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(t.ink)
        };
        let mut spans = vec![
            Span::styled(format!("{marker}{:<24}", knob.key), key_style),
            Span::styled(format!("{now:>7}"), Style::default().fg(t.value)),
        ];
        if now != was {
            spans.push(Span::styled(
                format!("  (was {was})"),
                Style::default().fg(t.warning),
            ));
        } else if selected {
            spans.push(Span::styled(format!("  {}", knob.help), dim));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(""));
    let footer = if app.operator {
        "  ↑↓ pick · ←→ turn · Enter apply · Esc close"
    } else {
        "  Read-only: an operator's token turns these (antumbra-mcp --mint-token --operator)."
    };
    lines.push(Line::from(Span::styled(footer, dim)));
    f.render_widget(Paragraph::new(lines), inner);
}

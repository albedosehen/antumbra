//! A standing expert's training runs, as a workspace with accepted behaviors
//! holds them: the server trains an expert again when its behaviors change
//! and records each training as a candidate. The candidates' list tells one
//! expert's runs apart and says which is in use, a run kept without the base
//! model's reading says so, and the chart is of the expert's runs.

use super::*;
use crate::theme::NIGHT;

/// One graduated run of the standing expert for `scope`, `days` ago, with
/// the readings it kept: the base model's and the trained expert's, or the
/// trained one alone.
fn run(scope: &str, set: &str, days: i64, readings: &[f32]) -> Shadow {
    Shadow {
        id: antumbra_core::behavior::shadow_id(
            &antumbra_core::TenantId::new("ws:default"),
            &antumbra_core::UserId::new("user:default"),
            scope,
            set,
        ),
        parent_expert: None,
        adapter_uri: Some("adapters/standing.safetensors".into()),
        status: ShadowStatus::Graduated,
        generation: Generation::ZERO,
        reward_curve: readings.to_vec(),
        created_at: Utc::now() - chrono::Duration::days(days),
    }
}

/// The console on the candidates: three runs of one expert, the oldest kept
/// without the base model's reading, one run of another, and a candidate of
/// the loop's. Newest first, as the server gives them.
fn runs_app() -> App {
    let mut app = demo_app();
    let mut loops = graduated_shadow();
    loops.created_at = Utc::now() - chrono::Duration::days(40);
    app.shadows = vec![
        run("github.com/acme/orders", "aaaaaaaaaaaa", 2, &[0.42, 0.91]),
        run("github.com/acme/billing", "bbbbbbbbbbbb", 5, &[0.50, 0.80]),
        run("github.com/acme/orders", "cccccccccccc", 14, &[0.40, 0.78]),
        run("github.com/acme/orders", "dddddddddddd", 30, &[0.71]),
        loops,
    ];
    app.focus = Focus::Shadows;
    app
}

/// The frame's row that holds every one of `parts`.
fn row_with<'a>(text: &'a str, parts: &[&str]) -> Option<&'a str> {
    text.lines()
        .find(|row| parts.iter().all(|part| row.contains(part)))
}

#[test]
fn one_experts_runs_are_told_apart_by_when_and_by_which_is_in_use() -> anyhow::Result<()> {
    let mut app = runs_app();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    // The newest run of each expert is the one that graduated into use.
    for parts in [
        ["orders", "2d ago", "0.42 → 0.91", "● graduated"],
        ["billing", "5d ago", "0.50 → 0.80", "● graduated"],
        // The earlier ones were replaced by it.
        ["orders", "14d ago", "0.40 → 0.78", "× replaced"],
        ["orders", "30d ago", "0.71", "× replaced"],
    ] {
        assert!(row_with(&text, &parts).is_some(), "{parts:?}:\n{text}");
    }
    // A reading kept alone is the trained one, and sits under the others'
    // trained readings, not under their first.
    let column = |when: &str, reading: &str| {
        let row = row_with(&text, &[when])?;
        Some(row[..row.find(reading)?].chars().count())
    };
    let (both, alone) = (column("14d ago", "0.78"), column("30d ago", "0.71"));
    assert!(
        both.is_some() && both == alone,
        "{both:?} {alone:?}:\n{text}"
    );
    Ok(())
}

#[test]
fn a_later_run_that_did_not_graduate_replaces_nothing() -> anyhow::Result<()> {
    let mut app = runs_app();
    let mut pruned = run("github.com/acme/billing", "eeeeeeeeeeee", 1, &[0.50, 0.31]);
    pruned.status = ShadowStatus::Pruned;
    app.shadows.insert(0, pruned);
    let billing = &app.shadows[2];
    assert!(app.replaced_by(billing).is_none(), "still the one in use");
    // The loop's candidate is nobody's run, and nothing replaces it.
    let loops = &app.shadows[5];
    assert!(app.runs_with(loops).is_empty());
    assert!(app.replaced_by(loops).is_none());
    // An expert's runs come oldest first, whatever order they arrived in.
    let orders: Vec<i64> = app
        .runs_with(&app.shadows[1])
        .iter()
        .map(|run| (Utc::now() - run.created_at).num_days())
        .collect();
    assert_eq!(orders, vec![30, 14, 2]);
    Ok(())
}

#[test]
fn a_run_kept_without_the_base_models_reading_says_so() -> anyhow::Result<()> {
    let mut app = runs_app();
    app.selected_shadow = 3;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("base model not kept for this run"), "{text}");
    assert!(text.contains("replaced by the run 2d ago"), "{text}");
    assert!(text.contains("when       30d ago"), "{text}");
    // Never a reading of nothing, nor a chart of one point.
    assert!(!text.contains("not recorded"), "{text}");

    // The drill-down says what is and is not missing.
    app.open_detail();
    let text = to_text(&render(&mut app, 120, 40, 1600.0)?);
    assert!(text.contains("run        1 of 3, oldest first"), "{text}");
    assert!(text.contains("It was trained and it graduated"), "{text}");
    Ok(())
}

#[test]
fn the_chart_is_of_the_experts_runs_and_marks_the_one_selected() -> anyhow::Result<()> {
    let mut app = runs_app();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let text = to_text(&buf);
    assert!(
        text.contains("Pass rate over this expert's 3 training runs"),
        "{text}"
    );
    // Its lines are named under it, and its ends say how long ago they were.
    assert!(
        text.contains("── trained   ── base model   • this run"),
        "{text}"
    );
    for word in ["30d ago", "2d ago"] {
        assert!(text.contains(word), "`{word}`:\n{text}");
    }
    // The legend and the axes are drawn in the theme's inks on its ground.
    let area = buf.area;
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let cell = &buf[(x, y)];
            assert_ne!(
                cell.bg,
                ratatui::style::Color::Reset,
                "({x},{y}) `{}` is left to the terminal",
                cell.symbol()
            );
        }
    }
    assert_eq!(app.theme().name, NIGHT.name);
    Ok(())
}

// Two runs on one day are the same number of days ago: the chart's ends then
// give each its time, where both had read `1d ago`.
#[test]
fn two_runs_on_one_day_are_told_apart_at_the_charts_ends() -> anyhow::Result<()> {
    let mut app = demo_app();
    let mut earlier = run("github.com/acme/orders", "aaaaaaaaaaaa", 1, &[0.40, 0.78]);
    earlier.created_at -= chrono::Duration::minutes(30);
    app.shadows = vec![
        run("github.com/acme/orders", "bbbbbbbbbbbb", 1, &[0.42, 0.91]),
        earlier,
    ];
    app.focus = Focus::Shadows;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    let Some(axis) = text.lines().rev().find(|row| row.contains(':')) else {
        anyhow::bail!("no times at the chart's ends:\n{text}");
    };
    assert!(!axis.contains("1d ago"), "{axis}");
    Ok(())
}

#[test]
fn an_expert_trained_once_has_its_two_readings_and_no_chart() -> anyhow::Result<()> {
    let mut app = runs_app();
    app.selected_shadow = 1;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("This expert has been trained once."),
        "{text}"
    );
    assert!(!text.contains("Pass rate over"), "{text}");
    assert!(
        text.contains("base model") && text.contains("0.50"),
        "{text}"
    );
    // The loop's candidate keeps its own reward, step by step.
    app.selected_shadow = 4;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("Reward · step by step"), "{text}");
    Ok(())
}

#[test]
fn a_narrow_list_keeps_the_name_and_when_before_the_first_reading() -> anyhow::Result<()> {
    let mut app = runs_app();
    app.shadows.insert(
        0,
        run(
            "github.com/acme/warehouse-stock-reconciler",
            "ffffffffffff",
            1,
            &[0.30, 0.66],
        ),
    );
    let text = to_text(&render(&mut app, 80, 24, 1600.0)?);
    assert!(text.contains("CANDIDATE") && text.contains("AGE"), "{text}");
    let row = row_with(&text, &["warehouse-stock-reconciler", "1d", "0.66"]);
    assert!(row.is_some(), "{text}");
    assert!(
        row_with(&text, &["orders", "30d", "0.71", "×"]).is_some(),
        "{text}"
    );
    Ok(())
}

// Golden the candidates with an expert's runs: the list, the selected run's
// two readings, and the chart of the expert's runs.
#[test]
fn golden_candidate_runs() -> anyhow::Result<()> {
    let mut app = runs_app();
    app.selected_shadow = 2;
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert_golden("candidate_runs", &text)
}

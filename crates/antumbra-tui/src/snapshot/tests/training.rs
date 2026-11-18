//! The Training and Evaluations pages on a workspace whose experts are
//! standing ones, which the server trains by itself: the Training page says
//! what it is teaching and what it will do next, and the Evaluations page
//! names each training and each audit in words.

use super::*;
use crate::app::Page;
use crate::standing::tests::{expert, recorded, training};
use antumbra_core::behavior::Status;

/// The console over a user with three scopes: one whose expert holds its
/// behavior, one with a behavior accepted and not yet trained, and one with
/// only a proposed behavior, twenty hours old.
fn taught_app() -> App {
    let mut app = demo_app();
    let everywhere = vec![recorded("memory:a", Status::Trained, "everywhere", 200)];
    let mut behaviors = everywhere.clone();
    behaviors.push(recorded(
        "memory:b",
        Status::Accepted,
        "github.com/acme/orders",
        2,
    ));
    behaviors.push(recorded(
        "memory:c",
        Status::Proposed,
        "github.com/acme/billing",
        20,
    ));
    app.experts = vec![expert("everywhere", &everywhere)];
    app.shadows = vec![training("everywhere", 3)];
    app.memories.extend(behaviors);
    app.loop_heads.clear();
    app.evals.clear();
    reread(&mut app);
    app
}

/// Work the standing experts out again, as a read of the server does.
fn reread(app: &mut App) {
    app.standing = crate::standing::read(
        &app.memories,
        &app.experts,
        &app.shadows,
        app.policy.accept_after_hours,
        Utc::now(),
    );
}

/// Whether a row of `text` holds every one of `parts`.
fn row(text: &str, parts: &[&str]) -> bool {
    text.lines()
        .any(|row| parts.iter().all(|part| row.contains(part)))
}

#[test]
fn the_training_page_says_what_the_server_is_teaching_and_what_it_does_next() -> anyhow::Result<()>
{
    let mut app = taught_app();
    app.set_page(Page::Loop);
    let text = to_text(&render(&mut app, 130, 34, 1600.0)?);
    assert!(
        text.contains("Standing experts 3 · taught by the server"),
        "{text}"
    );
    assert!(
        row(&text, &["everywhere", "3d ago", "● up to date"]),
        "{text}"
    );
    assert!(
        row(&text, &["orders", "○ trained at the server's next pass"]),
        "{text}"
    );
    assert!(
        row(&text, &["billing", "○ its first is accepted in 3h"])
            || row(&text, &["billing", "○ its first is accepted in 4h"]),
        "{text}"
    );
    assert!(
        text.contains("A proposed behavior that can be taught is accepted after 24 hours."),
        "{text}"
    );
    // The loop a person runs is under it, and says a standing expert needs none.
    assert!(text.contains("No training loop has run yet: "), "{text}");
    assert!(text.contains("A standing expert needs none"), "{text}");
    Ok(())
}

#[test]
fn a_workspace_that_keeps_accepting_to_itself_says_so() -> anyhow::Result<()> {
    let mut app = taught_app();
    app.policy.accept_after_hours = 0.0;
    reread(&mut app);
    app.set_page(Page::Loop);
    let text = to_text(&render(&mut app, 130, 34, 1600.0)?);
    assert!(
        row(&text, &["billing", "○ waits for you to accept it"]),
        "{text}"
    );
    assert!(
        text.contains("Proposed behaviors wait for you to accept them."),
        "{text}"
    );
    Ok(())
}

#[test]
fn a_workspace_with_no_behaviors_has_only_the_loop() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.loop_heads.clear();
    app.set_page(Page::Loop);
    let text = to_text(&render(&mut app, 130, 34, 1600.0)?);
    assert!(!text.contains("Standing experts"), "{text}");
    assert!(text.contains("No training loop has run yet: "), "{text}");
    Ok(())
}

/// One evaluation the server kept of `subject`.
fn kept(
    subject: &str,
    task: &str,
    status: EvalStatus,
    metrics: serde_json::Value,
) -> EvaluationRun {
    EvaluationRun {
        run_id: RunId::new("standing:0123456789ab"),
        subject_kind: SubjectKind::Expert,
        subject_id: subject.into(),
        corpus_task_id: task.into(),
        status,
        metrics: Some(metrics),
        regression_fingerprint: None,
        created_at: Utc::now() - chrono::Duration::days(2),
    }
}

#[test]
fn a_training_and_an_audit_are_named_in_words_with_what_each_read() -> anyhow::Result<()> {
    let mut app = taught_app();
    let subject = "expert:ws:demo:user:demo:behavior:everywhere";
    app.evals = vec![
        kept(
            subject,
            "training:everywhere",
            EvalStatus::Success,
            serde_json::json!({ "kind": "training", "base": 0.4, "expert": 0.8 }),
        ),
        kept(
            subject,
            "audit:everywhere",
            EvalStatus::Failure,
            serde_json::json!({ "kind": "audit", "examples": 4, "base": 3, "expert": 2 }),
        ),
    ];
    app.set_page(Page::Evals);
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    // The expert by the name the console shows it under, not its id.
    assert!(!text.contains("expert:ws:demo"), "{text}");
    assert!(
        row(
            &text,
            &[
                "everywhere",
                "training · everywhere",
                "0.40 → 0.80",
                "2d ago"
            ]
        ),
        "{text}"
    );
    assert!(
        row(
            &text,
            &["everywhere", "audit · everywhere", "2 of 4 · base 3"]
        ),
        "{text}"
    );
    // The loop's own runs keep their task's id and their fingerprint.
    app.evals = demo_evals(Utc::now());
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert!(row(&text, &["task:arith", "a1b2c3d4"]), "{text}");
    Ok(())
}

// Golden the Training page over a workspace of standing experts.
#[test]
fn golden_training_standing() -> anyhow::Result<()> {
    let mut app = taught_app();
    // A wait nothing in the fixture is close to the end of, so the frame
    // reads the same an hour on.
    app.memories.retain(|m| m.id.as_str() != "memory:c");
    reread(&mut app);
    app.set_page(Page::Loop);
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert_golden("training_standing", &text)
}

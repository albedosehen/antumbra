//! A workspace whose experts are standing ones, which is what accepting
//! behaviors produces: each is named by its scope and not by its id, and the
//! ask, which can route a task to none of them, says why and says which of
//! them shape an answer where the console was started.

use super::*;
use crate::backend::Routed;
use crate::words;

/// A standing expert for `scope`, as the server holds one: its name is its
/// id, and its card says it is standing and for which scope.
fn standing(scope: &str, fitness: f32) -> Expert {
    let now = Utc::now();
    let id = format!("expert:ws:default:user:default:behavior:{scope}");
    Expert {
        id: ExpertId::new(id.clone()),
        name: id,
        base_model: "Qwen/Qwen2.5-Coder-1.5B-Instruct".into(),
        artifact_uri: "adapters/standing.safetensors".into(),
        capability_card: serde_json::json!({ "standing": true, "scope": scope }),
        capability_vec: None,
        fitness,
        frozen_at: Some(now),
        generation: Generation::ZERO,
        owner: Some(antumbra_core::UserId::new("user:default")),
        compartment: None,
        placed_on: None,
        created_at: now,
    }
}

/// The console over a workspace that has only standing experts, started in
/// the repository `github.com/acme/orders` (the fixture's).
fn standing_app() -> App {
    let mut app = demo_app();
    app.experts = vec![
        standing("github.com/acme/orders", 0.87),
        standing("github.com/acme/billing", 0.91),
        standing("github.com/other/warehouse-stock-reconciler", 0.74),
        standing("everywhere", 0.66),
    ];
    app.sort_experts();
    // The first row, as the console opens on.
    app.selected = 0;
    // Read from a server, not the demo.
    app.real_embedder = true;
    app
}

/// The server's answer to a task no expert is routed to, as it sends it.
fn no_router() -> serde_json::Value {
    serde_json::json!({
        "covered": false,
        "escalate": true,
        "reason": "no learned router yet, and none of your private experts is close to this task",
        "routes": [],
    })
}

#[test]
fn a_standing_expert_is_named_by_its_scope_and_not_by_its_id() -> anyhow::Result<()> {
    let mut app = standing_app();
    let names: Vec<String> = app.experts.iter().map(words::expert_name).collect();
    assert_eq!(
        names,
        vec![
            "billing",
            "orders",
            "warehouse-stock-reconciler",
            "everywhere"
        ],
        "strongest first, each by its scope in brief"
    );

    // In the list every row can be told from the next, at any width.
    for (width, height) in [(120, 36), (80, 24)] {
        let text = to_text(&render(&mut app, width, height, 1600.0)?);
        for name in &names {
            assert!(text.contains(name.as_str()), "{width}: `{name}`:\n{text}");
        }
        assert!(
            !text.contains("expert:ws:default:user:defau"),
            "{width}: no row is the id's prefix:\n{text}"
        );
    }

    // The detail says what kind it is, and gives the scope whole.
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("kind       standing"), "{text}");
    assert!(
        text.contains("scope      github.com/acme/billing"),
        "{text}"
    );
    // And the drill-down has the id, which the name no longer carries.
    app.open_detail();
    let text = to_text(&render(&mut app, 120, 40, 1600.0)?);
    assert!(text.contains("Expert · billing"), "{text}");
    assert!(
        text.contains("expert:ws:default:user:default:behavior:github.com/acme/billing"),
        "{text}"
    );
    assert!(
        text.contains("standing: it shapes every answer in its scope"),
        "{text}"
    );
    Ok(())
}

#[test]
fn an_expert_that_is_not_standing_keeps_its_own_name() -> anyhow::Result<()> {
    let mut app = demo_app();
    assert_eq!(words::expert_name(&app.experts[0]), "arith-specialist");
    assert_eq!(words::expert_kind(&app.experts[0]), "routed");
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("kind       routed"), "{text}");
    assert!(
        !text.contains("scope   "),
        "a routed expert has no scope:\n{text}"
    );
    Ok(())
}

#[test]
fn the_servers_answer_is_read_with_its_reason() -> anyhow::Result<()> {
    let routed = Routed::read(&no_router());
    assert!(routed.routes.is_empty());
    assert_eq!(
        routed.reason.as_deref(),
        Some("no learned router yet, and none of your private experts is close to this task")
    );
    // An answer that routes carries no reason, and an older server's none.
    let routed = Routed::read(&serde_json::json!({
        "covered": true, "escalate": false,
        "routes": [{ "expert_id": "expert:arith", "probability": 0.82, "private": false }],
    }));
    assert_eq!(routed.routes.len(), 1);
    assert_eq!(routed.reason, None);
    assert_eq!(Routed::read(&serde_json::json!({})), Routed::default());
    Ok(())
}

#[test]
fn ask_says_why_no_expert_is_routed_and_which_standing_experts_shape_an_answer_here(
) -> anyhow::Result<()> {
    let mut app = standing_app();
    app.open_ask();
    // Before anything is asked, it already says who shapes an answer here.
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("Standing experts, in every answer here (orders):"),
        "{text}"
    );

    for c in "Which expert for orders?".chars() {
        app.ask_input(c);
    }
    app.set_ask_result(&Routed::read(&no_router()));
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("No expert is routed to it: it goes to your agent."),
        "{text}"
    );
    // The server's reason, whole, over as many lines as it needs.
    assert!(
        text.contains("Why: no learned router yet, and none of your"),
        "{text}"
    );
    assert!(text.contains("close to this task"), "{text}");

    // The one for everywhere, then the one for this repository, as the
    // server composes them; each with its scope.
    let everywhere = text.find("● everywhere");
    let orders = text.find("● orders");
    assert!(everywhere.is_some() && orders.is_some(), "{text}");
    assert!(everywhere < orders, "everywhere first:\n{text}");
    assert!(text.contains("github.com/acme/orders"), "{text}");
    // The ones for other repositories are counted, not listed.
    assert!(!text.contains("● billing"), "{text}");
    assert!(
        text.contains("and 2 more, for other repositories"),
        "{text}"
    );
    // Nothing is cut off at the bottom of the panel.
    assert!(
        text.lines()
            .any(|row| row.contains("and 2 more") && row.contains('│')),
        "inside the border:\n{text}"
    );
    Ok(())
}

#[test]
fn ask_outside_a_repository_lists_the_expert_for_everywhere() -> anyhow::Result<()> {
    let mut app = standing_app();
    app.git.repo = None;
    app.open_ask();
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("in every answer here (no repository):"),
        "{text}"
    );
    assert!(text.contains("● everywhere"), "{text}");
    assert!(!text.contains("● orders"), "{text}");
    assert!(
        text.contains("and 3 more, for other repositories"),
        "{text}"
    );

    // In a repository none of them is for, and with none for everywhere.
    app.git.repo = Some("github.com/acme/unknown".into());
    app.experts
        .retain(|e| e.standing_scope() != Some("everywhere"));
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(text.contains("none of them is for this place"), "{text}");
    Ok(())
}

#[test]
fn ask_with_no_standing_expert_says_nothing_of_them() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.open_ask();
    app.set_ask_result(&Routed {
        routes: Vec::new(),
        reason: Some("outside what the shared population covers".into()),
        ..Default::default()
    });
    let text = to_text(&render(&mut app, 120, 36, 1600.0)?);
    assert!(
        text.contains("Why: outside what the shared population covers"),
        "{text}"
    );
    assert!(!text.contains("Standing experts"), "{text}");
    Ok(())
}

#[test]
fn a_prompt_and_a_notice_name_a_standing_expert_by_its_scope() -> anyhow::Result<()> {
    let mut app = standing_app();
    app.request_action();
    anyhow::ensure!(app.mode == Mode::Confirm, "x on an expert asks first");
    assert_eq!(
        app.pending_prompt().as_deref(),
        Some("Thaw expert billing?")
    );
    app.cancel_action();
    // Sorted by name, it is the name shown that orders the list.
    app.sort = crate::app::SortKey::Name;
    app.sort_experts();
    let names: Vec<String> = app.experts.iter().map(words::expert_name).collect();
    assert_eq!(
        names,
        vec![
            "billing",
            "everywhere",
            "orders",
            "warehouse-stock-reconciler"
        ]
    );
    Ok(())
}

// Golden the ask over a workspace of standing experts, after a task no
// expert is routed to: the reason, and who shapes an answer here.
#[test]
fn golden_ask_standing() -> anyhow::Result<()> {
    let mut app = standing_app();
    app.open_ask();
    for c in "Which expert for orders?".chars() {
        app.ask_input(c);
    }
    app.set_ask_result(&Routed::read(&no_router()));
    golden_overlay("ask_standing", &mut app, 120, 36)
}

/// The server's answer to the same kind of task from a server that also
/// reads the user's standing experts and behaviors for it.
fn read_for_the_task() -> serde_json::Value {
    let id = |scope: &str| format!("expert:ws:default:user:default:behavior:{scope}");
    let mut out = no_router();
    out["standing"] = serde_json::json!([
        { "expert_id": id("everywhere"), "scope": "everywhere",
          "similarity": 0.41, "in_scope": true, "answers": true },
        { "expert_id": id("github.com/acme/billing"), "scope": "github.com/acme/billing",
          "similarity": 0.36, "in_scope": false, "answers": false },
        { "expert_id": id("github.com/acme/orders"), "scope": "github.com/acme/orders",
          "similarity": 0.12, "in_scope": true, "answers": false },
        { "expert_id": id("github.com/other/warehouse-stock-reconciler"),
          "scope": "github.com/other/warehouse-stock-reconciler",
          "similarity": 0.02, "in_scope": false, "answers": false },
    ]);
    out["taught"] = serde_json::json!([
        { "behavior_id": "memory:1", "scope": "everywhere", "status": "accepted",
          "similarity": 0.17, "learned": false,
          "rule": "Draw a game character for a side-scrolling or platformer game in a \
                   three-quarter or side view facing its direction of travel, with both \
                   hands visible, so the rig can be mirrored for the other direction." },
        { "behavior_id": "memory:2", "scope": "github.com/acme/orders", "status": "trained",
          "similarity": 0.11, "learned": true,
          "rule": "Name a branch for an issue feat/{issue}-{slug}." },
        { "behavior_id": "memory:3", "scope": "everywhere", "status": "proposed",
          "similarity": 0.09,
          "rule": "Write a commit subject in the imperative." },
    ]);
    out
}

#[test]
fn the_servers_reading_of_standing_experts_and_behaviors_is_read() -> anyhow::Result<()> {
    let routed = Routed::read(&read_for_the_task());
    let Some(standing) = &routed.near.standing else {
        anyhow::bail!("the standing experts the server sent were not read");
    };
    assert_eq!(standing.len(), 4);
    assert!(standing[0].answers && standing[0].in_scope);
    assert_eq!(standing[1].scope, "github.com/acme/billing");
    assert!((standing[1].similarity - 0.36).abs() < 1e-6);
    let learned: Vec<Option<bool>> = routed.near.taught.iter().map(|t| t.learned).collect();
    assert_eq!(learned, vec![Some(false), Some(true), None]);
    // An older server says nothing of them, which is not the same as a user
    // who has none: the console then lists the ones for this place itself.
    assert_eq!(Routed::read(&no_router()).near.standing, None);
    let mut none = no_router();
    none["standing"] = serde_json::json!([]);
    assert_eq!(Routed::read(&none).near.standing, Some(Vec::new()));
    Ok(())
}

// Asked about a task, the console says which standing expert would answer it
// and which are only near it, then the nearest of what was taught with where
// each stands: the rule that bears on the task and was not learned is the
// answer to why no expert takes it.
#[test]
fn ask_says_which_standing_expert_answers_and_what_was_taught_about_the_task() -> anyhow::Result<()>
{
    let mut app = standing_app();
    app.open_ask();
    for c in "draw a game character".chars() {
        app.ask_input(c);
    }
    app.set_ask_result(&Routed::read(&read_for_the_task()));
    let text = to_text(&render(&mut app, 120, 44, 1600.0)?);
    let row = |parts: &[&str]| {
        text.lines()
            .any(|row| parts.iter().all(|part| row.contains(part)))
    };
    assert!(
        text.contains("Standing experts, the closest to this task first:"),
        "{text}"
    );
    assert!(
        text.contains("No expert is routed to it: your standing expert everywhere answers it."),
        "{text}"
    );
    assert!(row(&["● everywhere", "0.41", "answers it"]), "{text}");
    assert!(
        row(&["○ billing", "0.36", "for another repository"]),
        "{text}"
    );
    assert!(
        row(&["○ orders", "0.12", "in scope, too far to answer alone"]),
        "{text}"
    );
    assert!(text.contains("and 1 more, further from it"), "{text}");
    // The list for this place gives way to the server's reading.
    assert!(!text.contains("in every answer here"), "{text}");

    assert!(
        text.contains("The nearest of what you have taught:"),
        "{text}"
    );
    // A long rule is two rows, cut with an ellipsis, and none is left out.
    assert!(
        text.contains("Draw a game character for a side-scrolling"),
        "{text}"
    );
    assert!(
        row(&["everywhere · not learned at its last training"]),
        "{text}"
    );
    assert!(
        text.contains("Name a branch for an issue feat/{issue}-{slug}."),
        "{text}"
    );
    assert!(row(&["orders · learned"]), "{text}");
    assert!(
        row(&["everywhere · proposed, waiting to be accepted"]),
        "{text}"
    );

    // Typing again drops the reading with the result it came with.
    app.ask_input('s');
    assert_eq!(app.ask_near, crate::backend::Near::default());
    let text = to_text(&render(&mut app, 120, 44, 1600.0)?);
    assert!(text.contains("in every answer here (orders)"), "{text}");
    assert!(
        !text.contains("The nearest of what you have taught"),
        "{text}"
    );
    Ok(())
}

// A server that reads the standing experts and finds the user has none says
// nothing of them, and a task some expert is routed to still gets the reading.
#[test]
fn ask_with_a_route_still_reads_the_standing_experts() -> anyhow::Result<()> {
    let mut app = standing_app();
    app.open_ask();
    let mut out = read_for_the_task();
    out["covered"] = true.into();
    out["escalate"] = false.into();
    out["reason"] = serde_json::Value::Null;
    out["routes"] = serde_json::json!([
        { "expert_id": "expert:ws:default:user:default:behavior:everywhere",
          "probability": 0.8, "private": true }
    ]);
    app.set_ask_result(&Routed::read(&out));
    let text = to_text(&render(&mut app, 120, 44, 1600.0)?);
    assert!(text.contains("80%"), "{text}");
    assert!(text.contains("answers it"), "{text}");

    let mut none = no_router();
    none["standing"] = serde_json::json!([]);
    app.set_ask_result(&Routed::read(&none));
    let text = to_text(&render(&mut app, 120, 44, 1600.0)?);
    assert!(!text.contains("Standing experts"), "{text}");
    Ok(())
}

// Golden the ask with the server's reading for the task.
#[test]
fn golden_ask_reading() -> anyhow::Result<()> {
    let mut app = standing_app();
    app.open_ask();
    for c in "draw a game character".chars() {
        app.ask_input(c);
    }
    app.set_ask_result(&Routed::read(&read_for_the_task()));
    golden_overlay("ask_reading", &mut app, 120, 44)
}

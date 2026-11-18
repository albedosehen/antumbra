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
        standing("github.com/other/data-plane-transaction-hub", 0.74),
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
            "data-plane-transaction-hub",
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
            "data-plane-transaction-hub",
            "everywhere",
            "orders"
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

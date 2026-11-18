//! The Services page: the workspace's dependency claims as a graph, and what
//! a change to the selected service reaches. Read from the demo's store,
//! whose claims are memories as a server keeps them.

use super::*;
use crate::app::Page;
use crate::backend::Backend;
use crate::drive::press;
use crate::keys::Key;
use antumbra_core::depgraph::{Direction, Source};

/// The demo's console on the Services page.
async fn on_services() -> anyhow::Result<(Backend, App)> {
    let backend = Backend::demo(crate::demo::seed().await?);
    let mut app = App::load(&backend).await?;
    app.set_page(Page::Services);
    Ok((backend, app))
}

/// Whether a row of `text` holds every one of `parts`.
fn row(text: &str, parts: &[&str]) -> bool {
    text.lines()
        .any(|row| parts.iter().all(|part| row.contains(part)))
}

#[tokio::test]
async fn the_page_opens_on_the_service_the_most_depend_on() -> anyhow::Result<()> {
    let (_backend, mut app) = on_services().await?;
    assert_eq!(
        app.selected_service_name(),
        Some("github.com/acme/ledger-core")
    );
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert!(text.contains("6 Services"), "the tab:\n{text}");
    assert!(text.contains("Services 8 · 9 dependencies"), "{text}");
    // Every service is drawn, by its short name and whole.
    for name in [
        "billing-ui",
        "payments-webhook",
        "payouts-worker",
        "reports-job",
        "ledger-api",
        "ledger-core",
        "auth-service",
        "ledger-db",
    ] {
        assert!(text.contains(&format!(" {name} ")), "`{name}`:\n{text}");
    }
    // An arrowhead marks what a line arrives at, and the key says what each
    // stroke is.
    assert!(text.contains("▸ ledger-core"), "{text}");
    assert!(
        text.contains("── declared   ╌╌ observed   ┄┄ learned or claimed"),
        "{text}"
    );
    // The blast radius: what depends on it, strongest first, each with what
    // says so or what it was reached through.
    assert!(
        text.contains("what depends on it · within 3 hops"),
        "{text}"
    );
    assert!(row(&text, &["ledger-api", "0.90 · 1 hop"]), "{text}");
    assert!(text.contains("declared · Cargo.toml names it"), "{text}");
    assert!(row(&text, &["billing-ui", "0.81 · 2 hops"]), "{text}");
    assert!(text.contains("via ledger-api"), "{text}");
    Ok(())
}

#[tokio::test]
async fn tab_turns_the_walk_round_and_h_lets_it_go_further() -> anyhow::Result<()> {
    let (backend, mut app) = on_services().await?;
    let mut transition = None;
    press(&mut app, &backend, &mut transition, Key::Tab).await;
    assert_eq!(app.service_direction, Direction::Dependencies);
    let said = app.current_notice().map(|n| n.text.clone());
    assert_eq!(said.as_deref(), Some("blast radius · what it depends on"));
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert!(
        text.contains("what it depends on · within 3 hops"),
        "{text}"
    );
    assert!(row(&text, &["ledger-db", "0.90 · 1 hop"]), "{text}");
    press(&mut app, &backend, &mut transition, Key::Tab).await;
    assert_eq!(app.service_direction, Direction::Dependents);

    // Three hops, six, one, and round again.
    let mut seen = vec![app.service_hops];
    for _ in 0..3 {
        press(&mut app, &backend, &mut transition, Key::Char('h')).await;
        seen.push(app.service_hops);
    }
    assert_eq!(seen, vec![3, 6, 1, 3]);
    press(&mut app, &backend, &mut transition, Key::Char('h')).await;
    press(&mut app, &backend, &mut transition, Key::Char('h')).await;
    let said = app.current_notice().map(|n| n.text.clone());
    assert_eq!(said.as_deref(), Some("blast radius · within 1 hop"));
    // One hop reaches the three that name it, and not what is behind them.
    let reached: Vec<String> = app
        .service_reach()
        .into_iter()
        .map(|r| crate::services::short(&r.service).to_string())
        .collect();
    assert_eq!(
        reached,
        ["ledger-api", "payments-webhook", "payouts-worker"]
    );
    // Both keys are on the footer, which is drawn from the bindings.
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert!(
        text.contains("tab  turn round") && text.contains("h  hops"),
        "{text}"
    );
    Ok(())
}

#[tokio::test]
async fn the_keys_move_the_walks_start_and_a_read_leaves_it_there() -> anyhow::Result<()> {
    let (backend, mut app) = on_services().await?;
    let mut transition = None;
    let before = app.selected_service_name().map(str::to_string);
    press(&mut app, &backend, &mut transition, Key::Down).await;
    let after = app.selected_service_name().map(str::to_string);
    assert_ne!(before, after, "the selection moved");
    // The lists are replaced by a read, in whatever order; the service
    // selected is found again by its name.
    for _ in 0..3 {
        app.reload(&backend).await?;
        assert_eq!(app.selected_service_name().map(str::to_string), after);
    }
    assert_eq!(app.filter_subject(), "services");
    Ok(())
}

#[tokio::test]
async fn enter_lists_every_claim_about_the_service() -> anyhow::Result<()> {
    let (backend, mut app) = on_services().await?;
    let mut transition = None;
    press(&mut app, &backend, &mut transition, Key::Enter).await;
    assert_eq!(app.mode, Mode::Detail);
    let text = to_text(&render(&mut app, 130, 44, 1600.0)?);
    assert!(text.contains("Service · ledger-core"), "{text}");
    assert!(text.contains("Depends on · 1"), "{text}");
    assert!(text.contains("Depended on by · 3"), "{text}");
    assert!(text.contains("github.com/acme/ledger-db"), "{text}");
    assert!(
        text.contains("declared · 0.90 · Cargo.toml names it · last seen now"),
        "{text}"
    );
    Ok(())
}

#[test]
fn a_workspace_with_no_claims_says_how_one_is_recorded() -> anyhow::Result<()> {
    let mut app = demo_app();
    app.set_page(Page::Services);
    let text = to_text(&render(&mut app, 120, 30, 1600.0)?);
    assert!(text.contains("No dependencies recorded yet."), "{text}");
    assert!(text.contains("antumbra claude dependencies"), "{text}");
    // Nothing to walk, so nothing offered to walk it with.
    assert!(!text.contains("turn round"), "{text}");
    Ok(())
}

// A graph with more services in a column than the panel has rows is drawn as
// the selected service and what is within reach of it, and says so.
#[test]
fn a_graph_too_big_for_the_panel_is_drawn_around_the_selection() -> anyhow::Result<()> {
    let name = |kind: &str, n: usize| format!("github.com/acme/{kind}-{n:02}");
    let mut claims = Vec::new();
    for n in 0..30 {
        claims.push(crate::services::tests::claim(
            &name("leaf", n),
            &name("mid", n),
            Source::Declared,
            "Cargo.toml names it",
        ));
        claims.push(crate::services::tests::claim(
            &name("mid", n),
            "github.com/acme/core",
            Source::Declared,
            "Cargo.toml names it",
        ));
    }
    let mut app = demo_app();
    app.services = crate::services::Services::read(&claims, Utc::now());
    app.set_page(Page::Services);
    let Some(at) = app
        .services
        .names
        .iter()
        .position(|n| n.ends_with("mid-03"))
    else {
        anyhow::bail!("the service is not listed");
    };
    app.selected_service = at;
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert!(text.contains("Services 61 · 60 dependencies"), "{text}");
    assert!(
        text.contains("showing 3 of 61 services: the selected one and what is within reach"),
        "{text}"
    );
    assert!(
        row(&text, &[" leaf-03 ", "▸ mid-03 ", "▸ core "]),
        "the three in one row:\n{text}"
    );
    assert!(!text.contains("leaf-04"), "{text}");
    Ok(())
}

// A terminal too small for even that says so, and the list beside still
// says what is reached.
#[test]
fn a_panel_too_small_for_the_graph_says_so() -> anyhow::Result<()> {
    let mut app = demo_app();
    with_services(&mut app);
    app.set_page(Page::Services);
    let text = to_text(&render(&mut app, 60, 20, 1600.0)?);
    assert!(text.contains("This panel is too small"), "{text}");
    assert!(text.contains("Blast radius"), "{text}");
    Ok(())
}

// Golden the Services page: the graph and the blast radius beside it.
#[test]
fn golden_services_page() -> anyhow::Result<()> {
    let mut app = demo_app();
    with_services(&mut app);
    app.set_page(Page::Services);
    let text = to_text(&render(&mut app, 130, 30, 1600.0)?);
    assert_golden("services_page", &text)
}

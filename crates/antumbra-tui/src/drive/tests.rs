//! Key presses carried out against the demo's store, start to finish: what a
//! key asks for, what the console then holds, and what it says on screen.

use super::*;
use crate::app::{Focus, Page, Pending};
use crate::theme::Tone;
use antumbra_core::generational::{GenerationHead, LoopCommand};
use antumbra_core::RunId;
use antumbra_store::repo::{generation, loop_control};
use antumbra_store::Store;
use anyhow::Context as _;

/// The console over the seeded demo, as an operator's.
async fn console() -> anyhow::Result<(Store, Backend, App)> {
    let store = crate::demo::seed().await?;
    let backend = Backend::demo(store.clone());
    let app = App::load(&backend).await?;
    Ok((store, backend, app))
}

/// Press each of `keys` in turn.
async fn keys_pressed(app: &mut App, backend: &Backend, keys: &[Key]) {
    let mut transition = None;
    for &key in keys {
        press(app, backend, &mut transition, key).await;
    }
}

/// Type `text`, a key a character.
async fn typed(app: &mut App, backend: &Backend, text: &str) {
    let keys: Vec<Key> = text.chars().map(Key::Char).collect();
    keys_pressed(app, backend, &keys).await;
}

/// The notice on screen, or fail saying there is none.
fn said(app: &App) -> anyhow::Result<(Tone, &str)> {
    let notice = app.current_notice().context("nothing is said on screen")?;
    Ok((notice.tone, notice.text.as_str()))
}

#[tokio::test]
async fn esc_closes_an_overlay_and_never_quits() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('?')]).await;
    assert_eq!(app.mode, Mode::Help);
    keys_pressed(&mut app, &backend, &[Key::Esc]).await;
    assert_eq!(app.mode, Mode::Normal);
    // Once more than needed: still here.
    keys_pressed(&mut app, &backend, &[Key::Esc, Key::Esc]).await;
    assert!(!app.should_quit, "Esc on the main view does not quit");
    keys_pressed(&mut app, &backend, &[Key::Char('q')]).await;
    assert!(app.should_quit);
    Ok(())
}

#[tokio::test]
async fn ctrl_c_quits_from_inside_an_overlay() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char(':')]).await;
    assert_eq!(app.mode, Mode::Palette);
    // A plain `c` is typed into the query; Ctrl-C is not.
    keys_pressed(&mut app, &backend, &[Key::Char('c')]).await;
    assert_eq!(app.palette.query, "c");
    assert!(!app.should_quit);
    keys_pressed(&mut app, &backend, &[Key::Ctrl('c')]).await;
    assert!(app.should_quit);
    Ok(())
}

#[tokio::test]
async fn x_on_an_expert_freezes_or_thaws_it_behind_a_confirm() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    let (id, was_frozen) = {
        let expert = app.selected_expert().context("no expert is selected")?;
        (expert.id.clone(), expert.is_frozen())
    };
    keys_pressed(&mut app, &backend, &[Key::Char('x')]).await;
    assert_eq!(app.mode, Mode::Confirm);
    assert!(
        matches!(&app.pending, Some(Pending::FreezeExpert(e, freeze)) if e.as_str() == id.as_str() && *freeze != was_frozen),
        "it asks to do the opposite of what the expert is"
    );
    // Enter and a stray key leave the prompt open and the expert as it was.
    keys_pressed(&mut app, &backend, &[Key::Enter, Key::Char('z')]).await;
    assert_eq!(app.mode, Mode::Confirm);

    keys_pressed(&mut app, &backend, &[Key::Char('y')]).await;
    assert_eq!(app.mode, Mode::Normal);
    let now = app
        .experts
        .iter()
        .find(|e| e.id.as_str() == id.as_str())
        .context("the expert is still listed")?;
    assert_eq!(now.is_frozen(), !was_frozen, "the server did it");
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Settled);
    let word = if was_frozen { "thawed" } else { "frozen" };
    assert!(text.contains(word) && text.contains(id.as_str()), "{text}");
    Ok(())
}

#[tokio::test]
async fn declining_leaves_everything_as_it_was() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    let before = app.boundaries.len();
    keys_pressed(&mut app, &backend, &[Key::Tab, Key::Tab, Key::Char('x')]).await;
    assert_eq!(app.focus, Focus::Boundaries);
    assert!(matches!(app.pending, Some(Pending::DeleteBoundary(_))));
    keys_pressed(&mut app, &backend, &[Key::Char('n')]).await;
    assert_eq!(app.mode, Mode::Normal);
    assert!(app.pending.is_none());
    assert_eq!(app.boundaries.len(), before);
    Ok(())
}

#[tokio::test]
async fn deleting_a_boundary_says_which_on_screen() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Tab, Key::Tab]).await;
    let behavior = app
        .selected_boundary()
        .context("no boundary is selected")?
        .behavior
        .clone();
    let before = app.boundaries.len();
    keys_pressed(&mut app, &backend, &[Key::Char('x'), Key::Char('y')]).await;
    assert_eq!(app.boundaries.len(), before - 1);
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Settled);
    assert!(
        text.contains("boundary removed") && text.contains(&behavior),
        "{text}"
    );
    // And the history keeps it.
    assert!(app
        .events
        .iter()
        .any(|e| e.text.contains("boundary removed")));
    Ok(())
}

#[tokio::test]
async fn a_read_only_console_answers_x_on_screen() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    app.operator = false;
    keys_pressed(&mut app, &backend, &[Key::Char('x')]).await;
    assert_eq!(app.mode, Mode::Normal, "nothing is staged");
    assert!(app.pending.is_none());
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Failed, "a refusal");
    assert!(
        text.contains("Read-only") && text.contains("--operator"),
        "{text}"
    );
    Ok(())
}

#[tokio::test]
async fn find_on_the_memory_page_selects_a_memory() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('2')]).await;
    assert_eq!(app.page, Page::Memory);
    // The last memory listed, so the jump is not where the selection began.
    let listed = app.listed_memories();
    let last = *listed.last().context("the demo has memories")?;
    let wanted = app.memories[last].content.clone();
    anyhow::ensure!(
        listed.len() > 1 && listed.first() == Some(&app.selected_memory),
        "the page opens on the first memory it lists"
    );
    keys_pressed(&mut app, &backend, &[Key::Char('/')]).await;
    assert_eq!(app.mode, Mode::Filter);
    assert_eq!(app.filter_subject(), "memories");
    typed(&mut app, &backend, &wanted).await;
    keys_pressed(&mut app, &backend, &[Key::Enter]).await;
    assert_eq!(app.mode, Mode::Normal);
    assert_eq!(
        app.selected_memory().map(|m| m.content.as_str()),
        Some(wanted.as_str()),
        "the memory that was found is the one selected"
    );
    Ok(())
}

#[tokio::test]
async fn tab_on_the_memory_page_lists_the_next_kind_and_keeps_the_keys_within_it(
) -> anyhow::Result<()> {
    use antumbra_core::MemoryNetwork;
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('2')]).await;
    assert_eq!(app.memory_kind, None, "it opens on every kind");
    let all = app.listed_memories().len();
    anyhow::ensure!(all == app.memories.len() && all > 3);

    // Tab: the facts alone. The selection is one of them.
    keys_pressed(&mut app, &backend, &[Key::Tab]).await;
    assert_eq!(app.memory_kind, Some(MemoryNetwork::World));
    let world = app.listed_memories();
    assert!(!world.is_empty() && world.len() < all);
    assert!(world
        .iter()
        .all(|&i| app.memories[i].network == MemoryNetwork::World));
    assert!(world.contains(&app.selected_memory));

    // The keys stay within the kind: past the last is the first again.
    for _ in 0..world.len() {
        keys_pressed(&mut app, &backend, &[Key::Down]).await;
        assert!(world.contains(&app.selected_memory));
    }
    keys_pressed(&mut app, &backend, &[Key::End]).await;
    assert_eq!(Some(&app.selected_memory), world.last());

    // So does find: a memory of another kind is not found.
    let bank = app
        .memories
        .iter()
        .find(|m| m.network == MemoryNetwork::Bank)
        .context("the demo has an experience")?
        .content
        .clone();
    keys_pressed(&mut app, &backend, &[Key::Char('/')]).await;
    typed(&mut app, &backend, &bank).await;
    assert!(app.filtered().is_empty(), "an experience is not a fact");
    keys_pressed(&mut app, &backend, &[Key::Esc]).await;

    // Three more Tabs come round to every kind, the selection kept.
    let kept = app.selected_memory;
    keys_pressed(&mut app, &backend, &[Key::Tab]).await;
    assert_eq!(app.memory_kind, Some(MemoryNetwork::Bank));
    assert_ne!(app.selected_memory, kept, "a fact is not an experience");
    keys_pressed(&mut app, &backend, &[Key::Tab, Key::Tab]).await;
    assert_eq!(app.memory_kind, None);
    Ok(())
}

#[tokio::test]
async fn l_swaps_the_memory_list_for_the_graph_and_back() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('2'), Key::Tab]).await;
    anyhow::ensure!(app.memory_kind.is_some(), "a kind is chosen");
    keys_pressed(&mut app, &backend, &[Key::Char('l')]).await;
    assert!(app.memory_graph);
    assert_eq!(app.memory_kind, None, "the graph draws every kind");
    assert!(said(&app)?.1.contains("graph"));
    // Tab has no list to move along while the graph is shown.
    keys_pressed(&mut app, &backend, &[Key::Tab]).await;
    assert_eq!(app.memory_kind, None);
    keys_pressed(&mut app, &backend, &[Key::Char('l')]).await;
    assert!(!app.memory_graph);
    assert!(said(&app)?.1.contains("list"));
    Ok(())
}

#[tokio::test]
async fn l_on_the_population_page_reaches_the_graph_and_comes_back() -> anyhow::Result<()> {
    use crate::app::LayoutMode;
    let (_store, backend, mut app) = console().await?;
    assert_eq!(app.layout, LayoutMode::List, "it opens on the list");
    let mut seen = vec![app.layout];
    for _ in 0..4 {
        keys_pressed(&mut app, &backend, &[Key::Char('l')]).await;
        seen.push(app.layout);
    }
    assert_eq!(
        seen,
        vec![
            LayoutMode::List,
            LayoutMode::Dashboard,
            LayoutMode::Graph,
            LayoutMode::Table,
            LayoutMode::List,
        ]
    );
    Ok(())
}

#[tokio::test]
async fn p_graduates_the_shadow_that_was_found() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Tab, Key::Char('/')]).await;
    assert_eq!(app.focus, Focus::Shadows);
    typed(&mut app, &backend, "shadow:g4").await;
    keys_pressed(&mut app, &backend, &[Key::Enter]).await;
    let found = app.selected_shadow().context("no shadow is selected")?;
    assert_eq!(found.id.as_str(), "shadow:g4");
    anyhow::ensure!(
        found.status.as_str() != "graduated",
        "g4 is still in flight"
    );

    keys_pressed(&mut app, &backend, &[Key::Char('p')]).await;
    assert!(matches!(app.pending, Some(Pending::GraduateShadow(_))));
    keys_pressed(&mut app, &backend, &[Key::Char('y')]).await;
    let graduated = app
        .shadows
        .iter()
        .find(|s| s.id.as_str() == "shadow:g4")
        .context("g4 is still listed")?;
    assert_eq!(graduated.status.as_str(), "graduated");
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Settled);
    assert!(text.contains("graduated"), "{text}");

    // Asked again, it says why nothing happens.
    keys_pressed(&mut app, &backend, &[Key::Char('p')]).await;
    assert_eq!(app.mode, Mode::Normal);
    assert!(said(&app)?.1.contains("already graduated"));
    Ok(())
}

#[tokio::test]
async fn the_palette_runs_the_command_under_the_highlight() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    let before = app.theme().name;
    keys_pressed(&mut app, &backend, &[Key::Char(':')]).await;
    typed(&mut app, &backend, "theme").await;
    keys_pressed(&mut app, &backend, &[Key::Enter]).await;
    assert_eq!(app.mode, Mode::Normal);
    assert_ne!(app.theme().name, before, "the theme moved on");
    // A change the eye might miss says what it changed to.
    let (_, text) = said(&app)?;
    assert!(text.contains(app.theme().name), "{text}");
    Ok(())
}

#[tokio::test]
async fn the_palettes_freeze_brings_the_experts_into_view_first() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('4')]).await;
    assert_eq!(app.page, Page::Evals);
    apply(&mut app, &backend, &mut None, Action::FreezeExpert).await;
    assert_eq!(app.mode, Mode::Confirm);
    assert_eq!(app.page, Page::Population, "what it acts on is in view");
    assert_eq!(app.focus, Focus::Experts);
    Ok(())
}

#[tokio::test]
async fn x_halts_the_selected_run_and_then_offers_to_cancel() -> anyhow::Result<()> {
    let (store, backend, mut app) = console().await?;
    generation::save_head(
        &store,
        &GenerationHead::new(RunId::new("run:second"), chrono::Utc::now()),
    )
    .await?;
    keys_pressed(&mut app, &backend, &[Key::Char('r'), Key::Char('3')]).await;
    assert_eq!(app.page, Page::Loop);
    anyhow::ensure!(app.loop_heads.len() == 2, "two runs are listed");
    keys_pressed(&mut app, &backend, &[Key::Down]).await;
    let run = app
        .selected_loop_head()
        .context("no run is selected")?
        .run_id
        .clone();
    let first = app.loop_heads[0].run_id.clone();
    anyhow::ensure!(run.as_str() != first.as_str(), "the second run is selected");

    keys_pressed(&mut app, &backend, &[Key::Char('x'), Key::Char('y')]).await;
    assert_eq!(loop_control::load(&store, &run).await?, LoopCommand::Halt);
    assert_eq!(loop_control::load(&store, &first).await?, LoopCommand::Run);
    assert!(app.selected_halt_pending(), "the page says it of this run");
    assert!(said(&app)?.1.contains("halt requested"));

    // The same key now offers to cancel; declining keeps the halt.
    keys_pressed(&mut app, &backend, &[Key::Char('x')]).await;
    assert!(matches!(app.pending, Some(Pending::CancelHalt(_))));
    keys_pressed(&mut app, &backend, &[Key::Char('n')]).await;
    assert_eq!(loop_control::load(&store, &run).await?, LoopCommand::Halt);
    keys_pressed(&mut app, &backend, &[Key::Char('x'), Key::Char('y')]).await;
    assert_eq!(loop_control::load(&store, &run).await?, LoopCommand::Run);
    assert!(!app.selected_halt_pending());
    assert!(said(&app)?.1.contains("halt canceled"));
    Ok(())
}

#[tokio::test]
async fn a_failed_read_is_said_once_and_stays_until_reads_recover() -> anyhow::Result<()> {
    let (_store, _backend, mut app) = console().await?;
    let events = app.events.len();
    app.read_finished(Err(anyhow::anyhow!("connection refused")));
    assert!(app.reads_failing);
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Failed);
    assert!(text.contains("reload failed") && text.contains("connection refused"));
    assert_eq!(app.events.len(), events + 1);

    // Long after any other notice would have aged out, and after more failed
    // reads, it is still on screen and was said once.
    app.tick(60_000.0);
    app.read_finished(Err(anyhow::anyhow!("connection refused")));
    assert_eq!(said(&app)?.0, Tone::Failed);
    assert_eq!(
        app.events.len(),
        events + 1,
        "a run of failures is said once"
    );

    app.read_finished(Ok(None));
    assert!(!app.reads_failing);
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Settled);
    assert!(text.contains("again"), "{text}");
    Ok(())
}

#[tokio::test]
async fn a_key_that_changes_the_view_says_what_it_changed_to() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('t')]).await;
    assert!(said(&app)?.1.contains("theme · mist"));

    app.pin_fps(144);
    keys_pressed(&mut app, &backend, &[Key::Char('+')]).await;
    assert_eq!(app.target_fps, 165);
    assert!(said(&app)?.1.contains("frame-rate cap · 165"));
    keys_pressed(&mut app, &backend, &[Key::Char('-')]).await;
    assert!(said(&app)?.1.contains("frame-rate cap · 144"));
    keys_pressed(&mut app, &backend, &[Key::Char('0')]).await;
    assert!(app.auto_fps, "0 hands the cap back to the monitor");
    assert!(said(&app)?.1.contains("follows the monitor"));

    keys_pressed(&mut app, &backend, &[Key::Char('l')]).await;
    assert!(said(&app)?.1.contains("layout · dashboard"));
    keys_pressed(&mut app, &backend, &[Key::Char('s')]).await;
    assert!(said(&app)?.1.contains("sorted by name"));
    keys_pressed(&mut app, &backend, &[Key::Char('r')]).await;
    let (tone, text) = said(&app)?;
    assert_eq!(tone, Tone::Settled);
    assert!(text.contains("again"), "{text}");
    Ok(())
}

#[tokio::test]
async fn a_notice_ages_out() -> anyhow::Result<()> {
    let (_store, backend, mut app) = console().await?;
    keys_pressed(&mut app, &backend, &[Key::Char('t')]).await;
    assert!(app.current_notice().is_some());
    // An hour on: long past the few seconds a notice is kept.
    app.tick(3_600_000.0);
    assert!(app.current_notice().is_none());
    Ok(())
}

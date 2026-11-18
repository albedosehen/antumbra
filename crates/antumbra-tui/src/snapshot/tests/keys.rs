//! The key map, read against itself and against what the screen draws from
//! it: every key a view lists does what it is listed to do, no view binds a
//! key twice, and the footer and the help show the keys of the view they are
//! drawn for.

use super::*;
use crate::app::Page;
use crate::keys::{self, Key};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// The console with something in every list.
fn full_app() -> App {
    let mut app = demo_app();
    app.shadows = vec![graduated_shadow(), kept_shadow()];
    app.boundaries = vec![actionable_boundary(), missed_boundary()];
    with_services(&mut app);
    app
}

const LAYOUTS: [LayoutMode; 4] = [
    LayoutMode::List,
    LayoutMode::Dashboard,
    LayoutMode::Graph,
    LayoutMode::Table,
];
const LISTS: [Focus; 3] = [Focus::Experts, Focus::Shadows, Focus::Boundaries];
const OVERLAYS: [Mode; 10] = [
    Mode::Help,
    Mode::Palette,
    Mode::Filter,
    Mode::Events,
    Mode::Detail,
    Mode::Gate,
    Mode::Connect,
    Mode::Ask,
    Mode::Confirm,
    Mode::Knobs,
];

/// Every view the keys are read in: each page, in each layout and on each
/// list, and each overlay.
fn views() -> Vec<(String, App)> {
    let mut views = Vec::new();
    for page in Page::ALL {
        for layout in LAYOUTS {
            for focus in LISTS {
                let mut app = full_app();
                app.set_page(page);
                app.set_focus(focus);
                // After the list is chosen: a layout that draws the experts
                // alone moves the keys to them.
                app.set_layout(layout);
                views.push((
                    format!("{} · {} · {:?}", page.title(), layout.name(), app.focus),
                    app,
                ));
            }
        }
    }
    for mode in OVERLAYS {
        let mut app = full_app();
        app.mode = mode;
        views.push((format!("{mode:?}"), app));
    }
    views
}

#[test]
fn every_key_a_view_lists_does_what_it_is_listed_to_do() -> anyhow::Result<()> {
    for (view, app) in views() {
        let mut seen: Vec<Key> = Vec::new();
        for binding in keys::bindings(&app) {
            assert!(!binding.chip.is_empty(), "{view}: a binding with no key");
            assert!(
                !binding.does.is_empty(),
                "{view}: `{}` says nothing",
                binding.chip
            );
            for (key, action) in &binding.keys {
                assert!(
                    !seen.contains(key),
                    "{view}: {key:?} is bound twice, so one binding never fires"
                );
                seen.push(*key);
                assert_eq!(
                    keys::action_for(&app, *key),
                    Some(*action),
                    "{view}: {key:?} is listed under `{}`",
                    binding.chip
                );
            }
        }
    }
    Ok(())
}

#[test]
fn esc_does_nothing_on_the_main_view_and_ctrl_c_quits_from_every_view() -> anyhow::Result<()> {
    for (view, app) in views() {
        assert_eq!(
            keys::action_for(&app, Key::Ctrl('c')),
            Some(Action::Quit),
            "{view}"
        );
        if app.mode == Mode::Normal {
            assert_eq!(keys::action_for(&app, Key::Esc), None, "{view}");
            assert_eq!(
                keys::action_for(&app, Key::Char('q')),
                Some(Action::Quit),
                "{view}"
            );
        } else {
            // Esc gets out of every overlay, and never out of the console.
            let out = keys::action_for(&app, Key::Esc);
            assert!(
                matches!(out, Some(Action::Close | Action::Decline)),
                "{view}: Esc asks for {out:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn only_y_goes_ahead_and_a_stray_key_leaves_the_prompt_open() -> anyhow::Result<()> {
    let mut app = full_app();
    app.set_focus(Focus::Boundaries);
    app.request_action();
    anyhow::ensure!(app.mode == Mode::Confirm, "the confirm prompt opened");
    assert_eq!(
        keys::action_for(&app, Key::Char('y')),
        Some(Action::Confirm)
    );
    for cancel in [Key::Char('n'), Key::Esc, Key::Char('q')] {
        assert_eq!(
            keys::action_for(&app, cancel),
            Some(Action::Decline),
            "{cancel:?}"
        );
    }
    // Enter opens a drill-down on the main view; here it must not confirm.
    for stray in [
        Key::Enter,
        Key::Char('x'),
        Key::Tab,
        Key::Down,
        Key::Char(' '),
    ] {
        assert_eq!(keys::action_for(&app, stray), None, "{stray:?}");
    }
    Ok(())
}

#[test]
fn a_view_that_takes_text_types_its_letters() -> anyhow::Result<()> {
    for mode in [Mode::Palette, Mode::Filter, Mode::Ask] {
        let mut app = full_app();
        app.mode = mode;
        // Keys that do something on the main view are text here.
        for c in ['q', 'j', 'k', 'x', '?', ':', '1'] {
            assert_eq!(
                keys::action_for(&app, Key::Char(c)),
                Some(Action::Type(c)),
                "{mode:?}: `{c}`"
            );
        }
        assert_eq!(
            keys::action_for(&app, Key::Backspace),
            Some(Action::Backspace),
            "{mode:?}"
        );
        assert_eq!(
            keys::action_for(&app, Key::Enter),
            Some(Action::Submit),
            "{mode:?}"
        );
        assert_eq!(
            keys::action_for(&app, Key::Esc),
            Some(Action::Close),
            "{mode:?}"
        );
    }
    // The two with a list under the query move its highlight.
    for mode in [Mode::Palette, Mode::Filter] {
        let mut app = full_app();
        app.mode = mode;
        assert_eq!(keys::action_for(&app, Key::Up), Some(Action::Move(-1)));
        assert_eq!(
            keys::action_for(&app, Key::Ctrl('n')),
            Some(Action::Move(1))
        );
    }
    Ok(())
}

#[test]
fn tab_is_bound_only_where_the_list_it_moves_to_is_drawn() -> anyhow::Result<()> {
    for (view, app) in views() {
        if app.mode != Mode::Normal {
            continue;
        }
        // The population's three lists, where they are drawn; the kinds of
        // memory, where the list of them is.
        let lists = app.page == Page::Population
            && matches!(app.layout, LayoutMode::List | LayoutMode::Dashboard);
        let kinds = app.page == Page::Memory && !app.memory_graph;
        // And on the Services page it turns the walk round, where there is
        // a graph to walk.
        let walk = app.page == Page::Services && !app.services.is_empty();
        assert_eq!(
            keys::action_for(&app, Key::Tab),
            (lists || kinds || walk).then_some(Action::FocusNext),
            "{view}"
        );
        // Where it is not, the keys drive the list that is drawn.
        if app.page == Page::Population && !lists {
            assert_eq!(app.focus, Focus::Experts, "{view}");
        }
    }
    // The memory page's graph draws every kind, so there is none to move to.
    let mut app = full_app();
    app.set_page(Page::Memory);
    app.toggle_memory_graph();
    assert_eq!(keys::action_for(&app, Key::Tab), None);
    assert_eq!(
        keys::action_for(&app, Key::Char('l')),
        Some(Action::LayoutNext),
        "and `l` swaps it back for the list"
    );
    Ok(())
}

#[test]
fn x_is_bound_where_there_is_something_to_act_on() -> anyhow::Result<()> {
    for (view, app) in views() {
        if app.mode != Mode::Normal {
            continue;
        }
        let acts = matches!(app.page, Page::Population | Page::Loop);
        assert_eq!(
            keys::action_for(&app, Key::Char('x')),
            acts.then_some(Action::Act),
            "{view}"
        );
        let graduates = app.page == Page::Population && app.focus == Focus::Shadows;
        assert_eq!(
            keys::action_for(&app, Key::Char('p')),
            graduates.then_some(Action::GraduateShadow),
            "{view}"
        );
    }
    Ok(())
}

#[test]
fn the_footer_offers_to_act_only_to_an_operator() -> anyhow::Result<()> {
    let mut app = full_app();
    app.set_focus(Focus::Boundaries);
    let footer = |app: &mut App| -> anyhow::Result<String> {
        let text = to_text(&render(app, 120, 36, 1600.0)?);
        Ok(text.lines().last().unwrap_or_default().to_string())
    };
    assert!(footer(&mut app)?.contains("delete"), "an operator's footer");
    app.operator = false;
    let read_only = footer(&mut app)?;
    assert!(!read_only.contains("delete"), "{read_only}");
    // The key is still bound, so pressing it says why it does nothing.
    assert_eq!(keys::action_for(&app, Key::Char('x')), Some(Action::Act));
    Ok(())
}

#[test]
fn the_footer_fits_and_keeps_help_and_quit_at_any_width() -> anyhow::Result<()> {
    for width in [200, 120, 100, 80, 60, 40, 24] {
        for (view, mut app) in views() {
            let buf = render(&mut app, width, 24, 1600.0)?;
            let text = to_text(&buf);
            let footer = text.lines().last().unwrap_or_default();
            let wanted: &[&str] = match app.mode {
                Mode::Normal => &["help", "quit"],
                Mode::Confirm => &["cancel"],
                _ => &["close"],
            };
            for word in wanted {
                assert!(
                    footer.contains(word),
                    "{view} at {width} columns: no `{word}` in `{footer}`"
                );
            }
            // Nothing is cut off: the row ends on a whole hint.
            let whole = keys::bindings(&app)
                .iter()
                .any(|b| footer.trim_end().ends_with(b.hint));
            assert!(whole, "{view} at {width} columns: `{footer}`");
        }
    }
    Ok(())
}

#[test]
fn the_footer_at_80_columns_keeps_the_keys_a_newcomer_needs_first() -> anyhow::Result<()> {
    let mut app = full_app();
    let text = to_text(&render(&mut app, 80, 24, 1600.0)?);
    let footer = text.lines().last().unwrap_or_default();
    for word in ["move", "details", "help", "quit"] {
        assert!(footer.contains(word), "no `{word}` in `{footer}`");
    }
    Ok(())
}

#[test]
fn the_help_lists_every_key_of_the_page_it_was_opened_on() -> anyhow::Result<()> {
    for page in Page::ALL {
        for focus in LISTS {
            let mut app = full_app();
            app.set_page(page);
            app.set_focus(focus);
            let bindings = keys::page_bindings(&app);
            app.toggle_help();
            // Tall enough that the help does not need to scroll.
            let text = to_text(&render(&mut app, 120, 70, 1600.0)?);
            assert!(
                text.contains(&format!("Help · {}", page.title())),
                "{}: titled for its page:\n{text}",
                page.title()
            );
            for b in bindings {
                assert!(
                    text.contains(b.does),
                    "{} · {focus:?}: the help leaves out `{}` ({}):\n{text}",
                    page.title(),
                    b.chip,
                    b.does
                );
            }
        }
    }
    Ok(())
}

#[test]
fn the_help_scrolls_where_it_does_not_fit() -> anyhow::Result<()> {
    let mut app = full_app();
    app.toggle_help();
    // The help ends with the glossary, whose last entry is the operator.
    let last = "reads its own workspace and acts on nothing";
    let top = to_text(&render(&mut app, 100, 20, 1600.0)?);
    assert!(top.contains("move the selection"), "{top}");
    assert!(!top.contains(last), "the end is below the fold:\n{top}");
    app.detail_move(100);
    let end = to_text(&render(&mut app, 100, 20, 1600.0)?);
    assert!(end.contains(last), "scrolled to the end:\n{end}");
    assert!(!end.contains("move the selection"), "{end}");
    Ok(())
}

#[test]
fn a_palette_command_shows_the_key_that_does_the_same() -> anyhow::Result<()> {
    // The key beside a command is the one bound to its action on the main
    // view, so the palette cannot teach a key that does something else.
    let app = full_app();
    for command in crate::command::COMMANDS {
        let mut chars = command.key.chars();
        let key = match (chars.next(), chars.next()) {
            (None, _) => continue,
            (Some(c), None) => Key::Char(c),
            _ => anyhow::bail!("`{}`: `{}` is not one key", command.label, command.key),
        };
        assert_eq!(
            keys::action_for(&app, key),
            Some(command.action),
            "`{}` shows `{}`",
            command.label,
            command.key
        );
    }
    Ok(())
}

#[test]
fn the_line_above_the_footer_says_what_the_console_last_did() -> anyhow::Result<()> {
    use crate::theme::{Tone, NIGHT};
    let mut app = full_app();
    let second_last = |app: &mut App| -> anyhow::Result<(String, ratatui::buffer::Buffer)> {
        let buf = render(app, 120, 36, 1600.0)?;
        let text = to_text(&buf);
        let line = text.lines().rev().nth(1).unwrap_or_default().to_string();
        Ok((line, buf))
    };
    // Nothing said: the line is empty.
    assert_eq!(second_last(&mut app)?.0.trim(), "");

    app.note(Tone::Settled, "pruned · shadow:g3");
    let (line, _) = second_last(&mut app)?;
    assert_eq!(line.trim(), "● pruned · shadow:g3");

    // A refusal is under the failure's shape, in the danger colour.
    app.problem("not done · it is no longer there");
    let (line, buf) = second_last(&mut app)?;
    assert_eq!(line.trim(), "▲ not done · it is no longer there");
    let row = buf.area.height - 2;
    assert_eq!(buf[(1, row)].fg, NIGHT.danger, "the shape");
    assert_eq!(buf[(4, row)].fg, NIGHT.danger, "and the words");

    // It is still there, undimmed, with an overlay open over the page.
    app.toggle_help();
    let (line, buf) = second_last(&mut app)?;
    assert_eq!(line.trim(), "▲ not done · it is no longer there");
    assert_eq!(buf[(4, row)].fg, NIGHT.danger, "not dimmed with the page");
    Ok(())
}

#[test]
fn the_page_keys_name_every_page() -> anyhow::Result<()> {
    let app = full_app();
    let bindings = keys::page_bindings(&app);
    let Some(pages) = bindings.iter().find(|b| b.hint == "pages") else {
        anyhow::bail!("no binding goes to a page");
    };
    for page in Page::ALL {
        assert!(
            pages.does.contains(page.title()),
            "`{}` leaves out {}",
            pages.does,
            page.title()
        );
        let key = page.key().map(Key::Char);
        assert_eq!(
            key.and_then(|key| keys::action_for(&app, key)),
            Some(Action::Page(page)),
            "{}",
            page.title()
        );
    }
    Ok(())
}

#[test]
fn a_terminal_event_becomes_the_key_it_is() -> anyhow::Result<()> {
    let event = |code, modifiers| Key::from_event(KeyEvent::new(code, modifiers));
    assert_eq!(
        event(KeyCode::Char('c'), KeyModifiers::CONTROL),
        Some(Key::Ctrl('c'))
    );
    assert_eq!(
        event(
            KeyCode::Char('C'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT
        ),
        Some(Key::Ctrl('c'))
    );
    // Ctrl with Alt is how some keyboards type a character.
    assert_eq!(
        event(
            KeyCode::Char('q'),
            KeyModifiers::CONTROL | KeyModifiers::ALT
        ),
        Some(Key::Char('q'))
    );
    assert_eq!(
        event(KeyCode::Char('G'), KeyModifiers::SHIFT),
        Some(Key::Char('G'))
    );
    assert_eq!(event(KeyCode::Esc, KeyModifiers::NONE), Some(Key::Esc));
    assert_eq!(event(KeyCode::F(5), KeyModifiers::NONE), None);
    Ok(())
}

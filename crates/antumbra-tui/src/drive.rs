//! Carrying out what a key asked for: the one place an [`Action`] takes
//! effect, whether a key or the palette asked for it.
//!
//! [`press`] is a key press from start to finish, with no terminal in it, so
//! a test presses keys against the demo's store and reads what the console
//! did.

use crate::app::{App, Mode, Page};
use crate::backend::Backend;
use crate::command::Action;
use crate::keys::{self, Key};
use crate::theme::Tone;
use crate::transition;

/// The view-switch effect in flight, if any; an action may start one.
pub(crate) type Transition = Option<transition::Pending>;

/// A key press: what `key` asks for in the view the console is in, carried
/// out. A key that asks for nothing there does nothing.
pub(crate) async fn press(app: &mut App, backend: &Backend, transition: &mut Transition, key: Key) {
    let Some(mut action) = keys::action_for(app, key) else {
        return;
    };
    // Enter in the palette runs the command under the highlight, as if its
    // own key had been pressed on the main view.
    if action == Action::Submit && app.mode == Mode::Palette {
        let chosen = app.palette_action();
        app.close_overlay();
        match chosen {
            Some(chosen) => action = chosen,
            None => return,
        }
    }
    apply(app, backend, transition, action).await;
}

/// Carry out `action`, starting the view transition it implies.
pub(crate) async fn apply(
    app: &mut App,
    backend: &Backend,
    transition: &mut Transition,
    action: Action,
) {
    let before = app.mode;
    match action {
        Action::Quit => app.should_quit = true,
        Action::Reload => {
            app.refresh(backend).await;
            if !app.reads_failing {
                let source = app.source.clone();
                app.note(Tone::Settled, format!("read {source} again"));
            }
        }
        Action::Help => app.toggle_help(),
        Action::Palette => app.open_palette(),
        Action::Filter => app.open_filter(),
        Action::Events => app.open_events(),
        Action::Connect => app.open_connect(),
        Action::Knobs => app.open_knobs(),
        Action::Gate => app.open_gate(),
        Action::Ask => app.open_ask(),

        Action::Page(page) => {
            app.set_page(page);
            *transition = Some(transition::layout_switch());
        }
        Action::PageStep(delta) => {
            app.cycle_page(delta);
            *transition = Some(transition::layout_switch());
        }
        Action::Focus(target) => {
            app.show(target);
            *transition = Some(transition::focus_switch());
        }
        // The next list: the next kind on the Memory page, else the next of
        // the population's three.
        Action::FocusNext if app.page == Page::Memory => app.next_memory_kind(),
        Action::FocusNext => {
            app.toggle_focus();
            *transition = Some(transition::focus_switch());
        }
        // A change of view that is not plain to the eye says what it changed
        // to, since the footer no longer carries a readout for each.
        Action::Layout(mode) => {
            app.set_layout(mode);
            app.note(Tone::Open, format!("layout · {}", app.layout.name()));
            *transition = Some(transition::layout_switch());
        }
        Action::LayoutNext if app.page == Page::Memory => {
            app.toggle_memory_graph();
            let shown = if app.memory_graph { "graph" } else { "list" };
            app.note(Tone::Open, format!("memory · {shown}"));
            *transition = Some(transition::layout_switch());
        }
        Action::LayoutNext => {
            app.cycle_layout();
            app.note(Tone::Open, format!("layout · {}", app.layout.name()));
            *transition = Some(transition::layout_switch());
        }
        Action::CycleSort => {
            app.cycle_sort();
            app.note(Tone::Open, format!("experts sorted by {}", app.sort.name()));
        }
        Action::CycleTheme => {
            app.cycle_theme();
            app.note(Tone::Open, format!("theme · {}", app.theme().name));
            *transition = Some(transition::theme_wash(&app.theme()));
        }
        Action::FpsUp => {
            app.fps_up();
            app.note(Tone::Open, format!("frame-rate cap · {}", app.target_fps));
        }
        Action::FpsDown => {
            app.fps_down();
            app.note(Tone::Open, format!("frame-rate cap · {}", app.target_fps));
        }
        Action::FollowMonitor => {
            app.follow_monitor();
            app.note(
                Tone::Open,
                format!("frame-rate cap follows the monitor · {}", app.target_fps),
            );
        }

        Action::Move(delta) => match app.mode {
            Mode::Normal if delta > 0 => app.select_next(),
            Mode::Normal => app.select_prev(),
            Mode::Palette => app.palette_move(delta),
            Mode::Filter => app.filter_move(delta),
            Mode::Events => app.events_move(delta),
            Mode::Knobs => app.knob_move(delta),
            Mode::Help | Mode::Detail | Mode::Gate | Mode::Connect => app.detail_move(delta),
            Mode::Ask | Mode::Confirm => {}
        },
        Action::PageMove(delta) => match app.mode {
            Mode::Normal => app.select_page(delta),
            Mode::Events => app.events_move(delta * 10),
            Mode::Help | Mode::Detail | Mode::Gate | Mode::Connect => app.detail_move(delta * 10),
            Mode::Palette | Mode::Filter | Mode::Knobs | Mode::Ask | Mode::Confirm => {}
        },
        Action::First => app.select_first(),
        Action::Last => app.select_last(),
        Action::Submit => match app.mode {
            Mode::Normal => app.open_detail(),
            Mode::Filter => app.filter_apply(),
            Mode::Ask => ask_route(app, backend).await,
            Mode::Knobs => app.request_knobs(),
            Mode::Detail | Mode::Gate | Mode::Connect => app.close_overlay(),
            // The palette's Enter is resolved to its command before it gets
            // here; the rest have nothing to submit.
            Mode::Palette | Mode::Help | Mode::Events | Mode::Confirm => {}
        },

        Action::Act => app.request_action(),
        Action::GraduateShadow => app.request_graduate(),
        Action::FreezeExpert => app.request_freeze(),
        Action::CancelHalt => app.request_cancel_halt(),

        Action::Close => match app.mode {
            Mode::Knobs => app.close_knobs(),
            Mode::Confirm => app.cancel_action(),
            _ => app.close_overlay(),
        },
        Action::Type(c) => match app.mode {
            Mode::Palette => app.palette_input(c),
            Mode::Filter => app.filter_input(c),
            Mode::Ask => app.ask_input(c),
            _ => {}
        },
        Action::Backspace => match app.mode {
            Mode::Palette => app.palette_backspace(),
            Mode::Filter => app.filter_backspace(),
            Mode::Ask => app.ask_backspace(),
            _ => {}
        },
        Action::Turn(direction) => app.knob_turn(direction),
        Action::Confirm => app.apply_pending(backend).await,
        Action::Decline => app.cancel_action(),
    }
    // Whatever opened an overlay, it coalesces into view.
    if app.mode != before && app.mode != Mode::Normal {
        *transition = Some(transition::overlay_open());
    }
}

/// Route the ask's task, storing the distribution on the app. A failure is
/// said on the main view and yields an empty (escalate) result.
pub(crate) async fn ask_route(app: &mut App, backend: &Backend) {
    let routed = match backend.route(&app.ask_query, app.router.as_ref()).await {
        Ok(routed) => routed,
        Err(e) => {
            app.problem(format!("ask failed · {e:#}"));
            crate::backend::Routed::default()
        }
    };
    app.set_ask_result(&routed);
}

#[cfg(test)]
mod tests;

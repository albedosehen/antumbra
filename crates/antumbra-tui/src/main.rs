//! Antumbra operator console (ADR-0009): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, over the
//! same SurrealDB store the CLI drives. Read-only observatory for now;
//! interactive route/ask is the next layer.

mod app;
mod command;
mod events;
mod overlay;
mod pacing;
mod scroll;
mod snapshot;
mod theme;
mod transition;
mod ui;

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use tachyonfx::{Duration as FxDuration, EffectRenderer};

use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

use crate::app::{App, Mode};
use crate::command::Action;

#[derive(Parser)]
#[command(name = "antumbra-tui", about = "Antumbra operator console")]
struct Args {
    /// SurrealDB url (same store the CLI uses).
    #[arg(long, default_value = "surrealkv://./data/antumbra.skv", global = true)]
    url: String,
    /// Pin the frame-rate cap (Hz) to a fixed value. Omit to follow the active
    /// monitor's refresh rate automatically; adjust live with `+`/`-`, `a` to
    /// resume following.
    #[arg(long)]
    fps: Option<u32>,
    /// Ease the render rate down to 60fps after a few idle seconds to spare the
    /// CPU. Off by default so the console runs at the full cap continuously.
    #[arg(long)]
    power_save: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Headless: render one frame to an in-memory buffer (no terminal) and write
    /// `<out>.txt` (a Unicode grid, for e2e + plain inspection) and `<out>.png`
    /// (a colour screenshot). Use `--demo` to render a seeded population.
    Snapshot {
        /// Output path stem; `.txt` and `.png` are appended.
        #[arg(long, default_value = "antumbra-tui-snapshot")]
        out: String,
        #[arg(long, default_value_t = 140)]
        width: u16,
        #[arg(long, default_value_t = 40)]
        height: u16,
        /// Animation clock (ms) to freeze the frame at (deterministic output).
        #[arg(long, default_value_t = 1600.0)]
        at_ms: f64,
        /// Body layout to render: focused, dashboard, or graph.
        #[arg(long, default_value = "focused")]
        layout: String,
        /// Focused region: experts, shadows, or boundaries.
        #[arg(long, default_value = "experts")]
        focus: String,
        /// Overlay to render on top: none, help, or palette.
        #[arg(long, default_value = "none")]
        overlay: String,
        /// Render a seeded in-memory demo population instead of reading `--url`.
        #[arg(long)]
        demo: bool,
    },
    /// Diagnose the multi-monitor refresh detection: list every display and its
    /// rate, mark the one the console window resolves to (what the cap follows),
    /// and show the rate it would snap to. Run it on each monitor to confirm the
    /// active display is tracked (vs. pinned to the primary under ConPTY).
    Monitors,
}

fn main() -> Result<()> {
    // Host the runtime on a large-stack thread: SurrealDB's ACL subqueries
    // recurse past the 1 MB Windows main-thread stack (see the CLI note).
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(app_main())
        })?
        .join()
        .map_err(|_| anyhow::anyhow!("antumbra-tui worker thread panicked"))?
}

async fn app_main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Some(Command::Snapshot {
            out,
            width,
            height,
            at_ms,
            layout,
            focus,
            overlay,
            demo,
        }) => {
            let store = if demo {
                seed_demo().await?
            } else {
                connect(&args.url).await?
            };
            let mut app = App::load(&store).await?;
            app.set_layout(match layout.as_str() {
                "dashboard" => app::LayoutMode::Dashboard,
                "graph" => app::LayoutMode::Graph,
                _ => app::LayoutMode::Focused,
            });
            app.set_focus(match focus.as_str() {
                "shadows" => app::Focus::Shadows,
                "boundaries" => app::Focus::Boundaries,
                _ => app::Focus::Experts,
            });
            match overlay.as_str() {
                "help" => app.toggle_help(),
                "palette" => app.open_palette(),
                "events" => app.open_events(),
                "detail" => app.open_detail(),
                _ => {}
            }
            let buf = snapshot::render(&mut app, width, height, at_ms)?;
            let text = snapshot::to_text(&buf);
            std::fs::write(format!("{out}.txt"), &text)?;
            snapshot::save_png(&buf, &format!("{out}.png"), 13, 26)?;
            print!("{text}");
            eprintln!("wrote {out}.txt ({width}x{height}) and {out}.png");
            return Ok(());
        }
        Some(Command::Monitors) => {
            print_monitors();
            return Ok(());
        }
        None => {}
    }

    let store = connect(&args.url).await?;
    let mut app = App::load(&store).await?;
    // Capture the terminal window now, while it's focused, so the follow tracks
    // this window between monitors rather than re-reading focus each tick.
    app.capture_window();
    app.power_save = args.power_save;
    match args.fps {
        Some(fps) => app.pin_fps(fps),
        None => app.follow_monitor(),
    }
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &store).await;
    ratatui::restore();
    result
}

/// Print the multi-monitor refresh diagnostic: every display and its rate, the
/// one the terminal resolves to, and the rate the cap would follow.
fn print_monitors() {
    let active = pacing::active_device(0);
    let mons = pacing::monitors();
    if mons.is_empty() {
        println!("no displays detected (off Windows, or enumeration unavailable)");
    } else {
        println!("displays:");
        for m in &mons {
            let here = active.as_deref() == Some(m.device.as_str());
            println!(
                "  {} {:<14} {:>3} Hz{}{}",
                if here { ">" } else { " " },
                m.device,
                m.hz,
                if m.primary { "  primary" } else { "" },
                if here { "  <- terminal" } else { "" },
            );
        }
    }
    match pacing::detect_refresh(0) {
        Some(hz) => println!(
            "\nfollowing the terminal's monitor: {hz} Hz (cap snaps to {})",
            pacing::snap_refresh(hz)
        ),
        None => println!("\ndetection failed; the cap keeps its default value"),
    }
}

/// A seeded in-memory population so the headless snapshot (and its e2e test) has
/// content without a live store.
async fn seed_demo() -> Result<Store> {
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::{
        BoundaryId, Expert, ExpertId, FailureBoundary, Generation, Grain, Shadow, ShadowId,
        ShadowStatus,
    };
    use antumbra_store::repo::{boundary, expert, router, shadow};
    use chrono::Utc;

    let store = Store::connect_memory(EMBED_DIM).await?;
    let demo = [
        ("arith-specialist", 0.92f32, true),
        ("string-specialist", 0.81, true),
        ("datetime-specialist", 0.74, true),
        ("json-shaper", 0.55, false),
        ("regex-smith", 0.63, true),
    ];
    let now = Utc::now();
    for (name, fitness, frozen) in demo {
        expert::insert(
            &store,
            &Expert {
                id: ExpertId::new(format!("expert:{name}")),
                name: name.into(),
                base_model: "Qwen/Qwen2.5-Coder-1.5B".into(),
                artifact_uri: format!("mem://{name}"),
                capability_card: serde_json::json!({
                    "description": format!("demo specialist for {name}"),
                    "exemplars": ["ex-1", "ex-2", "ex-3"],
                }),
                capability_vec: Some(vec![0.0; EMBED_DIM]),
                fitness,
                frozen_at: frozen.then_some(now),
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                created_at: now,
            },
        )
        .await?;
    }
    router::save(
        &store,
        &LearnedRouter {
            weights: vec![1.0; EMBED_DIM],
            experts: demo
                .iter()
                .map(|(n, _, _)| RouterExpert {
                    id: ExpertId::new(format!("expert:{n}")),
                    centroid: vec![0.0; EMBED_DIM],
                })
                .collect(),
            temperature: 0.2,
            floor: 0.1,
        },
    )
    .await?;
    // A couple of boundaries (the antumbra) so the inspector has content: one
    // actionable (a recovered C/C' contrast that gates routing) and one open
    // (recorded from a collapse, not yet scoped).
    let scopes = [
        (
            "run `npm install`",
            "runtime",
            serde_json::json!({ "runtime": "deno" }),
            Some(serde_json::json!({ "runtime": "node" })),
            true,
        ),
        (
            "approach of shadow:g3",
            "",
            serde_json::json!({ "generation": 3 }),
            None,
            false,
        ),
    ];
    for (i, (behavior, feature, fail, near_ok, actionable)) in scopes.into_iter().enumerate() {
        boundary::upsert(
            &store,
            &FailureBoundary {
                id: BoundaryId::new(format!("boundary:demo-{i}")),
                behavior: behavior.into(),
                fail_context: fail,
                near_ok_context: near_ok,
                governing_features: if feature.is_empty() {
                    Vec::new()
                } else {
                    vec![feature.into()]
                },
                grain: actionable.then_some(Grain::Project),
                context_vec: None,
                ok_context_vec: None,
                confidence: if actionable { 0.8 } else { 0.3 },
                generation: Generation::ZERO,
                created_at: now,
            },
        )
        .await?;
    }
    // A few shadows (the penumbra) across the lifecycle, so the training view has
    // content: one still exploring, one graduated (rising reward), one collapsed.
    let shadows = [
        ("shadow:g4", ShadowStatus::Exploring, vec![0.10, 0.32, 0.55]),
        (
            "shadow:g3",
            ShadowStatus::Graduated,
            vec![0.20, 0.60, 0.85, 0.93],
        ),
        ("shadow:g2", ShadowStatus::Pruned, vec![0.05, 0.04, 0.06]),
    ];
    for (i, (id, status, reward_curve)) in shadows.into_iter().enumerate() {
        shadow::upsert(
            &store,
            &Shadow {
                id: ShadowId::new(id),
                parent_expert: None,
                adapter_uri: Some(format!("mem://{id}")),
                status,
                generation: Generation(4 - i as u32),
                reward_curve,
                created_at: now,
            },
        )
        .await?;
    }
    Ok(store)
}

async fn connect(url: &str) -> Result<Store> {
    let config = ConnectionConfig::builder()
        .url(url)
        .namespace("antumbra")
        .database("main")
        .build()?;
    Ok(Store::connect(config, EMBED_DIM).await?)
}

/// Run a palette-chosen [`Action`], queuing any view transition it implies. The
/// single place actions take effect, shared by the palette (keybindings call the
/// same app methods directly).
async fn apply_action(
    app: &mut App,
    store: &Store,
    transition: &mut Option<transition::Pending>,
    action: Action,
) -> Result<()> {
    match action {
        Action::Reload => app.reload(store).await?,
        Action::CycleTheme => {
            app.cycle_theme();
            *transition = Some(transition::theme_wash(&app.theme()));
        }
        Action::Focus(target) => {
            app.set_focus(target);
            *transition = Some(transition::focus_switch());
        }
        Action::Layout(mode) => {
            app.set_layout(mode);
            *transition = Some(transition::layout_switch());
        }
        Action::FpsUp => app.fps_up(),
        Action::FpsDown => app.fps_down(),
        Action::FollowMonitor => app.follow_monitor(),
        Action::Help => {
            app.toggle_help();
            if app.mode == Mode::Help {
                *transition = Some(transition::overlay_open());
            }
        }
        Action::Quit => app.should_quit = true,
    }
    Ok(())
}

async fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App, store: &Store) -> Result<()> {
    // Honour sub-16ms frame budgets on Windows (restored on drop).
    let _timer = pacing::TimerResolution::acquire();
    let mut last = Instant::now();
    // The active view-switch effect (and the scope it animates), processed
    // against the frame buffer and cleared when it finishes.
    let mut transition: Option<transition::Pending> = None;
    // Re-check the active monitor's refresh rate about once a second (cheap).
    let mut since_monitor_ms = 0.0;
    loop {
        let frame_start = Instant::now();
        let dt = frame_start.duration_since(last);
        last = frame_start;
        let dt_ms = dt.as_secs_f64() * 1000.0;
        app.tick(dt_ms);
        app.record_frame(dt_ms);
        since_monitor_ms += dt_ms;
        if since_monitor_ms >= 1000.0 {
            app.poll_monitor();
            since_monitor_ms = 0.0;
        }

        // Full rate while a transition animates or input is recent; ease to the
        // idle rate otherwise so a still view doesn't peg a core.
        let cap = app.frame_cap(transition.is_some());
        app.idle = cap < app.target_fps;

        let tick = FxDuration::from(dt);
        terminal.draw(|f| {
            ui::render(f, app);
            if let Some((effect, scope)) = transition.as_mut() {
                let area = match scope {
                    transition::Scope::Overlay => ui::overlay_area(app, f.area()),
                    fixed => Some(transition::scope_area(fixed, f.area())),
                };
                if let Some(area) = area {
                    f.render_effect(effect, area, tick);
                }
            }
        })?;
        if transition.as_ref().is_some_and(|(effect, _)| effect.done()) {
            transition = None;
        }

        // Pace to the (possibly idle) cap, spending the rest of the frame budget
        // waiting on (and draining) input so keys stay responsive at any rate.
        let budget = pacing::frame_budget(cap);
        while let Some(remaining) = budget.checked_sub(frame_start.elapsed()) {
            if remaining.is_zero() || !event::poll(remaining)? {
                break;
            }
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.note_input();
                    match app.mode {
                        Mode::Help => match key.code {
                            KeyCode::Char('?') | KeyCode::Esc | KeyCode::Char('q') => {
                                app.close_overlay()
                            }
                            _ => {}
                        },
                        Mode::Palette => match key.code {
                            KeyCode::Esc => app.close_overlay(),
                            KeyCode::Enter => {
                                let action = app.palette_action();
                                app.close_overlay();
                                if let Some(action) = action {
                                    apply_action(app, store, &mut transition, action).await?;
                                }
                            }
                            KeyCode::Backspace => app.palette_backspace(),
                            KeyCode::Up => app.palette_move(-1),
                            KeyCode::Down => app.palette_move(1),
                            KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                match c {
                                    'n' => app.palette_move(1),
                                    'p' => app.palette_move(-1),
                                    _ => {}
                                }
                            }
                            KeyCode::Char(c) => app.palette_input(c),
                            _ => {}
                        },
                        Mode::Filter => match key.code {
                            KeyCode::Esc => app.close_overlay(),
                            KeyCode::Enter => app.filter_apply(),
                            KeyCode::Backspace => app.filter_backspace(),
                            KeyCode::Up => app.filter_move(-1),
                            KeyCode::Down => app.filter_move(1),
                            KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                match c {
                                    'n' => app.filter_move(1),
                                    'p' => app.filter_move(-1),
                                    _ => {}
                                }
                            }
                            KeyCode::Char(c) => app.filter_input(c),
                            _ => {}
                        },
                        Mode::Events => match key.code {
                            KeyCode::Char('e') | KeyCode::Char('q') | KeyCode::Esc => {
                                app.close_overlay()
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.events_move(1),
                            KeyCode::Up | KeyCode::Char('k') => app.events_move(-1),
                            _ => {}
                        },
                        Mode::Detail => match key.code {
                            KeyCode::Enter | KeyCode::Char('q') | KeyCode::Esc => {
                                app.close_overlay()
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.detail_move(1),
                            KeyCode::Up | KeyCode::Char('k') => app.detail_move(-1),
                            KeyCode::PageDown => app.detail_move(10),
                            KeyCode::PageUp => app.detail_move(-10),
                            _ => {}
                        },
                        Mode::Normal => match key.code {
                            KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                            KeyCode::Down | KeyCode::Char('j') => app.select_next(),
                            KeyCode::Up | KeyCode::Char('k') => app.select_prev(),
                            KeyCode::Enter => {
                                app.open_detail();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Home | KeyCode::Char('g') => app.select_first(),
                            KeyCode::End | KeyCode::Char('G') => app.select_last(),
                            KeyCode::PageDown => app.select_page(1),
                            KeyCode::PageUp => app.select_page(-1),
                            KeyCode::Char(':') => {
                                app.open_palette();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Char('/') => {
                                app.open_filter();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Char('e') => {
                                app.open_events();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Tab => {
                                app.toggle_focus();
                                transition = Some(transition::focus_switch());
                            }
                            KeyCode::Char('l') | KeyCode::Char('L') => {
                                app.cycle_layout();
                                transition = Some(transition::layout_switch());
                            }
                            KeyCode::Char('t') => {
                                app.cycle_theme();
                                transition = Some(transition::theme_wash(&app.theme()));
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => app.fps_up(),
                            KeyCode::Char('-') | KeyCode::Char('_') => app.fps_down(),
                            KeyCode::Char('a') => app.follow_monitor(),
                            KeyCode::Char('r') => app.reload(store).await?,
                            KeyCode::Char('?') => {
                                app.toggle_help();
                                if app.mode == Mode::Help {
                                    transition = Some(transition::overlay_open());
                                }
                            }
                            _ => {}
                        },
                    }
                }
            }
        }
        if app.wants_reload() {
            app.reload(store).await?;
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

//! Antumbra operator console (`antumbra tui`): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, read from
//! the Antumbra server with the same address and token the hooks use.
//! Interactive: route-ask through the gate, a live event stream of what
//! changed, drill-down inspection, and operator actions (prune / delete)
//! behind a confirm, for an operator's token.

mod app;
mod backend;
mod command;
mod events;
mod here;
mod overlay;
mod pacing;
mod render;
mod scroll;
mod settings;
mod snapshot;
mod sovereign;
mod theme;
mod transition;
mod ui;

/// The workspace `--demo` seeds and reads its memories as.
pub(crate) const DEMO_TENANT: &str = "ws:demo";

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Subcommand;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use tachyonfx::{Duration as FxDuration, EffectRenderer};

use antumbra_core::ports::Embedder;
use antumbra_core::testing::FixedEmbedder;
use antumbra_store::{Store, EMBED_DIM};

use crate::app::{App, Mode};
use crate::backend::{Backend, Server};
use crate::command::Action;

/// `antumbra tui`'s options.
#[derive(clap::Args)]
pub struct Args {
    /// The Antumbra server to read from. Without it: `ANTUMBRA_URL`, else the
    /// one Claude Code's settings name (where `antumbra setup` writes it),
    /// else the one `antumbra setup` recorded, else this machine's.
    #[arg(long, global = true)]
    surface: Option<String>,
    /// The token to present. Without it, every token on hand is a candidate
    /// (`ANTUMBRA_TOKEN`, the file `ANTUMBRA_TOKEN_FILE` names,
    /// `~/.antumbra/token.txt`, and those in Claude Code's settings), and an
    /// operator's is preferred: it also shows the substrate's pages and may
    /// act on them (`antumbra-mcp --mint-token --operator`).
    #[arg(long, global = true)]
    token: Option<String>,
    /// Pin the frame-rate cap (Hz) to a fixed value. Omit to follow the active
    /// monitor's refresh rate automatically; adjust live with `+`/`-`, `a` to
    /// resume following.
    #[arg(long)]
    fps: Option<u32>,
    /// Ease the render rate down to 60fps after a few idle seconds to spare the
    /// CPU. Off by default so the console runs at the full cap continuously.
    #[arg(long)]
    power_save: bool,
    /// How visuals render: `auto` probes the terminal once and picks the richest
    /// tier; `canvas` forces the universal Braille/vector path (the safe default
    /// everywhere); `ascii` forces coarse dot markers for dumb terminals;
    /// `raster` forces the graphics-protocol image path (needs the `raster`
    /// build + a capable terminal).
    #[arg(
        long,
        value_enum,
        env = "ANTUMBRA_RENDER",
        default_value = "auto",
        global = true
    )]
    render: render::RenderMode,
    /// Show a seeded demo population held in memory instead of reading the
    /// server, so the console can be explored before anything is set up.
    /// Nothing persists.
    #[arg(long)]
    demo: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Headless: render one frame to an in-memory buffer (no terminal) and write
    /// `<out>.txt` (a Unicode grid, for e2e + plain inspection) and `<out>.png`
    /// (a color screenshot). Use `--demo` to render a seeded population.
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
        /// Top-level page to render: population, memory, loop, evals, or sovereign.
        #[arg(long, default_value = "population")]
        page: String,
        /// Body layout to render: focused, dashboard, or graph.
        #[arg(long, default_value = "focused")]
        layout: String,
        /// Focused region: experts, shadows, or boundaries.
        #[arg(long, default_value = "experts")]
        focus: String,
        /// Overlay to render on top: none, help, palette, events, detail, ask,
        /// or confirm.
        #[arg(long, default_value = "none")]
        overlay: String,
        /// Render the seeded demo population instead of reading the server.
        #[arg(long)]
        demo: bool,
    },
    /// Diagnose the multi-monitor refresh detection: list every display and its
    /// rate, mark the one the console window resolves to (what the cap follows),
    /// and show the rate it would snap to. Run it on each monitor to confirm the
    /// active display is tracked (vs. pinned to the primary under ConPTY).
    Monitors,
}

/// Where `args` says to read: the seeded demo, or the server with its token,
/// found where the agent's connection is configured when not given
/// ([`settings`]).
async fn backend(args: &Args, demo: bool) -> Result<Backend> {
    Ok(if demo {
        Backend::demo(seed_demo().await?)
    } else {
        let places = settings::Places::here();
        Backend::Server(Arc::new(Server::new(
            settings::surface(args.surface.clone(), &places),
            settings::token(args.token.clone(), &places)?,
        )))
    })
}

/// Run `antumbra tui`. Called on the CLI's runtime, whose thread already has
/// the large stack the demo's in-memory store needs.
pub async fn run(args: Args) -> Result<()> {
    match args.command {
        Some(Command::Snapshot {
            ref out,
            width,
            height,
            at_ms,
            ref page,
            ref layout,
            ref focus,
            ref overlay,
            demo,
        }) => {
            let backend = backend(&args, demo || args.demo).await?;
            let mut app = App::load(&backend)
                .await
                .with_context(|| backend.unreachable())?;
            app.set_page(match page.as_str() {
                "memory" => app::Page::Memory,
                "loop" => app::Page::Loop,
                "evals" => app::Page::Evals,
                "sovereign" => app::Page::Sovereign,
                _ => app::Page::Population,
            });
            app.set_layout(match layout.as_str() {
                "dashboard" => app::LayoutMode::Dashboard,
                "graph" => app::LayoutMode::Graph,
                "table" => app::LayoutMode::Table,
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
                "gate" => app.open_gate(),
                "connect" => app.open_connect(),
                "detail" => app.open_detail(),
                "confirm" => {
                    app.set_focus(app::Focus::Boundaries);
                    app.request_action();
                }
                "ask" => {
                    app.open_ask();
                    for c in "string handling".chars() {
                        app.ask_input(c);
                    }
                    ask_route(&mut app, &backend).await;
                }
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

    // `--demo` seeds a throwaway in-memory population so the console can be
    // explored before anything is set up; otherwise read the server.
    let backend = backend(&args, args.demo).await?;
    let mut app = App::load(&backend)
        .await
        .with_context(|| backend.unreachable())?;
    app.demo = args.demo;
    // Capture the terminal window now, while it's focused, so the follow tracks
    // this window between monitors rather than re-reading focus each tick.
    app.capture_window();
    app.power_save = args.power_save;
    // Resolve the render tier once, before the loop (probe-safe: degrades to the
    // universal Canvas path off-TTY, in a hostile multiplexer, or on Windows).
    app.render_tier = render::resolve_tier(args.render);
    match args.fps {
        Some(fps) => app.pin_fps(fps),
        None => app.follow_monitor(),
    }
    let mut terminal = ratatui::init();
    let result = run_loop(&mut terminal, &mut app, &backend).await;
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
    use antumbra_core::generational::{GenerationHead, LoopState};
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::{
        BoundaryId, EdgeType, EvalStatus, EvaluationRun, Expert, ExpertId, FailureBoundary,
        Generation, Grain, Memory, MemoryEdge, MemoryNetwork, RunId, Shadow, ShadowId,
        ShadowStatus, SubjectKind,
    };
    use antumbra_store::repo::{
        boundary, edge, evaluation, expert, generation, memory, router, shadow,
    };
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
    // Embed each expert's name for its capability vector and router centroid, so
    // the route-ask (which embeds the query the same way) routes coherently and
    // the drill-down's route preview shows a real distribution.
    let embedder = FixedEmbedder::new(EMBED_DIM);
    let mut probe = std::collections::HashMap::new();
    for (name, _, _) in &demo {
        probe.insert(*name, embedder.embed(name).await?);
    }
    for (name, fitness, frozen) in demo.iter().copied() {
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
                capability_vec: Some(probe[name].clone()),
                fitness,
                frozen_at: frozen.then_some(now),
                generation: Generation::ZERO,
                owner: None,
                compartment: None,
                // A `mem://` artifact is on nobody's disk: placed nowhere,
                // servable anywhere, which is what a demo wants.
                placed_on: None,
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
                    centroid: probe[*n].clone(),
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
                owner: None,
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
    // Penumbra memory traces (the Memory page) across the three networks, with one
    // consolidated into an expert, plus edges including a contradiction and a
    // supersession; the native consolidation→retire signal.
    let tenant = DEMO_TENANT;
    let mems = [
        (
            "memory:deno",
            MemoryNetwork::World,
            "prefer deno over node for new scripts",
            0.86,
            4u32,
            false,
            false,
        ),
        (
            "memory:surql",
            MemoryNetwork::World,
            "use surql-rs builders, never raw SurrealQL",
            0.93,
            7,
            false,
            true,
        ),
        (
            "memory:node-incident",
            MemoryNetwork::Bank,
            "npm install failed under the deno runtime",
            0.70,
            2,
            true,
            false,
        ),
        (
            "memory:retry",
            MemoryNetwork::Bank,
            "retrying the flaky step fixed the test",
            0.58,
            1,
            true,
            false,
        ),
        (
            "memory:terse",
            MemoryNetwork::Opinion,
            "keep operator docs terse, no emojis",
            0.80,
            3,
            false,
            false,
        ),
        (
            "memory:canvas",
            MemoryNetwork::Opinion,
            "canvas-first rendering beats raster over SSH",
            0.75,
            2,
            false,
            false,
        ),
    ];
    for (id, network, content, confidence, reinforcement, volatile, consolidated) in mems {
        let mut m = Memory::new(id, tenant, network, content, confidence, now)
            .by("user:operator", "windows")
            .volatile(volatile)
            .with_evidence(vec!["session-log".into()]);
        m.reinforcement = reinforcement;
        if consolidated {
            m.mark_consolidated(ExpertId::new("expert:json-shaper"), now);
        }
        memory::upsert(&store, &m).await?;
    }
    let edges = [
        (
            "memory:node-incident",
            "memory:deno",
            EdgeType::Contradicts,
            0.8,
        ),
        ("memory:deno", "memory:node-incident", EdgeType::Caused, 0.6),
        (
            "memory:retry",
            "memory:node-incident",
            EdgeType::Follows,
            0.5,
        ),
        ("memory:surql", "memory:deno", EdgeType::References, 0.4),
        ("memory:terse", "memory:canvas", EdgeType::Supersedes, 0.5),
    ];
    for (from, to, edge_type, weight) in edges {
        edge::relate(
            &store,
            &MemoryEdge::new(tenant, from, to, edge_type, weight, now),
        )
        .await?;
    }
    // The generational loop head (mid-cycle) for the Loop page.
    let mut head = GenerationHead::new(RunId::new("run:demo"), now);
    head.generation = Generation(4);
    head.state = LoopState::Score;
    generation::save_head(&store, &head).await?;
    // A few evaluation runs across subjects/statuses for the Evals page (one
    // failure = the regression tripwire firing).
    let runs = [
        (
            "run:e1",
            SubjectKind::Expert,
            "expert:arith-specialist",
            EvalStatus::Success,
            Some("a1b2c3d4"),
        ),
        (
            "run:e2",
            SubjectKind::Expert,
            "expert:json-shaper",
            EvalStatus::Success,
            Some("9f8e7d6c"),
        ),
        (
            "run:e3",
            SubjectKind::Shadow,
            "shadow:g3",
            EvalStatus::Failure,
            None,
        ),
        (
            "run:e4",
            SubjectKind::Router,
            "router",
            EvalStatus::Running,
            None,
        ),
    ];
    for (id, kind, subject, status, fp) in runs {
        evaluation::insert(
            &store,
            &EvaluationRun {
                run_id: RunId::new(id),
                subject_kind: kind,
                subject_id: subject.into(),
                corpus_task_id: "task:arith".into(),
                status,
                metrics: None,
                regression_fingerprint: fp.map(str::to_string),
                created_at: now,
            },
        )
        .await?;
    }
    Ok(store)
}

/// Run a palette-chosen [`Action`], queuing any view transition it implies. The
/// single place actions take effect, shared by the palette (keybindings call the
/// same app methods directly).
async fn apply_action(
    app: &mut App,
    backend: &Backend,
    transition: &mut Option<transition::Pending>,
    action: Action,
) -> Result<()> {
    match action {
        Action::Reload => app.refresh(backend).await,
        Action::CycleTheme => {
            app.cycle_theme();
            *transition = Some(transition::theme_wash(&app.theme()));
        }
        Action::Page(page) => {
            app.set_page(page);
            *transition = Some(transition::layout_switch());
        }
        Action::CycleSort => app.cycle_sort(),
        Action::CancelHalt => app.cancel_halt(backend).await,
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
        Action::Ask => {
            app.open_ask();
            *transition = Some(transition::overlay_open());
        }
        Action::Events => {
            app.open_events();
            *transition = Some(transition::overlay_open());
        }
        Action::Gate => {
            app.open_gate();
            *transition = Some(transition::overlay_open());
        }
        Action::Connect => {
            app.open_connect();
            *transition = Some(transition::overlay_open());
        }
        Action::GraduateShadow => {
            app.request_graduate();
            if app.mode == Mode::Confirm {
                *transition = Some(transition::overlay_open());
            }
        }
        Action::FreezeExpert => {
            app.request_freeze();
            if app.mode == Mode::Confirm {
                *transition = Some(transition::overlay_open());
            }
        }
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

/// Route the ask query through the gate, storing the distribution on the app.
/// A failure is said in the event stream and yields an empty (escalate) result.
async fn ask_route(app: &mut App, backend: &Backend) {
    let routed = match backend.route(&app.ask_query, app.router.as_ref()).await {
        Ok(routed) => routed,
        Err(e) => {
            app.operator_event(format!("ask failed · {e:#}"));
            Vec::new()
        }
    };
    app.set_ask_result(&routed);
}

async fn run_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    backend: &Backend,
) -> Result<()> {
    // Honor sub-16ms frame budgets on Windows (restored on drop).
    let _timer = pacing::TimerResolution::acquire();
    // The periodic refresh runs OFF the render thread: a snapshot of thousands
    // of memories takes long enough to read that awaiting it inline stutters
    // input. At most one read is in flight; its result is applied (cheaply)
    // when ready. A run of failed reads is said once, when it starts.
    let mut reload_inflight: Option<
        tokio::task::JoinHandle<anyhow::Result<Option<app::Snapshot>>>,
    > = None;
    let mut reads_failing = false;
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
                                    apply_action(app, backend, &mut transition, action).await?;
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
                        Mode::Gate => match key.code {
                            KeyCode::Enter | KeyCode::Char('q') | KeyCode::Esc => {
                                app.close_overlay()
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.detail_move(1),
                            KeyCode::Up | KeyCode::Char('k') => app.detail_move(-1),
                            KeyCode::PageDown => app.detail_move(10),
                            KeyCode::PageUp => app.detail_move(-10),
                            _ => {}
                        },
                        Mode::Connect => match key.code {
                            KeyCode::Enter | KeyCode::Char('q') | KeyCode::Esc => {
                                app.close_overlay()
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.detail_move(1),
                            KeyCode::Up | KeyCode::Char('k') => app.detail_move(-1),
                            KeyCode::PageDown => app.detail_move(10),
                            KeyCode::PageUp => app.detail_move(-10),
                            _ => {}
                        },
                        Mode::Ask => match key.code {
                            KeyCode::Esc => app.close_overlay(),
                            KeyCode::Backspace => app.ask_backspace(),
                            KeyCode::Enter => ask_route(app, backend).await,
                            KeyCode::Char(c) => app.ask_input(c),
                            _ => {}
                        },
                        Mode::Confirm => match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                app.apply_pending(backend).await
                            }
                            _ => app.cancel_action(),
                        },
                        Mode::Knobs => match key.code {
                            KeyCode::Char('K') | KeyCode::Char('q') | KeyCode::Esc => {
                                app.close_knobs()
                            }
                            KeyCode::Down | KeyCode::Char('j') => app.knob_move(1),
                            KeyCode::Up | KeyCode::Char('k') => app.knob_move(-1),
                            KeyCode::Right | KeyCode::Char('l') => app.knob_turn(1),
                            KeyCode::Left | KeyCode::Char('h') => app.knob_turn(-1),
                            KeyCode::Enter => app.request_knobs(),
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
                            KeyCode::Char('c') => {
                                app.open_connect();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Char('K') => {
                                app.open_knobs();
                                transition = Some(transition::overlay_open());
                            }
                            KeyCode::Char('x') => {
                                app.request_action();
                                if app.mode == Mode::Confirm {
                                    transition = Some(transition::overlay_open());
                                }
                            }
                            KeyCode::Tab => {
                                app.toggle_focus();
                                transition = Some(transition::focus_switch());
                            }
                            KeyCode::Char(']') => {
                                app.cycle_page(1);
                                transition = Some(transition::layout_switch());
                            }
                            KeyCode::Char('[') => {
                                app.cycle_page(-1);
                                transition = Some(transition::layout_switch());
                            }
                            KeyCode::Char(c) if app::Page::index_for_key(c).is_some() => {
                                if let Some(index) = app::Page::index_for_key(c) {
                                    app.goto_page(index);
                                }
                                transition = Some(transition::layout_switch());
                            }
                            KeyCode::Char('l') | KeyCode::Char('L') => {
                                app.cycle_layout();
                                transition = Some(transition::layout_switch());
                            }
                            KeyCode::Char('s') => app.cycle_sort(),
                            KeyCode::Char('t') => {
                                app.cycle_theme();
                                transition = Some(transition::theme_wash(&app.theme()));
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => app.fps_up(),
                            KeyCode::Char('-') | KeyCode::Char('_') => app.fps_down(),
                            KeyCode::Char('a') => app.follow_monitor(),
                            KeyCode::Char('r') => app.refresh(backend).await,
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
        // Apply a finished off-thread read (cheap), then start a new one when the
        // periodic tick is due. The read never blocks the render loop; the
        // explicit `r` / operator-action reloads stay inline.
        if reload_inflight.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(handle) = reload_inflight.take() {
                match handle.await {
                    Ok(Ok(read)) => {
                        if reads_failing {
                            reads_failing = false;
                            app.operator_event(format!("reading {} again", backend.describe()));
                        }
                        if let Some(snap) = read {
                            app.apply_snapshot(snap);
                        }
                    }
                    Ok(Err(e)) if !reads_failing => {
                        reads_failing = true;
                        app.operator_event(format!("reload failed · {e:#}"));
                    }
                    _ => {}
                }
            }
        }
        if reload_inflight.is_none() && app.wants_reload() {
            app.since_reload_ms = 0.0; // restart the interval from this request
            let backend = backend.clone();
            reload_inflight = Some(tokio::spawn(async move { backend.snapshot().await }));
        }
        if app.should_quit {
            return Ok(());
        }
    }
}

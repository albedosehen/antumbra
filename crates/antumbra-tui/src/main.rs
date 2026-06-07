//! Antumbra operator console (ADR-0009): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, over the
//! same SurrealDB store the CLI drives. Read-only observatory for now;
//! interactive route/ask is the next layer.

mod app;
mod pacing;
mod snapshot;
mod theme;
mod transition;
mod ui;

use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use tachyonfx::{Duration as FxDuration, Effect, EffectRenderer};

use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

use crate::app::App;

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
        /// Render a seeded in-memory demo population instead of reading `--url`.
        #[arg(long)]
        demo: bool,
    },
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
    if let Some(Command::Snapshot {
        out,
        width,
        height,
        at_ms,
        demo,
    }) = args.command
    {
        let store = if demo {
            seed_demo().await?
        } else {
            connect(&args.url).await?
        };
        let mut app = App::load(&store).await?;
        let buf = snapshot::render(&mut app, width, height, at_ms)?;
        let text = snapshot::to_text(&buf);
        std::fs::write(format!("{out}.txt"), &text)?;
        snapshot::save_png(&buf, &format!("{out}.png"), 13, 26)?;
        print!("{text}");
        eprintln!("wrote {out}.txt ({width}x{height}) and {out}.png");
        return Ok(());
    }

    let store = connect(&args.url).await?;
    let mut app = App::load(&store).await?;
    match args.fps {
        Some(fps) => app.pin_fps(fps),
        None => app.follow_monitor(),
    }
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &store).await;
    ratatui::restore();
    result
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

async fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App, store: &Store) -> Result<()> {
    // Honour sub-16ms frame budgets on Windows (restored on drop).
    let _timer = pacing::TimerResolution::acquire();
    let mut last = Instant::now();
    // The active view-switch effect, processed against the frame buffer and
    // cleared when it finishes.
    let mut transition: Option<Effect> = None;
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

        let tick = FxDuration::from(dt);
        terminal.draw(|f| {
            ui::render(f, app);
            if let Some(effect) = transition.as_mut() {
                f.render_effect(effect, transition::detail_area(f.area()), tick);
            }
        })?;
        if transition.as_ref().is_some_and(Effect::done) {
            transition = None;
        }

        // Pace to the target rate, spending the rest of the frame budget waiting
        // on (and draining) input so keys stay responsive at any cap.
        let budget = pacing::frame_budget(app.target_fps);
        while let Some(remaining) = budget.checked_sub(frame_start.elapsed()) {
            if remaining.is_zero() || !event::poll(remaining)? {
                break;
            }
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Down | KeyCode::Char('j') => app.select_next(),
                        KeyCode::Up | KeyCode::Char('k') => app.select_prev(),
                        KeyCode::Tab => {
                            app.toggle_focus();
                            transition = Some(transition::focus_switch());
                        }
                        KeyCode::Char('t') => {
                            app.cycle_theme();
                            transition = Some(transition::theme_wash(&app.theme()));
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => app.fps_up(),
                        KeyCode::Char('-') | KeyCode::Char('_') => app.fps_down(),
                        KeyCode::Char('a') => app.follow_monitor(),
                        KeyCode::Char('r') => app.reload(store).await?,
                        _ => {}
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

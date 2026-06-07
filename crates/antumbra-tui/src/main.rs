//! Antumbra operator console (ADR-0009): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, over the
//! same SurrealDB store the CLI drives. Read-only observatory for now;
//! interactive route/ask is the next layer.

mod app;
mod snapshot;
mod ui;

use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};

use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

use crate::app::App;

#[derive(Parser)]
#[command(name = "antumbra-tui", about = "Antumbra operator console")]
struct Args {
    /// SurrealDB url (same store the CLI uses).
    #[arg(long, default_value = "surrealkv://./data/antumbra.skv", global = true)]
    url: String,
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
    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &store).await;
    ratatui::restore();
    result
}

/// A seeded in-memory population so the headless snapshot (and its e2e test) has
/// content without a live store.
async fn seed_demo() -> Result<Store> {
    use antumbra_core::router::{LearnedRouter, RouterExpert};
    use antumbra_core::{BoundaryId, Expert, ExpertId, FailureBoundary, Generation, Grain};
    use antumbra_store::repo::{boundary, expert, router};
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
    let mut last = Instant::now();
    loop {
        let now = Instant::now();
        app.tick(now.duration_since(last).as_secs_f64() * 1000.0);
        last = now;

        terminal.draw(|f| ui::render(f, app))?;

        // Poll paces the loop (~25 fps) and reads input when present.
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => app.should_quit = true,
                        KeyCode::Down | KeyCode::Char('j') => app.select_next(),
                        KeyCode::Up | KeyCode::Char('k') => app.select_prev(),
                        KeyCode::Tab => app.toggle_focus(),
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

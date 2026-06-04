//! Antumbra operator console (ADR-0009): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, over the
//! same SurrealDB store the CLI drives. Read-only observatory for now;
//! interactive route/ask is the next layer.

mod app;
mod ui;

use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};

use antumbra_store::{ConnectionConfig, Store, EMBED_DIM};

use crate::app::App;

#[derive(Parser)]
#[command(name = "antumbra-tui", about = "Antumbra operator console")]
struct Args {
    /// SurrealDB url (same store the CLI uses).
    #[arg(long, default_value = "surrealkv://./data/antumbra.skv")]
    url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let store = connect(&args.url).await?;
    let mut app = App::load(&store).await?;

    let mut terminal = ratatui::init();
    let result = run(&mut terminal, &mut app, &store).await;
    ratatui::restore();
    result
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

//! Antumbra operator console (`antumbra tui`): a live, animated view of the
//! population (umbra), boundaries (antumbra), and the learned gate, read from
//! the Antumbra server with the same address and token the hooks use.
//! Interactive: route-ask through the gate, a live event stream of what
//! changed, drill-down inspection, and operator actions (prune / delete)
//! behind a confirm, for an operator's token.

mod app;
mod backend;
mod command;
mod demo;
mod drive;
mod events;
mod here;
mod keys;
mod named;
mod netgraph;
mod overlay;
mod pacing;
mod render;
mod scroll;
mod services;
mod settings;
mod snapshot;
mod sovereign;
mod theme;
mod transition;
mod ui;
mod words;

/// The workspace `--demo` seeds and reads its memories as.
pub(crate) const DEMO_TENANT: &str = "ws:demo";

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Subcommand;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use tachyonfx::{Duration as FxDuration, EffectRenderer};

use crate::app::App;
use crate::backend::{Backend, Server};

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
    /// monitor's refresh rate automatically; adjust live with `+`/`-`, `0` to
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
    /// The palette to open in: `night` (the dashboard's dark palette),
    /// `mist` (its light one, for a light terminal) or `mono` (greys on the
    /// terminal's own background). `t` cycles them while the console runs.
    #[arg(
        long,
        env = "ANTUMBRA_THEME",
        default_value = theme::NIGHT.name,
        value_parser = clap::builder::PossibleValuesParser::new(theme::ALL.map(|t| t.name)),
        global = true
    )]
    theme: String,
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
        /// Top-level page to render: population, memory, training, evaluations,
        /// agent, or services (the former names loop, evals and sovereign are
        /// taken too).
        #[arg(long, default_value = "population")]
        page: String,
        /// The population page's layout: list, dashboard, graph, or table
        /// (focused, the list's former name, is taken too).
        #[arg(long, default_value = "list")]
        layout: String,
        /// The list in front: experts, candidates, or boundaries (shadows,
        /// the candidates' former name, is taken too).
        #[arg(long, default_value = "experts")]
        focus: String,
        /// Panel to render on top: none, help, palette, events, detail,
        /// router, connect, knobs, ask, or confirm (gate, the router's former
        /// name, is taken too).
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
        Backend::demo(demo::seed().await?)
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
            // Read what is asked for before anything is read from a server: a
            // name nothing has is refused at once, and never drawn as
            // something else.
            let page = named::page(page)?;
            let layout = named::layout(layout)?;
            let list = named::list(focus)?;
            let panel = named::panel(overlay)?;
            let backend = backend(&args, demo || args.demo).await?;
            let mut app = App::load(&backend)
                .await
                .with_context(|| backend.unreachable())?;
            app.demo = demo || args.demo;
            app.theme_idx = theme::Theme::index_of(&args.theme);
            app.set_page(page);
            app.set_layout(layout);
            // The Memory page has two views of its own, and `graph` asks for
            // the one that is not its list.
            if page == app::Page::Memory && layout == app::LayoutMode::Graph {
                app.toggle_memory_graph();
            }
            app.set_focus(list);
            match panel {
                named::Panel::None => {}
                named::Panel::Help => app.toggle_help(),
                named::Panel::Palette => app.open_palette(),
                named::Panel::Events => app.open_events(),
                named::Panel::Router => app.open_gate(),
                named::Panel::Connect => app.open_connect(),
                named::Panel::Knobs => app.open_knobs(),
                named::Panel::Detail => app.open_detail(),
                named::Panel::Confirm => {
                    app.set_focus(app::Focus::Boundaries);
                    app.request_action();
                }
                named::Panel::Ask => {
                    app.open_ask();
                    for c in "string handling".chars() {
                        app.ask_input(c);
                    }
                    drive::ask_route(&mut app, &backend).await;
                }
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
    app.theme_idx = theme::Theme::index_of(&args.theme);
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
    // when ready.
    let mut reload_inflight: Option<
        tokio::task::JoinHandle<anyhow::Result<Option<app::Snapshot>>>,
    > = None;
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
            if let Event::Key(event) = event::read()? {
                if event.kind == KeyEventKind::Press {
                    app.note_input();
                    if let Some(key) = keys::Key::from_event(event) {
                        drive::press(app, backend, &mut transition, key).await;
                    }
                }
            }
        }
        // Apply a finished off-thread read (cheap), then start a new one when the
        // periodic tick is due. The read never blocks the render loop; the
        // explicit `r` / operator-action reloads stay inline.
        if reload_inflight.as_ref().is_some_and(|h| h.is_finished()) {
            if let Some(handle) = reload_inflight.take() {
                // A read whose task was cancelled has nothing to apply.
                if let Ok(read) = handle.await {
                    app.read_finished(read);
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

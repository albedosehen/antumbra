//! How the console looks, as far as a buffer can say: what is behind every
//! cell, which glyphs are drawn, and that a state keeps its colour where the
//! selection lies over it. The words and the layout are the goldens'.

use super::*;
use crate::app::Page;
use crate::theme::{Theme, Tone, MIST, MONO, NIGHT};
use ratatui::buffer::Buffer;
use ratatui::style::Color;

/// The console with something of every kind in it, in `theme`.
fn full_app(theme: &Theme) -> App {
    let mut app = demo_app();
    app.theme_idx = Theme::index_of(theme.name);
    app.shadows = vec![graduated_shadow(), kept_shadow()];
    app.boundaries = vec![actionable_boundary(), missed_boundary()];
    with_services(&mut app);
    app.demo = true;
    app
}

/// Every view the console has: the pages and layouts, then each overlay.
fn every_view(theme: &Theme) -> anyhow::Result<Vec<(String, Buffer)>> {
    let mut views = plain_views(theme)?;
    views.extend(overlay_views(theme)?);
    Ok(views)
}

/// Each page, and each layout and focus of the population page, with no
/// overlay open, so nothing is dimmed.
fn plain_views(theme: &Theme) -> anyhow::Result<Vec<(String, Buffer)>> {
    let mut views = Vec::new();
    for page in Page::ALL {
        let mut app = full_app(theme);
        app.set_page(page);
        views.push((
            format!("page {}", page.title()),
            render(&mut app, 120, 36, 1600.0)?,
        ));
    }
    for layout in [
        LayoutMode::List,
        LayoutMode::Dashboard,
        LayoutMode::Graph,
        LayoutMode::Table,
    ] {
        for focus in [Focus::Experts, Focus::Shadows, Focus::Boundaries] {
            let mut app = full_app(theme);
            app.set_layout(layout);
            app.set_focus(focus);
            views.push((
                format!("layout {} focus {focus:?}", layout.name()),
                render(&mut app, 120, 36, 1600.0)?,
            ));
        }
    }
    Ok(views)
}

/// What opens an overlay.
type Open = fn(&mut App);

/// Each overlay, open over the population page.
fn overlay_views(theme: &Theme) -> anyhow::Result<Vec<(String, Buffer)>> {
    let mut views = Vec::new();
    let overlays: [(&str, Open); 9] = [
        ("help", App::toggle_help),
        ("palette", App::open_palette),
        ("filter", App::open_filter),
        ("events", App::open_events),
        ("detail", App::open_detail),
        ("gate", App::open_gate),
        ("connect", App::open_connect),
        ("ask", App::open_ask),
        ("knobs", App::open_knobs),
    ];
    for (name, open) in overlays {
        let mut app = full_app(theme);
        app.operator_event("string-specialist emerged".into());
        open(&mut app);
        views.push((
            format!("overlay {name}"),
            render(&mut app, 120, 36, 1600.0)?,
        ));
    }
    let mut app = full_app(theme);
    app.set_focus(Focus::Boundaries);
    app.request_action();
    anyhow::ensure!(app.mode == Mode::Confirm, "the confirm prompt opened");
    views.push(("overlay confirm".into(), render(&mut app, 120, 36, 1600.0)?));
    Ok(views)
}

#[test]
fn night_and_mist_paint_their_ground_behind_every_cell() -> anyhow::Result<()> {
    for theme in [NIGHT, MIST] {
        for (view, buf) in every_view(&theme)? {
            for y in 0..buf.area.height {
                for x in 0..buf.area.width {
                    assert_ne!(
                        buf[(x, y)].bg,
                        Color::Reset,
                        "{} · {view}: the cell at {x},{y} is left to the terminal",
                        theme.name
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn a_page_in_night_or_mist_is_the_ground_and_the_fills_on_it() -> anyhow::Result<()> {
    // With no overlay open nothing is dimmed, so a cell's background is the
    // ground, or one of the fills a role names.
    for theme in [NIGHT, MIST] {
        let fills = [theme.ground, theme.raised, theme.accent_soft, theme.accent];
        for (view, buf) in plain_views(&theme)? {
            for y in 0..buf.area.height {
                for x in 0..buf.area.width {
                    let bg = buf[(x, y)].bg;
                    assert!(
                        fills.contains(&bg),
                        "{} · {view}: {bg:?} at {x},{y} is not a fill of the theme",
                        theme.name
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn an_overlay_is_filled_with_the_surface_over_a_dimmed_page() -> anyhow::Result<()> {
    for theme in [NIGHT, MIST] {
        for (view, buf) in overlay_views(&theme)? {
            let on_surface = buf
                .content()
                .iter()
                .filter(|cell| cell.bg == theme.surface)
                .count();
            assert!(on_surface > 100, "{} · {view}: {on_surface}", theme.name);
            // The corner of the frame is page, and the page's ground dims to
            // itself.
            assert_eq!(buf[(0, 0)].bg, theme.ground, "{} · {view}", theme.name);
        }
    }
    Ok(())
}

#[test]
fn mono_leaves_the_page_to_the_terminal() -> anyhow::Result<()> {
    // Only a selection, a chip or the active tab sets a background.
    let fills = [MONO.raised, MONO.accent_soft, MONO.accent];
    for (view, buf) in plain_views(&MONO)? {
        let mut left = 0usize;
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                match buf[(x, y)].bg {
                    Color::Reset => left += 1,
                    bg => assert!(
                        fills.contains(&bg),
                        "{view}: {bg:?} at {x},{y} is not a selection or a chip"
                    ),
                }
            }
        }
        let cells = usize::from(buf.area.width) * usize::from(buf.area.height);
        assert!(
            left * 10 > cells * 9,
            "{view}: {left} of {cells} cells left"
        );
    }
    Ok(())
}

#[test]
fn every_glyph_is_from_a_block_a_terminal_font_carries() -> anyhow::Result<()> {
    // No emoji and no dingbat: either is two columns wide in some terminals
    // and a missing-glyph box in others.
    let carried = |c: char| {
        let code = u32::from(c);
        code < 0x100
            || (0x2010..=0x203A).contains(&code) // dashes, quotes, the ellipsis, ‖, ›
            || (0x2190..=0x21FF).contains(&code) // arrows
            || (0x2500..=0x259F).contains(&code) // box drawing and blocks
            || (0x25A0..=0x25FF).contains(&code) // geometric shapes
            || (0x2800..=0x28FF).contains(&code) // braille, the canvas
    };
    for (view, buf) in every_view(&NIGHT)? {
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                for c in buf[(x, y)].symbol().chars() {
                    assert!(
                        carried(c),
                        "{view}: `{c}` (U+{:04X}) at {x},{y}",
                        u32::from(c)
                    );
                }
            }
        }
    }
    Ok(())
}

/// The first cell on row `y` drawn as `glyph`.
fn cell_of<'a>(buf: &'a Buffer, y: u16, glyph: &str) -> Option<&'a ratatui::buffer::Cell> {
    (0..buf.area.width)
        .map(|x| &buf[(x, y)])
        .find(|cell| cell.symbol() == glyph)
}

#[test]
fn a_state_keeps_its_colour_under_the_selection() -> anyhow::Result<()> {
    let mut app = full_app(&NIGHT);
    app.set_page(Page::Evals);
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let text = to_text(&buf);
    // The selected run is the first, a failure; the next is a pass.
    let row_of = |needle: &str| {
        text.lines()
            .position(|line| line.contains(needle))
            .and_then(|y| u16::try_from(y).ok())
            .ok_or_else(|| anyhow::anyhow!("no row with `{needle}`:\n{text}"))
    };
    let failed = row_of("▸ expert:arith-specialist")?;
    let Some(mark) = cell_of(&buf, failed, Tone::Failed.glyph()) else {
        anyhow::bail!("the selected row has no failure mark:\n{text}");
    };
    assert_eq!(mark.fg, NIGHT.danger, "the failure keeps the danger colour");
    assert_eq!(mark.bg, NIGHT.accent_soft, "on the selection's fill");

    let passed = row_of("● pass")?;
    let Some(mark) = cell_of(&buf, passed, Tone::Settled.glyph()) else {
        anyhow::bail!("no pass mark:\n{text}");
    };
    assert_eq!(mark.fg, NIGHT.accent);
    Ok(())
}

#[test]
fn only_a_failure_is_drawn_in_the_danger_colour() -> anyhow::Result<()> {
    // The population page holds an actionable boundary, a graduated shadow
    // and a frozen expert, none of which failed.
    for focus in [Focus::Experts, Focus::Shadows, Focus::Boundaries] {
        let mut app = full_app(&NIGHT);
        app.set_focus(focus);
        let buf = render(&mut app, 120, 36, 1600.0)?;
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                let cell = &buf[(x, y)];
                assert!(
                    cell.fg != NIGHT.danger && cell.bg != NIGHT.danger,
                    "{focus:?}: `{}` at {x},{y} is in the danger colour",
                    cell.symbol()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn the_key_that_removes_is_the_one_chip_in_the_danger_colour() -> anyhow::Result<()> {
    // Deleting a boundary removes it; freezing an expert does not.
    let mut app = full_app(&NIGHT);
    app.set_focus(Focus::Boundaries);
    app.request_action();
    let buf = render(&mut app, 120, 36, 1600.0)?;
    let chips = |buf: &Buffer, fill: Color| {
        buf.content()
            .iter()
            .filter(|cell| cell.bg == fill && cell.symbol() == "y")
            .count()
    };
    assert_eq!(chips(&buf, NIGHT.danger), 1, "delete confirms in danger");

    let mut app = full_app(&NIGHT);
    app.request_freeze();
    anyhow::ensure!(app.mode == Mode::Confirm, "the confirm prompt opened");
    let buf = render(&mut app, 120, 36, 1600.0)?;
    assert_eq!(chips(&buf, NIGHT.danger), 0, "a freeze is not a removal");
    assert_eq!(chips(&buf, NIGHT.accent), 1, "it confirms in the accent");
    Ok(())
}

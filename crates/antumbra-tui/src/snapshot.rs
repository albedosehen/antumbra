//! Headless rendering of the operator console: draw a frame to an in-memory
//! buffer (no terminal) so the console can be e2e-tested and screenshotted.
//!
//! - [`to_text`] preserves every cell's glyph (braille graph, box-drawing,
//!   symbols), so it reads back exactly, for assertions and plain inspection.
//! - [`save_png`] adds the RGB colors for true visual fidelity, rasterizing the
//!   embedded Cascadia Mono (OFL; see `assets/CascadiaMono.LICENSE`).

use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};
use anyhow::{anyhow, Result};
use image::{Rgb, RgbImage};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
#[cfg(test)]
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::Terminal;

use crate::app::App;
use crate::ui;

/// Cascadia Mono (OFL), rasterized for the PNG snapshot so braille / box-drawing
/// / symbol glyphs render true.
const FONT: &[u8] = include_bytes!("../assets/CascadiaMono.ttf");

/// The dark-terminal background and default ink the console is designed against.
const DEFAULT_BG: [u8; 3] = [12, 14, 18];
const DEFAULT_FG: [u8; 3] = [180, 190, 200];

/// Render one frame of the console to an in-memory buffer at a fixed animation
/// clock, so the output is deterministic.
pub fn render(app: &mut App, width: u16, height: u16, at_ms: f64) -> Result<Buffer> {
    app.clock_ms = at_ms;
    let mut terminal = Terminal::new(TestBackend::new(width, height))?;
    terminal.draw(|f| ui::render(f, app))?;
    Ok(terminal.backend().buffer().clone())
}

/// The buffer as a Unicode text grid (rows joined by newlines, trailing spaces
/// trimmed). Every cell's symbol is preserved.
pub fn to_text(buf: &Buffer) -> String {
    let area = buf.area;
    let mut out = String::with_capacity((area.width as usize + 1) * area.height as usize);
    for y in 0..area.height {
        let mut row = String::with_capacity(area.width as usize);
        for x in 0..area.width {
            if let Some(cell) = buf.cell((x, y)) {
                row.push_str(cell.symbol());
            }
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}

/// The text of a sub-region of the buffer (rows trimmed), for golden-testing a
/// modal without the animated background behind it.
#[cfg(test)]
pub fn to_text_in(buf: &Buffer, area: Rect) -> String {
    let mut out = String::new();
    for y in area.top()..area.bottom() {
        let mut row = String::new();
        for x in area.left()..area.right() {
            if let Some(cell) = buf.cell((x, y)) {
                row.push_str(cell.symbol());
            }
        }
        out.push_str(row.trim_end());
        out.push('\n');
    }
    out
}

/// Render the buffer to a PNG at `cell_w` x `cell_h` pixels per cell and save it.
pub fn save_png(buf: &Buffer, path: &str, cell_w: u32, cell_h: u32) -> Result<()> {
    let area = buf.area;
    let mut img = RgbImage::from_pixel(
        area.width as u32 * cell_w,
        area.height as u32 * cell_h,
        Rgb(DEFAULT_BG),
    );
    let font = FontRef::try_from_slice(FONT).map_err(|e| anyhow!("load font: {e}"))?;
    let scale = PxScale::from(cell_h as f32);
    let ascent = font.as_scaled(scale).ascent();

    for y in 0..area.height {
        for x in 0..area.width {
            let Some(cell) = buf.cell((x, y)) else {
                continue;
            };
            let bg = color_rgb(cell.bg, DEFAULT_BG);
            let fg = color_rgb(cell.fg, DEFAULT_FG);
            let (px0, py0) = (x as u32 * cell_w, y as u32 * cell_h);

            // Cell background.
            for yy in 0..cell_h {
                for xx in 0..cell_w {
                    img.put_pixel(px0 + xx, py0 + yy, Rgb(bg));
                }
            }

            // Glyph, blended over the background by per-pixel coverage.
            let ch = cell.symbol().chars().next().unwrap_or(' ');
            if ch == ' ' {
                continue;
            }
            let glyph = font
                .glyph_id(ch)
                .with_scale_and_position(scale, point(px0 as f32, py0 as f32 + ascent));
            if let Some(outline) = font.outline_glyph(glyph) {
                let bounds = outline.px_bounds();
                outline.draw(|gx, gy, c| {
                    let ix = bounds.min.x as i32 + gx as i32;
                    let iy = bounds.min.y as i32 + gy as i32;
                    if ix < 0 || iy < 0 || ix as u32 >= img.width() || iy as u32 >= img.height() {
                        return;
                    }
                    let px = img.get_pixel(ix as u32, iy as u32).0;
                    let blended = [
                        blend(px[0], fg[0], c),
                        blend(px[1], fg[1], c),
                        blend(px[2], fg[2], c),
                    ];
                    img.put_pixel(ix as u32, iy as u32, Rgb(blended));
                });
            }
        }
    }
    img.save(path)
        .map_err(|e| anyhow!("save png {path}: {e}"))?;
    Ok(())
}

/// `over` blended onto `under` by coverage `c` in `[0,1]`.
fn blend(under: u8, over: u8, c: f32) -> u8 {
    (under as f32 * (1.0 - c) + over as f32 * c)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Map a ratatui color to RGB, falling back to `default` for `Reset`/unknown.
fn color_rgb(color: Color, default: [u8; 3]) -> [u8; 3] {
    match color {
        Color::Rgb(r, g, b) => [r, g, b],
        Color::Black => [0, 0, 0],
        Color::White => [235, 235, 235],
        Color::Red => [200, 70, 70],
        Color::Green => [70, 190, 110],
        Color::Yellow => [210, 200, 90],
        Color::Blue => [80, 130, 230],
        Color::Magenta => [200, 110, 220],
        Color::Cyan => [90, 200, 230],
        Color::Gray => [150, 160, 170],
        Color::DarkGray => [80, 90, 100],
        _ => default,
    }
}

#[cfg(test)]
mod tests;

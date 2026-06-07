//! Frame pacing for high-refresh terminals. The console paces to a target FPS
//! (adjustable live with `+`/`-`), and on Windows raises the multimedia timer
//! resolution to 1ms so short frame budgets are actually honoured — the default
//! ~15.6ms scheduler tick would otherwise cap the loop near 64fps however short
//! the budget, so 144/165/244Hz monitors would never be fed.

use std::time::Duration;

/// Bounds the live `+`/`-` adjustment clamps to (covers every common monitor).
pub const MIN_FPS: u32 = 30;
pub const MAX_FPS: u32 = 480;

/// Common monitor refresh rates the `+`/`-` keys snap through, so the target
/// lands exactly on 165 or 244 rather than near it.
pub const PRESETS: [u32; 9] = [30, 60, 90, 120, 144, 165, 240, 244, 360];

/// The next preset above `fps` (or `fps` itself if already at the top).
pub fn next_preset(fps: u32) -> u32 {
    PRESETS
        .iter()
        .copied()
        .find(|&p| p > fps)
        .unwrap_or(fps)
        .min(MAX_FPS)
}

/// The next preset below `fps` (or `fps` itself if already at the bottom).
pub fn prev_preset(fps: u32) -> u32 {
    PRESETS
        .iter()
        .rev()
        .copied()
        .find(|&p| p < fps)
        .unwrap_or(fps)
        .max(MIN_FPS)
}

/// The per-frame time budget for a target rate.
pub fn frame_budget(fps: u32) -> Duration {
    Duration::from_secs_f64(1.0 / fps.clamp(MIN_FPS, MAX_FPS) as f64)
}

/// Clean a detected refresh rate: snap to a [`PRESETS`] value when within a few
/// Hz (drivers report 59/143/164 for 60/144/165), else take it as-is, clamped.
pub fn snap_refresh(hz: u32) -> u32 {
    let nearest = PRESETS
        .iter()
        .copied()
        .min_by_key(|&p| p.abs_diff(hz))
        .unwrap_or(hz);
    let chosen = if nearest.abs_diff(hz) <= 6 {
        nearest
    } else {
        hz
    };
    chosen.clamp(MIN_FPS, MAX_FPS)
}

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
    fn timeEndPeriod(period: u32) -> u32;
}

/// Raises (and on drop restores) the OS timer resolution so short frame budgets
/// are honoured. A no-op off Windows, where sleeps are already fine-grained.
pub struct TimerResolution {
    #[cfg(windows)]
    period_ms: u32,
}

impl TimerResolution {
    /// Request 1ms timer resolution for the lifetime of the guard.
    pub fn acquire() -> Self {
        #[cfg(windows)]
        {
            let period_ms = 1;
            // SAFETY: a valid millisecond period; paired with `timeEndPeriod` in Drop.
            unsafe { timeBeginPeriod(period_ms) };
            Self { period_ms }
        }
        #[cfg(not(windows))]
        {
            Self {}
        }
    }
}

impl Drop for TimerResolution {
    fn drop(&mut self) {
        #[cfg(windows)]
        // SAFETY: restores exactly the period acquired above.
        unsafe {
            timeEndPeriod(self.period_ms);
        }
    }
}

/// The refresh rate (Hz) of the monitor the console window is currently on, so a
/// multi-monitor setup can follow the active display. Best-effort: resolving the
/// window's monitor depends on `GetConsoleWindow`, which under Windows Terminal /
/// ConPTY may report the primary monitor rather than the one the window sits on.
/// Returns `None` off Windows or when the query fails (caller keeps its rate).
#[cfg(windows)]
pub fn detect_refresh() -> Option<u32> {
    // SAFETY: standard Win32 monitor queries; every buffer is stack-owned and
    // sized through its `cb_size` / `dm_size` field as the API requires.
    unsafe {
        let monitor = MonitorFromWindow(GetConsoleWindow(), MONITOR_DEFAULTTONEAREST);
        let mut info: MonitorInfoExW = std::mem::zeroed();
        info.cb_size = std::mem::size_of::<MonitorInfoExW>() as u32;
        if GetMonitorInfoW(monitor, &mut info) == 0 {
            return None;
        }
        let mut mode: DevModeW = std::mem::zeroed();
        mode.dm_size = std::mem::size_of::<DevModeW>() as u16;
        if EnumDisplaySettingsW(info.sz_device.as_ptr(), ENUM_CURRENT_SETTINGS, &mut mode) == 0 {
            return None;
        }
        match mode.dm_display_frequency {
            hz if hz > 1 => Some(hz),
            _ => None,
        }
    }
}

#[cfg(not(windows))]
pub fn detect_refresh() -> Option<u32> {
    None
}

#[cfg(windows)]
type Handle = *mut std::ffi::c_void;

#[cfg(windows)]
const MONITOR_DEFAULTTONEAREST: u32 = 2;
#[cfg(windows)]
const ENUM_CURRENT_SETTINGS: u32 = 0xFFFF_FFFF;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetConsoleWindow() -> Handle;
}

#[cfg(windows)]
#[link(name = "user32")]
extern "system" {
    fn MonitorFromWindow(hwnd: Handle, flags: u32) -> Handle;
    fn GetMonitorInfoW(monitor: Handle, info: *mut MonitorInfoExW) -> i32;
    fn EnumDisplaySettingsW(device: *const u16, mode_num: u32, mode: *mut DevModeW) -> i32;
}

/// `RECT` (four `LONG`s).
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct WinRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

/// `MONITORINFOEXW`: the base plus the `szDevice` name `EnumDisplaySettings` keys on.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct MonitorInfoExW {
    cb_size: u32,
    rc_monitor: WinRect,
    rc_work: WinRect,
    dw_flags: u32,
    sz_device: [u16; 32],
}

/// `DEVMODEW`: only `dm_display_frequency` is read, but the whole struct must be
/// declared so the driver writes within the buffer it's told the size of.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct DevModeW {
    dm_device_name: [u16; 32],
    dm_spec_version: u16,
    dm_driver_version: u16,
    dm_size: u16,
    dm_driver_extra: u16,
    dm_fields: u32,
    dm_position_x: i32,
    dm_position_y: i32,
    dm_display_orientation: u32,
    dm_display_fixed_output: u32,
    dm_color: i16,
    dm_duplex: i16,
    dm_y_resolution: i16,
    dm_tt_option: i16,
    dm_collate: i16,
    dm_form_name: [u16; 32],
    dm_log_pixels: u16,
    dm_bits_per_pel: u32,
    dm_pels_width: u32,
    dm_pels_height: u32,
    dm_display_flags: u32,
    dm_display_frequency: u32,
    dm_icm_method: u32,
    dm_icm_intent: u32,
    dm_media_type: u32,
    dm_dither_type: u32,
    dm_reserved1: u32,
    dm_reserved2: u32,
    dm_panning_width: u32,
    dm_panning_height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_snap_to_common_refresh_rates() {
        // Stepping up off 144 lands exactly on 165, then 240/244 — not near them.
        assert_eq!(next_preset(144), 165);
        assert_eq!(next_preset(165), 240);
        assert_eq!(next_preset(240), 244);
        // Down walks back the same ladder.
        assert_eq!(prev_preset(244), 240);
        assert_eq!(prev_preset(60), 30);
        // The ends clamp rather than wrap.
        assert_eq!(next_preset(360), 360);
        assert_eq!(prev_preset(30), 30);
        // An arbitrary `--fps` value snaps to a neighbouring preset.
        assert_eq!(next_preset(100), 120);
        assert_eq!(prev_preset(200), 165);
    }

    #[test]
    fn frame_budget_matches_the_rate() {
        assert_eq!(frame_budget(60), Duration::from_secs_f64(1.0 / 60.0));
        // Out-of-range targets clamp before becoming a budget.
        assert_eq!(frame_budget(10_000), frame_budget(MAX_FPS));
    }

    #[test]
    fn detected_rates_snap_to_presets_within_tolerance() {
        // Drivers under-report the round number by a Hz or two.
        assert_eq!(snap_refresh(59), 60);
        assert_eq!(snap_refresh(143), 144);
        assert_eq!(snap_refresh(164), 165);
        assert_eq!(snap_refresh(239), 240);
        // Far from any preset: kept as-is (a genuine 200Hz panel stays 200).
        assert_eq!(snap_refresh(200), 200);
        // Out of range clamps.
        assert_eq!(snap_refresh(9_999), MAX_FPS);
        assert_eq!(snap_refresh(5), MIN_FPS);
    }

    #[test]
    fn detect_refresh_never_panics() {
        // Returns Some(real Hz) on a display, None headless/off-Windows — but
        // must always be safe to call.
        if let Some(hz) = detect_refresh() {
            assert!(hz > 1);
        }
    }
}

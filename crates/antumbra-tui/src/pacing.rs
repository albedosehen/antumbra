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
}

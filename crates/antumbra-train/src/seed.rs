//! The run's generation seed. Unseeded draws and the training shuffle take
//! their randomness from a process-wide counter, so a process that does the
//! same things draws the same way: a run repeated is the same run. That makes
//! a measurement repeatable, and it makes a repeat useless for seeing how much
//! training varies. The generation seed shifts every such stream. Seed 0 is the
//! stream every run drew before the seed existed, so earlier runs reproduce.

use std::sync::atomic::{AtomicU64, Ordering};

static GENERATION_SEED: AtomicU64 = AtomicU64::new(0);

/// Take `seed` for every unseeded draw and shuffle from here on.
pub fn seed_generation(seed: u64) {
    GENERATION_SEED.store(seed, Ordering::Relaxed);
}

/// The seed set with [`seed_generation`], 0 by default.
pub fn generation_seed() -> u64 {
    GENERATION_SEED.load(Ordering::Relaxed)
}

/// `stream`, shifted by the run's seed: unchanged at seed 0, a different
/// stream at any other.
pub fn shifted(stream: u64, seed: u64) -> u64 {
    stream ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_zero_is_the_stream_runs_drew_before_it() {
        for stream in [0u64, 1, 0xA17, u64::MAX] {
            assert_eq!(shifted(stream, 0), stream);
        }
    }

    #[test]
    fn another_seed_is_another_stream_and_the_same_seed_the_same() {
        let (a, b) = (shifted(0xA17, 1), shifted(0xA17, 2));
        assert_ne!(a, 0xA17);
        assert_ne!(a, b);
        assert_eq!(shifted(0xA17, 1), a);
    }
}

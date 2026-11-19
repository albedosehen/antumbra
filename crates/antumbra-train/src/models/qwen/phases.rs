//! Work for every core in phases, in one dispatch to candle's barrier pool.
//!
//! A decoded token's steps depend on one another: a product's adapter needs
//! `A · x` before any row of `B`, the MLP's second product needs the first
//! one's gate. Each as a dispatch of its own hands the rows between them to
//! the calling thread and back, and a row that eight cores computed is pulled
//! a cache line at a time to one core and then out again to all eight, which
//! on a decoded token costs more than the arithmetic. Here the steps are
//! phases of one dispatch: every thread works through a phase's units, waits
//! until the phase is done, and goes on to the next, so what a core computed
//! in one phase is what it reads in the next.
//!
//! A phase's units are shared out a run to each thread, which takes its run
//! from end to end: one stream of memory a thread, which the CPU reads ahead
//! of. A thread that is done takes from the back of the run with the most
//! left, so one that falls behind is not waited for longer than a claim.

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};

/// The units one thread has still to do: the first in the high half, one
/// past the last in the low. On lines of its own, since its thread writes it
/// at every claim.
#[repr(align(128))]
struct Share(AtomicU64);

impl Share {
    fn of(first: usize, last: usize) -> Self {
        Self(AtomicU64::new(((first as u64) << 32) | last as u64))
    }

    fn left(word: u64) -> (usize, usize) {
        ((word >> 32) as usize, (word & 0xffff_ffff) as usize)
    }

    /// Take up to `most` units from the front: the thread whose share this
    /// is, going on from where it was.
    fn front(&self, most: usize) -> Option<(usize, usize)> {
        let mut word = self.0.load(Ordering::Acquire);
        loop {
            let (first, last) = Self::left(word);
            if first >= last {
                return None;
            }
            let n = most.min(last - first);
            let taken = (((first + n) as u64) << 32) | last as u64;
            match self
                .0
                .compare_exchange_weak(word, taken, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return Some((first, first + n)),
                Err(now) => word = now,
            }
        }
    }
}

/// Take up to `most` units from the back of the share with the most left: a
/// thread whose own is done, from the end its owner reaches last.
fn back(shares: &[Share], most: usize) -> Option<(usize, usize)> {
    loop {
        let mut fullest = (0usize, 0usize, 0u64);
        for (i, share) in shares.iter().enumerate() {
            let word = share.0.load(Ordering::Acquire);
            let (first, last) = Share::left(word);
            if last > first && last - first > fullest.0 {
                fullest = (last - first, i, word);
            }
        }
        let (left, i, word) = fullest;
        if left == 0 {
            return None;
        }
        let (first, last) = Share::left(word);
        let n = most.min(left);
        let taken = ((first as u64) << 32) | (last - n) as u64;
        if shares[i]
            .0
            .compare_exchange_weak(word, taken, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Some((last - n, last));
        }
    }
}

/// One phase of a dispatch.
pub(super) struct Phase<'a> {
    /// How many units of work it is.
    pub(super) units: usize,
    /// Units a claim covers at most: enough that claiming is nothing beside
    /// the work, few enough that the last claims even out.
    pub(super) most: usize,
    /// Do the units `first..last`. Each unit is done once, by one thread.
    pub(super) run: &'a (dyn Fn(usize, usize) + Sync),
    /// Run once, by one thread, when every unit is done and before any thread
    /// starts the next phase: what the next phase needs made of this one.
    pub(super) then: Option<&'a (dyn Fn() + Sync)>,
}

const WAITING: u8 = 0;
const RUNNING: u8 = 1;
const DONE: u8 = 2;

/// A phase as it is worked through.
struct Live<'a> {
    phase: &'a Phase<'a>,
    shares: Vec<Share>,
    done: AtomicUsize,
    then: AtomicU8,
}

/// Counts units as done when it is dropped, so a unit that panics is still
/// counted and the threads waiting on its phase go on to end the dispatch,
/// where the pool hands the panic to the caller.
struct Counted<'a>(&'a AtomicUsize, usize);

impl Drop for Counted<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(self.1, Ordering::Release);
    }
}

/// Marks a phase's `then` as run when it is dropped, for the same reason.
struct Ran<'a>(&'a AtomicU8);

impl Drop for Ran<'_> {
    fn drop(&mut self) {
        self.0.store(DONE, Ordering::Release);
    }
}

impl Live<'_> {
    fn work(&self, thread: usize, last: bool) {
        let run = |(first, until): (usize, usize)| {
            let _counted = Counted(&self.done, until - first);
            (self.phase.run)(first, until);
        };
        if let Some(own) = self.shares.get(thread) {
            while let Some(units) = own.front(self.phase.most) {
                run(units);
            }
        }
        while let Some(units) = back(&self.shares, self.phase.most) {
            run(units);
        }
        // The last phase with nothing after it ends with the dispatch, which
        // waits for every thread itself.
        let Some(then) = self.phase.then else {
            if !last {
                self.wait();
            }
            return;
        };
        // Every unit taken is finished by the thread that took it, so this
        // ends.
        self.wait();
        match self
            .then
            .compare_exchange(WAITING, RUNNING, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                let _ran = Ran(&self.then);
                then();
            }
            Err(_) => {
                while self.then.load(Ordering::Acquire) != DONE {
                    std::hint::spin_loop();
                }
            }
        }
    }

    fn wait(&self) {
        while self.done.load(Ordering::Acquire) < self.phase.units {
            std::hint::spin_loop();
        }
    }
}

/// Work through `phases` in order on every core: each phase's units, then
/// its `then`, before any of the next. With `inline`, on the calling thread
/// alone: for work too small to be worth waking the others.
pub(super) fn dispatch(phases: &[Phase], inline: bool) {
    let pool = candle_core::utils::barrier_pool();
    if inline || pool.n_workers() == 0 {
        for phase in phases {
            if phase.units > 0 {
                (phase.run)(0, phase.units);
            }
            if let Some(then) = phase.then {
                then();
            }
        }
        return;
    }
    let threads = pool.n_workers() + 1;
    let live: Vec<Live> = phases
        .iter()
        .map(|phase| Live {
            phase,
            shares: (0..threads)
                .map(|t| Share::of(phase.units * t / threads, phase.units * (t + 1) / threads))
                .collect(),
            done: AtomicUsize::new(0),
            then: AtomicU8::new(WAITING),
        })
        .collect();
    pool.execute(|thread| {
        for (i, phase) in live.iter().enumerate() {
            phase.work(thread, i + 1 == live.len());
        }
    });
}

/// A row that the units of a phase each write their own part of, and that
/// later phases read.
pub(super) struct Shared<T>(*mut T, usize);

// SAFETY: the pointer is only written through at places no two units of work
// share, and only read where a phase before has finished writing (see
// `part` and `all`), for as long as the slice it was made from is borrowed.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Shared<T> {
    pub(super) fn of(row: &mut [T]) -> Self {
        Self(row.as_mut_ptr(), row.len())
    }

    /// The `n` values from `at`, to write.
    ///
    /// SAFETY (for the caller): the values are inside the row, the slice the
    /// row was made from is still borrowed, and no other thread reads or
    /// writes these values while the part is held.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn part(&self, at: usize, n: usize) -> &mut [T] {
        debug_assert!(at + n <= self.1);
        std::slice::from_raw_parts_mut(self.0.add(at), n)
    }

    /// The whole row, to read.
    ///
    /// SAFETY (for the caller): the slice the row was made from is still
    /// borrowed, and no thread writes any of it while this is held: the
    /// phase that wrote it is done.
    pub(super) unsafe fn all(&self) -> &[T] {
        std::slice::from_raw_parts(self.0, self.1)
    }

    pub(super) fn at(&self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// Every unit of every phase is done exactly once, a phase's `then` runs
    /// once after all its units and before any of the next phase's, whether
    /// the phases are dispatched or run where the caller stands.
    #[test]
    fn phases_run_in_order_and_each_unit_once() {
        for inline in [false, true] {
            for (first_units, second_units) in [(0usize, 5usize), (1, 1), (37, 1000), (4096, 3)] {
                let first: Vec<AtomicU32> = (0..first_units).map(|_| AtomicU32::new(0)).collect();
                let second: Vec<AtomicU32> = (0..second_units).map(|_| AtomicU32::new(0)).collect();
                let between = AtomicU32::new(0);
                let early = AtomicU32::new(0);
                let run_first = |a: usize, b: usize| {
                    for unit in &first[a..b] {
                        unit.fetch_add(1, Ordering::Relaxed);
                    }
                };
                let then = || {
                    // Every unit of the first phase is in by now.
                    let all = first.iter().all(|u| u.load(Ordering::Relaxed) == 1);
                    between.fetch_add(if all { 1 } else { 100 }, Ordering::Relaxed);
                };
                let run_second = |a: usize, b: usize| {
                    if between.load(Ordering::Relaxed) != 1 {
                        early.fetch_add(1, Ordering::Relaxed);
                    }
                    for unit in &second[a..b] {
                        unit.fetch_add(1, Ordering::Relaxed);
                    }
                };
                dispatch(
                    &[
                        Phase {
                            units: first_units,
                            most: 3,
                            run: &run_first,
                            then: Some(&then),
                        },
                        Phase {
                            units: second_units,
                            most: 7,
                            run: &run_second,
                            then: None,
                        },
                    ],
                    inline,
                );
                assert!(first.iter().all(|u| u.load(Ordering::Relaxed) == 1));
                assert!(second.iter().all(|u| u.load(Ordering::Relaxed) == 1));
                assert_eq!(
                    between.load(Ordering::Relaxed),
                    1,
                    "then ran once, after its phase"
                );
                assert_eq!(
                    early.load(Ordering::Relaxed),
                    0,
                    "nothing of the next phase before it"
                );
            }
        }
    }

    /// A share is taken from the front by its owner and from the back by
    /// another, and no unit twice.
    #[test]
    fn a_share_is_taken_from_both_ends_and_no_unit_twice() {
        let shares = [Share::of(0, 10), Share::of(10, 13)];
        assert_eq!(shares[0].front(4), Some((0, 4)));
        assert_eq!(
            back(&shares, 4),
            Some((6, 10)),
            "the fullest, from its back"
        );
        assert_eq!(back(&shares, 2), Some((11, 13)));
        assert_eq!(shares[0].front(4), Some((4, 6)));
        assert_eq!(shares[0].front(4), None);
        assert_eq!(back(&shares, 4), Some((10, 11)));
        assert_eq!(back(&shares, 4), None);
    }
}

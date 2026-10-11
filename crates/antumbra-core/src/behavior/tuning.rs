//! The loop's hand on the recipe's regression confidence.
//!
//! Admission bets: a behavior the base model already followed slipped on its
//! held-out tasks, and the set was admitted because the slip read as less
//! likely than the confidence to be real. An [`Audit`](super::Audit) later
//! measures that served expert, beside the base model, on examples the user
//! wrote after the admission. Pooled over a workspace, those audits say
//! whether the bets held. The confidence moves on that, and on nothing the
//! loop produced itself.
//!
//! A reading moves it only when the reading before leaned the same way, by
//! [`STEP`], never past [`FLOOR`] or [`CEILING`], and the same audits are never
//! read twice. Every reading is kept as a row. Off unless the node is told to
//! tune.

use super::{audits, prob_beta_greater, Memory};

/// How far one reading moves the confidence.
pub const STEP: f64 = 0.02;
/// The confidence never moves below this.
pub const FLOOR: f64 = 0.6;
/// Nor above this.
pub const CEILING: f64 = 0.98;
/// The fewest later examples a reading needs.
pub const MIN_FRESH: usize = 6;
/// And the fewest bets they must be spread over.
pub const MIN_BETS: usize = 2;
/// How far from even the pooled odds must sit before they lean either way.
pub const MARGIN: f64 = 0.25;

/// The admitted bets a workspace's behaviors have been audited on, pooled:
/// every audit of a behavior admission found more likely worse than not.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pool {
    /// How many audits the pool is over.
    pub bets: usize,
    /// How many later examples they measured.
    pub fresh: usize,
    /// How many of them the base model passed.
    pub base: usize,
    /// How many the experts passed.
    pub expert: usize,
}

impl Pool {
    /// Pool the bets audited across `memories`, a workspace's behaviors.
    pub fn of(memories: &[Memory]) -> Pool {
        let mut pool = Pool::default();
        for a in memories.iter().flat_map(|m| audits(&m.evidence)) {
            if a.worse.is_some_and(|w| w > 0.5) {
                pool.bets += 1;
                pool.fresh += a.fresh;
                pool.base += a.base;
                pool.expert += a.expert;
            }
        }
        pool
    }

    /// How likely the base model's true pass rate on the bets' later examples
    /// is above the experts': the bets failing. 0.5 with nothing measured.
    pub fn realized(&self) -> f64 {
        if self.fresh == 0 {
            return 0.5;
        }
        let base = self.base.min(self.fresh);
        let expert = self.expert.min(self.fresh);
        prob_beta_greater(
            1 + base,
            1 + self.fresh - base,
            1 + expert,
            1 + self.fresh - expert,
        )
    }

    /// Which way the pool leans, when it holds enough to lean at all: the bets
    /// held, so admission may loosen; or they failed, so it must tighten.
    pub fn lean(&self) -> Option<Lean> {
        if self.bets < MIN_BETS || self.fresh < MIN_FRESH {
            return None;
        }
        let r = self.realized();
        if r <= 0.5 - MARGIN {
            Some(Lean::Looser)
        } else if r >= 0.5 + MARGIN {
            Some(Lean::Stricter)
        } else {
            None
        }
    }
}

/// What a pool argues for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lean {
    /// The bets held: admit at a higher chance of being worse.
    Looser,
    /// The bets failed: refuse sooner.
    Stricter,
}

impl Lean {
    pub fn as_str(self) -> &'static str {
        match self {
            Lean::Looser => "looser",
            Lean::Stricter => "stricter",
        }
    }

    pub fn parse(s: &str) -> Option<Lean> {
        match s {
            "looser" => Some(Lean::Looser),
            "stricter" => Some(Lean::Stricter),
            _ => None,
        }
    }
}

/// One reading of a workspace's pool against the confidence in force.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub lean: Lean,
    /// The confidence in force when read.
    pub from: f64,
    /// The confidence after: `from`, unless the reading before leaned the
    /// same way.
    pub to: f64,
    pub bets: usize,
    pub fresh: usize,
    pub base: usize,
    pub expert: usize,
    /// The pool's [`Pool::realized`] when read.
    pub realized: f64,
}

impl Reading {
    /// Whether this reading moved the confidence.
    pub fn moved(&self) -> bool {
        self.to != self.from
    }

    /// What it read and what it did, for the log and the user.
    pub fn describe(&self) -> String {
        let verdict = match self.lean {
            Lean::Looser => "the bets held",
            Lean::Stricter => "the bets failed",
        };
        let did = if self.moved() {
            format!("moved from {:.2} to {:.2}", self.from, self.to)
        } else {
            format!("stays at {:.2} until a second reading agrees", self.from)
        };
        format!(
            "{} bets over {} later examples, the base model passed {}, the experts {}, {:.0}% likely worse: {verdict}; {did}",
            self.bets,
            self.fresh,
            self.base,
            self.expert,
            self.realized * 100.0
        )
    }
}

/// Two decimals, so steps do not drift.
fn round(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Read `pool` against `current`. `None` when the pool does not lean. The
/// confidence moves only when `previous`, the reading before, leaned the same
/// way: one step, within the floor and the ceiling.
pub fn read(pool: &Pool, current: f64, previous: Option<Lean>) -> Option<Reading> {
    let lean = pool.lean()?;
    let to = if previous == Some(lean) {
        match lean {
            Lean::Looser => round((current + STEP).min(CEILING)),
            Lean::Stricter => round((current - STEP).max(FLOOR)),
        }
    } else {
        current
    };
    Some(Reading {
        lean,
        from: current,
        to,
        bets: pool.bets,
        fresh: pool.fresh,
        base: pool.base,
        expert: pool.expert,
        realized: pool.realized(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{mark_audit, Audit};
    use super::*;
    use crate::{MemoryNetwork, TenantId};

    fn audited(id: &str, fresh: usize, base: usize, expert: usize, worse: Option<f32>) -> Memory {
        let mut m = Memory::new(
            id,
            TenantId::new("ws:a"),
            MemoryNetwork::Opinion,
            "a behavior",
            1.0,
            chrono::Utc::now(),
        );
        mark_audit(
            &mut m.evidence,
            &Audit {
                set: "x".into(),
                fresh,
                base,
                expert,
                worse,
            },
        );
        m
    }

    #[test]
    fn the_pool_is_the_audits_of_admitted_bets_and_nothing_else() {
        let memories = vec![
            audited("memory:a", 3, 3, 3, Some(0.74)),
            audited("memory:b", 2, 2, 0, Some(0.6)),
            // No slip at admission: not a bet.
            audited("memory:c", 4, 0, 4, Some(0.02)),
            // A card from before the bet was kept: unknown, left out.
            audited("memory:d", 4, 4, 0, None),
        ];
        let pool = Pool::of(&memories);
        assert_eq!(
            pool,
            Pool {
                bets: 2,
                fresh: 5,
                base: 5,
                expert: 3
            }
        );
        assert_eq!(Pool::of(&[]), Pool::default());
        assert_eq!(Pool::default().realized(), 0.5);
    }

    #[test]
    fn a_pool_leans_only_with_enough_evidence_and_only_past_the_margin() {
        let held = Pool {
            bets: 2,
            fresh: 6,
            base: 3,
            expert: 6,
        };
        assert_eq!(held.lean(), Some(Lean::Looser), "{}", held.realized());
        let failed = Pool {
            bets: 2,
            fresh: 6,
            base: 6,
            expert: 3,
        };
        assert_eq!(failed.lean(), Some(Lean::Stricter));
        let even = Pool {
            bets: 2,
            fresh: 6,
            base: 4,
            expert: 4,
        };
        assert_eq!(even.lean(), None, "{}", even.realized());
        let thin = Pool {
            bets: 1,
            fresh: 6,
            base: 0,
            expert: 6,
        };
        assert_eq!(thin.lean(), None, "one bet is not a pattern");
        let few = Pool {
            bets: 3,
            fresh: 5,
            base: 0,
            expert: 5,
        };
        assert_eq!(few.lean(), None, "five examples are too few");
    }

    #[test]
    fn a_reading_moves_only_when_the_one_before_agreed_and_stays_in_bounds() {
        let held = Pool {
            bets: 2,
            fresh: 6,
            base: 2,
            expert: 6,
        };
        let first = read(&held, 0.9, None).unwrap();
        assert_eq!((first.lean, first.from, first.to), (Lean::Looser, 0.9, 0.9));
        assert!(!first.moved());
        assert!(first.describe().contains("the bets held; stays at 0.90"));
        let second = read(&held, 0.9, Some(Lean::Looser)).unwrap();
        assert_eq!(second.to, 0.92);
        assert!(second.moved());
        assert!(second.describe().contains("moved from 0.90 to 0.92"));
        assert_eq!(
            read(&held, 0.9, Some(Lean::Stricter)).unwrap().to,
            0.9,
            "a change of mind is a first reading"
        );
        assert_eq!(read(&held, 0.97, Some(Lean::Looser)).unwrap().to, CEILING);
        assert_eq!(
            read(&held, CEILING, Some(Lean::Looser)).unwrap().to,
            CEILING
        );

        let failed = Pool {
            bets: 2,
            fresh: 6,
            base: 6,
            expert: 2,
        };
        let r = read(&failed, 0.9, Some(Lean::Stricter)).unwrap();
        assert_eq!((r.lean, r.to), (Lean::Stricter, 0.88));
        assert!(r
            .describe()
            .contains("the bets failed; moved from 0.90 to 0.88"));
        assert_eq!(read(&failed, 0.61, Some(Lean::Stricter)).unwrap().to, FLOOR);

        let even = Pool {
            bets: 2,
            fresh: 6,
            base: 3,
            expert: 3,
        };
        assert!(read(&even, 0.9, Some(Lean::Looser)).is_none());
        assert_eq!(Lean::parse(Lean::Looser.as_str()), Some(Lean::Looser));
        assert_eq!(Lean::parse("sideways"), None);
    }
}

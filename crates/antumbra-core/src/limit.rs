//! Token buckets for what a server does on request without a token, or at a
//! cost it cannot take back: a public page read, a print, an email. Each
//! client, by whatever names it (its peer address), has a bucket, and so does
//! everyone together, so one client cannot spend the server's time at will and
//! many clients cannot either.
//!
//! A bucket holds `burst` tokens and refills at `per_minute`; a request takes
//! one or is refused. A client is charged before everyone is, so a client
//! refused at its own bucket costs the others nothing. Clients are kept in a
//! bounded map: a flood of names evicts the oldest rather than growing without
//! limit, and an evicted client comes back with a full bucket, which is what
//! everyone's bucket is there to bound.
//!
//! Plain `std` locking, held for a few arithmetic operations, so an async
//! server and a synchronous one use the same type.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// How many clients keep a bucket at once.
const CLIENTS: usize = 4096;

/// A bucket's size and refill rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rate {
    /// Requests allowed at once from a full bucket.
    pub burst: u32,
    /// Tokens a minute flow back in.
    pub per_minute: u32,
}

struct Bucket {
    tokens: f64,
    at: Instant,
}

impl Bucket {
    fn full(rate: Rate, now: Instant) -> Self {
        Self {
            tokens: f64::from(rate.burst),
            at: now,
        }
    }

    /// Refill for the time since the last look, then take a token if one is
    /// there.
    fn take(&mut self, rate: Rate, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.at).as_secs_f64();
        self.tokens =
            (self.tokens + elapsed * f64::from(rate.per_minute) / 60.0).min(f64::from(rate.burst));
        self.at = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Return a token taken a moment ago, when the request was refused after
    /// all for another reason.
    fn give_back(&mut self, rate: Rate) {
        self.tokens = (self.tokens + 1.0).min(f64::from(rate.burst));
    }
}

/// Buckets by client name, bounded: a name past the cap evicts the oldest.
struct Clients {
    buckets: HashMap<String, Bucket>,
    order: VecDeque<String>,
    cap: usize,
}

impl Clients {
    fn new(cap: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn get_mut(&mut self, client: &str) -> Option<&mut Bucket> {
        self.buckets.get_mut(client)
    }

    fn insert(&mut self, client: String, bucket: Bucket) {
        if self.buckets.insert(client.clone(), bucket).is_none() {
            self.order.push_back(client);
            while self.order.len() > self.cap {
                if let Some(oldest) = self.order.pop_front() {
                    self.buckets.remove(&oldest);
                }
            }
        }
    }
}

/// One budget per client and one for everyone.
pub struct Limiter {
    per_client: Rate,
    everyone: Rate,
    clients: Mutex<Clients>,
    all: Mutex<Bucket>,
}

impl Limiter {
    pub fn new(per_client: Rate, everyone: Rate) -> Self {
        Self {
            per_client,
            everyone,
            clients: Mutex::new(Clients::new(CLIENTS)),
            all: Mutex::new(Bucket::full(everyone, Instant::now())),
        }
    }

    /// Whether `client` may make one more request now.
    pub fn allow(&self, client: &str) -> bool {
        self.allow_at(client, Instant::now())
    }

    /// [`Self::allow`] at a given instant, so a test can move time.
    pub fn allow_at(&self, client: &str, now: Instant) -> bool {
        let mut clients = held(&self.clients);
        let client_allows = match clients.get_mut(client) {
            Some(bucket) => bucket.take(self.per_client, now),
            None => {
                let mut bucket = Bucket::full(self.per_client, now);
                let allowed = bucket.take(self.per_client, now);
                clients.insert(client.to_string(), bucket);
                allowed
            }
        };
        if !client_allows {
            return false;
        }
        if held(&self.all).take(self.everyone, now) {
            return true;
        }
        // Everyone's budget is spent; the client's token goes back, since the
        // request did not run.
        if let Some(bucket) = clients.get_mut(client) {
            bucket.give_back(self.per_client);
        }
        false
    }
}

/// The lock, poisoned or not: its critical sections are arithmetic on a few
/// numbers, which a panic cannot leave half done.
fn held<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    match lock.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{Limiter, Rate};

    const ROOMY: Rate = Rate {
        burst: 1_000,
        per_minute: 60_000,
    };

    #[test]
    fn a_client_has_its_burst_then_waits_for_the_refill() {
        let limiter = Limiter::new(
            Rate {
                burst: 3,
                per_minute: 60,
            },
            ROOMY,
        );
        let now = Instant::now();
        for _ in 0..3 {
            assert!(limiter.allow_at("one", now));
        }
        assert!(!limiter.allow_at("one", now), "the burst is spent");
        assert!(
            limiter.allow_at("two", now),
            "another client has a bucket of its own"
        );
        // One token flows back a second, at sixty a minute.
        assert!(limiter.allow_at("one", now + Duration::from_millis(1_100)));
        assert!(!limiter.allow_at("one", now + Duration::from_millis(1_100)));
    }

    #[test]
    fn everyone_shares_a_ceiling_a_flood_of_names_cannot_pass() {
        let limiter = Limiter::new(
            ROOMY,
            Rate {
                burst: 2,
                per_minute: 60,
            },
        );
        let now = Instant::now();
        assert!(limiter.allow_at("a", now));
        assert!(limiter.allow_at("b", now));
        assert!(!limiter.allow_at("c", now), "everyone's budget is spent");
        // The refused client was not charged for a request that did not run.
        assert!(
            limiter.allow_at("c", now + Duration::from_secs(1)),
            "a token flowed back for everyone, and c still has its own"
        );
    }

    #[test]
    fn a_client_refused_at_its_own_bucket_costs_the_others_nothing() {
        let limiter = Limiter::new(
            Rate {
                burst: 1,
                per_minute: 1,
            },
            Rate {
                burst: 2,
                per_minute: 1,
            },
        );
        let now = Instant::now();
        assert!(limiter.allow_at("loud", now));
        for _ in 0..10 {
            assert!(!limiter.allow_at("loud", now));
        }
        assert!(
            limiter.allow_at("quiet", now),
            "everyone's second token is still there"
        );
    }

    #[test]
    fn a_flood_of_names_evicts_the_oldest_bucket() {
        let limiter = Limiter::new(
            Rate {
                burst: 1,
                per_minute: 1,
            },
            // Wide enough for the whole flood, so only eviction is in play.
            Rate {
                burst: 10_000,
                per_minute: 600_000,
            },
        );
        let now = Instant::now();
        assert!(limiter.allow_at("first", now));
        assert!(!limiter.allow_at("first", now), "first's bucket is spent");
        for n in 0..super::CLIENTS {
            limiter.allow_at(&format!("name-{n}"), now);
        }
        assert!(
            limiter.allow_at("first", now),
            "evicted, first comes back with a full bucket; everyone's ceiling bounds that"
        );
    }
}

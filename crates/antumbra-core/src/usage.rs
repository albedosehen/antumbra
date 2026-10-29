//! How strongly a memory stands, judged by the system from how it is used.
//!
//! A memory's `confidence` is what its writer believed. Its **strength** is
//! what the system makes of it since: it fades with the time since it was last
//! of use, at a half-life set by its network, and it falls further the more
//! often it was put in front of the work and passed over. Nobody has to tend
//! it. The server watches for the signals ([`MemoryUse`]) and computes the
//! strength when it reads ([`strength`]), never storing it, so every
//! [`MemoryPolicy`] knob can be turned while the server runs and takes effect
//! on the next read.
//!
//! An agent may still judge a memory itself (`reinforce_memory`,
//! `penalize_memory`): that moves the memory's confidence and counts as a use
//! or a judgment here, the same lever the system pulls.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::MemoryId;
use crate::memory::{Memory, MemoryNetwork};

/// What the system has seen happen to one memory in one workspace. Every
/// count starts at zero; the server only ever adds to them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryUse {
    pub memory: MemoryId,
    /// Times recall returned it.
    #[serde(default)]
    pub recalled: u32,
    /// Times the work that followed a recall was checked against it.
    #[serde(default)]
    pub judged: u32,
    /// Times that work drew on it.
    #[serde(default)]
    pub used: u32,
    /// Times a new memory restated it.
    #[serde(default)]
    pub recurred: u32,
    /// Times an agent judged it good (`reinforce_memory`).
    #[serde(default)]
    pub good: u32,
    /// Times an agent judged it bad (`penalize_memory`).
    #[serde(default)]
    pub bad: u32,
    #[serde(default)]
    pub last_recalled_at: Option<DateTime<Utc>>,
    /// The last time it was used, restated or judged good: when its fade
    /// restarts.
    #[serde(default)]
    pub last_used_at: Option<DateTime<Utc>>,
}

impl MemoryUse {
    pub fn new(memory: impl Into<MemoryId>) -> Self {
        Self {
            memory: memory.into(),
            recalled: 0,
            judged: 0,
            used: 0,
            recurred: 0,
            good: 0,
            bad: 0,
            last_recalled_at: None,
            last_used_at: None,
        }
    }

    /// Times it was of use: drawn on, restated, or judged good.
    pub fn uses(&self) -> u32 {
        self.used
            .saturating_add(self.recurred)
            .saturating_add(self.good)
    }

    /// Times it was checked against the work and not drawn on.
    pub fn passed_over(&self) -> u32 {
        self.judged.saturating_sub(self.used)
    }
}

/// The knobs a workspace's operator turns to steer how the system governs its
/// memories. Read on every recall, so a change applies at once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryPolicy {
    /// The system watches use, links new memories and lets strength fade.
    /// Off, strength is the writer's confidence and nothing is recorded.
    pub auto: bool,
    /// Days for an unused fact to lose half its strength.
    pub world_half_life_days: f32,
    /// Days for an unused experience to lose half its strength.
    pub bank_half_life_days: f32,
    /// Days for an unused judgment or preference to lose half its strength.
    pub opinion_half_life_days: f32,
    /// Days for an unused volatile memory, of any network, to lose half.
    pub volatile_half_life_days: f32,
    /// How close the work after a recall must come to a memory (cosine) for
    /// the memory to count as used.
    pub use_threshold: f32,
    /// How close a new memory must come to an existing one to be linked to it.
    pub link_threshold: f32,
    /// How close a new memory must come to an existing one to count as
    /// restating it.
    pub recurrence_threshold: f32,
    /// The most links a new memory gets.
    pub max_links: u32,
    /// How much each pass-over weighs against a memory, next to one use.
    pub pass_over_weight: f32,
    /// How much strength moves recall's order, from 0 (relevance alone) to 1.
    pub strength_weight: f32,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            auto: true,
            world_half_life_days: 180.0,
            bank_half_life_days: 90.0,
            opinion_half_life_days: 180.0,
            volatile_half_life_days: 14.0,
            use_threshold: 0.55,
            link_threshold: 0.80,
            recurrence_threshold: 0.92,
            max_links: 3,
            pass_over_weight: 0.1,
            strength_weight: 0.25,
        }
    }
}

/// One knob: its name, what it does, its range and step, and how to read and
/// set it, so the console and the server agree on every one.
pub struct Knob {
    pub key: &'static str,
    pub help: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub get: fn(&MemoryPolicy) -> f32,
    pub set: fn(&mut MemoryPolicy, f32),
}

/// Every knob, in the order the console lists them.
pub const KNOBS: &[Knob] = &[
    Knob {
        key: "auto",
        help: "the system governs memories itself (1) or leaves them as written (0)",
        min: 0.0,
        max: 1.0,
        step: 1.0,
        get: |p| if p.auto { 1.0 } else { 0.0 },
        set: |p, v| p.auto = v >= 0.5,
    },
    Knob {
        key: "world_half_life_days",
        help: "days for an unused fact to lose half its strength",
        min: 1.0,
        max: 3650.0,
        step: 15.0,
        get: |p| p.world_half_life_days,
        set: |p, v| p.world_half_life_days = v,
    },
    Knob {
        key: "bank_half_life_days",
        help: "days for an unused experience to lose half its strength",
        min: 1.0,
        max: 3650.0,
        step: 15.0,
        get: |p| p.bank_half_life_days,
        set: |p, v| p.bank_half_life_days = v,
    },
    Knob {
        key: "opinion_half_life_days",
        help: "days for an unused judgment or preference to lose half its strength",
        min: 1.0,
        max: 3650.0,
        step: 15.0,
        get: |p| p.opinion_half_life_days,
        set: |p, v| p.opinion_half_life_days = v,
    },
    Knob {
        key: "volatile_half_life_days",
        help: "days for an unused volatile memory to lose half its strength",
        min: 1.0,
        max: 365.0,
        step: 1.0,
        get: |p| p.volatile_half_life_days,
        set: |p, v| p.volatile_half_life_days = v,
    },
    Knob {
        key: "use_threshold",
        help: "how close the work after a recall must come to a memory to count as using it",
        min: 0.1,
        max: 0.99,
        step: 0.05,
        get: |p| p.use_threshold,
        set: |p, v| p.use_threshold = v,
    },
    Knob {
        key: "link_threshold",
        help: "how close a new memory must come to an existing one to be linked to it",
        min: 0.5,
        max: 0.99,
        step: 0.02,
        get: |p| p.link_threshold,
        set: |p, v| p.link_threshold = v,
    },
    Knob {
        key: "recurrence_threshold",
        help: "how close a new memory must come to an existing one to count as restating it",
        min: 0.5,
        max: 0.99,
        step: 0.01,
        get: |p| p.recurrence_threshold,
        set: |p, v| p.recurrence_threshold = v,
    },
    Knob {
        key: "max_links",
        help: "the most links a new memory gets",
        min: 0.0,
        max: 10.0,
        step: 1.0,
        get: |p| p.max_links as f32,
        set: |p, v| p.max_links = v.round() as u32,
    },
    Knob {
        key: "pass_over_weight",
        help: "how much each pass-over weighs against a memory, next to one use",
        min: 0.0,
        max: 1.0,
        step: 0.05,
        get: |p| p.pass_over_weight,
        set: |p, v| p.pass_over_weight = v,
    },
    Knob {
        key: "strength_weight",
        help: "how much strength moves recall's order (0 = relevance alone)",
        min: 0.0,
        max: 1.0,
        step: 0.05,
        get: |p| p.strength_weight,
        set: |p, v| p.strength_weight = v,
    },
];

impl MemoryPolicy {
    /// Every knob brought inside its range; a non-finite value takes the
    /// default's. What the server stores and serves is always clamped.
    pub fn clamped(mut self) -> Self {
        let default = Self::default();
        for knob in KNOBS {
            let v = (knob.get)(&self);
            let v = if v.is_finite() {
                v
            } else {
                (knob.get)(&default)
            };
            (knob.set)(&mut self, v.clamp(knob.min, knob.max));
        }
        self
    }

    /// The knob named `key`.
    pub fn knob(key: &str) -> Option<&'static Knob> {
        KNOBS.iter().find(|k| k.key == key)
    }

    /// The half-life, in days, of a memory in `network`.
    pub fn half_life_days(&self, network: MemoryNetwork, volatile: bool) -> f32 {
        if volatile {
            return self.volatile_half_life_days;
        }
        match network {
            MemoryNetwork::World => self.world_half_life_days,
            MemoryNetwork::Bank => self.bank_half_life_days,
            MemoryNetwork::Opinion => self.opinion_half_life_days,
        }
    }

    /// How much a hit of `strength` keeps of its relevance in recall's order.
    pub fn rank_factor(&self, strength: f32) -> f32 {
        if !self.auto {
            return 1.0;
        }
        let w = self.strength_weight.clamp(0.0, 1.0);
        1.0 - w + w * strength.clamp(0.0, 1.0)
    }
}

/// How strongly `memory` stands at `now`: its confidence, halved for every
/// half-life since it was last of use (or written, if never), and lowered by
/// how often it was passed over against how often it was used. With the
/// policy's `auto` off it is the confidence alone.
pub fn strength(
    memory: &Memory,
    usage: Option<&MemoryUse>,
    policy: &MemoryPolicy,
    now: DateTime<Utc>,
) -> f32 {
    let confidence = memory.confidence.clamp(0.0, 1.0);
    if !policy.auto {
        return confidence;
    }
    let last = usage
        .and_then(|u| u.last_used_at)
        .map_or(memory.created_at, |t| t.max(memory.created_at));
    let days = (now - last).num_seconds().max(0) as f64 / 86_400.0;
    let half_life = f64::from(policy.half_life_days(memory.network, memory.volatile)).max(0.1);
    let fade = 0.5f64.powf(days / half_life);
    let (uses, passed) = usage.map_or((0, 0), |u| (u.uses(), u.passed_over()));
    let kept = (1.0 + f64::from(uses))
        / (1.0 + f64::from(uses) + f64::from(policy.pass_over_weight.max(0.0)) * f64::from(passed));
    (f64::from(confidence) * fade * kept) as f32
}

#[cfg(test)]
mod tests;

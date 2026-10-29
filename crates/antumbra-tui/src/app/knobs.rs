//! The knobs panel (`K`): the workspace's memory knobs, which the system
//! governs every memory by. Anyone sees them; an operator turns them, one step
//! at a time within each knob's range, and applies the lot with a confirm,
//! like any other operator action. Each memory's strength is read here too,
//! with the same function and knobs the server ranks recall by.

use antumbra_core::usage::{Knob, KNOBS};
use antumbra_core::{strength, Memory, MemoryPolicy, MemoryUse};

use super::{App, Mode, Pending};

impl App {
    /// Open the knobs panel on the workspace's current knobs.
    pub fn open_knobs(&mut self) {
        self.knob_draft = self.policy.clone();
        self.knob_selected = self.knob_selected.min(KNOBS.len() - 1);
        self.mode = Mode::Knobs;
    }

    /// Close the panel, dropping whatever was turned and not applied.
    pub fn close_knobs(&mut self) {
        self.knob_draft = self.policy.clone();
        self.mode = Mode::Normal;
    }

    /// Move the highlight up (`-1`) or down (`1`) the knobs.
    pub fn knob_move(&mut self, delta: i32) {
        let last = KNOBS.len() as i32 - 1;
        self.knob_selected = (self.knob_selected as i32 + delta).clamp(0, last) as usize;
    }

    /// Turn the highlighted knob one step down (`-1`) or up (`1`), within its
    /// range. Only an operator's console turns anything.
    pub fn knob_turn(&mut self, direction: i32) {
        if !self.operator {
            return;
        }
        let knob = &KNOBS[self.knob_selected];
        let now = (knob.get)(&self.knob_draft);
        let next = (now + knob.step * direction as f32).clamp(knob.min, knob.max);
        // Round off float drift so a value stepped up and back reads the same.
        let next = (next * 1000.0).round() / 1000.0;
        (knob.set)(&mut self.knob_draft, next);
    }

    /// Stage applying the turned knobs, opening the confirm prompt. Nothing to
    /// stage when nothing was turned; without an operator's token, nothing may
    /// be.
    pub fn request_knobs(&mut self) {
        if !self.may_act() {
            return;
        }
        if self.knob_draft == self.policy {
            self.mode = Mode::Normal;
            return;
        }
        self.pending = Some(Pending::SetMemoryPolicy(self.knob_draft.clone()));
        self.mode = Mode::Confirm;
    }

    /// What `turned` changes from the workspace's knobs, as `key a → b`.
    pub fn knob_changes(&self, turned: &MemoryPolicy) -> String {
        let changed: Vec<String> = KNOBS
            .iter()
            .filter(|k| (k.get)(turned) != (k.get)(&self.policy))
            .map(|k| {
                format!(
                    "{} {} → {}",
                    k.key,
                    knob_value(k, &self.policy),
                    knob_value(k, turned)
                )
            })
            .collect();
        if changed.is_empty() {
            "no change".into()
        } else {
            changed.join(", ")
        }
    }

    /// How many knobs `turned` changes.
    pub fn knobs_changed(&self, turned: &MemoryPolicy) -> usize {
        KNOBS
            .iter()
            .filter(|k| (k.get)(turned) != (k.get)(&self.policy))
            .count()
    }

    /// What the system has seen of `memory`'s use, if anything.
    pub fn usage_of(&self, memory: &Memory) -> Option<&MemoryUse> {
        self.usage.get(memory.id.as_str())
    }

    /// How strongly `memory` stands now, under the workspace's knobs.
    pub fn strength_of(&self, memory: &Memory) -> f32 {
        strength(
            memory,
            self.usage_of(memory),
            &self.policy,
            chrono::Utc::now(),
        )
    }
}

/// A knob's value as the panel shows it: a switch as on/off, a whole-number
/// knob without decimals.
pub fn knob_value(knob: &Knob, policy: &MemoryPolicy) -> String {
    let v = (knob.get)(policy);
    if knob.key == "auto" {
        return if v >= 0.5 { "on" } else { "off" }.into();
    }
    if knob.step >= 1.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.2}")
    }
}

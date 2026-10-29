use chrono::Duration;

use super::*;

fn memory(network: MemoryNetwork, confidence: f32, written: DateTime<Utc>) -> Memory {
    Memory::new("memory:1", "ws:1", network, "a fact", confidence, written)
}

#[test]
fn an_unused_memory_loses_half_its_strength_each_half_life() {
    let now = Utc::now();
    let policy = MemoryPolicy::default();
    let fresh = memory(MemoryNetwork::Bank, 0.8, now);
    assert!((strength(&fresh, None, &policy, now) - 0.8).abs() < 1e-6);

    let old = memory(MemoryNetwork::Bank, 0.8, now - Duration::days(90));
    assert!((strength(&old, None, &policy, now) - 0.4).abs() < 1e-3);
    let older = memory(MemoryNetwork::Bank, 0.8, now - Duration::days(180));
    assert!((strength(&older, None, &policy, now) - 0.2).abs() < 1e-3);
}

#[test]
fn each_network_and_a_volatile_memory_fade_at_their_own_rate() {
    let now = Utc::now();
    let policy = MemoryPolicy::default();
    let written = now - Duration::days(14);
    let fact = memory(MemoryNetwork::World, 1.0, written);
    let mut changing = memory(MemoryNetwork::World, 1.0, written);
    changing.volatile = true;
    assert!(strength(&fact, None, &policy, now) > 0.9);
    assert!((strength(&changing, None, &policy, now) - 0.5).abs() < 1e-3);
}

#[test]
fn a_use_restarts_the_fade() {
    let now = Utc::now();
    let policy = MemoryPolicy::default();
    let old = memory(MemoryNetwork::Bank, 0.8, now - Duration::days(180));
    let usage = MemoryUse {
        used: 1,
        judged: 1,
        last_used_at: Some(now),
        ..MemoryUse::new("memory:1")
    };
    assert!((strength(&old, Some(&usage), &policy, now) - 0.8).abs() < 1e-6);
}

#[test]
fn being_passed_over_weighs_against_a_memory_and_use_offsets_it() {
    let now = Utc::now();
    let policy = MemoryPolicy::default();
    let m = memory(MemoryNetwork::World, 1.0, now);
    let ignored = MemoryUse {
        judged: 10,
        ..MemoryUse::new("memory:1")
    };
    // One plus no uses, against one plus ten pass-overs at 0.1 each: half.
    assert!((strength(&m, Some(&ignored), &policy, now) - 0.5).abs() < 1e-6);

    let drawn_on = MemoryUse {
        judged: 10,
        used: 9,
        ..MemoryUse::new("memory:1")
    };
    assert!(strength(&m, Some(&drawn_on), &policy, now) > 0.99);
}

#[test]
fn with_auto_off_strength_is_the_confidence() {
    let now = Utc::now();
    let policy = MemoryPolicy {
        auto: false,
        ..MemoryPolicy::default()
    };
    let old = memory(MemoryNetwork::Bank, 0.7, now - Duration::days(900));
    assert!((strength(&old, None, &policy, now) - 0.7).abs() < 1e-6);
    assert_eq!(policy.rank_factor(0.0), 1.0);
}

#[test]
fn strength_moves_recall_order_by_its_weight() {
    let policy = MemoryPolicy::default();
    assert_eq!(policy.rank_factor(1.0), 1.0);
    assert!((policy.rank_factor(0.0) - 0.75).abs() < 1e-6);
}

#[test]
fn clamping_brings_every_knob_into_range() {
    let wild = MemoryPolicy {
        world_half_life_days: -5.0,
        use_threshold: f32::NAN,
        max_links: 400,
        strength_weight: 3.0,
        ..MemoryPolicy::default()
    }
    .clamped();
    assert_eq!(wild.world_half_life_days, 1.0);
    assert_eq!(wild.use_threshold, MemoryPolicy::default().use_threshold);
    assert_eq!(wild.max_links, 10);
    assert_eq!(wild.strength_weight, 1.0);
    assert_eq!(MemoryPolicy::default().clamped(), MemoryPolicy::default());
}

#[test]
fn every_knob_reads_back_what_it_sets() {
    for knob in KNOBS {
        let mut p = MemoryPolicy::default();
        let v = if knob.step >= 1.0 {
            knob.min.round()
        } else {
            knob.min
        };
        (knob.set)(&mut p, v);
        assert_eq!((knob.get)(&p), v, "{}", knob.key);
        assert!(MemoryPolicy::knob(knob.key).is_some());
    }
}

#[test]
fn a_policy_missing_knobs_reads_with_their_defaults() {
    let p: MemoryPolicy = serde_json::from_value(serde_json::json!({ "auto": false })).unwrap();
    assert!(!p.auto);
    assert_eq!(p.max_links, MemoryPolicy::default().max_links);
    let u: MemoryUse = serde_json::from_value(serde_json::json!({ "memory": "memory:1" })).unwrap();
    assert_eq!(u.recalled, 0);
}

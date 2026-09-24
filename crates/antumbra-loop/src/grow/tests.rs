use chrono::Utc;

use super::*;

fn region(name: &str, acceptability: f32, centroid: Vec<f32>) -> RegionCensus {
    RegionCensus {
        run_id: RunId::new("run"),
        generation: Generation(0),
        region: name.into(),
        tasks: 8,
        acceptability,
        centroid,
        at: Utc::now(),
    }
}

fn task(id: &str, region: &str) -> TaskPrompt {
    TaskPrompt {
        id: id.into(),
        prompt: format!("{region} {id}"),
        region: region.into(),
    }
}

#[test]
fn the_census_is_the_populations_score_by_region() {
    let tasks = [
        task("a1", "a"),
        task("a2", "a"),
        task("b1", "b"),
        task("c1", "c"),
    ];
    let e = ExpertId::new("e");
    let routed = vec![
        ("a1".to_string(), Some(e.clone())),
        ("a2".to_string(), None),
        ("b1".to_string(), Some(e.clone())),
        ("c1".to_string(), Some(e.clone())),
    ];
    let vectors: BTreeMap<String, Vec<f32>> = [
        ("a1".to_string(), vec![1.0, 0.0]),
        ("a2".to_string(), vec![0.0, 1.0]),
    ]
    .into_iter()
    .collect();
    let score = |to: &Option<ExpertId>, t: &str| match (to, t) {
        (Some(_), "a1") => Some(1.0),
        (None, "a2") => Some(0.5),
        (Some(_), "b1") => Some(0.25),
        _ => None,
    };
    let c = census(&tasks, &routed, &vectors, score);
    assert_eq!(c["a"].0, 2);
    assert_eq!(
        c["a"].1, 0.75,
        "the expert where routed, the base where escalated"
    );
    assert_eq!(c["a"].2, vec![0.5, 0.5]);
    assert_eq!((c["b"].0, c["b"].1), (1, 0.25));
    assert!(
        !c.contains_key("c"),
        "a region nothing was scored on has no census"
    );
}

#[test]
fn the_gate_keeps_out_what_the_population_cannot_reach() {
    let census = [
        region("never", 0.0, vec![1.0, 0.0]),
        region("reachable", 0.3, vec![0.0, 1.0]),
    ];
    let (candidates, chosen) = choose(&census, &[], &GrowPolicy::default());
    assert_eq!(chosen.as_deref(), Some("reachable"));
    assert!(!candidates[0].admitted && candidates[1].admitted);
    let (_, none) = choose(&census[..1], &[], &GrowPolicy::default());
    assert!(none.is_none(), "nothing admitted, nothing chosen");
}

#[test]
fn learnability_prefers_the_half_solved_region() {
    let census = [
        region("nearly-solved", 0.95, vec![1.0, 0.0, 0.0]),
        region("half", 0.5, vec![0.0, 1.0, 0.0]),
        region("hard", 0.1, vec![0.0, 0.0, 1.0]),
    ];
    let (candidates, chosen) = choose(&census, &[], &GrowPolicy::default());
    assert_eq!(chosen.as_deref(), Some("half"));
    assert!((candidates[1].learnability - 0.25).abs() < 1e-6);
}

/// A region just like the one chosen recently gives up part of its
/// learnability, and a less learnable but different one is chosen instead.
#[test]
fn redundancy_with_recent_choices_steers_elsewhere() {
    let census = [
        region("again", 0.5, vec![1.0, 0.0]),
        region("elsewhere", 0.3, vec![0.0, 1.0]),
    ];
    let (_, fresh) = choose(&census, &[], &GrowPolicy::default());
    assert_eq!(fresh.as_deref(), Some("again"));
    let (candidates, steered) = choose(&census, &[vec![1.0, 0.0]], &GrowPolicy::default());
    assert_eq!(steered.as_deref(), Some("elsewhere"));
    assert!((candidates[0].penalty - 0.5).abs() < 1e-6);
    assert!((candidates[0].score - 0.125).abs() < 1e-6);
}

fn decided(
    generation: u32,
    chosen: Option<&str>,
    gated_out: &[&str],
    admitted: &[&str],
) -> GrowRecord {
    let candidate = |r: &&str, admitted: bool| RegionCandidate {
        region: r.to_string(),
        acceptability: 0.0,
        admitted,
        learnability: 0.0,
        penalty: 0.0,
        score: 0.0,
    };
    GrowRecord {
        run_id: RunId::new("run"),
        generation: Generation(generation),
        census_generation: None,
        chosen: chosen.map(str::to_string),
        candidates: gated_out
            .iter()
            .map(|r| candidate(r, false))
            .chain(admitted.iter().map(|r| candidate(r, true)))
            .collect(),
        focus: 0,
        unfiltered: 0,
        credit: None,
        at: Utc::now(),
    }
}

/// The uniform baseline picks among the admitted regions only, the same one
/// for the same run and generation, and not always the same one.
#[test]
fn the_uniform_baseline_draws_among_the_admitted() {
    let census = [
        region("never", 0.0, vec![1.0, 0.0, 0.0]),
        region("a", 0.5, vec![0.0, 1.0, 0.0]),
        region("b", 0.9, vec![0.0, 0.0, 1.0]),
    ];
    let (candidates, _) = choose(&census, &[], &GrowPolicy::default());
    let run = RunId::new("run");
    let picks: Vec<String> = (0..16)
        .filter_map(|g| uniformly(&candidates, &run, Generation(g)))
        .collect();
    assert_eq!(picks.len(), 16);
    assert!(picks.iter().all(|p| p != "never"));
    assert!(picks.iter().any(|p| p == "a") && picks.iter().any(|p| p == "b"));
    assert_eq!(
        uniformly(&candidates, &run, Generation(3)),
        uniformly(&candidates, &run, Generation(3))
    );
    assert!(uniformly(&candidates[..1], &run, Generation(0)).is_none());
}

#[test]
fn diversity_reads_entropy_coverage_and_revived_regions() {
    let even = [
        decided(0, Some("a"), &[], &["a", "b"]),
        decided(1, Some("b"), &[], &["a", "b"]),
    ];
    let d = diversity(&even, 2);
    assert!((d.entropy - 1.0).abs() < 1e-6, "{d:?}");
    assert_eq!(d.coverage, 1.0);
    let narrow = [
        decided(0, Some("a"), &["c"], &["a"]),
        decided(1, Some("a"), &["c"], &["a"]),
        decided(2, Some("a"), &[], &["a", "c"]),
    ];
    let d = diversity(&narrow, 4);
    assert_eq!((d.entropy, d.coverage), (0.0, 0.25));
    assert!(d.entropy.is_sign_positive(), "not a negative zero");
    assert_eq!(d.revived, 1, "c was gated out and now passes");
    assert_eq!(diversity(&[], 3).coverage, 0.0);
}

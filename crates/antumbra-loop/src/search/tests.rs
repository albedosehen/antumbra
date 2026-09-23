use chrono::Utc;

use antumbra_core::{Generation, RunId, ShadowId};

use super::*;

fn recipe(learning_rate: f64, batch_size: u32) -> TrainingRecipe {
    TrainingRecipe {
        learning_rate,
        batch_size,
        kl_beta: 0.04,
    }
}

fn row(generation: u32, index: usize, r: TrainingRecipe, fitness: f32, evals: u32) -> RecipeRecord {
    RecipeRecord {
        shadow: ShadowId::new(format!("run:g{generation}:s{index}")),
        run_id: RunId::new("run"),
        generation: Generation(generation),
        recipe: r,
        parent: None,
        partition_seed: None,
        fitness_mean: fitness,
        fitness_variance: None,
        evaluations: evals,
        created_at: Utc::now(),
    }
}

#[test]
fn the_learning_rate_is_placed_on_a_log_scale() {
    let space = RecipeSpace::default();
    let at = |lr: f64| space.to_unit(&recipe(lr, 1))[0];
    assert!((at(1e-5) - 0.0).abs() < 1e-12);
    assert!((at(1e-3) - 1.0).abs() < 1e-12);
    assert!((at(1e-4) - 0.5).abs() < 1e-12, "the log midpoint");
}

#[test]
fn batch_sizes_snap_to_an_allowed_one_and_fixed_axes_are_not_searched() {
    let space = RecipeSpace::default();
    assert_eq!(space.dims(), 2, "the KL weight is fixed by default");
    assert_eq!(space.from_unit(&[0.5, 0.4]).batch_size, 2);
    assert_eq!(space.from_unit(&[0.5, 0.9]).batch_size, 4);
    assert_eq!(space.from_unit(&[0.5, 0.9]).kl_beta, 0.04);
    let r = recipe(3e-4, 4);
    let back = space.from_unit(&space.to_unit(&r));
    assert!((back.learning_rate - r.learning_rate).abs() < 1e-12);
    assert_eq!((back.batch_size, back.kl_beta), (4, 0.04));
}

/// One lucky evaluation does not beat a well-measured recipe that scored a
/// little lower: the optimizer's curse, answered by shrinkage.
#[test]
fn a_lucky_single_evaluation_does_not_win() {
    let lucky = row(0, 0, recipe(1e-3, 1), 0.9, 1);
    let measured = row(0, 1, recipe(1e-4, 2), 0.8, 4);
    let middling = row(0, 2, recipe(1e-5, 1), 0.2, 1);
    let history = [lucky, measured.clone(), middling];
    let best = incumbent(&history, 1.0).expect("one of them");
    assert_eq!(best.shadow, measured.shadow);
}

#[test]
fn the_first_member_is_the_incumbent_or_the_anchor() {
    let policy = SearchPolicy::default();
    let anchor = recipe(1e-4, 1);
    assert_eq!(propose(&policy, &[], 0, anchor)[0], anchor);
    let good = recipe(3e-4, 4);
    let history = [row(0, 0, anchor, 0.3, 1), row(0, 1, good, 0.9, 1)];
    assert_eq!(propose(&policy, &history, 1, anchor)[0], good);
}

#[test]
fn a_cohort_is_all_different_inside_the_space_and_reproducible() {
    let policy = SearchPolicy {
        cohort: 6,
        ..SearchPolicy::default()
    };
    let history = [
        row(0, 0, recipe(1e-4, 1), 0.4, 1),
        row(0, 1, recipe(5e-4, 2), 0.6, 1),
        row(0, 2, recipe(2e-5, 4), 0.2, 1),
    ];
    let cohort = propose(&policy, &history, 1, recipe(1e-4, 1));
    assert_eq!(cohort.len(), 6);
    for (i, a) in cohort.iter().enumerate() {
        assert!((1e-5..=1e-3).contains(&a.learning_rate), "{a:?}");
        assert!([1, 2, 4].contains(&a.batch_size), "{a:?}");
        assert_eq!(a.kl_beta, 0.04);
        assert!(cohort[i + 1..].iter().all(|b| b != a), "duplicate {a:?}");
    }
    assert_eq!(
        cohort,
        propose(&policy, &history, 1, recipe(1e-4, 1)),
        "the same history and seed propose the same cohort"
    );
}

#[test]
fn nothing_to_search_or_no_room_proposes_accordingly() {
    let fixed = SearchPolicy {
        space: RecipeSpace {
            learning_rate: (1e-4, 1e-4),
            batch_sizes: vec![1],
            kl_beta: (0.04, 0.04),
        },
        ..SearchPolicy::default()
    };
    assert_eq!(propose(&fixed, &[], 0, recipe(1e-4, 1)).len(), 1);
    let none = SearchPolicy {
        cohort: 0,
        ..SearchPolicy::default()
    };
    assert!(propose(&none, &[], 0, recipe(1e-4, 1)).is_empty());
}

/// A landscape with one good region: learning rate near 3e-4, and larger
/// batches better. Invented for the test; its only job is to have a region
/// worth finding.
fn true_fitness(r: &TrainingRecipe) -> f32 {
    let off = (r.learning_rate.log10() - 3e-4f64.log10()) / 1.0;
    let batch = match r.batch_size {
        4 => 0.0,
        2 => 0.05,
        _ => 0.15,
    };
    (0.9 - 0.3 * off * off - batch).clamp(0.0, 1.0) as f32
}

/// Six generations of four, observed with noise, and the recipe each method
/// would take forward judged by its true fitness.
fn searched(seed: u64) -> f32 {
    let policy = SearchPolicy {
        seed,
        ..SearchPolicy::default()
    };
    let mut noise = SplitMix(seed.wrapping_add(1000));
    let mut history = Vec::new();
    for generation in 0..6 {
        let cohort = propose(&policy, &history, generation, recipe(1e-4, 1));
        for (i, r) in cohort.into_iter().enumerate() {
            let observed = true_fitness(&r) + 0.06 * (noise.unit() as f32 - 0.5);
            history.push(row(generation, i, r, observed, 1));
        }
    }
    true_fitness(
        &incumbent(&history, policy.prior_weight)
            .expect("history")
            .recipe,
    )
}

fn random(seed: u64) -> f32 {
    let space = RecipeSpace::default();
    let mut rng = SplitMix(seed.wrapping_add(7));
    let mut noise = SplitMix(seed.wrapping_add(1000));
    let mut best = (f32::MIN, recipe(1e-4, 1));
    for _ in 0..24 {
        let r = space.from_unit(&[rng.unit(), rng.unit()]);
        let observed = true_fitness(&r) + 0.06 * (noise.unit() as f32 - 0.5);
        if observed > best.0 {
            best = (observed, r);
        }
    }
    true_fitness(&best.1)
}

/// The search has to earn its complexity: on the same budget of evaluations,
/// the recipe it carries forward must be better than random search's, on
/// average and in its worst run. Measured when written, over these twelve
/// seeds: 0.898 against 0.882 on average, 0.891 against 0.840 at worst, with
/// the true optimum at 0.900.
#[test]
fn the_search_beats_random_search_on_the_same_budget() {
    let seeds = 0..12u64;
    let n = seeds.clone().count() as f32;
    let searched_runs: Vec<f32> = seeds.clone().map(searched).collect();
    let random_runs: Vec<f32> = seeds.map(random).collect();
    let mean = |runs: &[f32]| runs.iter().sum::<f32>() / n;
    let worst = |runs: &[f32]| runs.iter().copied().fold(f32::MAX, f32::min);
    assert!(
        mean(&searched_runs) > mean(&random_runs),
        "searched {searched_runs:?} vs random {random_runs:?}"
    );
    assert!(
        worst(&searched_runs) > worst(&random_runs),
        "searched {searched_runs:?} vs random {random_runs:?}"
    );
}

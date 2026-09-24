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
    assert_eq!(propose(&policy, &[], 0, anchor, &[])[0], Some(anchor));
    let good = recipe(3e-4, 4);
    let history = [row(0, 0, anchor, 0.3, 1), row(0, 1, good, 0.9, 1)];
    assert_eq!(propose(&policy, &history, 1, anchor, &[])[0], Some(good));
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
    let proposed = propose(&policy, &history, 1, recipe(1e-4, 1), &[]);
    let cohort: Vec<TrainingRecipe> = proposed.iter().flatten().copied().collect();
    assert_eq!(cohort.len(), 6);
    for (i, a) in cohort.iter().enumerate() {
        assert!((1e-5..=1e-3).contains(&a.learning_rate), "{a:?}");
        assert!([1, 2, 4].contains(&a.batch_size), "{a:?}");
        assert_eq!(a.kl_beta, 0.04);
        assert!(cohort[i + 1..].iter().all(|b| b != a), "duplicate {a:?}");
    }
    assert_eq!(
        proposed,
        propose(&policy, &history, 1, recipe(1e-4, 1), &[]),
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
    assert_eq!(propose(&fixed, &[], 0, recipe(1e-4, 1), &[]).len(), 1);
    let none = SearchPolicy {
        cohort: 0,
        ..SearchPolicy::default()
    };
    assert!(propose(&none, &[], 0, recipe(1e-4, 1), &[]).is_empty());
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

/// What each member ran before `generation`, read from its rows, as the
/// loop reads it.
fn slots_before(history: &[RecipeRecord], generation: u32, cohort: usize) -> Vec<Option<Slot>> {
    let ran = |g: u32, i: usize| {
        history
            .iter()
            .find(|r| r.generation.0 == g && r.shadow.as_str().ends_with(&format!(":s{i}")))
            .map(|r| r.recipe)
    };
    (0..cohort)
        .map(|i| {
            let recipe = ran(generation.checked_sub(1)?, i)?;
            let mut held = 1;
            while let Some(g) = generation.checked_sub(held + 1) {
                if ran(g, i) != Some(recipe) {
                    break;
                }
                held += 1;
            }
            Some(Slot { recipe, held })
        })
        .collect()
}

/// `generations` of `policy`, observed with noise, and the recipe the search
/// would take forward judged by its true fitness.
fn searched_under(policy: &SearchPolicy, generations: u32, noise_width: f32) -> f32 {
    let mut noise = SplitMix(policy.seed.wrapping_add(1000));
    let mut history = Vec::new();
    for generation in 0..generations {
        let slots = slots_before(&history, generation, policy.cohort);
        let cohort = propose(policy, &history, generation, recipe(1e-4, 1), &slots);
        for (i, r) in cohort.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let observed = true_fitness(&r) + noise_width * (noise.unit() as f32 - 0.5);
            history.push(row(generation, i, r, observed, 1));
        }
    }
    true_fitness(
        &incumbent(&history, policy.prior_weight)
            .expect("history")
            .recipe,
    )
}

/// Six generations of four, observed with noise, and the recipe each method
/// would take forward judged by its true fitness.
fn searched(seed: u64) -> f32 {
    let policy = SearchPolicy {
        seed,
        ..SearchPolicy::default()
    };
    searched_under(&policy, 6, 0.06)
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

/// A recipe run in three generations is ranked on all three runs, so a
/// newcomer's single run a little higher does not displace it. Ranked row by
/// row, the newcomer would have won.
#[test]
fn a_recipe_run_again_is_ranked_on_all_its_runs() {
    let steady = recipe(1e-4, 1);
    let newcomer = recipe(3e-4, 2);
    let history = [
        row(0, 0, steady, 0.8, 1),
        row(0, 1, recipe(1e-5, 1), 0.4, 1),
        row(1, 0, steady, 0.8, 1),
        row(1, 1, recipe(2e-5, 1), 0.4, 1),
        row(2, 0, steady, 0.8, 1),
        row(2, 1, newcomer, 0.83, 1),
        row(2, 2, recipe(5e-5, 4), 0.37, 1),
    ];
    let best = incumbent(&history, 1.0).expect("one of them");
    assert_eq!(best.recipe, steady);
    assert_eq!(best.generation.0, 2, "as its latest row");
    let newest = &history[5];
    let alone = shrunk_fitness(newest, (0.8 + 0.83 + 0.37) / 3.0, 1.0);
    assert!((pooled_fitness(&history, &newcomer, 1.0) - alone).abs() < 1e-6);
    let row_by_row = |r: &RecipeRecord| shrunk_fitness(r, 2.0 / 3.0, 1.0);
    assert!(row_by_row(newest) > row_by_row(&history[4]));
}

#[test]
fn slow_members_are_the_last_and_never_the_first() {
    let policy = |cohort, slow| SearchPolicy {
        cohort,
        slow,
        ..SearchPolicy::default()
    };
    let slow_of = |p: SearchPolicy| (0..p.cohort).filter(|&i| p.is_slow(i)).collect::<Vec<_>>();
    assert_eq!(slow_of(policy(4, 0)), Vec::<usize>::new());
    assert_eq!(slow_of(policy(4, 1)), [3]);
    assert_eq!(slow_of(policy(4, 2)), [2, 3]);
    assert_eq!(
        slow_of(policy(4, 9)),
        [1, 2, 3],
        "the first carries the incumbent"
    );
}

#[test]
fn the_fast_interval_lengthens_over_the_run_and_stays_short_of_the_slow() {
    let policy = SearchPolicy {
        slow_interval: 3,
        anneal: 8,
        ..SearchPolicy::default()
    };
    let at: Vec<u32> = [0, 3, 4, 8, 100]
        .iter()
        .map(|&g| policy.fast_interval(g))
        .collect();
    assert_eq!(at, [1, 1, 2, 2, 2]);
    let unannealed = SearchPolicy {
        anneal: 0,
        ..policy.clone()
    };
    assert_eq!(unannealed.fast_interval(100), 1);
    let one = SearchPolicy {
        slow_interval: 1,
        ..policy
    };
    assert_eq!(one.fast_interval(100), 1);
}

fn slow_policy() -> SearchPolicy {
    SearchPolicy {
        cohort: 3,
        slow: 1,
        slow_interval: 3,
        ..SearchPolicy::default()
    }
}

/// The slow member keeps a poor recipe until its interval has passed, however
/// much better the incumbent is: the fast members cannot truncate it. The fast
/// member is proposed afresh.
#[test]
fn a_slow_member_keeps_its_recipe_until_its_interval_has_passed() {
    let policy = slow_policy();
    let (good, fast, poor) = (recipe(3e-4, 4), recipe(1e-5, 2), recipe(1e-3, 1));
    let history = [
        row(0, 0, good, 0.9, 1),
        row(0, 1, fast, 0.5, 1),
        row(0, 2, poor, 0.1, 1),
    ];
    let slots = |held| {
        [
            Some(Slot {
                recipe: good,
                held: 1,
            }),
            Some(Slot {
                recipe: fast,
                held: 1,
            }),
            Some(Slot { recipe: poor, held }),
        ]
    };
    let kept = propose(&policy, &history, 1, recipe(1e-4, 1), &slots(1));
    assert_eq!(kept[0], Some(good));
    assert_eq!(kept[2], Some(poor));
    assert!(kept[1].is_some_and(|r| r != fast), "{kept:?}");
    let due = propose(&policy, &history, 3, recipe(1e-4, 1), &slots(3));
    assert!(due[2].is_some_and(|r| r != poor), "{due:?}");
}

/// When a slow member's recipe has become the incumbent, it is not run twice:
/// the slow member keeps it, and the first member is proposed another.
#[test]
fn an_incumbent_a_slow_member_holds_is_not_run_twice() {
    let policy = slow_policy();
    let (held, other) = (recipe(3e-4, 4), recipe(1e-5, 1));
    let history = [
        row(0, 0, other, 0.3, 1),
        row(0, 1, recipe(1e-3, 2), 0.2, 1),
        row(0, 2, held, 0.9, 1),
    ];
    let slots = [
        Some(Slot {
            recipe: other,
            held: 1,
        }),
        None,
        Some(Slot {
            recipe: held,
            held: 1,
        }),
    ];
    let cohort = propose(&policy, &history, 1, recipe(1e-4, 1), &slots);
    assert_eq!(cohort[2], Some(held));
    let recipes: Vec<_> = cohort.iter().flatten().collect();
    assert_eq!(recipes.len(), 3);
    assert_eq!(
        recipes.iter().filter(|r| ***r == held).count(),
        1,
        "{cohort:?}"
    );
}

/// Held members are read back from the rows, and a run of the full policy
/// keeps the slow member's recipe for exactly its interval.
#[test]
fn a_run_holds_the_slow_members_recipe_for_its_interval() {
    let policy = slow_policy();
    let mut history = Vec::new();
    for generation in 0..7 {
        let slots = slots_before(&history, generation, policy.cohort);
        let cohort = propose(&policy, &history, generation, recipe(1e-4, 1), &slots);
        for (i, r) in cohort.into_iter().enumerate() {
            let r = r.expect("room in the space");
            history.push(row(generation, i, r, true_fitness(&r), 1));
        }
    }
    let slow: Vec<TrainingRecipe> = history
        .iter()
        .filter(|r| r.shadow.as_str().ends_with(":s2"))
        .map(|r| r.recipe)
        .collect();
    assert_eq!(slow.len(), 7);
    assert!(slow[0] == slow[1] && slow[1] == slow[2], "{slow:?}");
    assert!(slow[3] == slow[4] && slow[4] == slow[5], "{slow:?}");
    assert!(slow[2] != slow[3] && slow[5] != slow[6], "{slow:?}");
}

/// With a slow member and the fast interval annealed, the search still earns
/// its complexity against random search on the same budget. Over 300 seeds of
/// twelve generations the two frequencies measured level with the fast cohort
/// alone at every noise tried (0.1, 0.3 and 0.5 wide): this landscape does not
/// move between generations, which is where greed would cost.
#[test]
fn the_search_with_a_slow_cohort_still_beats_random_search() {
    let runs: Vec<f32> = (0..12u64)
        .map(|seed| {
            let policy = SearchPolicy {
                seed,
                slow: 1,
                anneal: 6,
                ..SearchPolicy::default()
            };
            searched_under(&policy, 6, 0.06)
        })
        .collect();
    let random_runs: Vec<f32> = (0..12u64).map(random).collect();
    let mean = |r: &[f32]| r.iter().sum::<f32>() / r.len() as f32;
    let worst = |r: &[f32]| r.iter().copied().fold(f32::MAX, f32::min);
    assert!(
        mean(&runs) > mean(&random_runs),
        "{runs:?} vs {random_runs:?}"
    );
    assert!(
        worst(&runs) > worst(&random_runs),
        "{runs:?} vs {random_runs:?}"
    );
}

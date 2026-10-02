//! How `antumbra train` prints a generation: a line for the shadow, then one
//! for each thing the loop measured or decided about it.

use antumbra_loop::GenerationReport;

/// Print one generation's report.
pub(crate) fn print(r: &GenerationReport) {
    let curve: Vec<String> = r.reward_curve.iter().map(|p| format!("{p:.2}")).collect();
    println!(
        "gen {:<3} shadow {:<16} pass-rate/round=[{}] final={:.2} graduated={}",
        r.generation.0,
        r.shadow,
        curve.join(", "),
        r.fitness,
        r.graduated
    );
    if let Some(m) = &r.instruments {
        let gap = m
            .widest_gap()
            .map(|(band, w)| format!("{w:+.2} ({} tasks)", band.as_str()))
            .unwrap_or_else(|| "not measured".into());
        let audit = m
            .audit
            .rate()
            .map(|a| format!("{a:.2} over {}", m.audit.measured))
            .unwrap_or_else(|| "not due".into());
        let trend = r.trend.map_or("not read", |t| t.as_str());
        println!("        widest held-out gap {gap}, audit {audit}, trend {trend}");
    }
    if let Some(recipe) = &r.recipe {
        println!(
            "        recipe lr {:.1e}, batch {}, kl {}",
            recipe.learning_rate, recipe.batch_size, recipe.kl_beta
        );
    }
    if r.cohort.len() > 1 {
        for m in &r.cohort {
            let recipe = m.recipe.map_or("unreported".to_string(), |x| {
                format!(
                    "lr {:.1e} batch {} kl {}",
                    x.learning_rate, x.batch_size, x.kl_beta
                )
            });
            println!(
                "        member {} {recipe} fitness {:.2}{}",
                m.shadow,
                m.fitness,
                if m.slow { " (slow)" } else { "" }
            );
        }
        if r.remeasured.is_none() {
            println!(
                "        graduation score {:.2} (the best, shrunk toward the cohort's mean)",
                r.graduation_score
            );
        }
    }
    if let Some(m) = &r.remeasured {
        let rates: Vec<String> = m.pass_rates.iter().map(|p| format!("{p:.2}")).collect();
        println!(
            "        re-measured on {} {} task(s) under {} seed(s): [{}], graduation score {:.2}",
            m.tasks,
            if m.held_out { "held-out" } else { "trained" },
            m.pass_rates.len(),
            rates.join(", "),
            r.graduation_score
        );
    }
    match &r.admission {
        Some(antumbra_loop::Admission::Superseded {
            archived,
            similarity,
            candidate,
            incumbent,
        }) => println!(
            "        admitted in place of {archived} (similarity {similarity:.3}): \
             {candidate:.2} against its {incumbent:.2}; {archived} archived"
        ),
        Some(antumbra_loop::Admission::Rejected {
            duplicate_of,
            similarity,
            candidate,
            incumbent,
        }) => {
            let head_to_head = match (candidate, incumbent) {
                (Some(c), Some(i)) => format!("{c:.2} against its {i:.2}"),
                _ => "not measurable here".to_string(),
            };
            println!(
                "        not admitted: duplicates {duplicate_of} (similarity {similarity:.3}), {head_to_head}"
            );
        }
        Some(antumbra_loop::Admission::Outserved { tasks, .. }) if *tasks == 0 => println!(
            "        not admitted: the gate would route none of the live tasks to it"
        ),
        Some(antumbra_loop::Admission::Outserved {
            tasks,
            escalated,
            candidate,
            serving,
        }) => println!(
            "        not admitted: the population scores {candidate:.2} with it on the {tasks} live task(s) it would reroute ({escalated} of them escalated to the base model), against {serving:.2} without it"
        ),
        Some(antumbra_loop::Admission::Admitted {
            nearest: Some((id, s)),
        }) => println!("        admitted: nearest expert {id} at similarity {s:.3}"),
        _ => {}
    }
    if let Some(w) = &r.critic {
        let read = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.2}"));
        println!(
            "        critic over {} answer(s): correlation {}, calibration error {} ({} recalibrated), twin agreement {}",
            w.n,
            read(w.correlation),
            read(w.ece),
            read(w.recalibrated_ece),
            read(w.twin_agreement)
        );
    }
    for c in &r.rechecks {
        let verdict = c
            .measurement
            .as_ref()
            .map_or("not measured".to_string(), |m| format!("{:?}", m.verdict));
        let moved = c.moved.map_or(String::new(), |to| {
            format!("; now {to:?}, {} expert(s) archived", c.archived.len())
        });
        println!(
            "        recheck {}: {} answer(s) anchored, {} not; {} rewarded answer(s) the anchor failed; judged over {} generation(s): {verdict}{moved}",
            c.verifier, c.anchored, c.unanchored, c.rewarded_wrong, c.generations
        );
    }
    if !r.withdrawn.is_empty() {
        let ids: Vec<&str> = r.withdrawn.iter().map(|v| v.as_str()).collect();
        println!(
            "        not graduated: trained under {}, which no longer grant reward",
            ids.join(", ")
        );
    }
    for (expert, warning) in &r.detection.warnings {
        println!("        warning (advisory) {expert}: {warning:?}");
    }
    for moved in &r.detection.demoted {
        println!(
            "        demoted {} to dormant on {:?}",
            moved.expert, moved.cause
        );
    }
    if let Some(g) = &r.growth {
        match &g.record.chosen {
            Some(region) => {
                let learnability = g
                    .record
                    .candidates
                    .iter()
                    .find(|c| &c.region == region)
                    .map_or(0.0, |c| c.learnability);
                let credit = g
                    .record
                    .credit
                    .map_or("none yet".to_string(), |c| format!("{c:+.2}"));
                let from = g
                    .record
                    .warm_from
                    .as_ref()
                    .map_or("fresh factors".to_string(), |e| format!("warm from {e}"));
                println!(
                    "        grow: learned from {region} (learnability {learnability:.3}; {} task(s), {} unfiltered; {from}); last choice's credit {credit}; entropy {:.2}, coverage {:.2}, revived {}",
                    g.record.focus, g.record.unfiltered, g.diversity.entropy, g.diversity.coverage, g.diversity.revived
                );
            }
            None => println!(
                "        grow: no region chosen (no census yet, or none passed the gate); learned from every visible task"
            ),
        }
    }
    match &r.merge {
        Some(antumbra_loop::Merge::Merged {
            into,
            pair,
            similarity,
            retained,
            merged,
            better,
        }) => println!(
            "        merged {} and {} into {into} (similarity {similarity:.3}, overlap {retained:.3}): {merged:.2} against the better's {better:.2}; both archived",
            pair.0, pair.1
        ),
        Some(antumbra_loop::Merge::NotSiblings {
            pair,
            similarity,
            retained,
        }) => println!(
            "        not merged: {} and {} (similarity {similarity:.3}) share {retained:.3} of their subspace",
            pair.0, pair.1
        ),
        Some(antumbra_loop::Merge::Costly {
            pair,
            retained,
            merged,
            better,
            ..
        }) => println!(
            "        not merged: {} and {} (overlap {retained:.3}) merge to {merged:.2} against the better's {better:.2}",
            pair.0, pair.1
        ),
        None => {}
    }
    if let Some(b) = &r.baseline {
        let best = match (&b.best, b.best_alone, b.delta()) {
            (Some(id), Some(alone), Some(d)) => {
                format!("its best single expert {id} {alone:.2} (routing adds {d:+.2})")
            }
            _ => "no single expert to compare".to_string(),
        };
        let headroom = b.headroom().map_or(String::new(), |h| {
            // Uncross-fitted, the oracle is the highest of noisy scores and
            // reads high even over identical experts.
            let how = if b.oracle_cross_fitted {
                "cross-fitted"
            } else {
                "highest scores, biased up"
            };
            format!(
                "; routed as well as it could be ({how}), {:.2} ({h:+.2})",
                b.population + h
            )
        });
        println!(
            "        population {:.2} over {} live task(s) against {best}{headroom}",
            b.population, b.tasks
        );
    }
    match r.critic_fallback {
        Some(antumbra_core::critic::Fallback::Uncorrelated { correlation }) => println!(
            "        critic set aside: correlation {correlation:+.2} with the verifier; verifier-only reward from the next generation"
        ),
        Some(antumbra_core::critic::Fallback::TwinDeclined { from, to }) => println!(
            "        critic set aside: twin agreement fell from {from:.2} to {to:.2}; verifier-only reward from the next generation"
        ),
        None => {}
    }
    if r.routing_outcomes > 0 {
        println!(
            "        routing: {} live task(s) with a clear winner; the gate was retrained on them",
            r.routing_outcomes
        );
    }
    for c in &r.contribution {
        let delta = match (c.with, c.without, c.delta()) {
            (Some(with), Some(without), Some(d)) => {
                format!("with {with:.2} without {without:.2} contribution {d:+.2}")
            }
            _ => "unused".to_string(),
        };
        println!(
            "        expert {} routed {}/{} {delta}",
            c.expert, c.routed, c.tasks
        );
    }
    let scored = r.contribution_scores;
    if scored.asked > 0 {
        println!(
            "        contribution scores: {} task score(s) asked, {} already known",
            scored.asked, scored.reused
        );
    }
}

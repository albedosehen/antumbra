//! `antumbra verifier`: proposing verifiers, building the
//! cases they are measured on, measuring them, and moving them.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Context};
use chrono::{TimeDelta, Utc};

use antumbra_core::ports::{Verifier, VerifyRequest};
use antumbra_core::{
    verifier_address, Anchor, Case, Label, RunId, TrustMeasurement, TrustPolicy, TrustState,
    TrustVerdict, VerifierId, VerifierOrigin, VerifierRecord, VerifierTier,
};
use antumbra_critic::trust::{challenge, measure};
use antumbra_critic::CommandVerifier;
use antumbra_store::repo::verifier;
use antumbra_store::Store;

use crate::verifier_args::VerifierAction;

pub async fn run(url: &str, action: VerifierAction) -> anyhow::Result<()> {
    let store = crate::connect(url).await?;
    match action {
        VerifierAction::Propose {
            spec,
            domain,
            task,
            tier,
            authored,
            by,
            batch,
        } => {
            let records = match batch {
                Some(path) => batch_records(&read_json(&format!("@{path}"))?)?,
                None => {
                    let spec = read_json(spec.as_deref().unwrap_or_default())?;
                    let origin = if authored {
                        VerifierOrigin::Authored
                    } else {
                        VerifierOrigin::Synthesized
                    };
                    let domain = domain.unwrap_or_default();
                    let mut record =
                        VerifierRecord::new(domain, task, tier.parse()?, origin, spec, Utc::now());
                    record.proposed_by = by;
                    vec![record]
                }
            };
            for record in records {
                let kept = verifier::propose(&store, &record).await?;
                let state = verifier::state_of(&store, &kept).await?;
                println!(
                    "{} ({}, {}) {}",
                    kept.id,
                    kept.origin.as_str(),
                    state.as_str(),
                    kept.task.as_deref().unwrap_or(&kept.domain)
                );
            }
        }
        VerifierAction::Synthesize {
            corpus,
            skill,
            task,
            inputs,
            draws,
            max_new_tokens,
            base_model,
            out,
        } => {
            crate::verifier_synth::synthesize(crate::verifier_synth::SynthesizeArgs {
                corpus,
                skill,
                task,
                inputs,
                draws,
                max_new_tokens,
                base_model,
                out,
            })
            .await?;
        }
        VerifierAction::Cases {
            corpus,
            task,
            skill,
            completions,
            out,
        } => {
            let tasks: Vec<serde_json::Value> =
                serde_json::from_value(read_json(&format!("@{corpus}"))?)?;
            let mut given: Vec<serde_json::Value> = Vec::new();
            for path in &completions {
                let more: Vec<serde_json::Value> =
                    serde_json::from_value(read_json(&format!("@{path}"))?)?;
                given.extend(more);
            }
            let stem = std::path::Path::new(&corpus)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("corpus")
                .to_string();
            let only = Only {
                task: task.as_deref(),
                skill: skill.as_deref(),
            };
            let cases = build_cases(&store, &tasks, &stem, only, &given).await?;
            let text = serde_json::to_string_pretty(&cases)?;
            match out {
                Some(path) => {
                    std::fs::write(&path, text).with_context(|| format!("writing {path}"))?;
                    eprintln!("{} case(s) written to {path}", cases.len());
                }
                None => println!("{text}"),
            }
        }
        VerifierAction::Measure {
            verifier: named,
            domain,
            cases,
            repeats,
            confidence,
            max_false_positive,
            min_accepted,
            ttl_days,
        } => {
            let records = match (named, domain) {
                (Some(named), _) => vec![resolve(&store, &named).await?],
                (None, Some(domain)) => measurable(&store, &domain).await?,
                (None, None) => bail!("name a verifier or a --domain"),
            };
            let cases = read_cases(&cases)?;
            let policy = TrustPolicy {
                repeats,
                confidence,
                max_false_positive,
                min_accepted,
                ttl: TimeDelta::days(ttl_days),
            };
            let mut verdicts: BTreeMap<&'static str, u32> = BTreeMap::new();
            for record in &records {
                check_anchors(&store, record, &cases).await?;
                let m = measure(&CommandVerifier, record, &cases, &policy, Utc::now()).await?;
                *verdicts.entry(verdict_kind(&m.verdict)).or_default() += 1;
                report(&store, record, &m).await?;
            }
            if records.len() > 1 {
                let counts: Vec<String> =
                    verdicts.iter().map(|(k, n)| format!("{n} {k}")).collect();
                println!(
                    "{} verifier(s) measured: {}",
                    records.len(),
                    counts.join(", ")
                );
            }
        }
        VerifierAction::Challenge { domain, cases } => {
            let cases = read_cases(&cases)?;
            let policy = TrustPolicy::default();
            let mut challenged = 0;
            for record in verifier::list(&store).await? {
                let trusted = verifier::state_of(&store, &record).await? == TrustState::Trusted;
                if record.domain != domain
                    || record.origin != VerifierOrigin::Synthesized
                    || !trusted
                {
                    continue;
                }
                check_anchors(&store, &record, &cases).await?;
                let m = challenge(&CommandVerifier, &record, &cases, &policy, Utc::now()).await?;
                report(&store, &record, &m).await?;
                challenged += 1;
            }
            println!("{challenged} trusted synthesized verifier(s) in {domain} challenged");
        }
        VerifierAction::Name {
            corpus,
            domain,
            out,
        } => {
            let text =
                std::fs::read_to_string(&corpus).with_context(|| format!("reading {corpus}"))?;
            let mut tasks: Vec<serde_json::Value> = serde_json::from_str(&text)?;
            let granting = crate::verifier_name::granting(&store, &domain, Utc::now()).await?;
            let named = crate::verifier_name::name_tasks(&mut tasks, &granting);
            std::fs::write(&out, serde_json::to_string_pretty(&tasks)?)
                .with_context(|| format!("writing {out}"))?;
            println!(
                "named {named} of {} task(s) after the {domain} verifiers that grant reward: {out}",
                tasks.len()
            );
        }
        VerifierAction::List { domain } => {
            let now = Utc::now();
            for record in verifier::list(&store).await? {
                if domain.as_deref().is_some_and(|d| d != record.domain) {
                    continue;
                }
                let state = verifier::state_of(&store, &record).await?;
                let grants = verifier::grants(&store, &record, now).await?;
                println!(
                    "{}  {:<11} {:<11} {:<9} {} {}{}",
                    record.id,
                    record.origin.as_str(),
                    state.as_str(),
                    record.tier.as_str(),
                    record.domain,
                    record.task.as_deref().unwrap_or("*"),
                    if grants { "  grants reward" } else { "" }
                );
            }
        }
        VerifierAction::Show { verifier: named } => {
            let record = resolve(&store, &named).await?;
            println!("{}", serde_json::to_string_pretty(&record)?);
            println!(
                "state: {}",
                verifier::state_of(&store, &record).await?.as_str()
            );
            for t in verifier::history(&store, &record.id).await? {
                println!("  {} {} -> {}", t.at, t.from.as_str(), t.to.as_str());
            }
            for m in verifier::measurements(&store, &record.id).await? {
                println!("  measured {}: {}", m.at, summary(&m));
            }
        }
        VerifierAction::Quarantine {
            verifier: named,
            note,
        } => {
            moved(&store, &named, TrustState::Quarantined, note).await?;
        }
        VerifierAction::Revoke {
            verifier: named,
            note,
        } => {
            moved(&store, &named, TrustState::Revoked, note).await?;
        }
    }
    Ok(())
}

/// The synthesized verifiers of `domain` a measurement can still move:
/// the proposed and the trusted.
async fn measurable(store: &Store, domain: &str) -> anyhow::Result<Vec<VerifierRecord>> {
    let mut out = Vec::new();
    for record in verifier::list(store).await? {
        if record.domain != domain || record.origin != VerifierOrigin::Synthesized {
            continue;
        }
        let state = verifier::state_of(store, &record).await?;
        if matches!(state, TrustState::Proposed | TrustState::Trusted) {
            out.push(record);
        }
    }
    Ok(out)
}

/// A batch of proposals: `{domain, task, tier, spec, by}` each, synthesized.
fn batch_records(batch: &serde_json::Value) -> anyhow::Result<Vec<VerifierRecord>> {
    let items = batch
        .as_array()
        .ok_or_else(|| anyhow!("a batch is a JSON array of proposals"))?;
    let now = Utc::now();
    items
        .iter()
        .map(|item| {
            let domain = item["domain"]
                .as_str()
                .ok_or_else(|| anyhow!("a proposal has no domain"))?;
            let tier: VerifierTier = item["tier"].as_str().unwrap_or("reducible").parse()?;
            if item["spec"].is_null() {
                bail!("a proposal for {domain} has no spec");
            }
            let mut record = VerifierRecord::new(
                domain,
                item["task"].as_str().map(str::to_string),
                tier,
                VerifierOrigin::Synthesized,
                item["spec"].clone(),
                now,
            );
            record.proposed_by = item["by"].as_str().map(str::to_string);
            Ok(record)
        })
        .collect()
}

fn verdict_kind(v: &TrustVerdict) -> &'static str {
    match v {
        TrustVerdict::Sound => "sound",
        TrustVerdict::Flaky { .. } => "flaky",
        TrustVerdict::Shortcut { .. } => "shortcut",
        TrustVerdict::Unmeasured { .. } => "unmeasured",
        TrustVerdict::FalsePositives { .. } => "over the bound",
        TrustVerdict::TooStrict { .. } => "too strict",
    }
}

async fn moved(
    store: &Store,
    named: &str,
    to: TrustState,
    note: Option<String>,
) -> anyhow::Result<()> {
    let record = resolve(store, named).await?;
    let moved = verifier::transition(store, &record.id, to, note).await?;
    println!("{}: {}", record.id, describe(&moved));
    Ok(())
}

/// A move, and the experts it archived.
fn describe(moved: &verifier::VerifierMove) -> String {
    let t = &moved.transition;
    let mut line = format!("{} -> {}", t.from.as_str(), t.to.as_str());
    if !moved.archived.is_empty() {
        let ids: Vec<&str> = moved.archived.iter().map(|e| e.as_str()).collect();
        line.push_str(&format!(
            "; archived {} expert(s) trained under it: {}",
            ids.len(),
            ids.join(", ")
        ));
    }
    line
}

/// A verifier by its address or a unique prefix of it.
async fn resolve(store: &Store, named: &str) -> anyhow::Result<VerifierRecord> {
    let prefix = if named.starts_with("verifier:") {
        named.to_string()
    } else {
        format!("verifier:{named}")
    };
    let mut matches: Vec<VerifierRecord> = verifier::list(store)
        .await?
        .into_iter()
        .filter(|r| r.id.as_str().starts_with(&prefix))
        .collect();
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => bail!("no verifier matches {named}"),
        n => bail!("{n} verifiers match {named}: give more of the address"),
    }
}

/// Every label's anchor must be something the loop did not synthesize: a
/// verifier named as one must be an authored verifier in the namespace, and
/// not the one being measured.
async fn check_anchors(
    store: &Store,
    record: &VerifierRecord,
    cases: &[Case],
) -> anyhow::Result<()> {
    let named: BTreeSet<&str> = cases
        .iter()
        .filter_map(|c| match &c.anchor {
            Anchor::Verifier { id } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    for id in named {
        if id == record.id.as_str() {
            bail!("{id} cannot anchor its own measurement");
        }
        match verifier::get(store, &VerifierId::new(id)).await? {
            Some(anchor) if anchor.origin == VerifierOrigin::Authored => {}
            Some(_) => bail!("{id} is synthesized: a label can never come from one"),
            None => bail!("{id} is not in the namespace: register the authored verifier first"),
        }
    }
    Ok(())
}

async fn report(
    store: &Store,
    record: &VerifierRecord,
    m: &TrustMeasurement,
) -> anyhow::Result<()> {
    let moved = verifier::record_measurement(store, record, m).await?;
    println!("{}: {}", record.id, summary(m));
    if let Some(moved) = moved {
        println!("  {}", describe(&moved));
    }
    Ok(())
}

fn summary(m: &TrustMeasurement) -> String {
    let verdict = match &m.verdict {
        TrustVerdict::Sound => "sound".to_string(),
        TrustVerdict::Flaky { cases } => format!("flaky on {}", cases.join(", ")),
        TrustVerdict::Shortcut { cases } => format!("shortcut on {}", cases.join(", ")),
        TrustVerdict::Unmeasured { reason } => format!("unmeasured: {reason}"),
        TrustVerdict::FalsePositives { upper, max } => {
            format!("false-positive bound {upper:.3} over {max:.3}")
        }
        TrustVerdict::TooStrict { accepted, min } => {
            format!("accepted {accepted:.2} of the known-good, under {min:.2}")
        }
    };
    format!(
        "{verdict} (good {}/{}, bad passed {}/{}, bound {:.3} at {:.0}%, {} impossible, {} adversarial, {} run(s) each)",
        m.good_passed,
        m.good,
        m.bad_passed,
        m.bad,
        m.false_positive_upper,
        m.confidence * 100.0,
        m.impossible,
        m.adversarial,
        m.repeats
    )
}

fn read_json(arg: &str) -> anyhow::Result<serde_json::Value> {
    let text = match arg.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
        None => arg.to_string(),
    };
    Ok(serde_json::from_str(&text)?)
}

fn read_cases(path: &str) -> anyhow::Result<Vec<Case>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    Ok(serde_json::from_str(&text)?)
}

/// Runs of the authored verifier on each completion: it must agree with
/// itself to be an anchor.
const ANCHOR_RUNS: u32 = 3;

/// Which of a corpus's tasks to build cases for.
#[derive(Clone, Copy, Default)]
struct Only<'a> {
    task: Option<&'a str>,
    skill: Option<&'a str>,
}

/// The cases for `tasks` (a corpus), labeling the completions `given` for
/// them. A task with no `skill` is in the domain `stem`.
async fn build_cases(
    store: &Store,
    tasks: &[serde_json::Value],
    stem: &str,
    only: Only<'_>,
    given: &[serde_json::Value],
) -> anyhow::Result<Vec<Case>> {
    let mut cases = Vec::new();
    for task in tasks {
        let id = task["id"]
            .as_str()
            .ok_or_else(|| anyhow!("a task has no id"))?;
        let skill = task["skill"].as_str().unwrap_or(stem);
        if only.task.is_some_and(|t| t != id) || only.skill.is_some_and(|s| s != skill) {
            continue;
        }
        let impossible = task["impossible"].as_bool().unwrap_or(false);
        // A completion marked deliberate was built to be wrong (a mutant, a
        // forgery), so one the authored verifier fails is adversarial.
        let for_task: Vec<(&str, bool)> = given
            .iter()
            .filter(|c| c["task"].as_str() == Some(id))
            .filter_map(|c| {
                let deliberate = c["deliberate"].as_bool().unwrap_or(false);
                c["completion"].as_str().map(|s| (s, deliberate))
            })
            .collect();
        if impossible {
            for (i, (completion, _)) in for_task.iter().enumerate() {
                cases.push(Case {
                    id: format!("{id}#{i}"),
                    task: id.to_string(),
                    completion: completion.to_string(),
                    label: Label::Impossible,
                    anchor: Anchor::Constructed,
                });
            }
            continue;
        }
        if let Some(reference) = task["completion"].as_str() {
            cases.push(Case {
                id: format!("{id}#ref"),
                task: id.to_string(),
                completion: reference.to_string(),
                label: Label::Good,
                anchor: Anchor::Reference,
            });
        }
        if for_task.is_empty() {
            continue;
        }
        let spec = task["verify"].clone();
        if spec.is_null() {
            bail!("{id} has no verify spec to label its completions with");
        }
        let anchor = authored(store, skill, id, spec.clone()).await?;
        for (i, (completion, deliberate)) in for_task.iter().enumerate() {
            let passed = label_with(&spec, id, completion).await?;
            let Some(passed) = passed else {
                bail!("{id}'s authored verifier disagreed with itself on completion {i}");
            };
            cases.push(Case {
                id: format!("{id}#{i}"),
                task: id.to_string(),
                completion: completion.to_string(),
                label: match (passed, *deliberate) {
                    (true, _) => Label::Good,
                    (false, true) => Label::Adversarial,
                    (false, false) => Label::Bad,
                },
                anchor: Anchor::Verifier { id: anchor.clone() },
            });
        }
    }
    Ok(cases)
}

/// Register a task's authored spec in the namespace, and return its address.
async fn authored(
    store: &Store,
    domain: &str,
    task: &str,
    spec: serde_json::Value,
) -> anyhow::Result<VerifierId> {
    let id = verifier_address(domain, Some(task), VerifierTier::Reducible, &spec);
    let record = VerifierRecord::new(
        domain,
        Some(task.to_string()),
        VerifierTier::Reducible,
        VerifierOrigin::Authored,
        spec,
        Utc::now(),
    );
    let kept = verifier::propose(store, &record).await?;
    if kept.origin != VerifierOrigin::Authored {
        bail!("{id} is already in the namespace as synthesized, so it cannot anchor");
    }
    Ok(id)
}

/// The authored verdict on a completion, or `None` when its runs disagree.
async fn label_with(
    spec: &serde_json::Value,
    task: &str,
    completion: &str,
) -> anyhow::Result<Option<bool>> {
    let mut runs = Vec::new();
    for step_idx in 0..ANCHOR_RUNS {
        let req = VerifyRequest {
            run_id: RunId::new(format!("anchor:{task}")),
            step_idx,
            dimension: "anchor".into(),
            artifact: serde_json::json!({ "task": task, "completion": completion, "verify": spec }),
        };
        runs.push(CommandVerifier.verify(&req).await?.passed);
    }
    Ok(runs.iter().all(|&r| r == runs[0]).then_some(runs[0]))
}

/// The answers built to be wrong in a `verifier cases` completions file at
/// `path`: the entries marked `deliberate`, as `(task, completion)`.
#[cfg(any(feature = "models", test))]
pub(crate) fn deliberate_answers(path: &str) -> anyhow::Result<Vec<(String, String)>> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?,
    )
    .with_context(|| format!("{path} is not a JSON array of completions"))?;
    Ok(entries
        .iter()
        .filter(|e| e["deliberate"].as_bool().unwrap_or(false))
        .filter_map(|e| {
            Some((
                e["task"].as_str()?.to_string(),
                e["completion"].as_str()?.to_string(),
            ))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_deliberate_completions_are_read_as_built_to_be_wrong() {
        let path = std::env::temp_dir().join(format!("deliberate-{}.json", std::process::id()));
        std::fs::write(
            &path,
            serde_json::json!([
                { "task": "swap", "completion": "mutant", "deliberate": true },
                { "task": "swap", "completion": "policy answer" },
                { "task": "pad", "completion": "forgery", "deliberate": true },
                { "task": "pad", "deliberate": true },
            ])
            .to_string(),
        )
        .unwrap();
        let got = deliberate_answers(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(
            got,
            [
                ("swap".to_string(), "mutant".to_string()),
                ("pad".to_string(), "forgery".to_string()),
            ]
        );
    }

    fn corpus() -> Vec<serde_json::Value> {
        serde_json::from_value(serde_json::json!([
            {
                "id": "s/rev",
                "skill": "strings",
                "completion": "def rev(s): return s[::-1]",
                "verify": { "contains_all": ["s[::-1]"] }
            },
            {
                "id": "s/never",
                "skill": "strings",
                "impossible": true,
                "verify": { "contains_all": ["x"] }
            },
            { "id": "n/add", "skill": "numbers", "verify": { "contains_all": ["+"] } }
        ]))
        .unwrap()
    }

    fn given() -> Vec<serde_json::Value> {
        serde_json::from_value(serde_json::json!([
            { "task": "s/rev", "completion": "def rev(s): return s" },
            { "task": "s/rev", "completion": "def rev(s): return ''.join(s[::-1])" },
            { "task": "s/never", "completion": "def never(): return 1" },
            { "task": "n/add", "completion": "a - b" }
        ]))
        .unwrap()
    }

    #[test]
    fn a_batch_is_proposed_as_synthesized_whatever_it_claims() {
        let batch = serde_json::json!([
            { "domain": "strings", "task": "s/rev", "spec": { "contains_all": ["x"] }, "by": "m" },
            { "domain": "strings", "tier": "partial", "spec": { "contains_all": ["y"] }, "origin": "authored" }
        ]);
        let records = batch_records(&batch).unwrap();
        assert_eq!(records.len(), 2);
        assert!(records
            .iter()
            .all(|r| r.origin == VerifierOrigin::Synthesized));
        assert_eq!(records[0].task.as_deref(), Some("s/rev"));
        assert_eq!(records[0].proposed_by.as_deref(), Some("m"));
        assert_eq!(records[1].tier, VerifierTier::Partial);
        assert!(batch_records(&serde_json::json!([{ "domain": "d" }])).is_err());
        assert!(batch_records(
            &serde_json::json!([{ "domain": "d", "tier": "derived", "spec": {} }])
        )
        .is_err());
        assert!(batch_records(&serde_json::json!({})).is_err());
    }

    #[tokio::test]
    async fn a_domain_measurement_takes_the_proposed_and_trusted_synthesized_ones() {
        let store = Store::connect_memory(4).await.unwrap();
        let batch = serde_json::json!([
            { "domain": "strings", "spec": { "contains_all": ["a"] } },
            { "domain": "strings", "spec": { "contains_all": ["b"] } },
            { "domain": "lists", "spec": { "contains_all": ["c"] } }
        ]);
        let mut ids = Vec::new();
        for record in batch_records(&batch).unwrap() {
            ids.push(verifier::propose(&store, &record).await.unwrap().id);
        }
        let authored = VerifierRecord::new(
            "strings",
            None,
            VerifierTier::Reducible,
            VerifierOrigin::Authored,
            serde_json::json!({ "contains_all": ["z"] }),
            Utc::now(),
        );
        verifier::propose(&store, &authored).await.unwrap();
        verifier::transition(&store, &ids[1], TrustState::Revoked, None)
            .await
            .unwrap();
        let picked: Vec<VerifierId> = measurable(&store, "strings")
            .await
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(picked, vec![ids[0].clone()]);
    }

    #[tokio::test]
    async fn cases_carry_the_reference_and_the_authored_verdicts() {
        let store = Store::connect_memory(4).await.unwrap();
        let cases = build_cases(
            &store,
            &corpus(),
            "c",
            Only {
                skill: Some("strings"),
                ..Only::default()
            },
            &given(),
        )
        .await
        .unwrap();
        let labels: Vec<(&str, Label)> = cases.iter().map(|c| (c.id.as_str(), c.label)).collect();
        assert_eq!(
            labels,
            vec![
                ("s/rev#ref", Label::Good),
                ("s/rev#0", Label::Bad),
                ("s/rev#1", Label::Good),
                ("s/never#0", Label::Impossible),
            ]
        );
        // The authored verifier that labeled them is in the namespace, as authored.
        let Anchor::Verifier { id } = &cases[1].anchor else {
            panic!("{:?}", cases[1].anchor);
        };
        let anchor = verifier::get(&store, id).await.unwrap().unwrap();
        assert_eq!(anchor.origin, VerifierOrigin::Authored);
        assert_eq!(anchor.task.as_deref(), Some("s/rev"));
    }

    #[tokio::test]
    async fn a_deliberate_artifact_the_authored_verifier_fails_is_adversarial() {
        let store = Store::connect_memory(4).await.unwrap();
        let given: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([
            { "task": "s/rev", "completion": "def rev(s): return s", "deliberate": true },
            { "task": "s/rev", "completion": "def rev(s): return s[::-1]", "deliberate": true },
            { "task": "s/rev", "completion": "def rev(s): return None" }
        ]))
        .unwrap();
        let only = Only {
            task: Some("s/rev"),
            ..Only::default()
        };
        let cases = build_cases(&store, &corpus(), "c", only, &given)
            .await
            .unwrap();
        let labels: Vec<Label> = cases.iter().map(|c| c.label).collect();
        // The reference, a failed mutant, an equivalent one, a plain failure.
        assert_eq!(
            labels,
            vec![Label::Good, Label::Adversarial, Label::Good, Label::Bad]
        );
    }

    #[tokio::test]
    async fn a_label_never_comes_from_a_synthesized_verifier_or_from_the_one_measured() {
        let store = Store::connect_memory(4).await.unwrap();
        let cases = build_cases(&store, &corpus(), "c", Only::default(), &given())
            .await
            .unwrap();
        let spec = serde_json::json!({ "contains_all": ["[::-1]"] });
        let proposed = VerifierRecord::new(
            "strings",
            None,
            VerifierTier::Reducible,
            VerifierOrigin::Synthesized,
            spec,
            Utc::now(),
        );
        let proposed = verifier::propose(&store, &proposed).await.unwrap();
        check_anchors(&store, &proposed, &cases).await.unwrap();

        let mut laundered = cases.clone();
        laundered[1].anchor = Anchor::Verifier {
            id: proposed.id.clone(),
        };
        let other = VerifierRecord::new(
            "strings",
            None,
            VerifierTier::Partial,
            VerifierOrigin::Synthesized,
            serde_json::json!({ "contains_all": ["rev"] }),
            Utc::now(),
        );
        let other = verifier::propose(&store, &other).await.unwrap();
        let refused = check_anchors(&store, &other, &laundered).await.unwrap_err();
        assert!(refused.to_string().contains("synthesized"), "{refused}");
        let own = check_anchors(&store, &proposed, &laundered)
            .await
            .unwrap_err();
        assert!(own.to_string().contains("its own"), "{own}");
    }

    #[tokio::test]
    async fn a_proposal_is_trusted_by_measurement_and_quarantined_by_a_challenge() {
        let store = Store::connect_memory(4).await.unwrap();
        let tasks: Vec<serde_json::Value> = serde_json::from_value(serde_json::json!([{
            "id": "s/rev",
            "skill": "strings",
            "completion": "return s[::-1]",
            "verify": { "contains_all": ["s[::-1]"] }
        }]))
        .unwrap();
        // Thirty wrong answers the authored verifier fails, and one it passes.
        let mut given: Vec<serde_json::Value> = (0..30)
            .map(|i| serde_json::json!({ "task": "s/rev", "completion": format!("return {i}") }))
            .collect();
        given.push(serde_json::json!({ "task": "s/rev", "completion": "return s[::-1] or s" }));
        let cases = build_cases(&store, &tasks, "c", Only::default(), &given)
            .await
            .unwrap();
        let proposed = verifier::propose(
            &store,
            &VerifierRecord::new(
                "strings",
                Some("s/rev".into()),
                VerifierTier::Reducible,
                VerifierOrigin::Synthesized,
                serde_json::json!({ "contains_all": ["[::-1]"] }),
                Utc::now(),
            ),
        )
        .await
        .unwrap();
        check_anchors(&store, &proposed, &cases).await.unwrap();
        let policy = TrustPolicy::default();
        let m = measure(&CommandVerifier, &proposed, &cases, &policy, Utc::now())
            .await
            .unwrap();
        assert_eq!(m.verdict, TrustVerdict::Sound, "{m:?}");
        report(&store, &proposed, &m).await.unwrap();
        assert!(verifier::grants(&store, &proposed, Utc::now())
            .await
            .unwrap());

        // A deliberately wrong artifact the authored verifier fails and the
        // synthesized one passes: the challenge quarantines it.
        let mut wrong = cases.clone();
        wrong.push(Case {
            id: "s/rev#wrong".into(),
            task: "s/rev".into(),
            completion: "return s[::-1][1:]".into(),
            label: Label::Bad,
            anchor: Anchor::Constructed,
        });
        let caught = challenge(&CommandVerifier, &proposed, &wrong, &policy, Utc::now())
            .await
            .unwrap();
        assert!(
            matches!(caught.verdict, TrustVerdict::Shortcut { .. }),
            "{caught:?}"
        );
        report(&store, &proposed, &caught).await.unwrap();
        assert_eq!(
            verifier::state_of(&store, &proposed).await.unwrap(),
            TrustState::Quarantined
        );
        assert!(!verifier::grants(&store, &proposed, Utc::now())
            .await
            .unwrap());
    }
}

//! `antumbra behave`: train a user's accepted behaviors into their private
//! standing expert. The work is `antumbra_serve::behave`; this
//! prints what it did and why. `--import` records behaviors from a file
//! instead, as the `record_behavior` tool would: accepted, or with
//! `--propose` proposed, for the user to accept.

use std::hash::{Hash, Hasher};

use clap::Args;
use serde_json::Value;

use antumbra_core::behavior::{normalize_scope, Spec, Status};
use antumbra_core::{MemoryId, TenantId, UserId};
use antumbra_store::repo::behavior as store_behavior;

#[derive(Args, Debug)]
pub struct BehaveArgs {
    #[arg(long)]
    pub tenant: String,
    #[arg(long)]
    pub user: String,
    /// A repository (`host/org/name`); omit for the behaviors that apply
    /// everywhere.
    #[arg(long)]
    pub scope: Option<String>,
    /// Record the behaviors in this JSON file, accepted, and stop: an array
    /// of `{rule, must, must_not, examples: [{task, answer}], violations,
    /// scope}`. Each is checked as `record_behavior` checks it. Importing the
    /// same rule again replaces it.
    #[arg(long)]
    pub import: Option<String>,
    /// With `--import`, record them proposed: for the user to accept, as a
    /// behavior drawn from older memories is, rather than one they stated.
    #[arg(long, default_value_t = false)]
    pub propose: bool,
    /// Epochs over the examples and the replay: three taught four behaviors
    /// when this procedure was first run by hand.
    #[arg(long, default_value_t = 3)]
    pub rounds: usize,
    /// The longest answer generated: the base model's replay answers among
    /// them. One cut short would teach the expert to stop mid-answer, since
    /// training ends every answer it is given.
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: usize,
    /// The learning rate. Above 1.5e-4, a set of more than a few behaviors
    /// pulls the expert off the base model's answers everywhere else.
    #[arg(long, default_value_t = 1.5e-4)]
    pub lr: f64,
}

/// One behavior in an `--import` file.
struct Imported {
    spec: Spec,
    scope: Option<String>,
}

/// An `--import` file's rows: each a spec, with the scope beside it.
fn rows(text: &str) -> anyhow::Result<Vec<Imported>> {
    let parsed: Vec<Value> = serde_json::from_str(text)?;
    parsed
        .into_iter()
        .map(|v| {
            let scope = v.get("scope").and_then(Value::as_str).map(str::to_string);
            Ok(Imported {
                spec: serde_json::from_value(v)?,
                scope,
            })
        })
        .collect()
}

/// The id an imported behavior is stored under: the same user, scope and
/// rule land on the same record, so importing a file again replaces it. The
/// prefix keeps the spelling the records already stored were written with,
/// or an import would add a second copy of each instead of replacing it.
fn import_id(user: &str, scope: &str, rule: &str) -> MemoryId {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (user, scope, rule.trim()).hash(&mut h);
    MemoryId::new(format!("memory:behaviour-{:x}", h.finish()))
}

async fn import(url: &str, a: &BehaveArgs, path: &str) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(path)?;
    let rows = rows(&text)?;
    let store = crate::connect(url).await?;
    let embedder = crate::make_embedder()?;
    let (tenant, user) = (
        TenantId::new(a.tenant.as_str()),
        UserId::new(a.user.as_str()),
    );
    let host = antumbra_core::this_host();
    let status = if a.propose {
        Status::Proposed
    } else {
        Status::Accepted
    };
    let mut refused = 0;
    for row in rows {
        let problems = row.spec.problems();
        if !problems.is_empty() {
            refused += 1;
            println!("refused: {}", row.spec.rule);
            for p in problems {
                println!("  {p}");
            }
            continue;
        }
        let scope = normalize_scope(row.scope.as_deref().or(a.scope.as_deref()));
        let id = import_id(user.as_str(), &scope, &row.spec.rule);
        let embedding = embedder
            .embed(&antumbra_core::behavior::content(&row.spec))
            .await?;
        store_behavior::record(
            &store,
            &tenant,
            &user,
            &host,
            id.clone(),
            &row.spec,
            status,
            &scope,
            None,
            embedding,
        )
        .await?;
        println!(
            "{} {} ({scope}): {}",
            status.as_str(),
            id.as_str(),
            row.spec.rule
        );
    }
    if refused > 0 {
        anyhow::bail!("{refused} behavior(s) refused; nothing was written for them");
    }
    Ok(())
}

pub async fn run(url: &str, a: BehaveArgs) -> anyhow::Result<()> {
    match a.import.clone() {
        Some(path) => import(url, &a, &path).await,
        None => train(url, a).await,
    }
}

#[cfg(feature = "models")]
async fn train(url: &str, a: BehaveArgs) -> anyhow::Result<()> {
    let store = crate::connect(url).await?;
    let embedder = crate::make_embedder()?;
    let scope = normalize_scope(a.scope.as_deref());
    let cfg = antumbra_serve::RaftConfig {
        rounds: a.rounds,
        max_new_tokens: a.max_new_tokens,
        learning_rate: a.lr,
        ..antumbra_serve::RaftConfig::default()
    };
    let report = antumbra_serve::behave::train_behaviors(
        &store,
        embedder.as_ref(),
        &TenantId::new(a.tenant.as_str()),
        &UserId::new(a.user.as_str()),
        &scope,
        &cfg,
    )
    .await?;
    let Some(r) = report else {
        println!("no accepted behaviors in scope {scope}: nothing to teach");
        return Ok(());
    };
    println!(
        "{} behavior(s) in scope {scope}, with {} base answers replayed",
        r.behaviors.len(),
        r.replay
    );
    for b in &r.verdict.behaviors {
        println!(
            "  {:<40} held out: base {:.2} -> expert {:.2} {}",
            b.id,
            b.base,
            b.expert,
            if b.admitted { "learned" } else { "missed" }
        );
    }
    for c in &r.verdict.controls {
        println!(
            "  {:<40} base {:.2} -> expert {:.2}",
            c.family, c.base, c.expert
        );
    }
    match &r.expert {
        Some((id, uri)) => println!(
            "admitted: private expert {} ({uri}), holding the {} of {} behavior(s) it learned",
            id.as_str(),
            r.verdict.learned().len(),
            r.behaviors.len()
        ),
        None => {
            println!("not admitted, nothing minted:");
            for reason in &r.verdict.reasons {
                println!("  {reason}");
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "models"))]
async fn train(_url: &str, _a: BehaveArgs) -> anyhow::Result<()> {
    anyhow::bail!("training behaviors requires building with --features models (candle + a GPU)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_imported_rule_keeps_its_id_and_a_scope_or_rule_changes_it() {
        let a = import_id("user:a", "everywhere", "Use --save-exact.");
        assert_eq!(a, import_id("user:a", "everywhere", " Use --save-exact.\n"));
        assert_ne!(
            a,
            import_id("user:a", "github.com/a/b", "Use --save-exact.")
        );
        assert_ne!(a, import_id("user:a", "everywhere", "Pin versions."));
        assert_ne!(a, import_id("user:b", "everywhere", "Use --save-exact."));
    }

    #[test]
    fn an_import_row_carries_a_spec_and_an_optional_scope() {
        let rows = rows(
            r#"[{"rule":"r","must":["x"],"examples":[{"task":"t","answer":"x"}],"violations":["y"],"scope":"github.com/a/b"},
                {"rule":"s","must":["x"],"examples":[],"violations":[]}]"#,
        )
        .unwrap();
        assert_eq!(rows[0].scope.as_deref(), Some("github.com/a/b"));
        assert_eq!(rows[0].spec.examples.len(), 1);
        assert!(rows[1].scope.is_none());
        assert!(!rows[1].spec.problems().is_empty());
    }
}

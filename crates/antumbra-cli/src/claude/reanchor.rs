//! `antumbra claude reanchor`: tell the surface which branches have merged, so
//! the memories learned on them move to the branch they merged into (the
//! `record_merges` tool, called once with every merge). The GitHub webhook does this by itself, but only for a
//! server GitHub can reach, which a server on a private network is not. The
//! hooks anchor every memory to the branch a session is on, so without this a
//! merged feature branch's memories read `other_branch` from `main` for good,
//! and recall ranks them below memories that were never anchored.
//!
//! The merges come from GitHub (`gh pr list --state merged`), not from git: a
//! squash merge leaves the branch's commits out of the base branch's history,
//! so git cannot tell a squash-merged branch from an abandoned one. Safe to run
//! again: a merge whose memories have already moved moves nothing.

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

use super::conventions::Call;

/// The `gh pr list --json` fields [`merges_in`] reads.
pub const GH_FIELDS: &str =
    "number,headRefName,baseRefName,mergeCommit,mergedAt,isCrossRepository,commits";

/// One merged pull request, as the surface is told about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merged {
    pub number: u64,
    pub head: String,
    pub base: String,
    pub commit: String,
    pub merged_at: String,
    /// The commits the branch carried when it merged.
    pub commits: Vec<String>,
}

/// The merges in `gh pr list --json` output, oldest first. In that order a
/// stacked branch lands where it should: merged into its parent branch first,
/// it is carried on when the parent merges. A pull request from a fork is left
/// out, because its branch lives in the fork, and a memory anchored to a branch
/// of that name here is about a different branch.
pub fn merges_in(listed: &str) -> anyhow::Result<Vec<Merged>> {
    let prs: Vec<Value> = serde_json::from_str(listed)
        .map_err(|e| anyhow::anyhow!("gh pr list answered something unreadable: {e}"))?;
    let text = |v: &Value| v.as_str().map(str::to_string);
    let mut merges: Vec<Merged> = prs
        .iter()
        .filter(|pr| !pr["isCrossRepository"].as_bool().unwrap_or(false))
        .filter_map(|pr| {
            Some(Merged {
                number: pr["number"].as_u64()?,
                head: text(&pr["headRefName"])?,
                base: text(&pr["baseRefName"])?,
                commit: text(&pr["mergeCommit"]["oid"])?,
                merged_at: text(&pr["mergedAt"])?,
                commits: pr["commits"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|c| text(&c["oid"]))
                    .collect(),
            })
        })
        .collect();
    // RFC 3339 in UTC, as gh writes it, sorts as text.
    merges.sort_by(|a, b| a.merged_at.cmp(&b.merged_at));
    Ok(merges)
}

/// The merges at or after `cutoff`. A merge whose time does not read as RFC
/// 3339 is kept: reporting a merge again moves nothing, and dropping one would
/// leave its memories where they are.
pub fn merged_since(merges: Vec<Merged>, cutoff: DateTime<Utc>) -> Vec<Merged> {
    merges
        .into_iter()
        .filter(|m| DateTime::parse_from_rfc3339(&m.merged_at).map_or(true, |at| at >= cutoff))
        .collect()
}

/// Report the merges to the surface in one call, in order, and say what each
/// moved. One call, because the surface reads every memory's anchor to find the
/// ones a report moves, and it can do that once for all of them. With
/// `dry_run`, say what would be reported and call nothing.
pub fn report(
    call: Call<'_>,
    repo: &str,
    merges: &[Merged],
    dry_run: bool,
) -> anyhow::Result<Vec<String>> {
    let which = |m: &Merged| format!("#{} {} -> {}", m.number, m.head, m.base);
    if dry_run {
        return Ok(merges
            .iter()
            .map(|m| format!("would report {}", which(m)))
            .collect());
    }
    let mut said = Vec::new();
    let mut total = 0;
    if !merges.is_empty() {
        let reports: Vec<Value> = merges
            .iter()
            .map(|m| {
                json!({
                    "head_branch": m.head,
                    "base_branch": m.base,
                    "merge_commit": m.commit,
                    "merged_at": m.merged_at,
                    "commits": m.commits,
                })
            })
            .collect();
        let out = call("record_merges", json!({ "repo": repo, "merges": reports }))?;
        let outcomes = out["merges"].as_array().cloned().unwrap_or_default();
        if outcomes.len() != merges.len() {
            anyhow::bail!(
                "record_merges answered for {} merge(s) of the {} reported",
                outcomes.len(),
                merges.len()
            );
        }
        for (m, outcome) in merges.iter().zip(&outcomes) {
            let moved = outcome["moved"].as_u64().unwrap_or(0);
            let not_writable = outcome["not_writable"].as_u64().unwrap_or(0);
            total += moved;
            match (moved, not_writable) {
                (0, 0) => {}
                (_, 0) => said.push(format!("{}: moved {moved}", which(m))),
                _ => said.push(format!(
                    "{}: moved {moved}, left {not_writable} shared with you read-only",
                    which(m)
                )),
            }
        }
    }
    said.push(format!(
        "{} merge(s) reported, {total} memory(ies) moved",
        merges.len()
    ));
    Ok(said)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    const LISTED: &str = r#"[
        {"number": 51, "headRefName": "fix/rerank-budget", "baseRefName": "main",
         "mergeCommit": {"oid": "4e1318f"}, "mergedAt": "2026-09-23T17:47:07Z",
         "isCrossRepository": false, "commits": [{"oid": "5df8993"}, {"oid": "229e6c4"}]},
        {"number": 50, "headRefName": "fix/rerank-batches", "baseRefName": "main",
         "mergeCommit": {"oid": "7d74311"}, "mergedAt": "2026-09-23T17:05:01Z",
         "isCrossRepository": false, "commits": [{"oid": "c167eb6"}]},
        {"number": 12, "headRefName": "main", "baseRefName": "main",
         "mergeCommit": {"oid": "aaaaaaa"}, "mergedAt": "2026-09-01T00:00:00Z",
         "isCrossRepository": true, "commits": []},
        {"number": 9, "headRefName": "old", "baseRefName": "main",
         "mergeCommit": null, "mergedAt": "2026-08-01T00:00:00Z",
         "isCrossRepository": false, "commits": []}
    ]"#;

    #[test]
    fn merges_are_read_oldest_first_without_forks_or_missing_commits() {
        let merges = merges_in(LISTED).unwrap();
        let numbers: Vec<u64> = merges.iter().map(|m| m.number).collect();
        assert_eq!(numbers, vec![50, 51]);
        assert_eq!(merges[1].commits, vec!["5df8993", "229e6c4"]);
        assert_eq!(merges[1].commit, "4e1318f");
    }

    #[test]
    fn a_window_keeps_the_recent_merges_and_any_it_cannot_date() {
        let mut merges = merges_in(LISTED).unwrap();
        let cutoff = DateTime::parse_from_rfc3339("2026-09-23T17:30:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let recent: Vec<u64> = merged_since(merges.clone(), cutoff)
            .iter()
            .map(|m| m.number)
            .collect();
        assert_eq!(recent, vec![51]);
        merges[0].merged_at = "yesterday".into();
        let kept: Vec<u64> = merged_since(merges, cutoff)
            .iter()
            .map(|m| m.number)
            .collect();
        assert_eq!(kept, vec![50, 51]);
    }

    #[test]
    fn unreadable_gh_output_is_an_error_that_says_so() {
        let err = merges_in("not json").unwrap_err().to_string();
        assert!(err.contains("gh pr list"), "{err}");
    }

    /// A stand-in surface that answers `record_merges` with `each` for every
    /// merge it was sent, and keeps what it was sent.
    fn surface(
        each: Value,
    ) -> (
        RefCell<Vec<Value>>,
        impl Fn(&str, Value) -> anyhow::Result<Value>,
    ) {
        let answer = move |tool: &str, args: &Value| {
            assert_eq!(tool, "record_merges");
            let n = args["merges"].as_array().map_or(0, Vec::len);
            json!({ "merges": vec![each.clone(); n] })
        };
        (RefCell::new(Vec::new()), move |tool: &str, args: Value| {
            Ok(answer(tool, &args))
        })
    }

    #[test]
    fn one_call_carries_every_merge_in_order_and_what_each_moved_is_said() {
        let (calls, answer) = surface(json!({ "moved": 2, "not_writable": 0 }));
        let call = |tool: &str, args: Value| {
            calls.borrow_mut().push(args.clone());
            answer(tool, args)
        };
        let merges = merges_in(LISTED).unwrap();
        let said = report(&call, "github.com/albedosehen/antumbra", &merges, false).unwrap();
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 1, "one call for every merge");
        assert_eq!(calls[0]["repo"], "github.com/albedosehen/antumbra");
        let sent = &calls[0]["merges"];
        assert_eq!(sent[0]["head_branch"], "fix/rerank-batches");
        assert_eq!(sent[1]["merged_at"], "2026-09-23T17:47:07Z");
        assert_eq!(sent[1]["commits"], json!(["5df8993", "229e6c4"]));
        assert_eq!(
            said,
            vec![
                "#50 fix/rerank-batches -> main: moved 2",
                "#51 fix/rerank-budget -> main: moved 2",
                "2 merge(s) reported, 4 memory(ies) moved",
            ]
        );
    }

    #[test]
    fn a_merge_that_moves_nothing_is_counted_but_not_listed() {
        let merges = merges_in(LISTED).unwrap();
        let (_, shared) = surface(json!({ "moved": 0, "not_writable": 1 }));
        let said = report(&shared, "r/o/n", &merges[..1], false).unwrap();
        assert_eq!(
            said,
            vec![
                "#50 fix/rerank-batches -> main: moved 0, left 1 shared with you read-only",
                "1 merge(s) reported, 0 memory(ies) moved",
            ]
        );
        let (_, quiet) = surface(json!({ "moved": 0, "not_writable": 0 }));
        let said = report(&quiet, "r/o/n", &merges[..1], false).unwrap();
        assert_eq!(said, vec!["1 merge(s) reported, 0 memory(ies) moved"]);
    }

    #[test]
    fn no_merges_calls_nothing() {
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            panic!("nothing to report, so nothing to call")
        };
        let said = report(&call, "r/o/n", &[], false).unwrap();
        assert_eq!(said, vec!["0 merge(s) reported, 0 memory(ies) moved"]);
    }

    /// An answer that does not account for every merge sent is an error, not a
    /// report that silently skips some.
    #[test]
    fn an_answer_short_of_the_merges_sent_is_an_error() {
        let call = |_: &str, _: Value| Ok(json!({ "merges": [{ "moved": 1 }] }));
        let merges = merges_in(LISTED).unwrap();
        let err = report(&call, "r/o/n", &merges, false).unwrap_err();
        assert!(err.to_string().contains("1 merge(s) of the 2"), "{err}");
    }

    #[test]
    fn a_dry_run_calls_nothing() {
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            panic!("a dry run must not call the surface")
        };
        let merges = merges_in(LISTED).unwrap();
        let said = report(&call, "r/o/n", &merges, true).unwrap();
        assert_eq!(
            said,
            vec![
                "would report #50 fix/rerank-batches -> main",
                "would report #51 fix/rerank-budget -> main",
            ]
        );
    }

    /// A surface too old to know the tool stops the run with its own words,
    /// rather than reporting nothing moved.
    #[test]
    fn a_refusal_stops_the_run() {
        let call = |_: &str, _: Value| -> anyhow::Result<Value> {
            anyhow::bail!("unknown tool: record_merges")
        };
        let merges = merges_in(LISTED).unwrap();
        let err = report(&call, "r/o/n", &merges, false).unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }
}

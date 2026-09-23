//! `antumbra claude reanchor`: tell the surface which branches have merged, so
//! the memories learned on them move to the branch they merged into (the
//! `record_merge` tool). The GitHub webhook does this by itself, but only for a
//! server GitHub can reach, which a server on a private network is not. The
//! hooks anchor every memory to the branch a session is on, so without this a
//! merged feature branch's memories read `other_branch` from `main` for good,
//! and recall ranks them below memories that were never anchored.
//!
//! The merges come from GitHub (`gh pr list --state merged`), not from git: a
//! squash merge leaves the branch's commits out of the base branch's history,
//! so git cannot tell a squash-merged branch from an abandoned one. Safe to run
//! again: a merge whose memories have already moved moves nothing.

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

/// Report each merge to the surface, in order, and say what moved. With
/// `dry_run`, say what would be reported and call nothing.
pub fn report(
    call: Call<'_>,
    repo: &str,
    merges: &[Merged],
    dry_run: bool,
) -> anyhow::Result<Vec<String>> {
    let mut said = Vec::new();
    let mut total = 0;
    for m in merges {
        let which = format!("#{} {} -> {}", m.number, m.head, m.base);
        if dry_run {
            said.push(format!("would report {which}"));
            continue;
        }
        let out = call(
            "record_merge",
            json!({
                "repo": repo,
                "head_branch": m.head,
                "base_branch": m.base,
                "merge_commit": m.commit,
                "merged_at": m.merged_at,
                "commits": m.commits,
            }),
        )?;
        let moved = out["moved"].as_u64().unwrap_or(0);
        let not_writable = out["not_writable"].as_u64().unwrap_or(0);
        total += moved;
        match (moved, not_writable) {
            (0, 0) => {}
            (_, 0) => said.push(format!("{which}: moved {moved}")),
            _ => said.push(format!(
                "{which}: moved {moved}, left {not_writable} shared with you read-only"
            )),
        }
    }
    if !dry_run {
        said.push(format!(
            "{} merge(s) reported, {total} memory(ies) moved",
            merges.len()
        ));
    }
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
    fn unreadable_gh_output_is_an_error_that_says_so() {
        let err = merges_in("not json").unwrap_err().to_string();
        assert!(err.contains("gh pr list"), "{err}");
    }

    #[test]
    fn each_merge_is_reported_in_order_and_what_moved_is_said() {
        let calls = RefCell::new(Vec::new());
        let call = |tool: &str, args: Value| {
            assert_eq!(tool, "record_merge");
            calls.borrow_mut().push(args);
            Ok(json!({ "moved": 2, "not_writable": 0 }))
        };
        let merges = merges_in(LISTED).unwrap();
        let said = report(&call, "github.com/albedosehen/antumbra", &merges, false).unwrap();
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["head_branch"], "fix/rerank-batches");
        assert_eq!(calls[1]["merged_at"], "2026-09-23T17:47:07Z");
        assert_eq!(calls[1]["commits"], json!(["5df8993", "229e6c4"]));
        assert_eq!(calls[1]["repo"], "github.com/albedosehen/antumbra");
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
        let call = |_: &str, _: Value| Ok(json!({ "moved": 0, "not_writable": 1 }));
        let merges = merges_in(LISTED).unwrap();
        let said = report(&call, "r/o/n", &merges[..1], false).unwrap();
        assert_eq!(
            said,
            vec![
                "#50 fix/rerank-batches -> main: moved 0, left 1 shared with you read-only",
                "1 merge(s) reported, 0 memory(ies) moved",
            ]
        );
        let quiet = |_: &str, _: Value| Ok(json!({ "moved": 0, "not_writable": 0 }));
        let said = report(&quiet, "r/o/n", &merges[..1], false).unwrap();
        assert_eq!(said, vec!["1 merge(s) reported, 0 memory(ies) moved"]);
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
            anyhow::bail!("unknown tool: record_merge")
        };
        let merges = merges_in(LISTED).unwrap();
        let err = report(&call, "r/o/n", &merges, false).unwrap_err();
        assert!(err.to_string().contains("unknown tool"));
    }
}

//! What an agent is told, at session start, about the session it is in.
//!
//! The doctor's report is for the user. This is for the agent, and says only
//! what changes what it should do: whether the project's instructions reached
//! it, which shell is the host's, and what it must not offer because it is gone.
//!
//! It is rendered from the compiled rules and never recalled, so it is the same
//! words every time, never a paraphrase. It is short because it has to be: a hook's
//! context is capped at 10,000 characters for everything the hook says, and this
//! shares that room with the memories the hook recalls.

use super::bridge::Instructions;
use super::{Finding, Os, Report, Standing, VERIFIED_AGAINST, VERIFIED_ON};

/// The most this block may take of a hook's 10,000 characters. Nothing enforces
/// it at run time, because cutting a path short would be worse than running
/// long; the worst case is held to it by a test, and the hook holds the total.
#[cfg(test)]
const BUDGET: usize = 2_000;

fn finding<'a>(report: &'a Report, id: &str) -> Option<&'a Finding> {
    report.findings.iter().find(|f| f.rule.id == id)
}

fn instructions_line(instructions: &Instructions) -> Option<String> {
    match instructions {
        Instructions::Unread(path) => Some(format!(
            "{} was NOT loaded as project instructions in this session. Read it now, before \
             anything else, and follow it as the project's instructions. Then tell the user \
             that `antumbra claude bridge` fixes this for every later session, and for \
             subagents, which this note does not reach.",
            path.display()
        )),
        Instructions::Bridged(path) => Some(format!(
            "AGENTS.md reaches you through {}, an untracked bridge Antumbra wrote. It is the \
             project's instructions. Do not commit, edit or delete the bridge.",
            path.display()
        )),
        Instructions::NotApplicable => None,
    }
}

fn shell_line(report: &Report, os: Os) -> Option<String> {
    let standing = finding(report, "powershell-tool").map(|f| &f.standing);
    match (os, standing) {
        (Os::Windows, Some(Standing::Missing { .. })) => Some(
            "The host's shell is PowerShell, but the PowerShell tool is off in this session. \
             Tell the user that `antumbra claude doctor` prints the line that turns it on."
                .to_string(),
        ),
        (Os::Windows, _) => Some(
            "The host's shell is PowerShell. Use the PowerShell tool for shell work unless \
             the user picks another shell."
                .to_string(),
        ),
        (Os::Other, _) => None,
    }
}

fn unavailable_line(report: &Report) -> Option<String> {
    let gone: Vec<&str> = report
        .findings
        .iter()
        .filter(|f| !matches!(f.standing, Standing::Fine(_)))
        .filter_map(|f| f.rule.unavailable)
        .collect();
    (!gone.is_empty()).then(|| {
        format!(
            "Not available in this session, so do not offer or attempt them: {}.",
            gone.join("; ")
        )
    })
}

/// The block, or `None` outside sovereign mode: nothing is lost there, and
/// nothing the agent already has is supplied twice.
pub fn brief(report: &Report, instructions: &Instructions, os: Os) -> Option<String> {
    let because = report
        .triggers
        .iter()
        .map(|t| t.variable.as_str())
        .collect::<Vec<_>>();
    if because.is_empty() {
        return None;
    }
    let head = format!(
        "## Sovereign mode\n\nThis session runs with the agent's feature-flag fetching off ({}). \
         Features gated on those flags are missing and nothing else says so. Checked against \
         Claude Code {VERIFIED_AGAINST} on {VERIFIED_ON}.",
        because.join(", ")
    );
    let stale = report
        .unverified_version
        .as_ref()
        .map(|installed| format!("Installed is {installed}, so this list may be out of date."));
    // An AGENTS.md without a bridge is only unread while that loss lasts.
    let instructions = match instructions {
        Instructions::Unread(_) if finding(report, "agents-md").is_none() => {
            &Instructions::NotApplicable
        }
        other => other,
    };
    let lines = [
        instructions_line(instructions),
        shell_line(report, os),
        unavailable_line(report),
        stale,
        Some("The user sees the whole list with `antumbra claude doctor`.".to_string()),
    ];
    let body = lines
        .into_iter()
        .flatten()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    Some(format!("{head}\n\n{body}"))
}

#[cfg(test)]
mod tests {
    use super::super::{examine, Inputs, Origin, Variable};
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn inputs(os: Os, pairs: &[(&str, &str)], instructions: Instructions) -> Inputs {
        let variables: BTreeMap<String, Variable> = pairs
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string(),
                    Variable {
                        value: value.to_string(),
                        origin: Origin::Environment,
                    },
                )
            })
            .collect();
        Inputs {
            os,
            variables,
            user_default_mode: None,
            instructions,
            installed_version: None,
        }
    }

    fn said(inputs: &Inputs) -> Option<String> {
        brief(&examine(inputs), &inputs.instructions, inputs.os)
    }

    const SOVEREIGN: (&str, &str) = ("DISABLE_TELEMETRY", "1");

    #[test]
    fn nothing_is_said_outside_sovereign_mode() {
        let quiet = inputs(
            Os::Windows,
            &[],
            Instructions::Unread(PathBuf::from("AGENTS.md")),
        );
        assert_eq!(said(&quiet), None);
    }

    #[test]
    fn an_unread_instruction_file_is_named_and_the_agent_is_told_to_read_it() -> anyhow::Result<()>
    {
        let path = PathBuf::from("/work/project/AGENTS.md");
        // A release before AGENTS.md was read with telemetry off.
        let mut here = inputs(Os::Other, &[SOVEREIGN], Instructions::Unread(path.clone()));
        here.installed_version = Some("2.1.278".to_string());
        let Some(text) = said(&here) else {
            anyhow::bail!("sovereign mode said nothing");
        };
        assert!(text.contains(&path.display().to_string()), "{text}");
        assert!(text.contains("Read it now"), "{text}");
        assert!(text.contains("antumbra claude bridge"), "{text}");
        Ok(())
    }

    #[test]
    fn a_bridge_is_explained_and_protected() -> anyhow::Result<()> {
        let bridge = PathBuf::from("/work/project/CLAUDE.local.md");
        let here = inputs(
            Os::Other,
            &[SOVEREIGN],
            Instructions::Bridged(bridge.clone()),
        );
        let Some(text) = said(&here) else {
            anyhow::bail!("sovereign mode said nothing");
        };
        assert!(text.contains(&bridge.display().to_string()), "{text}");
        assert!(text.contains("Do not commit, edit or delete"), "{text}");
        assert!(!text.contains("Read it now"), "{text}");
        Ok(())
    }

    #[test]
    fn the_shell_follows_the_host() -> anyhow::Result<()> {
        let said_for = |os, pairs: &[(&str, &str)]| {
            said(&inputs(os, pairs, Instructions::NotApplicable))
                .ok_or_else(|| anyhow::anyhow!("sovereign mode said nothing"))
        };
        let on = said_for(
            Os::Windows,
            &[SOVEREIGN, ("CLAUDE_CODE_USE_POWERSHELL_TOOL", "1")],
        )?;
        assert!(on.contains("Use the PowerShell tool"), "{on}");
        let off = said_for(Os::Windows, &[SOVEREIGN])?;
        assert!(off.contains("the PowerShell tool is off"), "{off}");
        assert!(!off.contains("Use the PowerShell tool"), "{off}");
        let elsewhere = said_for(Os::Other, &[SOVEREIGN])?;
        assert!(!elsewhere.contains("PowerShell"), "{elsewhere}");
        Ok(())
    }

    #[test]
    fn every_rule_an_agent_can_reach_for_is_listed_and_no_other() -> anyhow::Result<()> {
        // A traffic switch loses Remote Control too, so every rule is gone.
        let here = inputs(
            Os::Other,
            &[SOVEREIGN, ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")],
            Instructions::NotApplicable,
        );
        let Some(text) = said(&here) else {
            anyhow::bail!("sovereign mode said nothing");
        };
        let rules = super::super::rules();
        let told: Vec<&str> = rules.iter().filter_map(|r| r.unavailable).collect();
        assert!(told.len() >= 6, "{told:?}");
        for phrase in &told {
            assert!(text.contains(phrase), "missing `{phrase}` in {text}");
        }
        // A rule the agent never reaches for stays with the doctor.
        assert!(!text.contains("VS Code"), "{text}");
        assert!(!text.contains("claude import"), "{text}");
        Ok(())
    }

    #[test]
    fn a_version_the_rules_were_not_checked_against_is_admitted() -> anyhow::Result<()> {
        let mut here = inputs(Os::Other, &[SOVEREIGN], Instructions::NotApplicable);
        here.installed_version = Some("9.9.9".to_string());
        let Some(text) = said(&here) else {
            anyhow::bail!("sovereign mode said nothing");
        };
        assert!(text.contains("Installed is 9.9.9"), "{text}");
        Ok(())
    }

    #[test]
    fn the_worst_case_fits_its_share_of_the_hook_limit() -> anyhow::Result<()> {
        let deep = PathBuf::from(format!("C:\\{}\\AGENTS.md", "directory\\".repeat(24)));
        let mut here = inputs(Os::Windows, &[SOVEREIGN], Instructions::Unread(deep));
        here.variables.extend(
            inputs(
                Os::Windows,
                &[
                    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
                    ("DO_NOT_TRACK", "1"),
                    ("DISABLE_GROWTHBOOK", "1"),
                    ("CLAUDE_CODE_USE_BEDROCK", "1"),
                ],
                Instructions::NotApplicable,
            )
            .variables,
        );
        here.installed_version = Some("12.34.567".to_string());
        let Some(text) = said(&here) else {
            anyhow::bail!("sovereign mode said nothing");
        };
        let length = text.chars().count();
        assert!(length <= BUDGET, "{length} characters: {text}");
        Ok(())
    }
}

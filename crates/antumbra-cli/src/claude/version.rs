//! Claude Code releases, as far as the sovereign-mode rules care. A loss can
//! end in a later release, and one loss depends on which switch turned the
//! flags off, so the doctor compares versions and reads the triggers.

use super::{detect, Inputs, VERIFIED_AGAINST};

/// Whether `version` is `min` or later, comparing dotted numbers. A part that
/// does not parse counts as 0.
pub fn at_least(version: &str, min: &str) -> bool {
    let parts = |v: &str| -> Vec<u64> {
        v.split('.')
            .map(|p| p.trim().parse::<u64>().unwrap_or(0))
            .collect()
    };
    let (a, b) = (parts(version), parts(min));
    for i in 0..a.len().max(b.len()) {
        let (x, y) = (
            a.get(i).copied().unwrap_or(0),
            b.get(i).copied().unwrap_or(0),
        );
        if x != y {
            return x > y;
        }
    }
    true
}

/// The release the rules are judged for: the installed one when it is known,
/// and otherwise the one they were verified against. The session-start brief
/// does not ask, since asking starts the agent's runtime.
pub fn judged_for(inputs: &Inputs) -> &str {
    inputs
        .installed_version
        .as_deref()
        .unwrap_or(VERIFIED_AGAINST)
}

/// The first release in which Remote Control survives the flags being off,
/// when only the telemetry switches turned them off.
pub const REMOTE_CONTROL_SPARED_FROM: &str = "2.1.283";

/// Whether Remote Control is available with the flags off: from 2.1.283, when
/// only `DISABLE_TELEMETRY` or `DO_NOT_TRACK` turned them off. It still is not
/// where the organization requires Trusted Devices, which nothing local says.
pub fn remote_control_spared(inputs: &Inputs) -> bool {
    at_least(judged_for(inputs), REMOTE_CONTROL_SPARED_FROM)
        && detect(&inputs.variables)
            .iter()
            .all(|t| matches!(t.variable.as_str(), "DISABLE_TELEMETRY" | "DO_NOT_TRACK"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::super::bridge::Instructions;
    use super::super::brief::brief;
    use super::super::{examine, rules, Origin, Os, Standing, Variable};
    use super::*;

    fn inputs(version: Option<&str>, pairs: &[(&str, &str)]) -> Inputs {
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
            os: Os::Other,
            variables,
            user_default_mode: None,
            instructions: Instructions::NotApplicable,
            installed_version: version.map(str::to_string),
        }
    }

    fn standing(inputs: &Inputs, id: &str) -> Option<Standing> {
        examine(inputs)
            .findings
            .into_iter()
            .find(|f| f.rule.id == id)
            .map(|f| f.standing)
    }

    #[test]
    fn versions_compare_by_number_not_by_text() {
        assert!(at_least("2.1.283", "2.1.281"));
        assert!(at_least("2.1.283", "2.1.283"));
        assert!(!at_least("2.1.278", "2.1.281"));
        assert!(at_least("2.10.0", "2.9.9"));
        assert!(at_least("3", "2.1.283"));
        assert!(!at_least("2.1", "2.1.1"));
    }

    #[test]
    fn a_loss_that_ended_is_not_reported_for_releases_after_it() {
        let telemetry = [("DISABLE_TELEMETRY", "1")];
        let current = inputs(None, &telemetry);
        for id in ["agents-md", "auto-mode-default", "vscode-starting-mode"] {
            assert_eq!(standing(&current, id), None, "{id} on {VERIFIED_AGAINST}");
        }
        let older = inputs(Some("2.1.278"), &telemetry);
        for id in ["agents-md", "auto-mode-default", "vscode-starting-mode"] {
            assert!(standing(&older, id).is_some(), "{id} on 2.1.278");
        }
        assert_eq!(examine(&older).findings.len(), rules().len());
    }

    #[test]
    fn remote_control_survives_the_telemetry_switches_alone() {
        let spared = inputs(None, &[("DISABLE_TELEMETRY", "1"), ("DO_NOT_TRACK", "1")]);
        assert!(matches!(
            standing(&spared, "remote-control"),
            Some(Standing::Fine(_))
        ));
        for (why, lost) in [
            (
                "a traffic switch",
                inputs(
                    None,
                    &[
                        ("DISABLE_TELEMETRY", "1"),
                        ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
                    ],
                ),
            ),
            (
                "the flag switch",
                inputs(None, &[("DISABLE_GROWTHBOOK", "1")]),
            ),
            (
                "a release before it",
                inputs(Some("2.1.282"), &[("DISABLE_TELEMETRY", "1")]),
            ),
        ] {
            assert!(
                !matches!(standing(&lost, "remote-control"), Some(Standing::Fine(_))),
                "{why}"
            );
        }
    }

    #[test]
    fn the_brief_tells_the_agent_only_what_is_gone() -> Result<(), String> {
        let unread = Instructions::Unread(PathBuf::from("/work/AGENTS.md"));
        let mut current = inputs(None, &[("DISABLE_TELEMETRY", "1")]);
        current.instructions = unread.clone();
        let text = brief(&examine(&current), &current.instructions, current.os)
            .ok_or("sovereign mode said nothing")?;
        assert!(!text.contains("Remote Control"), "{text}");
        assert!(!text.contains("Read it now"), "{text}");
        let mut older = inputs(Some("2.1.278"), &[("DISABLE_TELEMETRY", "1")]);
        older.instructions = unread;
        let text = brief(&examine(&older), &older.instructions, older.os)
            .ok_or("sovereign mode said nothing")?;
        assert!(text.contains("Remote Control"), "{text}");
        assert!(text.contains("Read it now"), "{text}");
        Ok(())
    }
}

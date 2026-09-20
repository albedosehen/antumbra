//! Keyed memories: a memory whose text opens with `[<family>:<name>]`.
//!
//! The tool surface gives every write a fresh id and shows neither a memory's
//! evidence nor its compartment, so a writer that has to find its own memory
//! again can only go by the text. A key at the front of it is that handle. Two
//! families exist (ADR-0021), and everything that writes or reads them agrees on
//! the spelling here: the CLI that writes them and the console that shows them.

/// A rule about Claude Code with its feature flags off: `[claude-code:<rule>]`.
pub const CLAUDE_CODE_RULE: &str = "[claude-code:";

/// A counter of one skill's use: `[skill-use:<skill>]`.
pub const SKILL_USE: &str = "[skill-use:";

/// The key a memory of `family` opens with, for `name`.
pub fn key(family: &str, name: &str) -> String {
    format!("{family}{name}]")
}

/// The name in the key that `content` opens with, when it opens with one of
/// `family`. A key is only a key at the very front: the same text further in is
/// something a person wrote.
pub fn name_in<'a>(content: &'a str, family: &str) -> Option<&'a str> {
    content
        .strip_prefix(family)
        .and_then(|rest| rest.split_once(']'))
        .map(|(name, _)| name)
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_is_read_back_as_it_was_written() {
        for family in [CLAUDE_CODE_RULE, SKILL_USE] {
            let content = format!("{} and the rest", key(family, "agents-md"));
            assert_eq!(name_in(&content, family), Some("agents-md"));
        }
    }

    #[test]
    fn only_the_front_of_the_text_is_a_key_and_only_of_its_own_family() {
        assert_eq!(name_in("see [skill-use:deploy] above", SKILL_USE), None);
        assert_eq!(name_in("[skill-use:deploy] deploy", CLAUDE_CODE_RULE), None);
        assert_eq!(name_in("[skill-use:] nameless", SKILL_USE), None);
        assert_eq!(name_in("[skill-use:unclosed", SKILL_USE), None);
        assert_eq!(name_in("", SKILL_USE), None);
    }
}

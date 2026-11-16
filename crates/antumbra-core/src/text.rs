//! Cutting text by characters, not bytes. Four crates each wrote the
//! character-boundary cut, and that is the kind of helper one copy gets wrong:
//! a byte-indexed slice panics in the middle of a multi-byte character.

use std::borrow::Cow;

/// The leading `n` characters of `text` as a slice: no allocation, and the cut
/// lands on a `char` boundary by construction (`char_indices` yields only
/// boundaries), so multi-byte text is never split mid-codepoint. Shorter text
/// comes back whole.
pub fn first_chars(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// `text` in at most `max` characters, ending in an ellipsis when it had to be
/// cut; untouched (and unallocated) when it already fits. `max` counts the
/// ellipsis, so the result always fits where `max` characters do.
pub fn clipped(text: &str, max: usize) -> Cow<'_, str> {
    if text.chars().count() <= max {
        return Cow::Borrowed(text);
    }
    let mut out = first_chars(text, max.saturating_sub(1)).to_string();
    out.push('…');
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_chars_counts_characters_not_bytes_and_keeps_short_input_whole() {
        // Multi-byte text: the cap is a character count and the cut cannot
        // split a codepoint (a byte-indexed slice here would panic).
        let s = "é".repeat(10);
        assert_eq!(first_chars(&s, 4), "éééé");
        assert_eq!(first_chars("short", 1200), "short");
        assert_eq!(first_chars("short", 0), "");
        assert_eq!(first_chars("", 3), "");
    }

    #[test]
    fn clipped_fits_in_max_ellipsis_included() {
        assert_eq!(clipped("fits", 4), "fits");
        assert!(matches!(clipped("fits", 4), Cow::Borrowed(_)));
        assert_eq!(clipped("too long", 4), "too…");
        assert_eq!(clipped("too long", 4).chars().count(), 4);
        let s = "é".repeat(10);
        assert_eq!(clipped(&s, 5), "éééé…");
        // The degenerate widths still fit.
        assert_eq!(clipped("ab", 1), "…");
        assert_eq!(clipped("ab", 0), "…");
        assert_eq!(clipped("", 0), "");
    }
}

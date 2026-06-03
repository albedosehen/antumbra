//! Decoding helpers that shape raw model output into a clean completion.
//!
//! A base code model, prompted to complete a function, keeps generating past
//! the function (the next definition, a blank-line paragraph, fences). RAFT
//! verifies *and* trains on whatever `generate` returns, so trimming the
//! run-on here keeps the two consistent: the model is reinforced toward the
//! clean completion that actually passed.

/// Default stop markers for code completion: a new top-level definition, a
/// fence, or a `__main__` guard end the useful completion.
pub const DEFAULT_STOPS: &[&str] = &["```", "\ndef ", "\nclass ", "\nif __name__", "\n\n\n"];

/// Truncate `text` at the earliest occurrence of any stop marker, trimming
/// trailing whitespace. Returns the whole (trimmed) text if no marker is found.
pub fn truncate_at_stops(text: &str, stops: &[&str]) -> String {
    let mut cut = text.len();
    for stop in stops {
        if let Some(idx) = text.find(stop) {
            cut = cut.min(idx);
        }
    }
    text[..cut].trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_at_a_second_definition() {
        let raw = "def add(a, b):\n    return a + b\ndef other():\n    pass\n";
        assert_eq!(
            truncate_at_stops(raw, DEFAULT_STOPS),
            "def add(a, b):\n    return a + b"
        );
    }

    #[test]
    fn stops_at_a_fence() {
        let raw = "def add(a, b):\n    return a + b\n```\nsome prose";
        assert_eq!(
            truncate_at_stops(raw, DEFAULT_STOPS),
            "def add(a, b):\n    return a + b"
        );
    }

    #[test]
    fn keeps_clean_single_function() {
        let raw = "def reverse(s):\n    return s[::-1]";
        assert_eq!(truncate_at_stops(raw, DEFAULT_STOPS), raw);
    }

    #[test]
    fn does_not_cut_indented_nested_def() {
        // a nested (indented) def is "\n    def", not "\ndef", so it survives.
        let raw = "def outer():\n    def inner():\n        return 1\n    return inner";
        assert_eq!(truncate_at_stops(raw, DEFAULT_STOPS), raw);
    }
}

//! A memory cut into the pieces the chunk index embeds.
//!
//! One vector for a long memory is a blur of all of it, and a passage from its
//! middle is a small part of the blur. Measured on 400 of the user's memories,
//! a query cut from the middle of one found it in the top 30 55% of the time
//! with one vector per memory and 82% with a vector per 400-character piece.

use sha2::{Digest, Sha256};

/// The size of a piece, in characters: the best of 400, 600, 800 and 1,000 on
/// every cutoff measured.
pub const MEMORY_CHUNK_CHARS: usize = 400;

/// `text` in pieces of about `size` characters, cut between words, each
/// overlapping the one before by about a fifth, so a passage cut at a boundary
/// is whole in one of them. A text shorter than `size` is one piece. A word
/// longer than `size` is a piece of its own.
pub fn split(text: &str, size: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut out = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let mut end = start;
        let mut length = 0;
        while end < words.len() && (length == 0 || length + words[end].len() < size) {
            length += words[end].len() + 1;
            end += 1;
        }
        out.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        // Step back about a fifth of a piece for the overlap, never to where
        // this piece began, so every step moves forward.
        let mut back = end;
        let mut overlap = 0;
        while back > start + 1 && overlap < size / 5 {
            back -= 1;
            overlap += words[back].len() + 1;
        }
        start = back;
    }
    if out.is_empty() {
        out.push(text.trim().to_string());
    }
    out
}

/// A short, stable digest of `text`.
pub fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// What the chunk index records a memory's pieces were cut from: its content
/// and the piece size, so a changed text, or a new size, re-cuts it.
pub fn cut_hash(content: &str) -> String {
    content_hash(&format!("{MEMORY_CHUNK_CHARS}\n{content}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_cover_the_text_overlapping_and_move_forward() {
        let text = (0..200)
            .map(|i| format!("w{i:03}"))
            .collect::<Vec<_>>()
            .join(" ");
        let pieces = split(&text, 40);
        assert!(pieces.len() > 20, "{pieces:?}");
        assert!(pieces.iter().all(|p| p.len() <= 44), "{pieces:?}");
        assert!(pieces[0].starts_with("w000"));
        assert!(pieces.last().unwrap().ends_with("w199"));
        for pair in pieces.windows(2) {
            let last = pair[0].split(' ').next_back().unwrap();
            assert!(
                pair[1].contains(last),
                "consecutive pieces overlap: {pair:?}"
            );
            assert_ne!(pair[0], pair[1]);
        }
    }

    #[test]
    fn a_short_text_is_one_piece_and_a_long_word_is_its_own() {
        assert_eq!(
            split("a short memory", MEMORY_CHUNK_CHARS),
            ["a short memory"]
        );
        assert_eq!(split("   ", 40), [""]);
        let long = "x".repeat(100);
        let pieces = split(&format!("{long} tail words here"), 40);
        assert_eq!(pieces[0], long);
        assert!(pieces.last().unwrap().ends_with("here"));
    }

    #[test]
    fn the_hash_is_stable_and_follows_the_content() {
        assert_eq!(content_hash("abc"), content_hash("abc"));
        assert_ne!(content_hash("abc"), content_hash("abd"));
        assert_eq!(content_hash("abc").len(), 32);
        assert_eq!(content_hash("abc"), "ba7816bf8f01cfea414140de5dae2223");
        assert_ne!(
            cut_hash("abc"),
            content_hash("abc"),
            "the size is part of it"
        );
        assert_eq!(cut_hash("abc"), cut_hash("abc"));
    }
}

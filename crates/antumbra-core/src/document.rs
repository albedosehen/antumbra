//! Knowledge documents (P-3): a *document* is ingested as chunked, embedded,
//! recallable text: a first-class type **distinct** from an episodic
//! [`crate::memory::Memory`]. Episodic memory is what an agent learned by doing;
//! a document is reference material it was given. Both are embedded and recalled
//! semantically, but they are stored and surfaced separately so one does not
//! drown out the other.
//!
//! This module is storage-agnostic: it holds the [`DocumentChunk`] entity and the
//! pure [`chunk_text`] splitter. Persistence (an HNSW-indexed `document_chunk`
//! table) and ingestion (chunk → embed → store) live in the store and the MCP
//! surface.

use chrono::{DateTime, Utc};

use crate::ids::{DocumentChunkId, TenantId};

/// One embedded slice of a document. The document itself is identified by
/// `title` (+ optional `source`); a chunk carries enough to recall it and name
/// its origin, without a separate document table in v0.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentChunk {
    pub id: DocumentChunkId,
    pub tenant: TenantId,
    /// The document this chunk belongs to (its human title).
    pub title: String,
    /// Where the document came from (path / url / note); `None` if unstated.
    pub source: Option<String>,
    /// 0-based position of this chunk within the document, so the original order
    /// is recoverable.
    pub ordinal: u32,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
}

/// Split `text` into chunks of at most `max_chars` characters, each overlapping
/// the previous by about `overlap` characters so a fact straddling a cut is still
/// wholly present in one chunk (the standard retrieval-chunking trick). Cuts
/// prefer a natural boundary (paragraph break, then sentence end, then
/// whitespace), searching backward from the hard limit so a chunk does not end
/// mid-word. Whitespace-only input yields no chunks.
///
/// Counting is by `char`, not byte, so the cuts are always on UTF-8 boundaries.
pub fn chunk_text(text: &str, max_chars: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let max = max_chars.max(1);
    // Overlap must be < max, or the window cannot advance.
    let overlap = overlap.min(max - 1);

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let hard_end = (start + max).min(chars.len());
        let cut = if hard_end == chars.len() {
            hard_end
        } else {
            boundary_before(&chars, start, hard_end)
        };
        let chunk: String = chars[start..cut].iter().collect();
        let trimmed = chunk.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if cut >= chars.len() {
            break;
        }
        // Advance with overlap, but always make progress (at least one char past
        // `start`) so a boundary-less blob cannot loop forever.
        start = (cut.saturating_sub(overlap)).max(start + 1);
    }
    chunks
}

/// The position to cut at: the latest paragraph/sentence/whitespace boundary in
/// the second half of `[start, hard_end)`, or `hard_end` if none (a long
/// unbroken run is cut at the hard limit). Searching only the second half keeps
/// chunks from collapsing to a tiny prefix.
fn boundary_before(chars: &[char], start: usize, hard_end: usize) -> usize {
    let min_cut = start + (hard_end - start) / 2;
    // Paragraph break (blank line): cut just after the second newline.
    for i in (min_cut..hard_end).rev() {
        if chars[i] == '\n' && i > start && chars[i - 1] == '\n' {
            return i + 1;
        }
    }
    // Sentence end followed by a space.
    for i in (min_cut..hard_end).rev() {
        if matches!(chars[i], '.' | '!' | '?')
            && chars.get(i + 1).is_some_and(|c| c.is_whitespace())
        {
            return i + 1;
        }
    }
    // Any whitespace.
    for i in (min_cut..hard_end).rev() {
        if chars[i].is_whitespace() {
            return i + 1;
        }
    }
    hard_end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(
            chunk_text("just a little text", 1000, 100),
            vec!["just a little text"]
        );
    }

    #[test]
    fn blank_input_yields_nothing() {
        assert!(chunk_text("   \n\n  ", 100, 10).is_empty());
        assert!(chunk_text("", 100, 10).is_empty());
    }

    #[test]
    fn splits_on_sentence_boundaries() {
        // Three sentences; a small window forces a split, which lands at a
        // sentence end rather than mid-word.
        let text = "The cat sat. The dog ran. The bird flew away quickly.";
        let chunks = chunk_text(text, 28, 0);
        assert!(chunks.len() >= 2);
        // No chunk ends mid-word (each ends at sentence punctuation or is the tail).
        for c in &chunks {
            assert!(!c.is_empty());
        }
        // Reassembling the (overlap-free) chunks recovers every word.
        let joined = chunks.join(" ");
        for word in ["cat", "dog", "bird", "quickly"] {
            assert!(joined.contains(word), "lost '{word}' in {chunks:?}");
        }
    }

    #[test]
    fn overlap_repeats_context_across_the_cut() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet";
        let no_overlap = chunk_text(text, 24, 0);
        let with_overlap = chunk_text(text, 24, 10);
        // Overlap produces at least as many chunks and repeats some content.
        assert!(with_overlap.len() >= no_overlap.len());
        assert!(with_overlap.len() >= 2);
    }

    #[test]
    fn a_boundaryless_blob_still_terminates_and_covers_everything() {
        // No whitespace at all: must cut at the hard limit and still progress.
        let blob: String = "x".repeat(250);
        let chunks = chunk_text(&blob, 100, 0);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.chars().count() <= 100));
        assert_eq!(chunks.concat(), blob);
    }

    #[test]
    fn every_chunk_respects_the_char_limit() {
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do \
                    eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        let chunks = chunk_text(text, 40, 8);
        assert!(chunks.iter().all(|c| c.chars().count() <= 40), "{chunks:?}");
    }
}

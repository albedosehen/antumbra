//! The HTTP embedder now lives in the shared `antumbra-embed` crate, so the
//! operator console can use the same endpoint the population was built with.
//! Re-exported here for the existing call sites (`embed::HttpEmbedder`).

pub use antumbra_embed::HttpEmbedder;

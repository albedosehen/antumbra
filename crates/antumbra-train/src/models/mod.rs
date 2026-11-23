//! The real candle model stack (MT-1), behind the `models` feature.

pub mod qwen;

pub use qwen::{base_is_cached, fetch_base, GenTiming, QwenCausalLm, Together};

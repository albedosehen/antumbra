//! The real candle model stack (MT-1), behind the `models` feature.

pub mod qwen;

pub use qwen::QwenCausalLm;

//! Compose experts by blending their LoRA adapters (ADR-0009). The learned
//! router gives per-expert weights for a task; this realizes that mix as one
//! served adapter.
//!
//! Because every frozen expert shares the base's rank and scale, the weighted
//! sum of their deltas is **exact** as a rank-concatenated adapter: stack the
//! `sqrt(w_i)`-scaled `A`/`B` factors so block `i` contributes `w_i * delta_i`,
//! and the model's existing single-adapter forward serves the blend at the
//! higher rank. No weight-space interference, no model surgery — linear
//! composition of independently-grown, verified experts.

use std::collections::HashMap;

use candle_core::{Device, Tensor};

use antumbra_core::{AntumbraError, Result};

fn ce(e: candle_core::Error) -> AntumbraError {
    AntumbraError::other(format!("compose: {e}"))
}

/// Merge `(adapter_path, weight)` specs into one rank-concatenated adapter at
/// `out_path`; returns the merged rank. `A` (`rank x in`) stacks along its rank
/// rows, `B` (`out x rank`) along its rank columns, each scaled by `sqrt(w_i)`.
pub fn compose_adapters(specs: &[(String, f32)], out_path: &str) -> Result<usize> {
    if specs.is_empty() {
        return Err(AntumbraError::other("compose: no adapters"));
    }
    let device = Device::Cpu;
    let maps: Vec<HashMap<String, Tensor>> = specs
        .iter()
        .map(|(p, _)| candle_core::safetensors::load(p, &device).map_err(ce))
        .collect::<Result<_>>()?;

    let keys: Vec<String> = maps[0].keys().cloned().collect();
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for key in &keys {
        // The adapter holds only LoRA factors. A: stack rows (rank dim 0);
        // B: stack cols (rank dim 1). Both fold sqrt(w) so the product is w.
        let dim = if key.ends_with("lora_a") { 0 } else { 1 };
        let mut parts = Vec::with_capacity(specs.len());
        for (i, (_, w)) in specs.iter().enumerate() {
            let t = maps[i]
                .get(key)
                .ok_or_else(|| AntumbraError::other(format!("compose: key {key} missing")))?;
            parts.push((t * (w.max(0.0).sqrt() as f64)).map_err(ce)?);
        }
        out.insert(key.clone(), Tensor::cat(&parts, dim).map_err(ce)?);
    }

    candle_core::safetensors::save(&out, out_path).map_err(ce)?;
    let rank = out
        .iter()
        .find(|(k, _)| k.ends_with("lora_a"))
        .map(|(_, t)| t.dim(0).unwrap_or(0))
        .unwrap_or(0);
    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_adapter(path: &str, a: &[f32], b: &[f32], rank: usize, inn: usize, out: usize) {
        let device = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "p.lora_a".into(),
            Tensor::from_vec(a.to_vec(), (rank, inn), &device).unwrap(),
        );
        m.insert(
            "p.lora_b".into(),
            Tensor::from_vec(b.to_vec(), (out, rank), &device).unwrap(),
        );
        candle_core::safetensors::save(&m, path).unwrap();
    }

    #[test]
    fn concatenation_realizes_the_weighted_delta_sum() {
        let dir = std::env::temp_dir();
        let (p1, p2, pm) = (
            dir.join("ant_c1.safetensors"),
            dir.join("ant_c2.safetensors"),
            dir.join("ant_cm.safetensors"),
        );
        // rank 1, in 2, out 1. delta_i = b_i (a_i . x).
        write_adapter(p1.to_str().unwrap(), &[1.0, 0.0], &[2.0], 1, 2, 1);
        write_adapter(p2.to_str().unwrap(), &[0.0, 1.0], &[3.0], 1, 2, 1);
        let (w1, w2) = (0.25f32, 0.75f32);
        let rank = compose_adapters(
            &[
                (p1.to_str().unwrap().into(), w1),
                (p2.to_str().unwrap().into(), w2),
            ],
            pm.to_str().unwrap(),
        )
        .unwrap();
        assert_eq!(rank, 2); // 1 + 1

        // Merged delta on x = [x0, x1]: w1*2*x0 + w2*3*x1. Check via B*(A*x).
        let device = Device::Cpu;
        let m = candle_core::safetensors::load(pm.to_str().unwrap(), &device).unwrap();
        let a = &m["p.lora_a"]; // (2, 2)
        let b = &m["p.lora_b"]; // (1, 2)
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (2, 1), &device).unwrap();
        let delta = b.matmul(&a.matmul(&x).unwrap()).unwrap();
        let got = delta.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let want = w1 * 2.0 * 1.0 + w2 * 3.0 * 1.0;
        assert!((got - want).abs() < 1e-5, "got {got} want {want}");

        for p in [p1, p2, pm] {
            std::fs::remove_file(p).ok();
        }
    }
}

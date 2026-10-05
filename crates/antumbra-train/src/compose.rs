//! Compose experts by blending their LoRA adapters into one heterogeneous model. The learned
//! router gives per-expert weights for a task; this realizes that mix as one
//! served adapter.
//!
//! Because every frozen expert shares the base's rank and scale, the weighted
//! sum of their deltas is **exact** as a rank-concatenated adapter: stack the
//! `sqrt(w_i)`-scaled `A`/`B` factors so block `i` contributes `w_i * delta_i`,
//! and the model's existing single-adapter forward serves the blend at the
//! higher rank. No weight-space interference, no model surgery: linear
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
    let device = Device::Cpu;
    let parts: Vec<(HashMap<String, Tensor>, f32)> = specs
        .iter()
        .map(|(p, w)| Ok((candle_core::safetensors::load(p, &device).map_err(ce)?, *w)))
        .collect::<Result<_>>()?;
    let out = blend(&parts, None)?;
    let rank = rank_of(&out);
    // A rank-0 result (no `lora_a` rows) would scale to a NaN `alpha/rank` and
    // poison generation; reject it before writing a garbage adapter to disk.
    if rank == 0 {
        return Err(AntumbraError::other(
            "compose: composed adapter has rank 0 (no lora_a rows)",
        ));
    }
    candle_core::safetensors::save(&out, out_path).map_err(ce)?;
    Ok(rank)
}

/// The rank of an adapter's factors: the rows of any `lora_a`.
pub fn rank_of(factors: &HashMap<String, Tensor>) -> usize {
    factors
        .iter()
        .find(|(k, _)| k.ends_with("lora_a"))
        .map(|(_, t)| t.dim(0).unwrap_or(0))
        .unwrap_or(0)
}

/// Stack `(factors, weight)` adapters into one, block `i` scaled to contribute
/// `w_i * delta_i`. With `rank`, the result is zero-padded to that rank, so a
/// model built at a higher rank serves it exactly: the padded rows of `A` and
/// columns of `B` add nothing. A model built at `rank` with alpha raised in
/// step keeps the trained scale, and then serves one adapter or a blend of
/// several without being rebuilt.
pub fn blend(
    parts: &[(HashMap<String, Tensor>, f32)],
    rank: Option<usize>,
) -> Result<HashMap<String, Tensor>> {
    let Some((first, _)) = parts.first() else {
        return Err(AntumbraError::other("compose: no adapters"));
    };
    let mut out: HashMap<String, Tensor> = HashMap::new();
    for key in first.keys() {
        // The adapter holds only LoRA factors. A: stack rows (rank dim 0);
        // B: stack cols (rank dim 1). Both fold sqrt(w) so the product is w.
        let dim = if key.ends_with("lora_a") { 0 } else { 1 };
        let mut stacked = Vec::with_capacity(parts.len() + 1);
        for (factors, w) in parts {
            let t = factors
                .get(key)
                .ok_or_else(|| AntumbraError::other(format!("compose: key {key} missing")))?;
            stacked.push((t * (w.max(0.0).sqrt() as f64)).map_err(ce)?);
        }
        let mut t = Tensor::cat(&stacked, dim).map_err(ce)?;
        if let Some(rank) = rank {
            let have = t.dim(dim).map_err(ce)?;
            if have > rank {
                return Err(AntumbraError::other(format!(
                    "compose: the blend has rank {have}, above the model's {rank}"
                )));
            }
            if have < rank {
                t = t.pad_with_zeros(dim, 0, rank - have).map_err(ce)?;
            }
        }
        out.insert(key.clone(), t);
    }
    Ok(out)
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

    #[test]
    fn rejects_a_rank_zero_adapter() {
        let dir = std::env::temp_dir();
        let (p, pm) = (
            dir.join("ant_rank0.safetensors"),
            dir.join("ant_rank0_m.safetensors"),
        );
        // lora_a is (0, 2): zero rank rows. A composed rank-0 adapter would scale
        // to a NaN alpha/rank and poison generation, so compose must refuse it.
        write_adapter(p.to_str().unwrap(), &[], &[], 0, 2, 1);
        let err = compose_adapters(&[(p.to_str().unwrap().into(), 1.0)], pm.to_str().unwrap());
        assert!(err.is_err(), "rank-0 compose must be rejected");
        assert!(
            !pm.exists(),
            "a rejected compose must not leave a garbage adapter on disk"
        );
        for q in [p, pm] {
            std::fs::remove_file(q).ok();
        }
    }

    fn factors(
        a: &[f32],
        b: &[f32],
        rank: usize,
        inn: usize,
        out: usize,
    ) -> HashMap<String, Tensor> {
        let device = Device::Cpu;
        HashMap::from([
            (
                "p.lora_a".to_string(),
                Tensor::from_vec(a.to_vec(), (rank, inn), &device).unwrap(),
            ),
            (
                "p.lora_b".to_string(),
                Tensor::from_vec(b.to_vec(), (out, rank), &device).unwrap(),
            ),
        ])
    }

    /// `B (A x)` for `x = [1, 2]`.
    fn delta(f: &HashMap<String, Tensor>) -> f32 {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &Device::Cpu).unwrap();
        let ax = f["p.lora_a"].matmul(&x).unwrap();
        f["p.lora_b"]
            .matmul(&ax)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    }

    #[test]
    fn one_adapter_padded_to_a_higher_rank_serves_the_same_delta() {
        let one = factors(&[1.0, 3.0], &[2.0], 1, 2, 1);
        let padded = blend(&[(one.clone(), 1.0)], Some(3)).unwrap();
        assert_eq!(padded["p.lora_a"].dims(), &[3, 2]);
        assert_eq!(padded["p.lora_b"].dims(), &[1, 3]);
        assert_eq!(rank_of(&padded), 3);
        assert!((delta(&padded) - delta(&one)).abs() < 1e-6);
    }

    #[test]
    fn a_padded_blend_is_the_weighted_sum_and_halves_make_a_whole() {
        let p = factors(&[1.0, 0.0], &[2.0], 1, 2, 1);
        let q = factors(&[0.0, 1.0], &[3.0], 1, 2, 1);
        let mixed = blend(&[(p.clone(), 0.25), (q.clone(), 1.0)], Some(4)).unwrap();
        assert!((delta(&mixed) - (0.25 * delta(&p) + delta(&q))).abs() < 1e-5);
        let halves = blend(&[(p.clone(), 0.5), (p.clone(), 0.5)], Some(3)).unwrap();
        assert!((delta(&halves) - delta(&p)).abs() < 1e-5);
    }

    #[test]
    fn a_blend_above_the_model_rank_or_missing_a_factor_is_refused() {
        let p = factors(&[1.0, 0.0], &[2.0], 1, 2, 1);
        assert!(blend(&[(p.clone(), 1.0), (p.clone(), 1.0)], Some(1)).is_err());
        let mut partial = p.clone();
        partial.remove("p.lora_b");
        assert!(blend(&[(p, 1.0), (partial, 1.0)], Some(2)).is_err());
        assert!(blend(&[], Some(2)).is_err());
    }
}

//! Train the learned router's metric (ADR-0009). Prototypical: learn a
//! per-dimension weighting of the frozen sentence embedding so each expert's
//! exemplars cluster tightest around their own centroid in the reweighted
//! space. Labels are free — an expert's solved exemplars are its positives.
//! CPU-only (a tiny model over a few hundred vectors), so it runs anywhere.

use candle_core::{DType, Device, Result as CResult, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use antumbra_core::{AntumbraError, ExpertId, LearnedRouter, Result, RouterExpert};

const PROJ_TEMP: f32 = 0.07;
/// Minimum gap below the in-distribution mean for the OOD floor, so a
/// low-variance population (few exemplars per expert) still gets a usable band.
const FLOOR_MARGIN: f32 = 0.15;

fn rce(e: candle_core::Error) -> AntumbraError {
    AntumbraError::other(format!("router: {e}"))
}

fn l2_normalize(x: &Tensor) -> CResult<Tensor> {
    let norm = (x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()? + 1e-6)?;
    x.broadcast_div(&norm)
}

/// Compute each class's centroid (mean of its rows) in `feats`, stacked and
/// L2-normalized: `(n_classes, dim)`.
fn centroids(feats: &Tensor, class_rows: &[Vec<u32>], device: &Device) -> CResult<Tensor> {
    let mut cents = Vec::with_capacity(class_rows.len());
    for rows in class_rows {
        let idx = Tensor::from_vec(rows.clone(), (rows.len(),), device)?;
        cents.push(feats.index_select(&idx, 0)?.mean(0)?);
    }
    l2_normalize(&Tensor::stack(&cents, 0)?)
}

/// Learn the router metric from `(expert, embedding)` exemplars.
pub fn train_learned_router(
    exemplars: &[(ExpertId, Vec<f32>)],
    epochs: usize,
) -> Result<LearnedRouter> {
    if exemplars.is_empty() {
        return Err(AntumbraError::other("router: no exemplars to train on"));
    }
    let mut ids: Vec<ExpertId> = Vec::new();
    for (id, _) in exemplars {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    let (n_experts, dim, n) = (ids.len(), exemplars[0].1.len(), exemplars.len());
    let device = Device::Cpu;

    let xs: Vec<f32> = exemplars
        .iter()
        .flat_map(|(_, e)| e.iter().copied())
        .collect();
    let x = Tensor::from_vec(xs, (n, dim), &device).map_err(rce)?;
    let labels: Vec<u32> = exemplars
        .iter()
        .map(|(id, _)| ids.iter().position(|c| c == id).unwrap() as u32)
        .collect();
    let y = Tensor::from_vec(labels.clone(), (n,), &device).map_err(rce)?;
    let class_rows: Vec<Vec<u32>> = (0..n_experts)
        .map(|c| {
            (0..n as u32)
                .filter(|&i| labels[i as usize] as usize == c)
                .collect()
        })
        .collect();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let w = vb
        .get_with_hints(dim, "metric", candle_nn::init::Init::Const(1.0))
        .map_err(rce)?;
    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: 5e-2,
            ..Default::default()
        },
    )
    .map_err(rce)?;

    for _ in 0..epochs {
        let scaled = l2_normalize(&x.broadcast_mul(&w).map_err(rce)?).map_err(rce)?;
        let cents = centroids(&scaled, &class_rows, &device).map_err(rce)?;
        let logits = (scaled.matmul(&cents.t().map_err(rce)?).map_err(rce)?
            * (1.0 / PROJ_TEMP as f64))
            .map_err(rce)?;
        let loss = candle_nn::loss::cross_entropy(&logits, &y).map_err(rce)?;
        opt.backward_step(&loss).map_err(rce)?;
    }

    // Freeze the metric and snapshot the centroids in the reweighted space.
    let weights: Vec<f32> = w.to_vec1().map_err(rce)?;
    let scaled = l2_normalize(&x.broadcast_mul(&w).map_err(rce)?).map_err(rce)?;
    let cents = centroids(&scaled, &class_rows, &device).map_err(rce)?;
    let experts = ids
        .iter()
        .enumerate()
        .map(|(c, id)| -> Result<RouterExpert> {
            Ok(RouterExpert {
                id: id.clone(),
                centroid: cents.get(c).map_err(rce)?.to_vec1().map_err(rce)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Calibrate the in-distribution floor: how similar each exemplar sits to its
    // own centroid in the learned space; a task well below this is OOD.
    let cents_for_rows = cents.index_select(&y, 0).map_err(rce)?;
    let sims = scaled
        .mul(&cents_for_rows)
        .map_err(rce)?
        .sum(D::Minus1)
        .map_err(rce)?;
    let sims_v: Vec<f32> = sims.to_vec1().map_err(rce)?;
    let mean = sims_v.iter().sum::<f32>() / sims_v.len().max(1) as f32;
    let var = sims_v.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / sims_v.len().max(1) as f32;
    // Subtract at least FLOOR_MARGIN: with few exemplars per expert the centroid
    // is the exemplar, so std ~ 0 and `mean - 2*std` collapses to ~1.0 and
    // rejects everything. The margin keeps a usable band in that regime; when
    // the population is well-sampled, 2*std dominates and this is a no-op.
    let floor = (mean - (2.0 * var.sqrt()).max(FLOOR_MARGIN)).clamp(-1.0, 1.0);

    Ok(LearnedRouter {
        weights,
        experts,
        temperature: PROJ_TEMP,
        floor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two experts whose raw embeddings share a dominant first dimension (so
    /// cosine confuses them) but differ on a small second dimension. The
    /// learned metric must down-weight the shared dim and route by the rest.
    #[tokio::test]
    async fn learns_to_separate_experts_sharing_a_background() {
        let mut exemplars = Vec::new();
        for i in 0..8 {
            let j = i as f32 * 0.01;
            // "general": big shared dim, +second
            exemplars.push((ExpertId::new("general"), vec![1.0, 0.20 + j, 0.0]));
            // "specific": big shared dim, +third
            exemplars.push((ExpertId::new("specific"), vec![1.0, 0.0, 0.20 + j]));
        }
        let router = train_learned_router(&exemplars, 300).unwrap();

        // A query sharing the background but tilted to the third dim -> specific.
        let ranked = router.route(&[1.0, 0.02, 0.18]);
        assert_eq!(ranked[0].0, ExpertId::new("specific"));
        assert!(ranked[0].1 > 0.6, "confident: {:?}", ranked[0]);
    }
}

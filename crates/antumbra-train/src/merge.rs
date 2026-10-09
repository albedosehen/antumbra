//! Rank-preserving merge of two LoRA adapters: conservative
//! and reversible, and only for siblings whose subspaces overlap.
//!
//! [`crate::compose_adapters`] blends adapters exactly by concatenating their
//! ranks, which is right for serving a mix but not for minting an expert:
//! every loader and server builds a model at the population's one rank, so a
//! merged expert must have it too. So the two deltas are averaged in factored
//! form and truncated back to that rank. The product of the stacked factors
//! is taken apart by QR on each side and an SVD of the small core, and the top
//! `rank` directions are kept.
//!
//! What the truncation keeps is the overlap merging needs. Siblings whose
//! subspaces coincide keep nearly all of their averaged delta's energy; two
//! with orthogonal subspaces of equal weight keep half. [`MergeReport::retained`]
//! is that fraction, weighted by energy across modules: the test for whether
//! two experts are siblings, and the fidelity of their merge in one number.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use antumbra_core::{AntumbraError, Result};

fn ce(e: candle_core::Error) -> AntumbraError {
    AntumbraError::other(format!("merge: {e}"))
}

/// What a merge kept.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MergeReport {
    /// The rank of the merged adapter: the rank of the two it merged.
    pub rank: usize,
    /// The share of the averaged delta's energy the truncation kept, weighted
    /// by energy across modules. 1 when the two span the same subspaces.
    pub retained: f32,
    /// LoRA-wrapped projections merged.
    pub modules: usize,
}

/// A dense row-major matrix in f64, just enough linear algebra for the merge.
#[derive(Debug, Clone, PartialEq)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    fn transpose(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.at(r, c));
            }
        }
        t
    }

    fn mul(&self, other: &Mat) -> Mat {
        let mut out = Mat::zeros(self.rows, other.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(r, k);
                if a == 0.0 {
                    continue;
                }
                for c in 0..other.cols {
                    out.data[r * other.cols + c] += a * other.at(k, c);
                }
            }
        }
        out
    }

    fn column(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.at(r, c)).collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Thin QR of a tall matrix by modified Gram-Schmidt, orthogonalized twice
/// for accuracy. A column already spanned by those before it gets a zero
/// column in `Q` and a zero on `R`'s diagonal.
fn qr(m: &Mat) -> (Mat, Mat) {
    let (n, k) = (m.rows, m.cols);
    let mut q = Mat::zeros(n, k);
    let mut r = Mat::zeros(k, k);
    let scale = m.data.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
    for j in 0..k {
        let mut v = m.column(j);
        for _ in 0..2 {
            for i in 0..j {
                let qi = q.column(i);
                let proj = dot(&qi, &v);
                r.set(i, j, r.at(i, j) + proj);
                for (x, y) in v.iter_mut().zip(&qi) {
                    *x -= proj * y;
                }
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm > 1e-12 * scale {
            r.set(j, j, norm);
            for (row, x) in v.iter().enumerate() {
                q.set(row, j, x / norm);
            }
        }
    }
    (q, r)
}

/// SVD of a small square matrix by one-sided Jacobi rotations: `m = u s vᵀ`,
/// singular values descending.
fn svd(m: &Mat) -> (Mat, Vec<f64>, Mat) {
    let n = m.cols;
    let mut u = m.clone();
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v.set(i, i, 1.0);
    }
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let (up, uq) = (u.column(p), u.column(q));
                let (alpha, beta, gamma) = (dot(&up, &up), dot(&uq, &uq), dot(&up, &uq));
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for mat in [&mut u, &mut v] {
                    for row in 0..mat.rows {
                        let (a, b) = (mat.at(row, p), mat.at(row, q));
                        mat.set(row, p, c * a - s * b);
                        mat.set(row, q, s * a + c * b);
                    }
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut order: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, dot(&u.column(i), &u.column(i)).sqrt()))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut uu = Mat::zeros(m.rows, n);
    let mut vv = Mat::zeros(n, n);
    let mut s = Vec::with_capacity(n);
    for (to, &(from, sigma)) in order.iter().enumerate() {
        s.push(sigma);
        for row in 0..m.rows {
            let x = u.at(row, from);
            uu.set(row, to, if sigma > 0.0 { x / sigma } else { 0.0 });
        }
        for row in 0..n {
            vv.set(row, to, v.at(row, from));
        }
    }
    (uu, s, vv)
}

/// The two factors of one module, merged: `(A', B', kept energy, total)`.
fn merge_module(a1: &Mat, b1: &Mat, a2: &Mat, b2: &Mat, rank: usize) -> (Mat, Mat, f64, f64) {
    let w = 0.5f64.sqrt();
    // B stacked by columns, A by rows: their product is the averaged delta.
    let (out, inn, r2) = (b1.rows, a1.cols, b1.cols + b2.cols);
    let mut b = Mat::zeros(out, r2);
    let mut at = Mat::zeros(inn, r2);
    for (offset, (bi, ai)) in [(0, (b1, a1)), (b1.cols, (b2, a2))] {
        for row in 0..out {
            for c in 0..bi.cols {
                b.set(row, offset + c, w * bi.at(row, c));
            }
        }
        for row in 0..ai.rows {
            for c in 0..inn {
                at.set(c, offset + row, w * ai.at(row, c));
            }
        }
    }
    let (qb, rb) = qr(&b);
    let (qa, ra) = qr(&at);
    let core = rb.mul(&ra.transpose());
    let (u, s, v) = svd(&core);
    let total: f64 = s.iter().map(|x| x * x).sum();
    let kept: f64 = s.iter().take(rank).map(|x| x * x).sum();
    let mut left = Mat::zeros(r2, rank);
    let mut right = Mat::zeros(r2, rank);
    for (i, sigma) in s.iter().enumerate().take(rank) {
        let root = sigma.sqrt();
        for row in 0..r2 {
            left.set(row, i, u.at(row, i) * root);
            right.set(row, i, v.at(row, i) * root);
        }
    }
    let merged_b = qb.mul(&left);
    let merged_a = qa.mul(&right).transpose();
    (merged_a, merged_b, kept, total)
}

fn to_mat(t: &Tensor) -> Result<Mat> {
    let (rows, cols) = t.dims2().map_err(ce)?;
    let data: Vec<f64> = t
        .to_dtype(DType::F64)
        .map_err(ce)?
        .flatten_all()
        .map_err(ce)?
        .to_vec1()
        .map_err(ce)?;
    Ok(Mat { rows, cols, data })
}

fn to_tensor(m: &Mat, dtype: DType) -> Result<Tensor> {
    Tensor::from_vec(m.data.clone(), (m.rows, m.cols), &Device::Cpu)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(ce)
}

/// One LoRA factor of a module in a loaded adapter.
fn factor<'m>(m: &'m HashMap<String, Tensor>, prefix: &str, suffix: &str) -> Result<&'m Tensor> {
    m.get(&format!("{prefix}.{suffix}"))
        .ok_or_else(|| AntumbraError::other(format!("merge: {prefix}.{suffix} missing")))
}

/// Merge the adapters at `a` and `b` into one of the same rank at `out`,
/// returning what the truncation kept. With `out` of `None`, nothing is
/// written: the report alone, which is the overlap test.
pub fn merge_adapters(a: &str, b: &str, out: Option<&str>) -> Result<MergeReport> {
    let device = Device::Cpu;
    let left = candle_core::safetensors::load(a, &device).map_err(ce)?;
    let right = candle_core::safetensors::load(b, &device).map_err(ce)?;
    let mut merged: HashMap<String, Tensor> = HashMap::new();
    let (mut kept, mut total, mut modules, mut rank) = (0.0f64, 0.0f64, 0usize, 0usize);
    let mut prefixes: Vec<&str> = left
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_a"))
        .collect();
    prefixes.sort_unstable();
    for prefix in prefixes {
        let (a1, b1, a2, b2) = (
            factor(&left, prefix, "lora_a")?,
            factor(&left, prefix, "lora_b")?,
            factor(&right, prefix, "lora_a")?,
            factor(&right, prefix, "lora_b")?,
        );
        if a1.dims() != a2.dims() || b1.dims() != b2.dims() {
            return Err(AntumbraError::other(format!(
                "merge: {prefix} differs in shape between the two adapters"
            )));
        }
        let r = a1.dim(0).map_err(ce)?;
        rank = r;
        let (ma, mb, k, t) =
            merge_module(&to_mat(a1)?, &to_mat(b1)?, &to_mat(a2)?, &to_mat(b2)?, r);
        kept += k;
        total += t;
        modules += 1;
        if out.is_some() {
            merged.insert(format!("{prefix}.lora_a"), to_tensor(&ma, a1.dtype())?);
            merged.insert(format!("{prefix}.lora_b"), to_tensor(&mb, b1.dtype())?);
        }
    }
    if modules == 0 {
        return Err(AntumbraError::other(
            "merge: no LoRA modules in the adapter",
        ));
    }
    if let Some(path) = out {
        candle_core::safetensors::save(&merged, path).map_err(ce)?;
    }
    Ok(MergeReport {
        rank,
        retained: if total > 0.0 {
            (kept / total) as f32
        } else {
            1.0
        },
        modules,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Mat {
        let mut m = Mat::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m.set(r, c, f(r, c));
            }
        }
        m
    }

    fn max_diff(a: &Mat, b: &Mat) -> f64 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn pseudo(seed: u64) -> impl Fn(usize, usize) -> f64 {
        move |r, c| {
            let x = (r as u64 * 7919 + c as u64 * 104_729 + seed * 1_000_003) % 1000;
            x as f64 / 500.0 - 1.0
        }
    }

    #[test]
    fn qr_and_svd_reconstruct_what_they_took_apart() {
        let m = mat(12, 4, pseudo(1));
        let (q, r) = qr(&m);
        assert!(max_diff(&q.mul(&r), &m) < 1e-9);
        assert!(
            max_diff(
                &q.transpose().mul(&q),
                &mat(4, 4, |i, j| f64::from(u8::from(i == j)))
            ) < 1e-9
        );
        let sq = mat(5, 5, pseudo(2));
        let (u, s, v) = svd(&sq);
        let diag = mat(5, 5, |i, j| if i == j { s[i] } else { 0.0 });
        assert!(max_diff(&u.mul(&diag).mul(&v.transpose()), &sq) < 1e-9);
        assert!(s.windows(2).all(|w| w[0] >= w[1]), "descending: {s:?}");
    }

    /// An adapter merged with itself is itself: all of its energy is kept and
    /// the merged delta is the original one.
    #[test]
    fn an_adapter_merged_with_itself_keeps_everything() {
        let (a, b) = (mat(3, 10, pseudo(3)), mat(8, 3, pseudo(4)));
        let (ma, mb, kept, total) = merge_module(&a, &b, &a, &b, 3);
        assert!((kept / total - 1.0).abs() < 1e-9);
        assert_eq!((ma.rows, ma.cols, mb.rows, mb.cols), (3, 10, 8, 3));
        assert!(max_diff(&mb.mul(&ma), &b.mul(&a)) < 1e-9);
    }

    /// Siblings with orthogonal subspaces and equal weight keep half the
    /// energy: they are not siblings.
    #[test]
    fn orthogonal_adapters_keep_half() {
        let unit = |rows: usize, cols: usize, at: usize| {
            mat(rows, cols, move |r, c| f64::from(u8::from(r == c + at)))
        };
        let (a1, b1) = (unit(2, 8, 0).transpose().transpose(), unit(8, 2, 0));
        let (a2, b2) = (
            mat(2, 8, |r, c| f64::from(u8::from(c == r + 4))),
            unit(8, 2, 4),
        );
        let (_, _, kept, total) = merge_module(&a1, &b1, &a2, &b2, 2);
        assert!((kept / total - 0.5).abs() < 1e-9, "{}", kept / total);
    }

    #[test]
    fn adapters_merge_through_their_files_at_their_rank() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("antumbra-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| AntumbraError::other(e.to_string()))?;
        let write = |name: &str, seed: u64| -> Result<String> {
            let path = dir.join(name).to_string_lossy().into_owned();
            let mut m = HashMap::new();
            for p in ["layers.0.q", "layers.0.v"] {
                m.insert(
                    format!("{p}.lora_a"),
                    to_tensor(&mat(4, 16, pseudo(seed)), DType::F32)?,
                );
                m.insert(
                    format!("{p}.lora_b"),
                    to_tensor(&mat(12, 4, pseudo(seed + 1)), DType::F32)?,
                );
            }
            candle_core::safetensors::save(&m, &path).map_err(ce)?;
            Ok(path)
        };
        let (a, b) = (write("a.safetensors", 5)?, write("b.safetensors", 9)?);
        let out = dir
            .join("merged.safetensors")
            .to_string_lossy()
            .into_owned();
        let report = merge_adapters(&a, &b, Some(&out))?;
        assert_eq!((report.rank, report.modules), (4, 2));
        assert!(report.retained > 0.0 && report.retained <= 1.0);
        let merged = candle_core::safetensors::load(&out, &Device::Cpu).map_err(ce)?;
        assert_eq!(merged["layers.0.q.lora_a"].dims(), [4, 16]);
        assert_eq!(merged["layers.0.q.lora_b"].dims(), [12, 4]);
        let itself = merge_adapters(&a, &a, None)?;
        assert!((itself.retained - 1.0).abs() < 1e-5);
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }
}

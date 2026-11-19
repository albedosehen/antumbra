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

use candle_core::safetensors::SliceSafetensors;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;

use antumbra_core::{AntumbraError, Result};

fn ce(e: candle_core::Error) -> AntumbraError {
    crate::candle_err("compose")(e)
}

/// `work` on each of `items`: what each made, in the items' order, or the
/// first refusal in that order. With `spread`, on as many threads as there
/// are cores, a run of the items to each. An expert's factors are some two
/// hundred tensors, and a request whose blend is not the resident one waits
/// while every one of them is read, stacked and set: on one thread, most of
/// a second. Without `spread`, on the calling thread, one after another:
/// what is on a GPU is made and set by the one thread, as it always was.
fn each<T: Sync, R: Send>(
    items: &[T],
    spread: bool,
    work: impl Fn(&T) -> Result<R> + Sync,
) -> Result<Vec<R>> {
    if !spread {
        return items.iter().map(work).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let per = items.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> = items
            .chunks(per)
            .map(|run| {
                let work = &work;
                scope.spawn(move || run.iter().map(work).collect::<Result<Vec<R>>>())
            })
            .collect();
        let mut all = Vec::with_capacity(items.len());
        for handle in handles {
            match handle.join() {
                Ok(made) => all.extend(made?),
                Err(_) => return Err(AntumbraError::other("compose: a thread panicked")),
            }
        }
        Ok(all)
    })
}

/// Each `(adapter_path, weight)` adapter's factors read from its file onto
/// `device`: into host memory, the files side by side.
pub fn load_parts(
    specs: &[(String, f32)],
    device: &Device,
) -> Result<Vec<(HashMap<String, Tensor>, f32)>> {
    each(specs, device.is_cpu(), |(path, weight)| {
        let factors = candle_core::safetensors::load(path, device)
            .map_err(|e| AntumbraError::other(format!("load adapter `{path}`: {e}")))?;
        Ok((factors, *weight))
    })
}

/// Merge `(adapter_path, weight)` specs into one rank-concatenated adapter at
/// `out_path`; returns the merged rank. `A` (`rank x in`) stacks along its rank
/// rows, `B` (`out x rank`) along its rank columns, each scaled by `sqrt(w_i)`.
pub fn compose_adapters(specs: &[(String, f32)], out_path: &str) -> Result<usize> {
    let parts = load_parts(specs, &Device::Cpu)?;
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

/// Restore one saved adapter into `vars` by name, each factor cast to its
/// variable's dtype (see [`set_factors`]).
pub fn restore_adapter(vars: &VarMap, path: &str) -> Result<()> {
    let what = format!("load adapter `{path}`");
    // A poisoned lock holds the same map of vars: a panic elsewhere while it
    // was held cannot leave one half-set, so it is taken as it is.
    let vars = vars
        .data()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(device) = vars.values().next().map(|v| v.device().clone()) else {
        return Ok(());
    };
    if set_straight(&vars, &[(path.to_string(), 1.0)])? {
        return Ok(());
    }
    let saved = candle_core::safetensors::load(path, &device)
        .map_err(|e| AntumbraError::other(format!("{what}: {e}")))?;
    set_factors(&vars, &saved, &what)
}

/// Set each of `vars` from the factor of its name in `factors`, cast to the
/// variable's dtype: an expert saved in bf16 on the GPU then loads into a
/// model held in f32, as every model on the CPU is, where candle's own
/// `VarMap::load` refuses the mismatch. `what` names the load in a refusal,
/// which a factor missing from `factors` is.
pub fn set_factors(
    vars: &HashMap<String, Var>,
    factors: &HashMap<String, Tensor>,
    what: &str,
) -> Result<()> {
    let fail = |e: candle_core::Error| AntumbraError::other(format!("{what}: {e}"));
    // Every factor is found before any is set, so a load that names a
    // missing one has set nothing.
    let named = vars
        .iter()
        .map(|(name, var)| {
            factors
                .get(name)
                .map(|t| (var, t))
                .ok_or_else(|| AntumbraError::other(format!("{what}: no factor {name}")))
        })
        .collect::<Result<Vec<(&Var, &Tensor)>>>()?;
    let in_host_memory = named
        .iter()
        .all(|(var, t)| var.device().is_cpu() && t.device().is_cpu());
    each(&named, in_host_memory, |(var, t)| {
        var.set(&t.to_dtype(var.dtype()).map_err(fail)?)
            .map_err(fail)
    })?;
    Ok(())
}

/// Set `vars`, a model's LoRA factors, to the blend of the
/// `(adapter_path, weight)` adapters `specs`: what [`blend`] makes of them at
/// the vars' own rank and [`set_factors`] then sets, value for value.
pub fn set_blend(vars: &HashMap<String, Var>, specs: &[(String, f32)]) -> Result<()> {
    let Some(device) = vars.values().next().map(|v| v.device().clone()) else {
        return Err(AntumbraError::other(
            "load blend: the model has no LoRA factors",
        ));
    };
    if set_straight(vars, specs)? {
        return Ok(());
    }
    let rank = vars
        .iter()
        .find(|(k, _)| k.ends_with("lora_a"))
        .map(|(_, v)| v.dim(0).unwrap_or(0))
        .unwrap_or(0);
    let parts = load_parts(specs, &device)?;
    let blended = blend(&parts, Some(rank))?;
    set_factors(vars, &blended, "load blend")
}

/// One adapter's part of one factor, as its file holds it.
struct Piece<'a> {
    /// The factor's values, little-endian: `rank` rows of `A`, or a row of
    /// `rank` for each row of `B`.
    bytes: &'a [u8],
    /// Whether they are bf16; f32 otherwise.
    bf16: bool,
    rank: usize,
    /// The root of the adapter's weight.
    scale: f32,
}

/// One of a model's factors and the pieces a blend sets it from.
struct Plan<'a> {
    var: &'a Var,
    /// Whether it is an `A`, whose rank is its rows; a `B`'s is its columns.
    is_a: bool,
    rank: usize,
    /// Its other dimension: an `A`'s columns, a `B`'s rows.
    other: usize,
    pieces: Vec<Piece<'a>>,
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// The nearest bf16 to `value`, ties to even: the rounding candle's bf16
/// arithmetic ends every product with.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        // Not a number stays one.
        return ((bits >> 16) | 0x0040) as u16;
    }
    let half = 0x0000_8000u32;
    let up = bits & half != 0 && bits & (3 * half - 1) != 0;
    (bits >> 16) as u16 + u16::from(up)
}

/// `bytes` as f32 values into `to`, each times `piece`'s scale as candle's
/// own arithmetic on that dtype gives it: a bf16 factor times the scale held
/// in bf16, rounded to bf16; an f32 one times the scale. A scale of one is
/// the value as it is.
fn fill(to: &mut [f32], bytes: &[u8], piece: &Piece) {
    let whole = piece.scale == 1.0;
    if piece.bf16 {
        let by = from_bf16(to_bf16(piece.scale));
        for (to, b) in to.iter_mut().zip(bytes.chunks_exact(2)) {
            let value = from_bf16(u16::from_le_bytes([b[0], b[1]]));
            // The nothing added is candle's too: it leaves a product of
            // minus nothing as plain nothing.
            *to = match whole {
                true => value,
                false => from_bf16(to_bf16(value * by)) + 0.0,
            };
        }
    } else {
        for (to, b) in to.iter_mut().zip(bytes.chunks_exact(4)) {
            let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            *to = match whole {
                true => value,
                false => value * piece.scale + 0.0,
            };
        }
    }
}

/// [`set_blend`] where each factor can go from its file's bytes straight to
/// its place in the model: every var is f32 in host memory, as every model on
/// a CPU is, and every adapter holds every factor in bf16 or f32 at a rank
/// that fits, as every expert the trainer saves does. `false`, with nothing
/// set, where that does not hold: the caller goes through tensors, which
/// also says what is wrong with an adapter that is.
///
/// Through tensors a factor is read into one, scaled into another, stacked
/// into a third, padded into a fourth, cast into a fifth and copied into the
/// model, and each of those is as large as the model's own. A request whose
/// blend is not the resident one waits for all of it: here a factor's values
/// are read once and written once, on every core.
fn set_straight(vars: &HashMap<String, Var>, specs: &[(String, f32)]) -> Result<bool> {
    if specs.is_empty()
        || !vars
            .values()
            .all(|v| v.device().is_cpu() && v.dtype() == DType::F32)
    {
        return Ok(false);
    }
    let Ok(files) = specs
        .iter()
        .map(|(path, _)| std::fs::read(path))
        .collect::<std::io::Result<Vec<Vec<u8>>>>()
    else {
        return Ok(false);
    };
    let Ok(opened) = files
        .iter()
        .map(|bytes| SliceSafetensors::new(bytes))
        .collect::<candle_core::Result<Vec<_>>>()
    else {
        return Ok(false);
    };
    let mut plans = Vec::with_capacity(vars.len());
    for (name, var) in vars {
        let is_a = name.ends_with("lora_a");
        let &[rows, cols] = var.dims() else {
            return Ok(false);
        };
        let (rank, other) = if is_a { (rows, cols) } else { (cols, rows) };
        let mut pieces = Vec::with_capacity(specs.len());
        for (file, (_, weight)) in opened.iter().zip(specs) {
            let Ok(view) = file.get(name) else {
                return Ok(false);
            };
            let bf16 = match DType::try_from(view.dtype()) {
                Ok(DType::BF16) => true,
                Ok(DType::F32) => false,
                _ => return Ok(false),
            };
            let &[rows, cols] = view.shape() else {
                return Ok(false);
            };
            let (part, part_other) = if is_a { (rows, cols) } else { (cols, rows) };
            let bytes = view.data();
            if part_other != other || bytes.len() != rows * cols * if bf16 { 2 } else { 4 } {
                return Ok(false);
            }
            pieces.push(Piece {
                bytes,
                bf16,
                rank: part,
                scale: weight.max(0.0).sqrt(),
            });
        }
        if pieces.iter().map(|p| p.rank).sum::<usize>() > rank {
            return Ok(false);
        }
        plans.push(Plan {
            var,
            is_a,
            rank,
            other,
            pieces,
        });
    }
    each(&plans, true, |plan| {
        // What no adapter fills stays at nothing.
        let mut values = vec![0f32; plan.rank * plan.other];
        let mut at = 0;
        for piece in &plan.pieces {
            let width = if piece.bf16 { 2 } else { 4 };
            if plan.is_a {
                // Its rows of `A`, one after another.
                let to = &mut values[at * plan.other..(at + piece.rank) * plan.other];
                fill(to, piece.bytes, piece);
            } else if piece.rank > 0 {
                // Its columns of each row of `B`.
                for (row, bytes) in values
                    .chunks_exact_mut(plan.rank)
                    .zip(piece.bytes.chunks_exact(piece.rank * width))
                {
                    fill(&mut row[at..at + piece.rank], bytes, piece);
                }
            }
            at += piece.rank;
        }
        let (rows, cols) = match plan.is_a {
            true => (plan.rank, plan.other),
            false => (plan.other, plan.rank),
        };
        let values = Tensor::from_vec(values, (rows, cols), &Device::Cpu).map_err(ce)?;
        plan.var.set(&values).map_err(ce)
    })?;
    Ok(true)
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
    let keys: Vec<&String> = first.keys().collect();
    let in_host_memory = parts
        .iter()
        .all(|(factors, _)| factors.values().all(|t| t.device().is_cpu()));
    let stacked = each(&keys, in_host_memory, |key| stack(parts, key, rank))?;
    Ok(keys.into_iter().cloned().zip(stacked).collect())
}

/// One factor of a blend: every part's factor `key`, each scaled, stacked
/// and padded to `rank` as [`blend`] says.
fn stack(
    parts: &[(HashMap<String, Tensor>, f32)],
    key: &str,
    rank: Option<usize>,
) -> Result<Tensor> {
    // The adapter holds only LoRA factors. A: stack rows (rank dim 0);
    // B: stack cols (rank dim 1). Both fold sqrt(w) so the product is w.
    let dim = if key.ends_with("lora_a") { 0 } else { 1 };
    let mut stacked = Vec::with_capacity(parts.len() + 1);
    for (factors, w) in parts {
        let t = factors
            .get(key)
            .ok_or_else(|| AntumbraError::other(format!("compose: key {key} missing")))?;
        let scale = w.max(0.0).sqrt();
        // A weight of one is the factor as it stands, with no pass over it.
        stacked.push(match scale == 1.0 {
            true => t.clone(),
            false => (t * scale as f64).map_err(ce)?,
        });
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
    Ok(t)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    /// The path as the adapter functions take it.
    fn utf8(path: &Path) -> Result<&str> {
        path.to_str()
            .ok_or_else(|| AntumbraError::other("the temp path is not UTF-8"))
    }

    fn write_adapter(
        path: &Path,
        a: &[f32],
        b: &[f32],
        rank: usize,
        inn: usize,
        out: usize,
    ) -> Result<()> {
        let device = Device::Cpu;
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "p.lora_a".into(),
            Tensor::from_vec(a.to_vec(), (rank, inn), &device).map_err(ce)?,
        );
        m.insert(
            "p.lora_b".into(),
            Tensor::from_vec(b.to_vec(), (out, rank), &device).map_err(ce)?,
        );
        candle_core::safetensors::save(&m, path).map_err(ce)
    }

    #[test]
    fn concatenation_realizes_the_weighted_delta_sum() -> Result<()> {
        let dir = std::env::temp_dir();
        let (p1, p2, pm) = (
            dir.join("ant_c1.safetensors"),
            dir.join("ant_c2.safetensors"),
            dir.join("ant_cm.safetensors"),
        );
        // rank 1, in 2, out 1. delta_i = b_i (a_i . x).
        write_adapter(&p1, &[1.0, 0.0], &[2.0], 1, 2, 1)?;
        write_adapter(&p2, &[0.0, 1.0], &[3.0], 1, 2, 1)?;
        let (w1, w2) = (0.25f32, 0.75f32);
        let rank = compose_adapters(
            &[(utf8(&p1)?.into(), w1), (utf8(&p2)?.into(), w2)],
            utf8(&pm)?,
        )?;
        assert_eq!(rank, 2); // 1 + 1

        // Merged delta on x = [x0, x1]: w1*2*x0 + w2*3*x1. Check via B*(A*x).
        let device = Device::Cpu;
        let m = candle_core::safetensors::load(&pm, &device).map_err(ce)?;
        let a = &m["p.lora_a"]; // (2, 2)
        let b = &m["p.lora_b"]; // (1, 2)
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (2, 1), &device).map_err(ce)?;
        let delta = b.matmul(&a.matmul(&x).map_err(ce)?).map_err(ce)?;
        let got = delta
            .flatten_all()
            .map_err(ce)?
            .to_vec1::<f32>()
            .map_err(ce)?[0];
        let want = w1 * 2.0 * 1.0 + w2 * 3.0 * 1.0;
        assert!((got - want).abs() < 1e-5, "got {got} want {want}");

        for p in [p1, p2, pm] {
            std::fs::remove_file(p).ok();
        }
        Ok(())
    }

    #[test]
    fn set_factors_casts_each_factor_to_its_var_and_names_a_missing_one() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarMap::new();
        vars.get(
            (1, 2),
            "p.lora_a",
            candle_nn::Init::Const(0.0),
            candle_core::DType::F32,
            &device,
        )
        .map_err(ce)?;
        vars.get(
            (1, 1),
            "p.lora_b",
            candle_nn::Init::Const(0.0),
            candle_core::DType::F32,
            &device,
        )
        .map_err(ce)?;
        let held = vars
            .data()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Factors saved in bf16, as a GPU-trained expert's are.
        let mut factors: HashMap<String, Tensor> = HashMap::new();
        factors.insert(
            "p.lora_a".into(),
            Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &device)
                .map_err(ce)?
                .to_dtype(candle_core::DType::BF16)
                .map_err(ce)?,
        );
        let missing_b = set_factors(&held, &factors, "load test");
        let Err(e) = missing_b else {
            return Err(AntumbraError::other("a missing factor was accepted"));
        };
        if e.to_string() != "load test: no factor p.lora_b" {
            return Err(AntumbraError::other(format!("the wrong refusal: {e}")));
        }
        factors.insert(
            "p.lora_b".into(),
            Tensor::from_vec(vec![3.0f32], (1, 1), &device)
                .map_err(ce)?
                .to_dtype(candle_core::DType::BF16)
                .map_err(ce)?,
        );
        set_factors(&held, &factors, "load test")?;
        let Some(a) = held.get("p.lora_a") else {
            return Err(AntumbraError::other("p.lora_a vanished"));
        };
        if a.dtype() != candle_core::DType::F32 {
            return Err(AntumbraError::other(
                "the factor was not cast to the var's dtype",
            ));
        }
        let read = a.as_tensor().to_vec2::<f32>().map_err(ce)?;
        if read != vec![vec![1.0, 2.0]] {
            return Err(AntumbraError::other(format!(
                "the factor was not set: {read:?}"
            )));
        }
        Ok(())
    }

    #[test]
    fn rejects_a_rank_zero_adapter() -> Result<()> {
        let dir = std::env::temp_dir();
        let (p, pm) = (
            dir.join("ant_rank0.safetensors"),
            dir.join("ant_rank0_m.safetensors"),
        );
        // lora_a is (0, 2): zero rank rows. A composed rank-0 adapter would scale
        // to a NaN alpha/rank and poison generation, so compose must refuse it.
        write_adapter(&p, &[], &[], 0, 2, 1)?;
        let err = compose_adapters(&[(utf8(&p)?.into(), 1.0)], utf8(&pm)?);
        assert!(err.is_err(), "rank-0 compose must be rejected");
        assert!(
            !pm.exists(),
            "a rejected compose must not leave a garbage adapter on disk"
        );
        for q in [p, pm] {
            std::fs::remove_file(q).ok();
        }
        Ok(())
    }

    fn factors(
        a: &[f32],
        b: &[f32],
        rank: usize,
        inn: usize,
        out: usize,
    ) -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        Ok(HashMap::from([
            (
                "p.lora_a".to_string(),
                Tensor::from_vec(a.to_vec(), (rank, inn), &device).map_err(ce)?,
            ),
            (
                "p.lora_b".to_string(),
                Tensor::from_vec(b.to_vec(), (out, rank), &device).map_err(ce)?,
            ),
        ]))
    }

    /// `B (A x)` for `x = [1, 2]`.
    fn delta(f: &HashMap<String, Tensor>) -> Result<f32> {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &Device::Cpu).map_err(ce)?;
        let ax = f["p.lora_a"].matmul(&x).map_err(ce)?;
        Ok(f["p.lora_b"]
            .matmul(&ax)
            .map_err(ce)?
            .flatten_all()
            .map_err(ce)?
            .to_vec1::<f32>()
            .map_err(ce)?[0])
    }

    #[test]
    fn one_adapter_padded_to_a_higher_rank_serves_the_same_delta() -> Result<()> {
        let one = factors(&[1.0, 3.0], &[2.0], 1, 2, 1)?;
        let padded = blend(&[(one.clone(), 1.0)], Some(3))?;
        assert_eq!(padded["p.lora_a"].dims(), &[3, 2]);
        assert_eq!(padded["p.lora_b"].dims(), &[1, 3]);
        assert_eq!(rank_of(&padded), 3);
        assert!((delta(&padded)? - delta(&one)?).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn a_padded_blend_is_the_weighted_sum_and_halves_make_a_whole() -> Result<()> {
        let p = factors(&[1.0, 0.0], &[2.0], 1, 2, 1)?;
        let q = factors(&[0.0, 1.0], &[3.0], 1, 2, 1)?;
        let mixed = blend(&[(p.clone(), 0.25), (q.clone(), 1.0)], Some(4))?;
        assert!((delta(&mixed)? - (0.25 * delta(&p)? + delta(&q)?)).abs() < 1e-5);
        let halves = blend(&[(p.clone(), 0.5), (p.clone(), 0.5)], Some(3))?;
        assert!((delta(&halves)? - delta(&p)?).abs() < 1e-5);
        Ok(())
    }

    /// Work on each item comes back in the items' order, spread over threads
    /// or not, and so does the first refusal.
    #[test]
    fn work_on_each_item_keeps_the_items_order() -> Result<()> {
        let items: Vec<usize> = (0..100).collect();
        for spread in [false, true] {
            let doubled = each(&items, spread, |i| Ok(i * 2))?;
            assert_eq!(doubled, (0..100).map(|i| i * 2).collect::<Vec<_>>());
            let refused = each(&items, spread, |i| match *i >= 40 {
                true => Err(AntumbraError::other(format!("no {i}"))),
                false => Ok(*i),
            });
            assert!(
                refused.is_err_and(|e| e.to_string().contains("no 40")),
                "spread {spread}: the first refusal in the items' order"
            );
            assert!(each(&[] as &[usize], spread, |i| Ok(*i))?.is_empty());
        }
        Ok(())
    }

    /// A blend of many factors is each factor's own stack, whichever thread
    /// stacked it: each part scaled by the root of its weight, a part at a
    /// weight of one as it stands, and the rest of the rank at nothing.
    #[test]
    fn many_factors_are_each_stacked_as_one_is() -> Result<()> {
        let device = Device::Cpu;
        let part = |seed: f32| -> Result<HashMap<String, Tensor>> {
            let mut all = HashMap::new();
            for i in 0..40 {
                let a: Vec<f32> = (0..6).map(|j| seed + i as f32 + j as f32 * 0.5).collect();
                let b: Vec<f32> = (0..4).map(|j| seed - i as f32 * 0.25 + j as f32).collect();
                all.insert(
                    format!("l{i}.lora_a"),
                    Tensor::from_vec(a, (2, 3), &device).map_err(ce)?,
                );
                all.insert(
                    format!("l{i}.lora_b"),
                    Tensor::from_vec(b, (2, 2), &device).map_err(ce)?,
                );
            }
            Ok(all)
        };
        let flat = |t: &Tensor| -> Result<Vec<f32>> {
            t.flatten_all().map_err(ce)?.to_vec1::<f32>().map_err(ce)
        };
        let (p, q) = (part(1.0)?, part(-3.0)?);
        let mixed = blend(&[(p.clone(), 1.0), (q.clone(), 0.25)], Some(6))?;
        assert_eq!(mixed.len(), 80);
        for (key, got) in &mixed {
            let dim = if key.ends_with("lora_a") { 0 } else { 1 };
            let halved = (&q[key] * 0.5).map_err(ce)?;
            let want = Tensor::cat(&[p[key].clone(), halved], dim)
                .map_err(ce)?
                .pad_with_zeros(dim, 0, 2)
                .map_err(ce)?;
            assert_eq!(got.dims(), want.dims(), "{key}");
            assert_eq!(flat(got)?, flat(&want)?, "{key}");
        }
        Ok(())
    }

    /// Two projections' zeroed f32 factors at `rank`, as a CPU model holds
    /// them.
    fn two_projections(rank: usize, inn: usize, out: usize) -> Result<VarMap> {
        let vars = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        for name in ["x", "y"] {
            vb.get_with_hints(
                (rank, inn),
                &format!("{name}.lora_a"),
                candle_nn::init::ZERO,
            )
            .map_err(ce)?;
            vb.get_with_hints(
                (out, rank),
                &format!("{name}.lora_b"),
                candle_nn::init::ZERO,
            )
            .map_err(ce)?;
        }
        Ok(vars)
    }

    /// Every factor in `vars` as its name and its values' bits.
    fn bits(vars: &VarMap) -> Result<Vec<(String, Vec<u32>)>> {
        let held = vars
            .data()
            .lock()
            .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
        let mut all = Vec::new();
        for (name, var) in held.iter() {
            let values = var
                .as_tensor()
                .flatten_all()
                .map_err(ce)?
                .to_vec1::<f32>()
                .map_err(ce)?;
            all.push((name.clone(), values.iter().map(|v| v.to_bits()).collect()));
        }
        all.sort();
        Ok(all)
    }

    /// A blend set straight from its adapters' files is the blend set
    /// through tensors, to the bit: for adapters saved in bf16 and in f32,
    /// at a weight of one and at others, one adapter and three, at the
    /// model's rank and padded to it. What the straight way does not take
    /// (an adapter saved in f16, a blend above the model's rank) it leaves
    /// alone, and the way through tensors sets it or says why not.
    #[test]
    fn a_blend_set_straight_from_its_files_is_the_one_set_through_tensors() -> Result<()> {
        let dir = std::env::temp_dir();
        let (rank, inn, out) = (2usize, 7usize, 5usize);
        let write = |name: &str, seed: f32, dtype: DType| -> Result<String> {
            let path = dir.join(name);
            let mut all = HashMap::new();
            for (at, projection) in ["x", "y"].iter().enumerate() {
                let value = |i: usize| ((i as f32 + seed + at as f32 * 9.0) * 0.37).sin() * 0.05;
                let a: Vec<f32> = (0..rank * inn).map(value).collect();
                let b: Vec<f32> = (0..out * rank).map(|i| value(i + 100)).collect();
                for (factor, values, shape) in
                    [("lora_a", a, (rank, inn)), ("lora_b", b, (out, rank))]
                {
                    all.insert(
                        format!("{projection}.{factor}"),
                        Tensor::from_vec(values, shape, &Device::Cpu)
                            .map_err(ce)?
                            .to_dtype(dtype)
                            .map_err(ce)?,
                    );
                }
            }
            candle_core::safetensors::save(&all, &path).map_err(ce)?;
            Ok(utf8(&path)?.to_string())
        };
        let through_tensors = |vars: &VarMap, specs: &[(String, f32)], rank: usize| -> Result<()> {
            let held = vars
                .data()
                .lock()
                .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
            let blended = blend(&load_parts(specs, &Device::Cpu)?, Some(rank))?;
            set_factors(&held, &blended, "load blend")
        };
        let straight = |vars: &VarMap, specs: &[(String, f32)]| -> Result<bool> {
            let held = vars
                .data()
                .lock()
                .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
            set_straight(&held, specs)
        };
        let mut written = Vec::new();
        for (dtype, tag) in [(DType::BF16, "bf16"), (DType::F32, "f32")] {
            let files = [
                write(&format!("ant_straight_{tag}_1.safetensors"), 1.0, dtype)?,
                write(&format!("ant_straight_{tag}_2.safetensors"), 2.0, dtype)?,
                write(&format!("ant_straight_{tag}_3.safetensors"), 3.0, dtype)?,
            ];
            for (weights, model_rank) in [
                (vec![1.0f32], 2usize),
                (vec![1.0], 6),
                (vec![0.4, 1.0, 0.3], 6),
                (vec![0.5, 0.25], 6),
                (vec![0.0, 0.7], 4),
            ] {
                let specs: Vec<(String, f32)> = files.iter().cloned().zip(weights).collect();
                let (one, other) = (
                    two_projections(model_rank, inn, out)?,
                    two_projections(model_rank, inn, out)?,
                );
                assert!(
                    straight(&one, &specs)?,
                    "{tag}: {specs:?} was not set straight"
                );
                through_tensors(&other, &specs, model_rank)?;
                assert_eq!(
                    bits(&one)?,
                    bits(&other)?,
                    "{tag}: {specs:?} at rank {model_rank}"
                );
            }
            // Three adapters of rank 2 do not fit a model of rank 4.
            let specs: Vec<(String, f32)> = files.iter().cloned().map(|f| (f, 1.0)).collect();
            let small = two_projections(4, inn, out)?;
            assert!(!straight(&small, &specs)?);
            let held = small
                .data()
                .lock()
                .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
            assert!(
                set_blend(&held, &specs).is_err(),
                "a blend above the model's rank"
            );
            written.extend(files);
        }
        // An adapter in f16 is not set straight, and is set all the same.
        let half = write("ant_straight_f16.safetensors", 4.0, DType::F16)?;
        let specs = vec![(half.clone(), 0.5f32)];
        let (one, other) = (two_projections(6, inn, out)?, two_projections(6, inn, out)?);
        assert!(!straight(&one, &specs)?);
        {
            let held = one
                .data()
                .lock()
                .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
            set_blend(&held, &specs)?;
        }
        through_tensors(&other, &specs, 6)?;
        assert_eq!(bits(&one)?, bits(&other)?);
        written.push(half);
        for file in written {
            std::fs::remove_file(file).ok();
        }
        Ok(())
    }

    #[test]
    fn a_blend_above_the_model_rank_or_missing_a_factor_is_refused() -> Result<()> {
        let p = factors(&[1.0, 0.0], &[2.0], 1, 2, 1)?;
        assert!(blend(&[(p.clone(), 1.0), (p.clone(), 1.0)], Some(1)).is_err());
        let mut partial = p.clone();
        partial.remove("p.lora_b");
        assert!(blend(&[(p, 1.0), (partial, 1.0)], Some(2)).is_err());
        assert!(blend(&[], Some(2)).is_err());
        Ok(())
    }

    /// A model's LoRA factors as the CPU holds them: zeroed f32 `Var`s.
    fn f32_vars(rank: usize, inn: usize, out: usize) -> Result<VarMap> {
        let vars = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vars, candle_core::DType::F32, &Device::Cpu);
        vb.get_with_hints((rank, inn), "p.lora_a", candle_nn::init::ZERO)
            .map_err(ce)?;
        vb.get_with_hints((out, rank), "p.lora_b", candle_nn::init::ZERO)
            .map_err(ce)?;
        Ok(vars)
    }

    #[test]
    fn a_bf16_adapter_restores_into_f32_factors() -> Result<()> {
        // Every value is exact in bf16, so the cast must reproduce it.
        let path = std::env::temp_dir().join("ant_restore_bf16.safetensors");
        let saved = factors(&[0.5, -1.25], &[3.0], 1, 2, 1)?
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_dtype(candle_core::DType::BF16).map_err(ce)?)))
            .collect::<Result<HashMap<String, Tensor>>>()?;
        candle_core::safetensors::save(&saved, &path).map_err(ce)?;
        let vars = f32_vars(1, 2, 1)?;
        restore_adapter(&vars, utf8(&path)?)?;
        let data = vars
            .data()
            .lock()
            .map_err(|_| AntumbraError::other("the varmap lock was poisoned"))?;
        let a = data["p.lora_a"].as_tensor();
        assert_eq!(a.dtype(), candle_core::DType::F32);
        assert_eq!(
            a.flatten_all().map_err(ce)?.to_vec1::<f32>().map_err(ce)?,
            [0.5, -1.25]
        );
        let b = data["p.lora_b"].as_tensor();
        assert_eq!(
            b.flatten_all().map_err(ce)?.to_vec1::<f32>().map_err(ce)?,
            [3.0]
        );
        std::fs::remove_file(path).ok();
        Ok(())
    }

    #[test]
    fn an_adapter_missing_a_factor_names_it() -> Result<()> {
        let path = std::env::temp_dir().join("ant_restore_partial.safetensors");
        let mut partial = factors(&[0.5, -1.25], &[3.0], 1, 2, 1)?;
        partial.remove("p.lora_b");
        candle_core::safetensors::save(&partial, &path).map_err(ce)?;
        let Err(err) = restore_adapter(&f32_vars(1, 2, 1)?, utf8(&path)?) else {
            return Err(AntumbraError::other("a partial adapter was restored"));
        };
        assert!(err.to_string().contains("p.lora_b"), "{err}");
        std::fs::remove_file(path).ok();
        Ok(())
    }
}

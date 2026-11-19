//! One decoded token on the CPU through plain rows, with no tensor made
//! between its products.
//!
//! A decoded token is one row through every layer: 113 products of quantized
//! weights, and between them two norms, a rotation, attention against the
//! cache, the MLP's gate and two additions a layer. As tensor ops each of
//! those small steps allocates its result, takes a lock on its inputs,
//! checks shapes and wraps a row in a tensor to hand it on: more time than
//! the arithmetic, on one thread, while the cores that multiply wait and the
//! memory they read from sits idle. A served base on a CPU is bound by how
//! fast its weights are read, so whatever one thread spends between
//! products comes straight out of the rate.
//!
//! Here the row stays a plain slice from the embedding to the logits, in
//! buffers kept from token to token. A product and its adapter are one
//! dispatch of two phases (`phases`): the adapter's `A · x`, then each run
//! of the weight's rows through its tiles (`tiles`) with the adapter's `B`
//! rows and the bias added by the thread that multiplied them. The MLP is
//! one dispatch: its two wide products, then the gate and the quantizing of
//! its result a block at a time by the cores that hold those values, then
//! its narrow product. So the MLP's wide rows never reach the calling
//! thread, where pulling them a cache line at a time from eight cores and
//! handing the gated row back out cost more than the arithmetic. Attention
//! reads the cache where it lies (`attend`).
//!
//! It computes what the tensor path computes, to the order of a few float
//! sums, and gives the same values on any number of threads. It is taken
//! only where the tensor path would have computed without a gradient: a
//! served base in f32 on a CPU that multiplies through tiles, generating one
//! sequence. A prompt's rows, training, every GPU and every other CPU keep
//! the tensor ops, and a sequence may pass from one to the other: both write
//! the same cache.

use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::{bail, Layout, Storage};

use super::*;
use phases::{dispatch, Phase, Shared};
use slices::dot;

/// The rows a decoded token is computed in, kept from one token to the next.
#[derive(Default)]
pub(in crate::models::qwen) struct Scratch {
    /// The hidden row, from the embedding to the last layer.
    x: Vec<f32>,
    /// That row normed: what a layer's products read.
    normed: Vec<f32>,
    /// A layer's query, keys and values, one after another.
    qkv: Vec<f32>,
    attended: Vec<f32>,
    /// The MLP's gate and the side it gates, one after the other.
    both: Vec<f32>,
    /// What a layer's attention, then its MLP, adds to the hidden row.
    out: Vec<f32>,
    /// For each block of the MLP's gated row, the down adapter's `A` times
    /// that block.
    parts: Vec<f32>,
    /// The room attention works in.
    attention: attend::Work,
    /// The quantized input of the product in hand, and the MLP's gated row
    /// quantized.
    row: tiles::Row,
    gated: tiles::Row,
}

/// The f32 values of a tensor whose storage is held for reading.
fn held<'a>(storage: &'a Storage, layout: &Layout, what: &str) -> CResult<&'a [f32]> {
    let Storage::Cpu(storage) = storage else {
        bail!("{what} is not in host memory")
    };
    slices::values(storage, layout, what)
}

/// Whether `tensor` is plain f32 rows in host memory.
fn plain(tensor: &Tensor) -> bool {
    tensor.device().is_cpu() && tensor.dtype() == DType::F32
}

/// What a decoded token reads of one projection: its tiles, its bias, and
/// its adapter's factors when the adapter is on.
struct Reads<'a> {
    tiled: &'a tiles::Tiled,
    bias: Option<&'a Tensor>,
    factors: Option<(&'a Tensor, &'a Tensor)>,
    rank: usize,
    scale: f32,
}

impl LoraLinear {
    /// Whether a decoded token can read this projection as plain rows: its
    /// base is tiled, and its adapter is off or read straight from its
    /// factors.
    fn steps(&self) -> bool {
        matches!(self.base, BaseWeight::Tiled(_))
            && (!self.enabled || self.reads_directly())
            && self.base_b.as_ref().is_none_or(plain)
    }

    fn reads(&self) -> CResult<Reads<'_>> {
        let BaseWeight::Tiled(tiled) = &self.base else {
            bail!("a projection that is not tiled was stepped through")
        };
        Ok(Reads {
            tiled,
            bias: self.base_b.as_ref(),
            factors: self.enabled.then_some((&self.a, &self.b)),
            rank: self.a.dim(0)?,
            scale: self.scale as f32,
        })
    }
}

/// One projection as the units of a dispatch read it: its tiles, and as
/// plain rows its bias and its adapter's factors (`a` is `rank` rows of the
/// input's width, `b` a row of `rank` for each product; no rank, no adapter).
struct Weight<'a> {
    tiled: &'a tiles::Tiled,
    bias: Option<&'a [f32]>,
    a: &'a [f32],
    b: &'a [f32],
    rank: usize,
    scale: f32,
}

/// The storage of each projection's factors and bias, held while a dispatch
/// reads them: nothing is copied, so a loaded or blended adapter is what the
/// next token uses.
macro_rules! hold {
    ($reads:expr) => {
        $reads
            .iter()
            .map(|r| {
                (
                    r.factors
                        .map(|(a, b)| (a.storage_and_layout(), b.storage_and_layout())),
                    r.bias.map(Tensor::storage_and_layout),
                )
            })
            .collect::<Vec<_>>()
    };
}

/// Each projection in `reads` as plain rows, out of what `hold!` holds for
/// them: the projection reads rows of `cols` values.
macro_rules! weights {
    ($reads:expr, $held:expr, $cols:expr) => {{
        let mut weights = Vec::with_capacity($reads.len());
        for (r, (factors, bias)) in $reads.iter().zip($held.iter()) {
            let (a, b, rank) = match factors {
                Some(((a_storage, a_layout), (b_storage, b_layout))) => (
                    held(a_storage, a_layout, "lora_a")?,
                    held(b_storage, b_layout, "lora_b")?,
                    r.rank,
                ),
                None => (&[][..], &[][..], 0),
            };
            let bias = match bias {
                Some((storage, layout)) => Some(held(storage, layout, "a projection's bias")?),
                None => None,
            };
            let rows = r.tiled.rows();
            if r.tiled.cols() != $cols
                || a.len() != rank * $cols
                || b.len() != rows * rank
                || bias.is_some_and(|bias| bias.len() != rows)
            {
                bail!(
                    "a projection of {rows} by {} with an adapter of rank {rank} ({} and {} values) does not fit a row of {}",
                    r.tiled.cols(),
                    a.len(),
                    b.len(),
                    $cols
                )
            }
            weights.push(Weight {
                tiled: r.tiled,
                bias,
                a,
                b,
                rank,
                scale: r.scale,
            });
        }
        weights
    }};
}

/// Rank rows of an adapter's `A` a claim covers.
const RANK_ROWS: usize = 4;

/// Room for each weight's `A · x`, one weight's values after another's, and
/// where each weight's start.
fn through(weights: &[Weight]) -> (Vec<AtomicU32>, Vec<usize>) {
    let starts: Vec<usize> = weights
        .iter()
        .scan(0, |at, w| {
            let start = *at;
            *at += w.rank;
            Some(start)
        })
        .collect();
    let ranks: usize = weights.iter().map(|w| w.rank).sum();
    ((0..ranks).map(|_| AtomicU32::new(0)).collect(), starts)
}

/// The values `first..last` of the adapters' `A · x`, each times its
/// adapter's scale: value `j` is a rank row of the weight it falls in.
fn a_rows(
    weights: &[Weight],
    x: &[f32],
    starts: &[usize],
    t: &[AtomicU32],
    first: usize,
    last: usize,
) {
    if x.is_empty() {
        return;
    }
    for (w, start) in weights.iter().zip(starts) {
        let rows = w.a.chunks_exact(x.len()).zip(&t[*start..]);
        for (j, (row, value)) in rows.enumerate() {
            if (first..last).contains(&(start + j)) {
                value.store((dot(row, x) * w.scale).to_bits(), Ordering::Relaxed);
            }
        }
    }
}

/// One weight's `A · x` as the `B` rows read it.
fn read(t: &[AtomicU32]) -> Vec<f32> {
    t.iter()
        .map(|v| f32::from_bits(v.load(Ordering::Relaxed)))
        .collect()
}

/// The sixteens `first..last` of `w`'s rows: their products with `row`, the
/// adapter's delta and the bias, written to `out`.
///
/// SAFETY (for the caller): `row` holds `w`'s `cols` values, `out` is room
/// for its `rows` products, and no other thread writes these sixteens.
unsafe fn rows_of(
    w: &Weight,
    first: usize,
    last: usize,
    row: &tiles::Row,
    t: &[f32],
    out: &Shared<f32>,
) {
    w.tiled.multiply_sixteens(first, last, row, out.at());
    let (first, last) = (first * tiles::ROWS, last * tiles::ROWS);
    let out = out.part(first, last - first);
    if w.rank > 0 {
        let b = &w.b[first * w.rank..last * w.rank];
        for (o, b_row) in out.iter_mut().zip(b.chunks_exact(w.rank)) {
            *o += dot(b_row, t);
        }
    }
    if let Some(bias) = w.bias {
        for (o, b) in out.iter_mut().zip(&bias[first..last]) {
            *o += b;
        }
    }
}

/// Every projection in `group` applied to the one row `x`, each into its own
/// slice of `outs`: what [`LoraLinear::forward_group`] gives a decoded row.
/// The row is quantized once for the group, and the group is one dispatch:
/// the adapters' `A · x` where any is on, then the bases' rows with the
/// adapters' `B` rows and the biases.
fn products<const N: usize>(
    group: [&LoraLinear; N],
    x: &[f32],
    mut outs: [&mut [f32]; N],
    row: &mut tiles::Row,
) -> CResult<()> {
    let reads: Vec<Reads> = group.iter().map(|l| l.reads()).collect::<CResult<_>>()?;
    let held = hold!(reads);
    let weights = weights!(reads, held, x.len());
    if weights
        .iter()
        .zip(outs.iter())
        .any(|(w, out)| w.tiled.rows() != out.len())
    {
        bail!("a group of projections and the places for their products do not fit")
    }
    tiles::quantize_into(x, row);
    let row = &*row;

    let (t, starts) = through(&weights);
    let a_phase = |first: usize, last: usize| a_rows(&weights, x, &starts, &t, first, last);
    let shared: Vec<Shared<f32>> = outs.iter_mut().map(|out| Shared::of(out)).collect();
    let sixteens: usize = weights.iter().map(|w| w.tiled.sixteens()).sum();
    // The group's sixteens, one weight's after another's.
    let rows_phase = |first: usize, last: usize| {
        let mut before = 0;
        for ((w, out), start) in weights.iter().zip(&shared).zip(&starts) {
            let n = w.tiled.sixteens();
            let (from, to) = (first.max(before), last.min(before + n));
            if from < to {
                let t = read(&t[*start..start + w.rank]);
                // SAFETY: `weights!` checked the row and each adapter against
                // its weight, each place for products against its weight
                // just above, and a unit is done by one thread.
                unsafe { rows_of(w, from - before, to - before, row, &t, out) };
            }
            before += n;
        }
    };
    let per_sixteen = weights.first().map_or(1, |w| w.tiled.sixteen_bytes());
    let rows_phase = Phase {
        units: sixteens,
        most: (tiles::CLAIM / per_sixteen).max(1),
        run: &rows_phase,
        then: None,
    };
    if t.is_empty() {
        dispatch(&[rows_phase], false);
    } else {
        dispatch(
            &[
                Phase {
                    units: t.len(),
                    most: RANK_ROWS,
                    run: &a_phase,
                    then: None,
                },
                rows_phase,
            ],
            false,
        );
    }
    Ok(())
}

impl RmsNorm {
    fn steps(&self) -> bool {
        plain(&self.weight)
    }

    /// The norm of the row `x` into `out`.
    fn step(&self, x: &[f32], out: &mut Vec<f32>) -> CResult<()> {
        let (storage, layout) = self.weight.storage_and_layout();
        let weight = held(&storage, layout, "a norm's weight")?;
        if weight.len() != x.len() {
            bail!("a norm of {} values on a row of {}", weight.len(), x.len())
        }
        out.clear();
        out.resize(x.len(), 0.0);
        rowwise::norm_row(x, weight, self.eps as f32, out);
        Ok(())
    }
}

impl RotaryEmbedding {
    fn steps(&self) -> bool {
        plain(&self.sin) && plain(&self.cos)
    }

    /// Turn each head's `dim` values in `heads` to `position`: what
    /// [`RotaryEmbedding::apply`] gives one position, value for value.
    fn turn(&self, heads: &mut [f32], dim: usize, position: usize) -> CResult<()> {
        let (cos_storage, cos_layout) = self.cos.storage_and_layout();
        let (sin_storage, sin_layout) = self.sin.storage_and_layout();
        let cos = held(&cos_storage, cos_layout, "the rotary cosines")?;
        let sin = held(&sin_storage, sin_layout, "the rotary sines")?;
        let half = dim / 2;
        let at = position * half..(position + 1) * half;
        let (Some(cos), Some(sin)) = (cos.get(at.clone()), sin.get(at)) else {
            bail!("position {position} is past the rotary table")
        };
        for head in heads.chunks_exact_mut(dim) {
            let (low, high) = head.split_at_mut(half);
            for (((a, b), c), s) in low.iter_mut().zip(high.iter_mut()).zip(cos).zip(sin) {
                let (x1, x2) = (*a, *b);
                *a = x1 * c - x2 * s;
                *b = x1 * s + x2 * c;
            }
        }
        Ok(())
    }
}

impl Attention {
    fn steps(&self) -> bool {
        self.kv_store.is_some()
            && self.kv_cache.is_none()
            && self.rotary.steps()
            && [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
                .iter()
                .all(|l| l.steps())
    }

    /// One decoded token's attention: `s.normed` is its normed row at
    /// `position`, and what attention adds to the hidden row is left in
    /// `s.out`. The token's keys and values join the cache.
    fn step(&mut self, position: usize, s: &mut Scratch) -> CResult<()> {
        let Some(store) = &mut self.kv_store else {
            bail!("attention with no cache of plain rows was stepped through")
        };
        let (queries, cached) = (
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        );
        s.qkv.resize(queries + 2 * cached, 0.0);
        let (q, rest) = s.qkv.split_at_mut(queries);
        let (k, v) = rest.split_at_mut(cached);
        products(
            [&self.q_proj, &self.k_proj, &self.v_proj],
            &s.normed,
            [&mut *q, &mut *k, &mut *v],
            &mut s.row,
        )?;
        self.rotary.turn(q, self.head_dim, position)?;
        self.rotary.turn(k, self.head_dim, position)?;
        store.append(k, v)?;
        s.attended.resize(queries, 0.0);
        attend::attend(q, store, &mut s.attended, &mut s.attention)?;
        s.out.resize(self.hidden_size, 0.0);
        products([&self.o_proj], &s.attended, [&mut s.out], &mut s.row)
    }
}

/// `silu(gate) · up` into `out`.
fn gate_times(gate: &[f32], up: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if tiles::available() {
        // SAFETY: `available` found AVX-512F among the features it names.
        unsafe { wide::silu_times(gate, up, out) };
        return;
    }
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = g / (1.0 + (-g).exp()) * u;
    }
}

/// Values to a block of a quantized row.
const BLOCK: usize = 256;

impl Mlp {
    fn steps(&self) -> bool {
        matches!(self.act, Activation::Silu)
            && [&self.gate_proj, &self.up_proj, &self.down_proj]
                .iter()
                .all(|l| l.steps())
    }

    /// One decoded token's MLP, `down(silu(gate(x)) · up(x))`: `s.normed` is
    /// its normed row, and what the MLP adds to the hidden row is left in
    /// `s.out`. One dispatch: the gate's and the other side's rows, then the
    /// gating a block at a time by the cores that hold those rows, each
    /// block quantized where it is gated, then the narrow product of the
    /// quantized row.
    fn step(&self, s: &mut Scratch) -> CResult<()> {
        let x = s.normed.as_slice();
        let reads = [
            self.gate_proj.reads()?,
            self.up_proj.reads()?,
            self.down_proj.reads()?,
        ];
        let held = hold!(reads);
        let wide = weights!(reads[..2], held[..2], x.len());
        let [gate, up] = wide.as_slice() else {
            bail!("an MLP's gate and the side it gates are two projections")
        };
        let width = gate.tiled.rows();
        if up.tiled.rows() != width || width % BLOCK != 0 {
            bail!(
                "an MLP {width} and {} wide is not gated in blocks",
                up.tiled.rows()
            )
        }
        let narrow = weights!(reads[2..], held[2..], width);
        let [down] = narrow.as_slice() else {
            bail!("an MLP's way down is one projection")
        };
        let blocks = width / BLOCK;

        tiles::quantize_into(x, &mut s.row);
        let row = &s.row;
        s.both.resize(2 * width, 0.0);
        let both = Shared::of(&mut s.both);
        let gated = s.gated.blocks(width);
        let gated_row = &s.gated;
        s.parts.resize(blocks * down.rank, 0.0);
        let parts = Shared::of(&mut s.parts);
        s.out.resize(down.tiled.rows(), 0.0);
        let out = Shared::of(&mut s.out);

        // The adapters' `A · x` for the two wide projections.
        let (t_wide, starts) = through(&wide);
        let a_phase = |first: usize, last: usize| a_rows(&wide, x, &starts, &t_wide, first, last);

        // The wide rows: a unit is a sixteen of the gate's rows and the same
        // sixteen of the other side's, so the core that gates a block holds
        // both.
        let wide_phase = |first: usize, last: usize| {
            for (w, start) in wide.iter().zip(&starts) {
                let t = read(&t_wide[*start..start + w.rank]);
                let at = if std::ptr::eq(w, gate) { 0 } else { width };
                // SAFETY: `weights!` checked the row and the adapters against
                // the two projections; `both` holds `width` products for each
                // of them, this one's from `at`; and a unit is done by one
                // thread.
                unsafe {
                    let out = Shared::of(both.part(at, width));
                    rows_of(w, first, last, row, &t, &out);
                }
            }
        };

        // The gate, a block at a time: gated, quantized, and through the
        // down adapter's `A`.
        let gate_phase = |first: usize, last: usize| {
            let mut m = [0f32; BLOCK];
            for block in first..last {
                // SAFETY: the wide rows were all written before this phase
                // began, and nothing writes them now; the gated row has room
                // for `width` values, this block among them, and a block is
                // done by one thread, as is its part of `parts`.
                unsafe {
                    let wide_rows = both.all();
                    gate_times(
                        &wide_rows[block * BLOCK..(block + 1) * BLOCK],
                        &wide_rows[width + block * BLOCK..width + (block + 1) * BLOCK],
                        &mut m,
                    );
                    gated.quantize(block, &m);
                    for (i, part) in parts
                        .part(block * down.rank, down.rank)
                        .iter_mut()
                        .enumerate()
                    {
                        *part = dot(
                            &down.a[i * width + block * BLOCK..i * width + (block + 1) * BLOCK],
                            &m,
                        );
                    }
                }
            }
        };
        // The down adapter's `A` times the whole gated row: its blocks'
        // parts, added in their order.
        let t_down: Vec<AtomicU32> = (0..down.rank).map(|_| AtomicU32::new(0)).collect();
        let gathered = || {
            // SAFETY: every block's part was written before this ran, and
            // nothing writes them now.
            let parts = unsafe { parts.all() };
            for (i, t) in t_down.iter().enumerate() {
                let sum: f32 = (0..blocks).map(|block| parts[block * down.rank + i]).sum();
                t.store((sum * down.scale).to_bits(), Ordering::Relaxed);
            }
        };

        let down_phase = |first: usize, last: usize| {
            let t = read(&t_down);
            // SAFETY: every block of the gated row was written two phases
            // back; `out` is room for the down projection's products; and a
            // unit is done by one thread.
            unsafe { rows_of(down, first, last, gated_row, &t, &out) };
        };

        let mut all = Vec::with_capacity(4);
        if !t_wide.is_empty() {
            all.push(Phase {
                units: t_wide.len(),
                most: RANK_ROWS,
                run: &a_phase,
                then: None,
            });
        }
        all.push(Phase {
            units: gate.tiled.sixteens(),
            most: (tiles::CLAIM / (2 * gate.tiled.sixteen_bytes())).max(1),
            run: &wide_phase,
            then: None,
        });
        all.push(Phase {
            units: blocks,
            most: 1,
            run: &gate_phase,
            then: (down.rank > 0).then_some(&gathered as &(dyn Fn() + Sync)),
        });
        all.push(Phase {
            units: down.tiled.sixteens(),
            most: (tiles::CLAIM / down.tiled.sixteen_bytes()).max(1),
            run: &down_phase,
            then: None,
        });
        dispatch(&all, false);
        Ok(())
    }
}

fn add(x: &mut [f32], more: &[f32]) -> CResult<()> {
    if x.len() != more.len() {
        bail!("a row of {} values added to one of {}", more.len(), x.len())
    }
    for (a, b) in x.iter_mut().zip(more) {
        *a += b;
    }
    Ok(())
}

impl DecoderLayer {
    fn steps(&self) -> bool {
        self.input_ln.steps()
            && self.post_attn_ln.steps()
            && self.self_attn.steps()
            && self.mlp.steps()
    }

    /// One decoded token through this layer: `x` is its hidden row at
    /// `position`, before and after.
    fn step(&mut self, x: &mut [f32], position: usize, s: &mut Scratch) -> CResult<()> {
        self.input_ln.step(x, &mut s.normed)?;
        self.self_attn.step(position, s)?;
        add(x, &s.out)?;
        self.post_attn_ln.step(x, &mut s.normed)?;
        self.mlp.step(s)?;
        add(x, &s.out)
    }
}

impl QwenLora {
    /// The logits after `token` at `position`, as [`QwenLora::hidden`] and
    /// [`QwenLora::logits`] give them for that one token with the cache on,
    /// through plain rows. `None`, with nothing read or written, when this
    /// model does not decode a token that way: the caller takes the tensor
    /// path, which reads and writes the same cache.
    pub(in crate::models::qwen) fn step(
        &mut self,
        token: u32,
        position: usize,
    ) -> CResult<Option<Vec<f32>>> {
        let table = self.embed_tokens.embeddings();
        let steps = (self.lm_head_t.is_some() || self.lm_head_q.is_some())
            && self.device.is_cpu()
            && self.dtype == DType::F32
            && plain(table)
            && self.norm.steps()
            && self.layers.iter().all(DecoderLayer::steps);
        if !steps {
            return Ok(None);
        }
        let s = &mut self.scratch;
        let mut x = std::mem::take(&mut s.x);
        {
            let (rows, width) = table.dims2()?;
            if token as usize >= rows {
                bail!("token {token} is past a vocabulary of {rows}")
            }
            let (storage, layout) = table.storage_and_layout();
            let all = held(&storage, layout, "the embeddings")?;
            x.clear();
            x.extend_from_slice(&all[token as usize * width..(token as usize + 1) * width]);
        }
        for layer in &mut self.layers {
            layer.step(&mut x, position, s)?;
        }
        self.norm.step(&x, &mut s.normed)?;
        s.x = x;
        // The head through its tiles, or through candle's kernels where this
        // CPU has no tiles for the form it is in.
        let Some(head) = &self.lm_head_t else {
            let row = Tensor::from_slice(&s.normed, (1, 1, s.normed.len()), &Device::Cpu)?;
            return Ok(Some(self.logits(&row)?.flatten_all()?.to_vec1()?));
        };
        let mut logits = vec![0f32; head.rows()];
        tiles::quantize_into(&s.normed, &mut s.row);
        tiles::multiply(&[head], &s.row, &mut [&mut logits])?;
        Ok(Some(logits))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn config() -> Config {
        Config {
            vocab_size: 64,
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 256,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            tie_word_embeddings: true,
        }
    }

    /// A base of drawn weights under the names a Qwen checkpoint gives them.
    fn weights(cfg: &Config) -> CResult<HashMap<String, Tensor>> {
        let dev = Device::Cpu;
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let kv = cfg.num_key_value_heads * (h / cfg.num_attention_heads);
        let mut all = HashMap::new();
        let mut draw = |name: String, shape: &[usize], mean: f32, spread: f32| -> CResult<()> {
            all.insert(name, Tensor::randn(mean, spread, shape, &dev)?);
            Ok(())
        };
        draw(
            "model.embed_tokens.weight".into(),
            &[cfg.vocab_size, h],
            0.0,
            0.5,
        )?;
        draw("model.norm.weight".into(), &[h], 1.0, 0.1)?;
        for layer in 0..cfg.num_hidden_layers {
            let at = format!("model.layers.{layer}");
            for (name, out) in [("q_proj", h), ("k_proj", kv), ("v_proj", kv)] {
                draw(
                    format!("{at}.self_attn.{name}.weight"),
                    &[out, h],
                    0.0,
                    0.05,
                )?;
                draw(format!("{at}.self_attn.{name}.bias"), &[out], 0.0, 0.05)?;
            }
            draw(format!("{at}.self_attn.o_proj.weight"), &[h, h], 0.0, 0.05)?;
            draw(format!("{at}.mlp.gate_proj.weight"), &[i, h], 0.0, 0.05)?;
            draw(format!("{at}.mlp.up_proj.weight"), &[i, h], 0.0, 0.05)?;
            draw(format!("{at}.mlp.down_proj.weight"), &[h, i], 0.0, 0.05)?;
            draw(format!("{at}.input_layernorm.weight"), &[h], 1.0, 0.1)?;
            draw(
                format!("{at}.post_attention_layernorm.weight"),
                &[h],
                1.0,
                0.1,
            )?;
        }
        Ok(all)
    }

    /// A model over `weights`, served at `serve`, generating: its adapter is
    /// of rank 8, drawn from a seed on the A side and, with `trained`, set
    /// to values that are not nothing on the B side.
    fn model(
        cfg: &Config,
        weights: &HashMap<String, Tensor>,
        serve: Option<GgmlDType>,
        trained: bool,
    ) -> CResult<QwenLora> {
        let dev = Device::Cpu;
        let base = VarBuilder::from_tensors(weights.clone(), DType::F32, &dev);
        let varmap = VarMap::new();
        let lora = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = QwenLora::build(cfg, &base, &lora, varmap, 8, 2.0, serve)?;
        seed_lora_a(&model.varmap, 11)?;
        if trained {
            let vars = model
                .varmap
                .data()
                .lock()
                .map_err(|_| candle_core::Error::msg("the varmap lock is poisoned"))?;
            let mut names: Vec<&String> = vars.keys().filter(|n| n.ends_with("lora_b")).collect();
            names.sort();
            for (salt, name) in names.into_iter().enumerate() {
                let var = &vars[name];
                let values: Vec<f32> = (0..var.elem_count())
                    .map(|i| ((i + 31 * salt) as f32 * 0.37).sin() * 0.05)
                    .collect();
                var.set(&Tensor::from_vec(values, var.shape(), &dev)?)?;
            }
        }
        model.set_grad(false);
        Ok(model)
    }

    /// The logits after `tokens` read at `position` through the tensor path,
    /// with the cache on.
    fn forwarded(model: &mut QwenLora, tokens: &[u32], position: usize) -> CResult<Vec<f32>> {
        let input = Tensor::new(tokens, &Device::Cpu)?.unsqueeze(0)?;
        let hidden = model.hidden(&input, position, true)?;
        let last = hidden.narrow(1, tokens.len() - 1, 1)?;
        model.logits(&last)?.flatten_all()?.to_vec1::<f32>()
    }

    /// The largest difference between two rows of logits, of the largest
    /// logit.
    fn gap(a: &[f32], b: &[f32]) -> f32 {
        let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / peak
    }

    const PROMPT: [u32; 5] = [3, 17, 60, 0, 42];

    fn token(i: usize) -> u32 {
        ((i * 7 + 5) % 64) as u32
    }

    /// A token stepped through plain rows gives the logits the tensor path
    /// gives it, token after token behind a prompt, at both served forms:
    /// for the bare base, for a fresh adapter and for a trained one.
    #[test]
    fn a_token_stepped_is_the_token_forwarded() -> CResult<()> {
        if !tiles::available() {
            eprintln!("skipped: this CPU has no AVX-512 VNNI, so nothing is stepped");
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        for serve in [GgmlDType::Q4K, GgmlDType::Q8_0] {
            for (what, trained, adapter) in [
                ("bare", false, false),
                ("a fresh adapter", false, true),
                ("a trained adapter", true, true),
            ] {
                let mut by_tensors = model(&cfg, &weights, Some(serve), trained)?;
                let mut by_rows = model(&cfg, &weights, Some(serve), trained)?;
                by_tensors.set_lora(adapter);
                by_rows.set_lora(adapter);
                let first = forwarded(&mut by_tensors, &PROMPT, 0)?;
                assert_eq!(
                    forwarded(&mut by_rows, &PROMPT, 0)?,
                    first,
                    "one prompt pass"
                );
                for i in 0..40 {
                    let position = PROMPT.len() + i;
                    let want = forwarded(&mut by_tensors, &[token(i)], position)?;
                    let Some(got) = by_rows.step(token(i), position)? else {
                        bail!("{serve:?}, {what}: token {i} was not stepped")
                    };
                    assert_eq!(got.len(), cfg.vocab_size);
                    let off = gap(&got, &want);
                    assert!(
                        off < 2e-3,
                        "{serve:?}, {what}, token {i}: off by {off} of the peak"
                    );
                }
                // The adapter is in what was compared: trained, it moves the
                // logits.
                if trained {
                    let mut bare = model(&cfg, &weights, Some(serve), trained)?;
                    bare.set_lora(false);
                    forwarded(&mut bare, &PROMPT, 0)?;
                    by_rows.clear_cache();
                    forwarded(&mut by_rows, &PROMPT, 0)?;
                    let (Some(with), Some(without)) = (
                        by_rows.step(token(0), PROMPT.len())?,
                        bare.step(token(0), PROMPT.len())?,
                    ) else {
                        bail!("{serve:?}: not stepped")
                    };
                    assert!(
                        gap(&with, &without) > 1e-2,
                        "{serve:?}: the adapter changed nothing"
                    );
                }
            }
        }
        Ok(())
    }

    /// A sequence passes between the two paths: tokens stepped through plain
    /// rows leave the cache as the tensor path would have, so a token
    /// forwarded after them gives what it gives after tokens forwarded.
    #[test]
    fn both_paths_write_the_one_cache() -> CResult<()> {
        if !tiles::available() {
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let mut by_tensors = model(&cfg, &weights, Some(GgmlDType::Q4K), true)?;
        let mut mixed = model(&cfg, &weights, Some(GgmlDType::Q4K), true)?;
        forwarded(&mut by_tensors, &PROMPT, 0)?;
        forwarded(&mut mixed, &PROMPT, 0)?;
        for i in 0..6 {
            let position = PROMPT.len() + i;
            forwarded(&mut by_tensors, &[token(i)], position)?;
            if mixed.step(token(i), position)?.is_none() {
                bail!("token {i} was not stepped")
            }
        }
        let position = PROMPT.len() + 6;
        let want = forwarded(&mut by_tensors, &[token(6)], position)?;
        let got = forwarded(&mut mixed, &[token(6)], position)?;
        let off = gap(&got, &want);
        assert!(off < 2e-3, "off by {off} of the peak");

        // And a new sequence starts from nothing on both.
        by_tensors.clear_cache();
        mixed.clear_cache();
        let want = forwarded(&mut by_tensors, &[token(1)], 0)?;
        let Some(got) = mixed.step(token(1), 0)? else {
            bail!("a first token was not stepped")
        };
        assert!(gap(&got, &want) < 2e-3, "a sequence of one token");
        Ok(())
    }

    /// What does not decode through plain rows is left to the tensor path
    /// with nothing touched: a base that is not served, an adapter that is
    /// training. A token past the vocabulary is refused.
    #[test]
    fn what_cannot_step_is_left_to_the_tensor_path() -> CResult<()> {
        let cfg = config();
        let weights = weights(&cfg)?;
        let mut dense = model(&cfg, &weights, None, false)?;
        assert!(dense.step(1, 0)?.is_none(), "a dense base");
        if !tiles::available() {
            return Ok(());
        }
        let mut served = model(&cfg, &weights, Some(GgmlDType::Q4K), false)?;
        served.set_grad(true);
        assert!(served.step(1, 0)?.is_none(), "an adapter with a gradient");
        served.set_grad(false);
        assert!(served.step(64, 0).is_err(), "a token past the vocabulary");
        assert!(served.step(1, 0)?.is_some());
        assert!(served.step(2, 1)?.is_some());
        assert!(
            served.step(3, 256).is_err(),
            "a position past the rotary table"
        );
        Ok(())
    }
}

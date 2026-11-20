//! A sequence's rows on the CPU as plain values, with no tensor made between
//! their products: a decoded token, and a prompt.
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
//! Here a row stays a plain slice from the embedding to the logits, in
//! buffers kept from token to token. A product and its adapter are one
//! dispatch (`phases`): each run of the weight's rows through its tiles
//! (`tiles`) with the adapter's `A` times the row beside them, then the
//! adapter's `B` rows and the bias. The adapter is read from a copy of its
//! factors at eight bits, laid out for vectors (`pack`), less the ranks a
//! blend of fewer experts leaves at nothing. The MLP is one dispatch: its
//! two wide products, then the gate and the quantizing of its result a block
//! at a time by the cores that hold those values, then its narrow product.
//! So the MLP's wide rows never reach the calling thread, where pulling them
//! a cache line at a time from eight cores and handing the gated row back
//! out cost more than the arithmetic. Attention reads the cache where it
//! lies (`attend`).
//!
//! A prompt is the same steps over several rows at once: a weight is read
//! from memory once for all of them, and each row is computed exactly as it
//! would be alone, so a prompt read here and the same tokens decoded one at
//! a time leave the same cache and give the same logits, to the bit.
//!
//! It computes what the tensor path computes, to the order of a few float
//! sums, and gives the same values on any number of threads. It is taken
//! only where the tensor path would have computed without a gradient: a
//! served base in f32 on a CPU that multiplies through tiles, generating one
//! sequence. Training, every GPU and every other CPU keep the tensor ops,
//! and a sequence may pass from one path to the other: both write the same
//! cache.

use candle_core::{bail, Layout, Storage};

use super::*;
use pack::BLOCK;
use phases::{dispatch, Phase, Shared};

/// Rows whose block of the MLP is gated by one unit of work: what an adapter
/// reads for the block it reads once for them all, and their gated blocks
/// stay in the nearest cache until the way down's `A` has read them.
const GATED_TOGETHER: usize = 8;

/// Rows read at once at most: a prompt longer than this is read in runs of
/// it, each run through every layer before the next. Its rows share one
/// reading of the weights; what they are computed in stays within the
/// cores' own caches.
const ROWS_AT_ONCE: usize = 32;

/// What rows are computed in, kept from one token to the next. Where it
/// holds several rows it holds them one after another.
#[derive(Default)]
pub(in crate::models::qwen) struct Scratch {
    /// The hidden rows, from the embedding to the last layer.
    x: Vec<f32>,
    /// Those rows normed: what a layer's products read.
    normed: Vec<f32>,
    /// A layer's queries for every row, then its keys, then its values.
    qkv: Vec<f32>,
    attended: Vec<f32>,
    /// The MLP's gate for every row, then the side it gates.
    both: Vec<f32>,
    /// What a layer's attention, then its MLP, adds to the hidden rows.
    out: Vec<f32>,
    /// Each adapter's `scale · A · x` for every row, for the product in
    /// hand; and the same for the MLP's way down.
    through: Vec<f32>,
    through_down: Vec<f32>,
    /// For each block of the gated rows and each row, the way down's `A`
    /// times that block of the row.
    parts: Vec<f32>,
    /// The room one row's attention works in.
    attention: attend::Work,
    /// The quantized inputs of the product in hand, and the MLP's gated
    /// rows quantized: a row each.
    quantized: Vec<tiles::Row>,
    gated: Vec<tiles::Row>,
    /// Room for a token's logits, when the last token's was handed back.
    logits: Vec<f32>,
    /// What the head's two stages are worked out in, where a token is
    /// chosen through its sketch.
    sketched: sketch::Room,
}

/// What a step leaves its caller with.
pub(in crate::models::qwen) enum Stepped {
    /// The logits after the last token.
    Logits(Vec<f32>),
    /// The token the head ranks first after it, where that was all that was
    /// asked and the head is sketched.
    First(u32),
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

/// Each of `rows` rows of `x` quantized into a row of `into`.
fn quantize(x: &[f32], rows: usize, into: &mut Vec<tiles::Row>) {
    if into.len() < rows {
        into.resize_with(rows, Default::default);
    }
    for (row, x) in into.iter_mut().zip(x.chunks_exact(x.len() / rows.max(1))) {
        tiles::quantize_into(x, row);
    }
}

/// One projection as the units of a dispatch read it: its tiles, its bias as
/// plain values, and its adapter's packed factors when the adapter is on and
/// has a rank.
struct Weight<'a> {
    tiled: &'a tiles::Tiled,
    bias: Option<&'a [f32]>,
    pack: Option<&'a pack::Packed>,
}

impl Weight<'_> {
    fn rank(&self) -> usize {
        self.pack.map_or(0, |p| p.rank)
    }
}

impl LoraLinear {
    /// Whether rows can be read through this projection as plain values:
    /// its base is tiled, and its adapter is off or packed.
    fn steps(&self) -> bool {
        matches!(self.base, BaseWeight::Tiled(_))
            && (!self.enabled || self.packed.is_some())
            && self.base_b.as_ref().is_none_or(plain)
    }

    fn tiled(&self) -> CResult<&tiles::Tiled> {
        match &self.base {
            BaseWeight::Tiled(tiled) => Ok(tiled),
            _ => bail!("a projection that is not tiled was stepped through"),
        }
    }

    /// The adapter as a decoded token reads it, when it is on.
    fn adapter(&self) -> Option<&pack::Packed> {
        match self.enabled {
            true => self.packed.as_ref().filter(|p| p.rank > 0),
            false => None,
        }
    }
}

/// The storage of each projection's bias, held while a dispatch reads it.
macro_rules! hold {
    ($group:expr) => {
        $group
            .iter()
            .map(|l| l.base_b.as_ref().map(Tensor::storage_and_layout))
            .collect::<Vec<_>>()
    };
}

/// Each projection in `group` as the units of a dispatch read it, out of
/// what `hold!` holds for them: the projection reads rows of `cols` values.
macro_rules! weights {
    ($group:expr, $held:expr, $cols:expr) => {{
        let mut weights = Vec::with_capacity($group.len());
        for (l, bias) in $group.iter().zip($held.iter()) {
            let tiled = l.tiled()?;
            let bias = match bias {
                Some((storage, layout)) => Some(held(storage, layout, "a projection's bias")?),
                None => None,
            };
            let pack = l.adapter();
            let rows = tiled.rows();
            if tiled.cols() != $cols
                || bias.is_some_and(|bias| bias.len() != rows)
                || pack.is_some_and(|p| p.cols() != $cols || p.rows() != rows)
            {
                bail!(
                    "a projection of {rows} by {} and its adapter do not fit a row of {}",
                    tiled.cols(),
                    $cols
                )
            }
            weights.push(Weight { tiled, bias, pack });
        }
        weights
    }};
}

/// Room in `room` for each weight's `scale · A · x` for each of `rows` rows:
/// one weight's rows after another's, a row's `rank` values together. Where
/// each weight's start is what comes back.
fn through(weights: &[Weight], rows: usize, room: &mut Vec<f32>) -> Vec<usize> {
    let mut at = 0;
    let starts = weights
        .iter()
        .map(|w| {
            let start = at;
            at += rows * w.rank();
            start
        })
        .collect();
    room.clear();
    room.resize(at, 0.0);
    starts
}

/// A unit of work for each weight's adapter when any of `weights` has one
/// on, and none otherwise.
fn adapted(weights: &[Weight]) -> usize {
    match weights.iter().any(|w| w.pack.is_some()) {
        true => weights.len(),
        false => 0,
    }
}

/// Weight `w`'s `scale · A · x` for every one of the `rows` rows of `x`,
/// into its part of `t`, which starts at `start`.
///
/// SAFETY (for the caller): `t` is the room [`through`] made for `rows`
/// rows, and no other thread reads or writes this weight's part of it.
unsafe fn a_unit(w: &Weight, x: &[f32], rows: usize, t: &Shared<f32>, start: usize) {
    let Some(pack) = w.pack else { return };
    let out = t.part(start, rows * pack.rank);
    pack.through(0, pack.cols(), x, pack.cols(), rows, out);
    pack.scaled(out);
}

/// To the sixteens `first..last` of `w`'s products for every one of `rows`
/// rows in `out`: the adapter's `B` rows times each row's part of `t` (the
/// weight's `scale · A · x`, row after row), then the bias.
///
/// SAFETY (for the caller): `out` is room for `w`'s products of every row, a
/// row's after another's, and no other thread reads or writes these sixteens
/// of them.
unsafe fn finish(w: &Weight, rows: usize, first: usize, last: usize, t: &[f32], out: &Shared<f32>) {
    let (first, n) = (first * tiles::ROWS, (last - first) * tiles::ROWS);
    let width = w.tiled.rows();
    if let Some(pack) = w.pack {
        pack.add(first, n, t, rows, out.at().add(first), width);
    }
    if let Some(bias) = w.bias {
        for i in 0..rows {
            let out = out.part(i * width + first, n);
            for (o, b) in out.iter_mut().zip(&bias[first..first + n]) {
                *o += b;
            }
        }
    }
}

/// The part of `first..last` that falls in the `n` units from `before` on,
/// counted from `before`.
fn within(first: usize, last: usize, before: usize, n: usize) -> Option<(usize, usize)> {
    let (from, to) = (first.max(before), last.min(before + n));
    (from < to).then(|| (from - before, to - before))
}

/// Every projection in `group` applied to the `rows` rows of `x`, each into
/// its own slice of `outs` (a row's products after another's): what
/// [`LoraLinear::forward_group`] gives those rows. Each row is quantized
/// once for the group, and the group is one dispatch. Its first phase is
/// the bases' rows and, where an adapter is on, each adapter's `A` times
/// every row, which nothing waits for: no thread stands idle while `A` is
/// read. Its second, where an adapter is on, is the `B` rows and the
/// biases. `through` is room the adapters' `A · x` is kept in.
fn products<const N: usize>(
    group: [&LoraLinear; N],
    x: &[f32],
    rows: usize,
    mut outs: [&mut [f32]; N],
    quantized: &mut Vec<tiles::Row>,
    room: &mut Vec<f32>,
) -> CResult<()> {
    if rows == 0 || !x.len().is_multiple_of(rows) {
        bail!("{} values are not {rows} rows", x.len())
    }
    let held = hold!(group);
    let weights = weights!(group, held, x.len() / rows);
    if weights
        .iter()
        .zip(outs.iter())
        .any(|(w, out)| rows * w.tiled.rows() != out.len())
    {
        bail!("a group of projections and the places for their products do not fit")
    }
    quantize(x, rows, quantized);
    let quantized = &quantized[..rows];

    let starts = through(&weights, rows, room);
    let t = Shared::of(room.as_mut_slice());
    let adapted = adapted(&weights);
    let shared: Vec<Shared<f32>> = outs.iter_mut().map(|out| Shared::of(out)).collect();
    let sixteens: usize = weights.iter().map(|w| w.tiled.sixteens()).sum();
    // The adapters, then the group's sixteens, one weight's after another's.
    let bases = |first: usize, last: usize| {
        for unit in first..last.min(adapted) {
            // SAFETY: `t` is the room made just above for these weights and
            // rows, and a unit is done by one thread.
            unsafe { a_unit(&weights[unit], x, rows, &t, starts[unit]) };
        }
        let Some((first, last)) = within(first, last, adapted, sixteens) else {
            return;
        };
        let mut before = 0;
        for (w, out) in weights.iter().zip(&shared) {
            if let Some((from, to)) = within(first, last, before, w.tiled.sixteens()) {
                // SAFETY: `weights!` checked the rows against each weight and
                // the places for products were checked just above; a unit is
                // done by one thread.
                unsafe {
                    w.tiled.multiply_sixteens(from, to, quantized, out.at());
                    if adapted == 0 {
                        finish(w, rows, from, to, &[], out);
                    }
                }
            }
            before += w.tiled.sixteens();
        }
    };
    let adapters = |first: usize, last: usize| {
        // SAFETY: every adapter's unit was done in the phase before, and
        // nothing writes their values now.
        let t = unsafe { t.all() };
        let mut before = 0;
        for ((w, out), start) in weights.iter().zip(&shared).zip(starts.iter().copied()) {
            if let Some((from, to)) = within(first, last, before, w.tiled.sixteens()) {
                let t = &t[start..start + rows * w.rank()];
                // SAFETY: as in the phase before, in which every base row
                // was written.
                unsafe { finish(w, rows, from, to, t, out) };
            }
            before += w.tiled.sixteens();
        }
    };
    let per_sixteen = weights.first().map_or(1, |w| w.tiled.sixteen_bytes());
    let most = (tiles::CLAIM / (per_sixteen * rows)).max(1);
    let bases = Phase {
        units: adapted + sixteens,
        most,
        run: &bases,
        then: None,
    };
    if adapted == 0 {
        dispatch(&[bases], false);
    } else {
        dispatch(
            &[
                bases,
                Phase {
                    units: sixteens,
                    most: 4 * most,
                    run: &adapters,
                    then: None,
                },
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

    /// The norm of each of the `rows` rows of `x` into `out`.
    fn step(&self, x: &[f32], rows: usize, out: &mut Vec<f32>) -> CResult<()> {
        let (storage, layout) = self.weight.storage_and_layout();
        let weight = held(&storage, layout, "a norm's weight")?;
        if weight.is_empty() || weight.len() * rows != x.len() {
            bail!(
                "a norm of {} values on {rows} rows of {} in all",
                weight.len(),
                x.len()
            )
        }
        out.resize(x.len(), 0.0);
        for (x, out) in x
            .chunks_exact(weight.len())
            .zip(out.chunks_exact_mut(weight.len()))
        {
            rowwise::norm_row(x, weight, self.eps as f32, out);
        }
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

    /// Attention for `rows` rows read together: `s.normed` is their normed
    /// rows, the first of them at `position`, and what attention adds to the
    /// hidden rows is left in `s.out`. Their keys and values join the cache,
    /// and each row attends to the positions before it and its own.
    fn step(&mut self, position: usize, rows: usize, s: &mut Scratch) -> CResult<()> {
        let Some(store) = &mut self.kv_store else {
            bail!("attention with no cache of plain rows was stepped through")
        };
        let (queries, cached) = (
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        );
        s.qkv.resize(rows * (queries + 2 * cached), 0.0);
        let (q, rest) = s.qkv.split_at_mut(rows * queries);
        let (k, v) = rest.split_at_mut(rows * cached);
        products(
            [&self.q_proj, &self.k_proj, &self.v_proj],
            &s.normed,
            rows,
            [&mut *q, &mut *k, &mut *v],
            &mut s.quantized,
            &mut s.through,
        )?;
        // Every row is turned before any is cached, so a position past the
        // table leaves the cache as it was.
        for (i, (q, k)) in q
            .chunks_exact_mut(queries)
            .zip(k.chunks_exact_mut(cached))
            .enumerate()
        {
            self.rotary.turn(q, self.head_dim, position + i)?;
            self.rotary.turn(k, self.head_dim, position + i)?;
        }
        for (k, v) in k.chunks_exact(cached).zip(v.chunks_exact(cached)) {
            store.append(k, v)?;
        }
        s.attended.resize(rows * queries, 0.0);
        match rows {
            1 => attend::attend(q, store, &mut s.attended, &mut s.attention)?,
            _ => attend::attend_rows(q, rows, store, &mut s.attended)?,
        }
        s.out.resize(rows * self.hidden_size, 0.0);
        products(
            [&self.o_proj],
            &s.attended,
            rows,
            [&mut s.out],
            &mut s.quantized,
            &mut s.through,
        )
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

impl Mlp {
    fn steps(&self) -> bool {
        matches!(self.act, Activation::Silu)
            && [&self.gate_proj, &self.up_proj, &self.down_proj]
                .iter()
                .all(|l| l.steps())
    }

    /// The MLP for `rows` rows, `down(silu(gate(x)) · up(x))`: `s.normed` is
    /// their normed rows, and what the MLP adds to the hidden rows is left
    /// in `s.out`. One dispatch of three phases. The wide bases' rows, with
    /// the wide adapters' `A` times every row. Then the gate, a block of a
    /// few rows at a time by the cores that hold those values: the adapters'
    /// `B` rows added, the block gated and quantized where it is gated, and
    /// put through the way down's `A`. Then the narrow product of the
    /// quantized rows.
    fn step(&self, rows: usize, s: &mut Scratch) -> CResult<()> {
        let x = s.normed.as_slice();
        if rows == 0 || !x.len().is_multiple_of(rows) {
            bail!("{} values are not {rows} rows", x.len())
        }
        let wide_group = [&self.gate_proj, &self.up_proj];
        let wide_held = hold!(wide_group);
        let wide = weights!(wide_group, wide_held, x.len() / rows);
        let [gate, up] = wide.as_slice() else {
            bail!("an MLP's gate and the side it gates are two projections")
        };
        let width = gate.tiled.rows();
        if up.tiled.rows() != width || !width.is_multiple_of(BLOCK) {
            bail!(
                "an MLP {width} and {} wide is not gated in blocks",
                up.tiled.rows()
            )
        }
        let narrow_group = [&self.down_proj];
        let narrow_held = hold!(narrow_group);
        let narrow = weights!(narrow_group, narrow_held, width);
        let [down] = narrow.as_slice() else {
            bail!("an MLP's way down is one projection")
        };
        let (blocks, down_rank) = (width / BLOCK, down.rank());

        quantize(x, rows, &mut s.quantized);
        let quantized = &s.quantized[..rows];
        s.both.resize(2 * rows * width, 0.0);
        let both = Shared::of(&mut s.both);
        if s.gated.len() < rows {
            s.gated.resize_with(rows, Default::default);
        }
        let gated: Vec<tiles::Blocks> = s.gated[..rows]
            .iter_mut()
            .map(|row| row.blocks(width))
            .collect();
        let gated_rows = &s.gated[..rows];
        s.parts.resize(blocks * rows * down_rank, 0.0);
        let parts = Shared::of(&mut s.parts);
        s.out.resize(rows * down.tiled.rows(), 0.0);
        let out = Shared::of(&mut s.out);

        // The wide adapters, then the wide rows: a unit is a sixteen of the
        // gate's rows and the same sixteen of the other side's, so the core
        // that gates a block holds both.
        let starts = through(&wide, rows, &mut s.through);
        let t_wide = Shared::of(s.through.as_mut_slice());
        let adapted = adapted(&wide);
        let sixteens = gate.tiled.sixteens();
        let wide_phase = |first: usize, last: usize| {
            for unit in first..last.min(adapted) {
                // SAFETY: `t_wide` is the room made just above for these
                // weights and rows, and a unit is done by one thread.
                unsafe { a_unit(&wide[unit], x, rows, &t_wide, starts[unit]) };
            }
            let Some((first, last)) = within(first, last, adapted, sixteens) else {
                return;
            };
            for (side, w) in wide.iter().enumerate() {
                // SAFETY: `weights!` checked the rows against the two
                // projections; `both` holds every row's `width` products for
                // each of them, one side after the other; and a unit is done
                // by one thread.
                unsafe {
                    w.tiled.multiply_sixteens(
                        first,
                        last,
                        quantized,
                        both.at().add(side * rows * width),
                    );
                }
            }
        };

        // The gate, a block of a few rows at a time: unit `u` is block
        // `u / groups` of the rows of group `u % groups`, so a thread's
        // units are the blocks whose rows it multiplied, and what an
        // adapter reads for a block it reads once for the group.
        let groups = rows.div_ceil(GATED_TOGETHER);
        let gate_phase = |first: usize, last: usize| {
            let mut m = [[0f32; BLOCK]; GATED_TOGETHER];
            // SAFETY: the wide adapters' units were all done in the phase
            // before, and nothing writes their values now.
            let t = unsafe { t_wide.all() };
            for unit in first..last {
                let (block, group) = (unit / groups, unit % groups);
                let (from, to) = (
                    group * GATED_TOGETHER,
                    rows.min((group + 1) * GATED_TOGETHER),
                );
                // SAFETY: the wide rows were all written in the phase
                // before. This block of these rows on each side, their
                // blocks of the gated rows and their part of `parts` are
                // this unit's alone, and a unit is done by one thread.
                unsafe {
                    for ((side, w), start) in wide.iter().enumerate().zip(starts.iter().copied()) {
                        let at = both.at().add((side * rows + from) * width + block * BLOCK);
                        if let Some(pack) = w.pack {
                            let t = &t[start + from * pack.rank..start + to * pack.rank];
                            pack.add(block * BLOCK, BLOCK, t, to - from, at, width);
                        }
                        if let Some(bias) = w.bias {
                            let bias = &bias[block * BLOCK..(block + 1) * BLOCK];
                            for i in 0..to - from {
                                let side = std::slice::from_raw_parts_mut(at.add(i * width), BLOCK);
                                for (o, b) in side.iter_mut().zip(bias) {
                                    *o += b;
                                }
                            }
                        }
                    }
                    for (i, m) in (from..to).zip(m.iter_mut()) {
                        let g = both.part(i * width + block * BLOCK, BLOCK);
                        let u = both.part((rows + i) * width + block * BLOCK, BLOCK);
                        gate_times(g, u, m);
                        gated[i].quantize(block, &*m);
                    }
                    if let Some(pack) = down.pack {
                        let part =
                            parts.part((block * rows + from) * down_rank, (to - from) * down_rank);
                        pack.through(
                            block * BLOCK,
                            BLOCK,
                            m.as_flattened(),
                            BLOCK,
                            to - from,
                            part,
                        );
                    }
                }
            }
        };
        // The way down's `scale · A` times each whole gated row: its
        // blocks' parts, added in their order.
        s.through_down.clear();
        s.through_down.resize(rows * down_rank, 0.0);
        let t_down = Shared::of(s.through_down.as_mut_slice());
        let gathered = || {
            // SAFETY: every block's part was written before this ran, on one
            // thread, and nothing else reads or writes either until it is
            // done.
            let (parts, t_down) = unsafe { (parts.all(), t_down.part(0, rows * down_rank)) };
            for (at, t) in t_down.iter_mut().enumerate() {
                *t = (0..blocks)
                    .map(|block| parts[block * rows * down_rank + at])
                    .sum();
            }
            if let Some(pack) = down.pack {
                pack.scaled(t_down);
            }
        };

        let down_phase = |first: usize, last: usize| {
            // SAFETY: every block of every gated row was written in the
            // phase before, and `weights!` checked their width against the
            // way down; its adapter's values were gathered after that phase
            // and nothing writes them now; `out` is room for the way down's
            // products for every row; and a unit is done by one thread.
            unsafe {
                down.tiled
                    .multiply_sixteens(first, last, gated_rows, out.at());
                finish(down, rows, first, last, t_down.all(), &out);
            }
        };

        dispatch(
            &[
                Phase {
                    units: adapted + sixteens,
                    most: (tiles::CLAIM / (2 * gate.tiled.sixteen_bytes() * rows)).max(1),
                    run: &wide_phase,
                    then: None,
                },
                Phase {
                    units: blocks * groups,
                    most: 1,
                    run: &gate_phase,
                    then: (down_rank > 0).then_some(&gathered as &(dyn Fn() + Sync)),
                },
                Phase {
                    units: down.tiled.sixteens(),
                    most: (tiles::CLAIM / (down.tiled.sixteen_bytes() * rows)).max(1),
                    run: &down_phase,
                    then: None,
                },
            ],
            false,
        );
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

    /// `rows` rows read together through this layer: `x` is their hidden
    /// rows, the first of them at `position`, before and after.
    fn step(
        &mut self,
        x: &mut [f32],
        position: usize,
        rows: usize,
        s: &mut Scratch,
    ) -> CResult<()> {
        self.input_ln.step(x, rows, &mut s.normed)?;
        self.self_attn.step(position, rows, s)?;
        add(x, &s.out)?;
        self.post_attn_ln.step(x, rows, &mut s.normed)?;
        self.mlp.step(rows, s)?;
        add(x, &s.out)
    }
}

impl QwenLora {
    /// The logits after the last of `tokens`, the first of which is at
    /// `position`, as [`QwenLora::hidden`] and [`QwenLora::logits`] give
    /// them with the cache on, through plain rows: one decoded token, or a
    /// prompt. `None`, with nothing read or written, when this model does
    /// not read rows that way: the caller takes the tensor path, which
    /// reads and writes the same cache.
    ///
    /// With `first` the caller wants the token with the largest logit and
    /// nothing else of the logits. Where the head is sketched that token is
    /// what comes back, chosen in two stages ([`sketch`]) from under half the
    /// bytes the whole head is; where it is not, the logits come back as
    /// they would have.
    pub(in crate::models::qwen) fn step(
        &mut self,
        tokens: &[u32],
        position: usize,
        first: bool,
    ) -> CResult<Option<Stepped>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        if self.serves && self.layers.iter().any(DecoderLayer::packs) {
            // The adapters' factors as these rows read them, made once
            // after they are set: by every core, a run of layers each. One
            // core takes most of a tenth of a second over a rank-16 expert,
            // which a request that routes to another expert would wait for.
            let threads = std::thread::available_parallelism().map_or(1, usize::from);
            let per = self.layers.len().div_ceil(threads).max(1);
            std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .layers
                    .chunks_mut(per)
                    .map(|layers| {
                        scope.spawn(move || layers.iter_mut().try_for_each(DecoderLayer::pack))
                    })
                    .collect();
                for handle in handles {
                    match handle.join() {
                        Ok(done) => done?,
                        Err(_) => bail!("a thread packing an adapter panicked"),
                    }
                }
                Ok(())
            })?;
        }
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
        let (vocabulary, width) = table.dims2()?;
        if let Some(token) = tokens.iter().find(|t| **t as usize >= vocabulary) {
            bail!("token {token} is past a vocabulary of {vocabulary}")
        }
        let s = &mut self.scratch;
        let mut x = std::mem::take(&mut s.x);
        let mut read = 0;
        // A long prompt a run of rows at a time, each run through every
        // layer: a later run attends to the rows the earlier ones cached.
        for run in tokens.chunks(ROWS_AT_ONCE) {
            x.clear();
            {
                let (storage, layout) = table.storage_and_layout();
                let all = held(&storage, layout, "the embeddings")?;
                for token in run {
                    let at = *token as usize * width;
                    x.extend_from_slice(&all[at..at + width]);
                }
            }
            for layer in &mut self.layers {
                layer.step(&mut x, position + read, run.len(), s)?;
            }
            read += run.len();
        }
        // The head reads the last row alone.
        let last = &x[x.len() - width..];
        self.norm.step(last, 1, &mut s.normed)?;
        s.x = x;
        // Through its tiles, or through candle's kernels where this CPU has
        // no tiles for the form it is in.
        let Some(head) = &self.lm_head_t else {
            let row = Tensor::from_slice(&s.normed, (1, 1, s.normed.len()), &Device::Cpu)?;
            let logits = self.logits(&row)?.flatten_all()?.to_vec1()?;
            return Ok(Some(Stepped::Logits(logits)));
        };
        quantize(&s.normed, 1, &mut s.quantized);
        if let (true, Some(sketch)) = (first && plain(&self.lm_head_w), &self.lm_head_s) {
            let (storage, layout) = self.lm_head_w.storage_and_layout();
            let dense = held(&storage, layout, "the head's dense rows")?;
            let token = sketch.first(&s.quantized[0], &s.normed, dense, &mut s.sketched)?;
            return Ok(Some(Stepped::First(token)));
        }
        // Every logit is written, so what the room held is nothing to it.
        let mut logits = std::mem::take(&mut s.logits);
        logits.resize(head.rows(), 0.0);
        tiles::multiply(&[head], &s.quantized[0], &mut [&mut logits])?;
        Ok(Some(Stepped::Logits(logits)))
    }

    /// Take back the room a token's logits were handed out in, for the next
    /// token's: over a vocabulary of 150 thousand it is 600 KB, which the
    /// allocator maps afresh and unmaps again at every token otherwise.
    pub(in crate::models::qwen) fn keep(&mut self, logits: Vec<f32>) {
        self.scratch.logits = logits;
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

    /// The rank of the adapter these tests read through.
    const RANK: usize = 8;

    /// A base of weights drawn from a seed, under the names a Qwen checkpoint
    /// gives them: the same base in every run, so what these tests find they
    /// find every time.
    fn weights(cfg: &Config) -> CResult<HashMap<String, Tensor>> {
        use rand::RngExt as _;
        let dev = Device::Cpu;
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let kv = cfg.num_key_value_heads * (h / cfg.num_attention_heads);
        let mut all = HashMap::new();
        let mut rng = StdRng::seed_from_u64(0x57E9);
        // Box-Muller: two uniforms make one standard normal.
        let mut normal = move || {
            let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.random::<f64>();
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        let mut draw = |name: String, shape: &[usize], mean: f32, spread: f32| -> CResult<()> {
            let values: Vec<f32> = (0..shape.iter().product::<usize>())
                .map(|_| mean + spread * normal())
                .collect();
            all.insert(name, Tensor::from_vec(values, shape, &dev)?);
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
    /// to values that are not nothing on the B side. Every factor is put as
    /// the copy a decoded token reads holds it, at eight bits of its rank's
    /// largest (`pack::as_held`): the copy is then the factors themselves,
    /// and the two paths are compared on their arithmetic alone.
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
        let mut model = QwenLora::build(cfg, &base, &lora, varmap, RANK, 2.0, serve)?;
        seed_lora_a(&model.varmap, 11)?;
        {
            let vars = model
                .varmap
                .data()
                .lock()
                .map_err(|_| candle_core::Error::msg("the varmap lock is poisoned"))?;
            let mut names: Vec<&String> = vars.keys().collect();
            names.sort();
            for (salt, name) in names.into_iter().enumerate() {
                let var = &vars[name];
                let values: Vec<f32> = match trained && name.ends_with("lora_b") {
                    true => (0..var.elem_count())
                        .map(|i| ((i + 31 * salt) as f32 * 0.37).sin() * 0.05)
                        .collect(),
                    false => var.flatten_all()?.to_vec1()?,
                };
                let mut values = values;
                pack::as_held(&mut values, RANK, name.ends_with("lora_a"));
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

    /// The logits a step gives when every one of them is asked for.
    fn stepped(model: &mut QwenLora, tokens: &[u32], position: usize) -> CResult<Option<Vec<f32>>> {
        match model.step(tokens, position, false)? {
            Some(Stepped::Logits(logits)) => Ok(Some(logits)),
            Some(Stepped::First(_)) => bail!("a token came back where the logits were asked for"),
            None => Ok(None),
        }
    }

    /// How far two rows of logits may be apart, of the largest logit. The two
    /// paths sum in different orders, and a last-bit difference in a row can
    /// round one of its values the other way when the row is quantized: one
    /// step of 127 in one value of a row 256 wide, where a real base's rows
    /// are 1536 wide. A step left out or done wrongly is off by far more.
    const CLOSE: f32 = 1e-2;

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
                    let Some(got) = stepped(&mut by_rows, &[token(i)], position)? else {
                        bail!("{serve:?}, {what}: token {i} was not stepped")
                    };
                    assert_eq!(got.len(), cfg.vocab_size);
                    let off = gap(&got, &want);
                    assert!(
                        off < CLOSE,
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
                        stepped(&mut by_rows, &[token(0)], PROMPT.len())?,
                        stepped(&mut bare, &[token(0)], PROMPT.len())?,
                    ) else {
                        bail!("{serve:?}: not stepped")
                    };
                    let moved = gap(&with, &without);
                    assert!(
                        moved > 3.0 * CLOSE,
                        "{serve:?}: the adapter moved the logits by {moved} of the peak, too little to have been compared"
                    );
                }
            }
        }
        Ok(())
    }

    /// A prompt read at once is its tokens read one at a time: the same
    /// logits and the same cache, to the bit, for a handful of rows and for
    /// more than are read at once, with a trained adapter on; and it is what
    /// the tensor path makes of the prompt, to the order of its float sums.
    #[test]
    fn a_prompt_read_at_once_is_its_tokens_read_one_by_one() -> CResult<()> {
        if !tiles::available() {
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let mut forms = vec![GgmlDType::Q4K, GgmlDType::Q8_0];
        if tiles::six_bit() {
            forms.push(GgmlDType::Q6K);
        }
        for serve in forms {
            for length in [1usize, 5, ROWS_AT_ONCE, 2 * ROWS_AT_ONCE + 6] {
                let prompt: Vec<u32> = (0..length).map(token).collect();
                let mut at_once = model(&cfg, &weights, Some(serve), true)?;
                let mut one_by_one = model(&cfg, &weights, Some(serve), true)?;
                let Some(got) = stepped(&mut at_once, &prompt, 0)? else {
                    bail!("{serve:?}: a prompt of {length} was not stepped")
                };
                let mut want = Vec::new();
                for (position, token) in prompt.iter().enumerate() {
                    let Some(logits) = stepped(&mut one_by_one, &[*token], position)? else {
                        bail!("{serve:?}: token {position} was not stepped")
                    };
                    want = logits;
                }
                assert_eq!(got, want, "{serve:?}, a prompt of {length}");
                // The caches are the same too: the next token says so.
                let (Some(got), Some(want)) = (
                    stepped(&mut at_once, &[token(99)], length)?,
                    stepped(&mut one_by_one, &[token(99)], length)?,
                ) else {
                    bail!("{serve:?}: the token after the prompt was not stepped")
                };
                assert_eq!(got, want, "{serve:?}, after a prompt of {length}");

                let mut by_tensors = model(&cfg, &weights, Some(serve), true)?;
                let off = gap(
                    &at_once_logits(&cfg, &weights, serve, &prompt)?,
                    &forwarded(&mut by_tensors, &prompt, 0)?,
                );
                assert!(
                    off < CLOSE,
                    "{serve:?}, a prompt of {length}: off the tensor path by {off} of the peak"
                );
            }
        }
        Ok(())
    }

    /// The logits after `prompt` read at once by a fresh model.
    fn at_once_logits(
        cfg: &Config,
        weights: &HashMap<String, Tensor>,
        serve: GgmlDType,
        prompt: &[u32],
    ) -> CResult<Vec<f32>> {
        match stepped(&mut model(cfg, weights, Some(serve), true)?, prompt, 0)? {
            Some(logits) => Ok(logits),
            None => bail!("a prompt was not stepped"),
        }
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
            if stepped(&mut mixed, &[token(i)], position)?.is_none() {
                bail!("token {i} was not stepped")
            }
        }
        let position = PROMPT.len() + 6;
        let want = forwarded(&mut by_tensors, &[token(6)], position)?;
        let got = forwarded(&mut mixed, &[token(6)], position)?;
        let off = gap(&got, &want);
        assert!(off < CLOSE, "off by {off} of the peak");

        // And a new sequence starts from nothing on both.
        by_tensors.clear_cache();
        mixed.clear_cache();
        let want = forwarded(&mut by_tensors, &[token(1)], 0)?;
        let Some(got) = stepped(&mut mixed, &[token(1)], 0)? else {
            bail!("a first token was not stepped")
        };
        assert!(gap(&got, &want) < CLOSE, "a sequence of one token");
        Ok(())
    }

    /// Asked for the head's first alone, a step gives the token whose dense
    /// row has the largest product with the row the head is given, at every
    /// served form, and leaves what a step that gives the logits leaves.
    #[test]
    fn the_first_asked_for_is_the_dense_rows_first() -> CResult<()> {
        if !sketch::available() {
            eprintln!("skipped: this CPU multiplies no sketch of the head");
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        for serve in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let mut chosen = model(&cfg, &weights, Some(serve), true)?;
            let mut whole = model(&cfg, &weights, Some(serve), true)?;
            let dense: Vec<f32> = chosen.lm_head_w.flatten_all()?.to_vec1()?;
            for i in 0..40 {
                let Some(Stepped::First(got)) = chosen.step(&[token(i)], i, true)? else {
                    bail!("{serve:?}: token {i} did not come back chosen")
                };
                if stepped(&mut whole, &[token(i)], i)?.is_none() {
                    bail!("{serve:?}: token {i} was not stepped")
                }
                let normed = &chosen.scratch.normed;
                assert_eq!(
                    normed, &whole.scratch.normed,
                    "{serve:?}, token {i}: the row the head is given"
                );
                let mut want = (f32::NEG_INFINITY, 0);
                for (at, row) in dense.chunks_exact(normed.len()).enumerate() {
                    let product = slices::dot(row, normed);
                    if product > want.0 {
                        want = (product, at as u32);
                    }
                }
                assert_eq!(got, want.1, "{serve:?}, token {i}");
            }
        }
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
        assert!(dense.step(&[1], 0, false)?.is_none(), "a dense base");
        if !tiles::available() {
            return Ok(());
        }
        let mut served = model(&cfg, &weights, Some(GgmlDType::Q4K), false)?;
        served.set_grad(true);
        assert!(
            served.step(&[1], 0, false)?.is_none(),
            "an adapter with a gradient"
        );
        served.set_grad(false);
        assert!(
            served.step(&[64], 0, false).is_err(),
            "a token past the vocabulary"
        );
        assert!(served.step(&[1], 0, false)?.is_some());
        assert!(served.step(&[2], 1, false)?.is_some());
        assert!(
            served.step(&[3], 256, false).is_err(),
            "a position past the rotary table"
        );
        Ok(())
    }
}

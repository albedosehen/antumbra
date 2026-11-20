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
    /// The blend of experts each row in hand is of, or nothing where every
    /// row is of the adapter's own ([`LoraLinear::adapter_of`]); and the
    /// blend the rows of one sequence read alone are of.
    of: Vec<u16>,
    blend: u16,
    /// What the head's two stages are worked out in, where a token is
    /// chosen through its sketch: a room for each row chosen at once.
    sketched: Vec<sketch::Room>,
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
    /// The runs of rows that read an adapter here, in the rows' order: rows
    /// answered together may be of different blends of experts, and each
    /// run of one blend reads that blend's factors.
    runs: Vec<Run<'a>>,
}

/// Rows `from..to` of a dispatch, which read one blend's packed factors.
struct Run<'a> {
    from: usize,
    to: usize,
    pack: &'a pack::Packed,
    /// Where these rows' `scale · A · x` starts in the room [`through`]
    /// makes: a row's `rank` values together, row after row.
    at: usize,
}

impl Run<'_> {
    fn rows(&self) -> usize {
        self.to - self.from
    }

    /// The values these rows' `scale · A · x` take.
    fn room(&self) -> usize {
        self.rows() * self.pack.rank
    }

    /// The part of this run within rows `from..to`: its first row and its
    /// last, and where its first row's values start in the room.
    fn within(&self, from: usize, to: usize) -> Option<(usize, usize, usize)> {
        let (a, b) = (self.from.max(from), self.to.min(to));
        (a < b).then(|| (a, b, self.at + (a - self.from) * self.pack.rank))
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

    /// The factors a row of blend `blend` reads here: the adapter's own for
    /// blend 0, and the ones kept for any other ([`LoraLinear::keep_pack`]).
    fn adapter_of(&self, blend: u16) -> Option<&pack::Packed> {
        match blend {
            0 => self.adapter(),
            n => self
                .kept
                .get(usize::from(n) - 1)
                .and_then(Option::as_ref)
                .filter(|p| p.rank > 0),
        }
    }

    /// The runs of `rows` rows that read an adapter here: `of` is each
    /// row's blend, or nothing where every row is of the adapter's own.
    fn runs(&self, of: &[u16], rows: usize) -> Vec<Run<'_>> {
        let mut runs: Vec<Run> = Vec::new();
        let mut from = 0;
        while from < rows {
            let blend = of.get(from).copied().unwrap_or(0);
            let mut to = from + 1;
            while to < rows && of.get(to).copied().unwrap_or(0) == blend {
                to += 1;
            }
            if let Some(pack) = self.adapter_of(blend) {
                runs.push(Run {
                    from,
                    to,
                    pack,
                    at: 0,
                });
            }
            from = to;
        }
        runs
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
/// what `hold!` holds for them: the projection reads `rows` rows of `cols`
/// values, each of the blend `of` gives it.
macro_rules! weights {
    ($group:expr, $held:expr, $cols:expr, $of:expr, $rows:expr) => {{
        let mut weights = Vec::with_capacity($group.len());
        for (l, bias) in $group.iter().zip($held.iter()) {
            let tiled = l.tiled()?;
            let bias = match bias {
                Some((storage, layout)) => Some(held(storage, layout, "a projection's bias")?),
                None => None,
            };
            let runs = l.runs($of, $rows);
            let rows = tiled.rows();
            if tiled.cols() != $cols
                || bias.is_some_and(|bias| bias.len() != rows)
                || runs
                    .iter()
                    .any(|r| r.pack.cols() != $cols || r.pack.rows() != rows)
            {
                bail!(
                    "a projection of {rows} by {} and its adapter do not fit a row of {}",
                    tiled.cols(),
                    $cols
                )
            }
            weights.push(Weight { tiled, bias, runs });
        }
        weights
    }};
}

/// Room in `room` for every run's `scale · A · x`: one weight's runs after
/// another's, a row's `rank` values together. Each run is told where its
/// rows start.
fn through(weights: &mut [Weight], room: &mut Vec<f32>) {
    let mut at = 0;
    for run in weights.iter_mut().flat_map(|w| w.runs.iter_mut()) {
        run.at = at;
        at += run.room();
    }
    room.clear();
    room.resize(at, 0.0);
}

/// A unit of work for each weight's adapter when any of `weights` has one
/// on for any row, and none otherwise.
fn adapted(weights: &[Weight]) -> usize {
    match weights.iter().any(|w| !w.runs.is_empty()) {
        true => weights.len(),
        false => 0,
    }
}

/// Weight `w`'s `scale · A · x` for every row of `x` that reads an adapter,
/// into those rows' part of `t`.
///
/// SAFETY (for the caller): `t` is the room [`through`] made for these
/// weights, and no other thread reads or writes this weight's part of it.
unsafe fn a_unit(w: &Weight, x: &[f32], t: &Shared<f32>) {
    for run in &w.runs {
        let cols = run.pack.cols();
        let out = t.part(run.at, run.room());
        run.pack
            .through(0, cols, &x[run.from * cols..], cols, run.rows(), out);
        run.pack.scaled(out);
    }
}

/// To the sixteens `first..last` of `w`'s products for every one of `rows`
/// rows in `out`: for each run of rows that reads an adapter, its `B` rows
/// times those rows' part of `t` (their `scale · A · x`), then the bias.
///
/// SAFETY (for the caller): `out` is room for `w`'s products of every row, a
/// row's after another's, and no other thread reads or writes these sixteens
/// of them.
unsafe fn finish(w: &Weight, rows: usize, first: usize, last: usize, t: &[f32], out: &Shared<f32>) {
    let (first, n) = (first * tiles::ROWS, (last - first) * tiles::ROWS);
    let width = w.tiled.rows();
    for run in &w.runs {
        run.pack.add(
            first,
            n,
            &t[run.at..run.at + run.room()],
            run.rows(),
            out.at().add(run.from * width + first),
            width,
        );
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
    of: &[u16],
) -> CResult<()> {
    if rows == 0 || !x.len().is_multiple_of(rows) {
        bail!("{} values are not {rows} rows", x.len())
    }
    let held = hold!(group);
    let mut weights = weights!(group, held, x.len() / rows, of, rows);
    if weights
        .iter()
        .zip(outs.iter())
        .any(|(w, out)| rows * w.tiled.rows() != out.len())
    {
        bail!("a group of projections and the places for their products do not fit")
    }
    quantize(x, rows, quantized);
    let quantized = &quantized[..rows];

    through(&mut weights, room);
    let weights = weights;
    let t = Shared::of(room.as_mut_slice());
    let adapted = adapted(&weights);
    let shared: Vec<Shared<f32>> = outs.iter_mut().map(|out| Shared::of(out)).collect();
    let sixteens: usize = weights.iter().map(|w| w.tiled.sixteens()).sum();
    // The adapters, then the group's sixteens, one weight's after another's.
    let bases = |first: usize, last: usize| {
        for w in &weights[first.min(adapted)..last.min(adapted)] {
            // SAFETY: `t` is the room made just above for these weights and
            // rows, and a unit is done by one thread.
            unsafe { a_unit(w, x, &t) };
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
        for (w, out) in weights.iter().zip(&shared) {
            if let Some((from, to)) = within(first, last, before, w.tiled.sixteens()) {
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
            &s.of,
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
            _ => attend::attend_rows(q, rows, store, &mut s.attended, &mut s.attention)?,
        }
        s.out.resize(rows * self.hidden_size, 0.0);
        products(
            [&self.o_proj],
            &s.attended,
            rows,
            [&mut s.out],
            &mut s.quantized,
            &mut s.through,
            &s.of,
        )
    }
}

impl Attention {
    /// Room for the caches of `n` sequences decoded side by side. The first
    /// is the cache every other path reads and writes.
    fn sequences(&mut self, n: usize) {
        while self.others.len() + 1 < n {
            self.others
                .push(kv::KvStore::new(self.num_kv_heads, self.head_dim));
        }
    }

    /// Sequence `slot`'s cache.
    fn cache_of(&mut self, slot: usize) -> CResult<&mut kv::KvStore> {
        let store = match slot {
            0 => self.kv_store.as_mut(),
            n => self.others.get_mut(n - 1),
        };
        match store {
            Some(store) => Ok(store),
            None => bail!("no cache of plain rows was made for sequence {slot}"),
        }
    }

    /// Put sequence `slot`'s cache where the first's is, and the first's
    /// where that one's was: what reads one sequence then reads that one.
    fn exchange(&mut self, slot: usize) -> CResult<()> {
        if slot == 0 {
            return Ok(());
        }
        let (Some(first), Some(other)) = (self.kv_store.as_mut(), self.others.get_mut(slot - 1))
        else {
            bail!("no cache of plain rows was made for sequence {slot}")
        };
        std::mem::swap(first, other);
        Ok(())
    }

    /// Attention for rows of several sequences read in one pass: `s.normed`
    /// is their normed rows, row `i` the row of sequence `at[i].0` at
    /// position `at[i].1`. Each row's keys and values join its own
    /// sequence's cache, and it attends to that cache up to its own position
    /// and no other, so each is the row [`Attention::step`] computes for
    /// its sequence by itself. A sequence may have several rows in the pass,
    /// at positions one after another in the order they are given: a later
    /// one attends to the earlier ones as to the rest of the cache.
    fn step_each(&mut self, at: &[(usize, usize)], s: &mut Scratch) -> CResult<()> {
        let rows = at.len();
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
            &s.of,
        )?;
        // Every row is turned before any is cached, so a position past the
        // table leaves every cache as it was.
        for ((q, k), (_, position)) in q
            .chunks_exact_mut(queries)
            .zip(k.chunks_exact_mut(cached))
            .zip(at)
        {
            self.rotary.turn(q, self.head_dim, *position)?;
            self.rotary.turn(k, self.head_dim, *position)?;
        }
        s.attended.resize(rows * queries, 0.0);
        for ((k, v), (slot, _)) in k.chunks_exact(cached).zip(v.chunks_exact(cached)).zip(at) {
            self.cache_of(*slot)?.append(k, v)?;
        }
        // Each row against its own sequence's cache, less the rows of this
        // pass that come after it in the sequence: one dispatch, a few rows'
        // runs of positions shared out among the cores and many rows a core
        // each.
        let mut each = Vec::with_capacity(rows);
        for (i, (slot, _)) in at.iter().enumerate() {
            let store = match slot {
                0 => self.kv_store.as_ref(),
                n => self.others.get(n - 1),
            };
            let Some(store) = store else {
                bail!("no cache of plain rows was made for sequence {slot}")
            };
            let after = at[i + 1..]
                .iter()
                .take_while(|(next, _)| next == slot)
                .count();
            each.push((store, store.len() - after));
        }
        attend::attend_each(q, &each, &mut s.attended, &mut s.attention)?;
        s.out.resize(rows * self.hidden_size, 0.0);
        products(
            [&self.o_proj],
            &s.attended,
            rows,
            [&mut s.out],
            &mut s.quantized,
            &mut s.through,
            &s.of,
        )
    }
}

/// `silu(gate) · up` into `out`.
fn gate_times(gate: &[f32], up: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if tiles::wide() {
        // SAFETY: `available` found AVX-512F among the features it names.
        unsafe { wide::silu_times(gate, up, out) };
        return;
    } else if tiles::available() {
        // SAFETY: a CPU that tiles without AVX-512 has AVX2 and FMA.
        unsafe { wide::avx2::silu_times(gate, up, out) };
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
        let mut wide = weights!(wide_group, wide_held, x.len() / rows, &s.of, rows);
        through(&mut wide, &mut s.through);
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
        let mut narrow = weights!(narrow_group, narrow_held, width, &s.of, rows);
        // The way down's `A` is read a block at a time, so its room is made
        // for one block and kept for each: `per_block` values a block.
        through(&mut narrow, &mut s.through_down);
        let per_block = s.through_down.len();
        let [down] = narrow.as_slice() else {
            bail!("an MLP's way down is one projection")
        };
        let blocks = width / BLOCK;

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
        s.parts.resize(blocks * per_block, 0.0);
        let parts = Shared::of(&mut s.parts);
        s.out.resize(rows * down.tiled.rows(), 0.0);
        let out = Shared::of(&mut s.out);

        // The wide adapters, then the wide rows: a unit is a sixteen of the
        // gate's rows and the same sixteen of the other side's, so the core
        // that gates a block holds both.
        let t_wide = Shared::of(s.through.as_mut_slice());
        let adapted = adapted(&wide);
        let sixteens = gate.tiled.sixteens();
        let wide_phase = |first: usize, last: usize| {
            for w in &wide[first.min(adapted)..last.min(adapted)] {
                // SAFETY: `t_wide` is the room made just above for these
                // weights and rows, and a unit is done by one thread.
                unsafe { a_unit(w, x, &t_wide) };
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
                    for (side, w) in wide.iter().enumerate() {
                        let at = both.at().add((side * rows + from) * width + block * BLOCK);
                        for run in &w.runs {
                            let Some((a, b, start)) = run.within(from, to) else {
                                continue;
                            };
                            let t = &t[start..start + (b - a) * run.pack.rank];
                            run.pack.add(
                                block * BLOCK,
                                BLOCK,
                                t,
                                b - a,
                                at.add((a - from) * width),
                                width,
                            );
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
                    for run in &down.runs {
                        let Some((a, b, start)) = run.within(from, to) else {
                            continue;
                        };
                        let part = parts.part(block * per_block + start, (b - a) * run.pack.rank);
                        run.pack.through(
                            block * BLOCK,
                            BLOCK,
                            &m.as_flattened()[(a - from) * BLOCK..],
                            BLOCK,
                            b - a,
                            part,
                        );
                    }
                }
            }
        };
        // The way down's `scale · A` times each whole gated row: its
        // blocks' parts, added in their order.
        let t_down = Shared::of(s.through_down.as_mut_slice());
        let gathered = || {
            // SAFETY: every block's part was written before this ran, on one
            // thread, and nothing else reads or writes either until it is
            // done.
            let (parts, t_down) = unsafe { (parts.all(), t_down.part(0, per_block)) };
            for (at, t) in t_down.iter_mut().enumerate() {
                *t = (0..blocks).map(|block| parts[block * per_block + at]).sum();
            }
            for run in &down.runs {
                run.pack.scaled(&mut t_down[run.at..run.at + run.room()]);
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
                    then: (per_block > 0).then_some(&gathered as &(dyn Fn() + Sync)),
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

impl DecoderLayer {
    /// Rows of several sequences through this layer
    /// ([`Attention::step_each`]): `x` is their hidden rows, before and
    /// after.
    fn step_each(&mut self, x: &mut [f32], at: &[(usize, usize)], s: &mut Scratch) -> CResult<()> {
        let rows = at.len();
        self.input_ln.step(x, rows, &mut s.normed)?;
        self.self_attn.step_each(at, s)?;
        add(x, &s.out)?;
        self.post_attn_ln.step(x, rows, &mut s.normed)?;
        self.mlp.step(rows, s)?;
        add(x, &s.out)
    }
}

/// One row for [`QwenLora::step_each`]: the sequence it is of, the token,
/// and that token's position in the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::models::qwen) struct Next {
    pub(in crate::models::qwen) sequence: usize,
    pub(in crate::models::qwen) token: u32,
    pub(in crate::models::qwen) position: usize,
    /// The blend of experts the sequence is answered with: 0 for the
    /// adapter's own factors, or one kept with [`QwenLora::keep_blend`].
    pub(in crate::models::qwen) blend: u16,
}

impl QwenLora {
    /// Whether this model reads rows as plain values, with the adapters'
    /// factors packed for it when it does: what [`QwenLora::step`] and
    /// [`QwenLora::step_each`] ask before they read or write anything.
    fn steps_plainly(&mut self) -> CResult<bool> {
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
        Ok((self.lm_head_t.is_some() || self.lm_head_q.is_some())
            && self.device.is_cpu()
            && self.dtype == DType::F32
            && plain(self.embed_tokens.embeddings())
            && self.norm.steps()
            && self.layers.iter().all(DecoderLayer::steps))
    }

    /// Room for `n` sequences decoded side by side ([`QwenLora::step_each`]),
    /// each with a cache of its own in every layer.
    pub(in crate::models::qwen) fn sequences(&mut self, n: usize) {
        for layer in &mut self.layers {
            layer.self_attn.sequences(n);
        }
    }

    /// Forget what sequence `sequence` has read.
    pub(in crate::models::qwen) fn clear_sequence(&mut self, sequence: usize) -> CResult<()> {
        for layer in &mut self.layers {
            layer.self_attn.cache_of(sequence)?.reset();
        }
        Ok(())
    }

    /// Keep the adapter's factors as they are set now as blend `blend`'s
    /// (one or more): what a sequence answered with that blend reads from
    /// then on, whatever the factors are set to after. `false`, with nothing
    /// kept, when this model does not read rows as plain values.
    pub(in crate::models::qwen) fn keep_blend(&mut self, blend: u16) -> CResult<bool> {
        if blend == 0 {
            bail!("blend 0 is the adapter's own factors, which are not kept")
        }
        if !self.steps_plainly()? {
            return Ok(false);
        }
        for layer in &mut self.layers {
            layer.keep_packs(usize::from(blend));
        }
        Ok(true)
    }

    /// [`QwenLora::step`] for sequence `sequence`, answered with blend
    /// `blend`: its prompt, read into its own cache. `None` as that gives
    /// it, with nothing read or written.
    pub(in crate::models::qwen) fn step_in(
        &mut self,
        sequence: usize,
        blend: u16,
        tokens: &[u32],
        position: usize,
        first: bool,
    ) -> CResult<Option<Stepped>> {
        for layer in &mut self.layers {
            layer.self_attn.exchange(sequence)?;
        }
        self.scratch.blend = blend;
        let stepped = self.step(tokens, position, first);
        self.scratch.blend = 0;
        for layer in &mut self.layers {
            layer.self_attn.exchange(sequence)?;
        }
        stepped
    }

    /// A decoded token after each of several rows, in one pass over the
    /// weights: for each of `each`, what [`QwenLora::step`] gives that
    /// row's token read alone after the rows of its sequence before it, to
    /// the bit. A weight is read from memory once for all of them, which is
    /// what bounds a decoded token on a CPU; each row is computed as it would
    /// be alone, and attends to its own sequence's cache up to its own
    /// position. `None`, with nothing read or written, when this model does
    /// not read rows as plain values.
    ///
    /// A sequence may be given several rows, one after another in `each` at
    /// positions one after another: the token in hand and tokens guessed to
    /// follow it ([`ahead`]). The rows after a guess that was not the
    /// model's own token were read on text that was never written, and their
    /// places in the sequence's cache are the caller's to cut back
    /// ([`QwenLora::truncate_sequence`]).
    pub(in crate::models::qwen) fn step_each(
        &mut self,
        each: &[Next],
        first: bool,
    ) -> CResult<Option<Vec<Stepped>>> {
        if each.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if each.len() > ROWS_AT_ONCE {
            bail!(
                "{} sequences at once, where {ROWS_AT_ONCE} rows are read together",
                each.len()
            )
        }
        for (i, next) in each.iter().enumerate() {
            // A sequence's rows come together, each at the position after
            // the one before it, and with the one blend.
            match each[..i].iter().rposition(|n| n.sequence == next.sequence) {
                None => {}
                Some(before) if before + 1 == i => {
                    let last = &each[before];
                    if (last.position + 1, last.blend) != (next.position, next.blend) {
                        bail!(
                            "sequence {} is given rows at {} and {} at once",
                            next.sequence,
                            last.position,
                            next.position
                        )
                    }
                }
                Some(_) => bail!(
                    "sequence {} is given rows in two places at once",
                    next.sequence
                ),
            }
        }
        if !self.steps_plainly()? {
            return Ok(None);
        }
        let table = self.embed_tokens.embeddings();
        let (vocabulary, width) = table.dims2()?;
        if let Some(next) = each.iter().find(|n| n.token as usize >= vocabulary) {
            bail!("token {} is past a vocabulary of {vocabulary}", next.token)
        }
        let rows = each.len();
        let at: Vec<(usize, usize)> = each.iter().map(|n| (n.sequence, n.position)).collect();
        let s = &mut self.scratch;
        s.of.clear();
        if each.iter().any(|n| n.blend != 0) {
            s.of.extend(each.iter().map(|n| n.blend));
        }
        let mut x = std::mem::take(&mut s.x);
        x.clear();
        {
            let (storage, layout) = table.storage_and_layout();
            let all = held(&storage, layout, "the embeddings")?;
            for next in each {
                let from = next.token as usize * width;
                x.extend_from_slice(&all[from..from + width]);
            }
        }
        for layer in &mut self.layers {
            layer.step_each(&mut x, &at, s)?;
        }
        s.of.clear();
        self.norm.step(&x, rows, &mut s.normed)?;
        s.x = x;
        let mut stepped = Vec::with_capacity(rows);
        let Some(head) = &self.lm_head_t else {
            // Through candle's kernels, where this CPU has no tiles for the
            // form the head is in.
            let normed = s.normed.clone();
            for normed in normed.chunks_exact(width) {
                let row = Tensor::from_slice(normed, (1, 1, width), &Device::Cpu)?;
                let logits = self.logits(&row)?.flatten_all()?.to_vec1()?;
                stepped.push(Stepped::Logits(logits));
            }
            return Ok(Some(stepped));
        };
        quantize(&s.normed, rows, &mut s.quantized);
        let sketch = match (first && plain(&self.lm_head_w), &self.lm_head_s) {
            (true, Some(sketch)) => Some(sketch),
            _ => None,
        };
        if let Some(sketch) = sketch {
            // Every row's first in one reading of the sketch.
            let (storage, layout) = self.lm_head_w.storage_and_layout();
            let dense = held(&storage, layout, "the head's dense rows")?;
            if s.sketched.len() < rows {
                s.sketched.resize_with(rows, Default::default);
            }
            let mut tokens = vec![0u32; rows];
            sketch.first_of_each(
                &s.quantized[..rows],
                &s.normed,
                dense,
                &mut s.sketched[..rows],
                &mut tokens,
            )?;
            stepped.extend(tokens.into_iter().map(Stepped::First));
            return Ok(Some(stepped));
        }
        for quantized in &s.quantized[..rows] {
            let mut logits = vec![0f32; head.rows()];
            tiles::multiply(&[head], quantized, &mut [&mut logits])?;
            stepped.push(Stepped::Logits(logits));
        }
        Ok(Some(stepped))
    }

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
        let Some(width) = self.read(tokens, position)? else {
            return Ok(None);
        };
        let s = &mut self.scratch;
        // The head reads the last row alone.
        self.norm
            .step(&s.x[s.x.len() - width..], 1, &mut s.normed)?;
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
            if s.sketched.is_empty() {
                s.sketched.push(sketch::Room::default());
            }
            let token = sketch.first(&s.quantized[0], &s.normed, dense, &mut s.sketched[0])?;
            return Ok(Some(Stepped::First(token)));
        }
        // Every logit is written, so what the room held is nothing to it.
        let mut logits = std::mem::take(&mut s.logits);
        logits.resize(head.rows(), 0.0);
        tiles::multiply(&[head], &s.quantized[0], &mut [&mut logits])?;
        Ok(Some(Stepped::Logits(logits)))
    }

    /// Whether [`QwenLora::step_ahead`] has a head to answer with: the
    /// sketch, and the dense rows it decides by.
    pub(in crate::models::qwen) fn reads_ahead(&self) -> bool {
        self.lm_head_s.is_some() && plain(&self.lm_head_w)
    }

    /// The token the head ranks first after every one of `tokens`, the
    /// first of which is at `position`, read in one pass: what
    /// [`QwenLora::step`] gives for each of them read one after another, with
    /// the first asked for. Where [`QwenLora::step_each`] reads a row from
    /// each of several sequences, this reads several rows of one, each
    /// attending to the rows before it in the pass as to the cache. The rows
    /// after one whose token is not the next of `tokens` were read on text
    /// that is not the model's, and their places in the cache are the
    /// caller's to cut back ([`QwenLora::truncate`]).
    ///
    /// `None`, with nothing read or written, when the model does not read
    /// rows this way, its head is not sketched, or there are more tokens
    /// than are read at once.
    pub(in crate::models::qwen) fn step_ahead(
        &mut self,
        tokens: &[u32],
        position: usize,
    ) -> CResult<Option<Vec<u32>>> {
        if !self.reads_ahead() || tokens.len() > ROWS_AT_ONCE {
            return Ok(None);
        }
        let Some(width) = self.read(tokens, position)? else {
            return Ok(None);
        };
        let Some(sketch) = &self.lm_head_s else {
            bail!("a head that reads ahead has a sketch")
        };
        let (rows, s) = (tokens.len(), &mut self.scratch);
        self.norm
            .step(&s.x[s.x.len() - rows * width..], rows, &mut s.normed)?;
        quantize(&s.normed, rows, &mut s.quantized);
        let (storage, layout) = self.lm_head_w.storage_and_layout();
        let dense = held(&storage, layout, "the head's dense rows")?;
        if s.sketched.len() < rows {
            s.sketched.resize_with(rows, Default::default);
        }
        // Every row's first in one reading of the sketch.
        let mut firsts = vec![0u32; rows];
        sketch.first_of_each(
            &s.quantized[..rows],
            &s.normed,
            dense,
            &mut s.sketched[..rows],
            &mut firsts,
        )?;
        Ok(Some(firsts))
    }

    /// Forget the positions from `len` on in every layer's cache of plain
    /// rows: the first sequence's, which is the one [`QwenLora::step`] and
    /// [`QwenLora::step_ahead`] read and write.
    pub(in crate::models::qwen) fn truncate(&mut self, len: usize) {
        for layer in &mut self.layers {
            layer.truncate(len);
        }
    }

    /// Forget the positions from `len` on in sequence `sequence`'s cache,
    /// in every layer.
    pub(in crate::models::qwen) fn truncate_sequence(
        &mut self,
        sequence: usize,
        len: usize,
    ) -> CResult<()> {
        for layer in &mut self.layers {
            layer.self_attn.cache_of(sequence)?.truncate(len);
        }
        Ok(())
    }

    /// `tokens`, the first of which is at `position`, through every layer
    /// as plain rows, their keys and values joined to the cache: the hidden
    /// rows of the last run of them are left in the scratch, and what comes
    /// back is a row's width. `None`, with nothing read or written, when
    /// this model does not read rows that way.
    fn read(&mut self, tokens: &[u32], position: usize) -> CResult<Option<usize>> {
        if tokens.is_empty() || !self.steps_plainly()? {
            return Ok(None);
        }
        let table = self.embed_tokens.embeddings();
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
            s.of.clear();
            if s.blend != 0 {
                s.of.resize(run.len(), s.blend);
            }
            for layer in &mut self.layers {
                layer.step(&mut x, position + read, run.len(), s)?;
            }
            s.of.clear();
            read += run.len();
        }
        s.x = x;
        Ok(Some(width))
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
            for length in [1usize, 2, 3, 5, 7, ROWS_AT_ONCE, 2 * ROWS_AT_ONCE + 6] {
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

    /// Several sequences decoded in one pass give, token for token, the
    /// logits each gives decoded alone, at every served form: with prompts
    /// of different lengths, a sequence that joins after the others have
    /// begun, and one that ends before them.
    #[test]
    fn sequences_decoded_together_are_each_decoded_alone() -> CResult<()> {
        if !tiles::available() {
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let mut forms = vec![GgmlDType::Q4K, GgmlDType::Q8_0];
        if tiles::six_bit() {
            forms.push(GgmlDType::Q6K);
        }
        // Each sequence's prompt length, the step it joins at, and the
        // number of tokens it decodes.
        let plan = [(3usize, 0usize, 7usize), (1, 0, 3), (9, 2, 6), (4, 0, 9)];
        for serve in forms {
            let mut together = model(&cfg, &weights, Some(serve), true)?;
            together.sequences(plan.len());
            let mut alone = Vec::new();
            for _ in &plan {
                alone.push(model(&cfg, &weights, Some(serve), true)?);
            }
            // Where each sequence stands: the tokens it has read.
            let mut read: Vec<Vec<u32>> = vec![Vec::new(); plan.len()];
            for step in 0..12 {
                let mut each = Vec::new();
                for (i, (prompt, joins, decodes)) in plan.iter().enumerate() {
                    if step < *joins || step >= joins + decodes {
                        continue;
                    }
                    if step == *joins {
                        // The prompt, read into the sequence's own cache.
                        let tokens: Vec<u32> = (0..*prompt).map(|t| token(7 * i + t)).collect();
                        let (Some(Stepped::Logits(got)), Some(want)) = (
                            together.step_in(i, 0, &tokens, 0, false)?,
                            stepped(&mut alone[i], &tokens, 0)?,
                        ) else {
                            bail!("{serve:?}: sequence {i}'s prompt was not stepped")
                        };
                        assert_eq!(got, want, "{serve:?}: sequence {i}'s prompt");
                        read[i] = tokens;
                    }
                    let next = token(31 * i + step);
                    each.push(Next {
                        sequence: i,
                        token: next,
                        position: read[i].len(),
                        blend: 0,
                    });
                    read[i].push(next);
                }
                let Some(got) = together.step_each(&each, false)? else {
                    bail!("{serve:?}: step {step} was not stepped")
                };
                assert_eq!(got.len(), each.len());
                for (next, got) in each.iter().zip(got) {
                    let Stepped::Logits(got) = got else {
                        bail!("{serve:?}: logits were asked for")
                    };
                    let Some(want) =
                        stepped(&mut alone[next.sequence], &[next.token], next.position)?
                    else {
                        bail!(
                            "{serve:?}: sequence {} was not stepped alone",
                            next.sequence
                        )
                    };
                    assert_eq!(
                        got,
                        want,
                        "{serve:?}, step {step}: sequence {} beside {} others",
                        next.sequence,
                        each.len() - 1
                    );
                }
            }
            // A sequence let go starts again from nothing.
            together.clear_sequence(2)?;
            let fresh = [token(5), token(6)];
            let mut again = model(&cfg, &weights, Some(serve), true)?;
            let (Some(Stepped::Logits(got)), Some(want)) = (
                together.step_in(2, 0, &fresh, 0, false)?,
                stepped(&mut again, &fresh, 0)?,
            ) else {
                bail!("{serve:?}: a sequence begun again was not stepped")
            };
            assert_eq!(got, want, "{serve:?}: a sequence begun again");
            // And a sequence's rows in a step come together, each at the
            // position after the one before it and with the one blend.
            let row = |sequence: usize, position: usize, blend: u16| Next {
                sequence,
                token: token(1),
                position,
                blend,
            };
            for (what, rows) in [
                ("one position twice", vec![row(1, 4, 0), row(1, 4, 0)]),
                ("a position skipped", vec![row(1, 4, 0), row(1, 6, 0)]),
                (
                    "in two places",
                    vec![row(1, 4, 0), row(0, 9, 0), row(1, 5, 0)],
                ),
                ("two blends", vec![row(1, 4, 0), row(1, 5, 1)]),
            ] {
                assert!(together.step_each(&rows, false).is_err(), "{what}");
            }
        }
        Ok(())
    }

    /// Set the model's `B` factors to others, as a second expert's would be.
    fn other_factors(model: &mut QwenLora, phase: f32) -> CResult<()> {
        {
            let vars = model
                .varmap
                .data()
                .lock()
                .map_err(|_| candle_core::Error::msg("the varmap lock is poisoned"))?;
            let mut names: Vec<&String> = vars.keys().filter(|n| n.ends_with("lora_b")).collect();
            names.sort();
            for (salt, name) in names.into_iter().enumerate() {
                let var = &vars[name];
                let mut values: Vec<f32> = (0..var.elem_count())
                    .map(|i| ((i + 17 * salt) as f32 * 0.53 + phase).cos() * 0.05)
                    .collect();
                pack::as_held(&mut values, RANK, false);
                var.set(&Tensor::from_vec(values, var.shape(), &Device::Cpu)?)?;
            }
        }
        model.adapters_changed();
        Ok(())
    }

    /// Sequences answered with different blends of experts in one pass give,
    /// token for token, the logits each gives alone on a model that holds
    /// its blend and no other: one expert, another, and the base alone,
    /// with two sequences of one blend apart from each other.
    #[test]
    fn sequences_of_different_blends_are_each_decoded_alone() -> CResult<()> {
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
            // One model holding three blends: the first expert kept as blend
            // 1, the base alone as blend 2, and a second expert as its own.
            let mut together = model(&cfg, &weights, Some(serve), true)?;
            assert!(
                together.keep_blend(1)?,
                "{serve:?}: the first expert is kept"
            );
            together.set_lora(false);
            assert!(together.keep_blend(2)?, "{serve:?}: the base alone is kept");
            together.set_lora(true);
            other_factors(&mut together, 0.7)?;
            assert!(together.keep_blend(0).is_err(), "blend 0 is not kept");

            let first = || model(&cfg, &weights, Some(serve), true);
            let second = || {
                let mut m = model(&cfg, &weights, Some(serve), true)?;
                other_factors(&mut m, 0.7)?;
                Ok::<_, candle_core::Error>(m)
            };
            let bare = || {
                let mut m = model(&cfg, &weights, Some(serve), true)?;
                m.set_lora(false);
                Ok::<_, candle_core::Error>(m)
            };
            // Each sequence's blend, and a model that holds it alone.
            let blends = [1u16, 0, 2, 1, 0];
            let mut alone = [first()?, second()?, bare()?, first()?, second()?];
            together.sequences(blends.len());
            let mut read = vec![0usize; blends.len()];
            for (i, blend) in blends.iter().enumerate() {
                let prompt: Vec<u32> = (0..2 + 3 * i).map(|t| token(5 * i + t)).collect();
                let (Some(Stepped::Logits(got)), Some(want)) = (
                    together.step_in(i, *blend, &prompt, 0, false)?,
                    stepped(&mut alone[i], &prompt, 0)?,
                ) else {
                    bail!("{serve:?}: sequence {i}'s prompt was not stepped")
                };
                assert_eq!(got, want, "{serve:?}: sequence {i}'s prompt, blend {blend}");
                read[i] = prompt.len();
            }
            // The blends differ: the same prompt and token give other logits.
            let probe = [token(3), token(4)];
            let (Some(a), Some(b), Some(c)) = (
                stepped(&mut first()?, &probe, 0)?,
                stepped(&mut second()?, &probe, 0)?,
                stepped(&mut bare()?, &probe, 0)?,
            ) else {
                bail!("{serve:?}: a probe was not stepped")
            };
            assert!(
                a != b && b != c && a != c,
                "{serve:?}: three blends, three answers"
            );

            for step in 0..8 {
                // The sequences in another order at every step, so the runs
                // of one blend fall differently.
                let mut each: Vec<Next> = (0..blends.len())
                    .map(|i| Next {
                        sequence: i,
                        token: token(13 * i + step),
                        position: read[i],
                        blend: blends[i],
                    })
                    .collect();
                each.rotate_left(step % blends.len());
                let Some(got) = together.step_each(&each, false)? else {
                    bail!("{serve:?}: step {step} was not stepped")
                };
                for (next, got) in each.iter().zip(got) {
                    let Stepped::Logits(got) = got else {
                        bail!("{serve:?}: logits were asked for")
                    };
                    let Some(want) =
                        stepped(&mut alone[next.sequence], &[next.token], next.position)?
                    else {
                        bail!(
                            "{serve:?}: sequence {} was not stepped alone",
                            next.sequence
                        )
                    };
                    assert_eq!(
                        got, want,
                        "{serve:?}, step {step}: sequence {} of blend {}",
                        next.sequence, next.blend
                    );
                    read[next.sequence] += 1;
                }
            }
        }
        Ok(())
    }

    /// Asked for each sequence's first alone, a step of several gives each
    /// the token it gets alone.
    #[test]
    fn the_first_of_each_sequence_is_its_first_alone() -> CResult<()> {
        if !sketch::available() {
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let mut together = model(&cfg, &weights, Some(GgmlDType::Q6K), true)?;
        together.sequences(3);
        let mut alone = Vec::new();
        for _ in 0..3 {
            alone.push(model(&cfg, &weights, Some(GgmlDType::Q6K), true)?);
        }
        for step in 0..10 {
            let each: Vec<Next> = (0..3)
                .map(|i| Next {
                    sequence: i,
                    token: token(11 * i + step),
                    position: step,
                    blend: 0,
                })
                .collect();
            let Some(got) = together.step_each(&each, true)? else {
                bail!("step {step} was not stepped")
            };
            for (next, got) in each.iter().zip(got) {
                let (Stepped::First(got), Some(Stepped::First(want))) = (
                    got,
                    alone[next.sequence].step(&[next.token], next.position, true)?,
                ) else {
                    bail!("step {step}: a token did not come back chosen")
                };
                assert_eq!(got, want, "step {step}, sequence {}", next.sequence);
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

    /// A generation a token at a time, as the generation loop writes it:
    /// the prompt, then each token's own pass, until the end token, a loop
    /// or the most tokens allowed.
    fn a_token_at_a_time(
        model: &mut QwenLora,
        prompt: &[u32],
        eos: u32,
        most: usize,
    ) -> CResult<Vec<u32>> {
        let mut tokens = prompt.to_vec();
        for index in 0..most {
            let start = match index {
                0 => 0,
                _ => tokens.len() - 1,
            };
            let Some(Stepped::First(next)) = model.step(&tokens[start..], start, true)? else {
                bail!("token {index} was not stepped to its first")
            };
            if next == eos {
                break;
            }
            tokens.push(next);
            if crate::decode::loops(&tokens[prompt.len()..]) {
                break;
            }
        }
        Ok(tokens)
    }

    /// Rows read ahead in one pass each give the first token a step for the
    /// row alone gives, at every served form; and a cache cut back is the
    /// cache as it was before the rows cut off: another token read there
    /// gives what it gives after the rows before it were read one by one.
    #[test]
    fn rows_read_ahead_give_each_rows_own_first_token() -> CResult<()> {
        if !sketch::available() {
            eprintln!("skipped: this CPU multiplies no sketch of the head");
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let tokens: Vec<u32> = (0..12).map(token).collect();
        for serve in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let mut alone = model(&cfg, &weights, Some(serve), true)?;
            let mut want = Vec::new();
            for (at, token) in tokens.iter().enumerate() {
                let Some(Stepped::First(first)) = alone.step(&[*token], at, true)? else {
                    bail!("{serve:?}: token {at} was not stepped to its first")
                };
                want.push(first);
            }
            // Three as a prompt, then nine at once: eight and one.
            let mut ahead = model(&cfg, &weights, Some(serve), true)?;
            assert!(ahead.reads_ahead());
            let Some(Stepped::First(first)) = ahead.step(&tokens[..3], 0, true)? else {
                bail!("{serve:?}: the prompt was not stepped to its first")
            };
            assert_eq!(first, want[2], "{serve:?}: after the prompt");
            let Some(firsts) = ahead.step_ahead(&tokens[3..], 3)? else {
                bail!("{serve:?}: nine rows were not read ahead")
            };
            assert_eq!(firsts, &want[3..], "{serve:?}: nine rows at once");

            // Cut back to five positions; a sixth token that was never there.
            ahead.truncate(5);
            let mut again = model(&cfg, &weights, Some(serve), true)?;
            for (at, token) in tokens[..5].iter().enumerate() {
                again.step(&[*token], at, true)?;
            }
            let other = token(40);
            let (Some(Stepped::First(got)), Some(Stepped::First(want))) = (
                ahead.step(&[other], 5, true)?,
                again.step(&[other], 5, true)?,
            ) else {
                bail!("{serve:?}: the token after the cut was not stepped")
            };
            assert_eq!(got, want, "{serve:?}: after the cache was cut back");
            assert_eq!(
                ahead.scratch.normed, again.scratch.normed,
                "{serve:?}: the row the head is given after the cut"
            );
        }
        // More rows than are read at once are not read ahead, and a head
        // with no sketch reads nothing ahead.
        let mut served = model(&cfg, &weights, Some(GgmlDType::Q6K), false)?;
        let many: Vec<u32> = (0..ROWS_AT_ONCE + 1).map(token).collect();
        assert!(served.step_ahead(&many, 0)?.is_none());
        let mut dense = model(&cfg, &weights, None, false)?;
        assert!(!dense.reads_ahead() && dense.step_ahead(&[1], 0)?.is_none());
        Ok(())
    }

    /// A generation read ahead is the generation a token at a time, token
    /// for token, whatever is guessed: nothing, the text itself, the text
    /// with every third token wrong, and wrong tokens throughout. It stops
    /// where a token at a time stops: at the end token, and at the most
    /// tokens allowed. Guessed right, it takes fewer passes than tokens.
    #[test]
    fn a_generation_read_ahead_is_the_generation_a_token_at_a_time() -> CResult<()> {
        use crate::draft::Drafts;
        if !sketch::available() {
            eprintln!("skipped: this CPU multiplies no sketch of the head");
            return Ok(());
        }
        let cfg = config();
        let weights = weights(&cfg)?;
        let serve = GgmlDType::Q6K;
        // No token is 64: a generation that only its length ends.
        const NEVER: u32 = 64;
        let whole = a_token_at_a_time(
            &mut model(&cfg, &weights, Some(serve), true)?,
            &PROMPT,
            NEVER,
            48,
        )?;
        let written = whole.len() - PROMPT.len();
        assert!(
            written >= 12,
            "a generation of {written} tokens says little"
        );

        let remembering = |text: &[u32]| {
            let mut memory = Drafts::new();
            memory.learn(text, PROMPT.len());
            memory
        };
        let mut garbled = whole.clone();
        for token in garbled[PROMPT.len()..].iter_mut().skip(2).step_by(3) {
            *token = (*token + 1) % 64;
        }
        let wrong: Vec<u32> = whole
            .iter()
            .enumerate()
            .map(|(at, token)| match at < PROMPT.len() {
                true => *token,
                false => (*token + 1) % 64,
            })
            .collect();
        let kept = |text: &[u32]| {
            let mut memory = Drafts::new();
            memory.keep(&PROMPT, &text[PROMPT.len()..]);
            memory
        };
        // An end token the generation writes part of the way through.
        let ending = whole[PROMPT.len() + written / 2];
        // What a served model counts, from one generation to the next.
        let mut odds = ahead::Odds::default();
        for (what, far) in [
            ("nothing remembered", Drafts::new()),
            ("the text itself", remembering(&whole)),
            ("every third token wrong", remembering(&garbled)),
            ("every token wrong", remembering(&wrong)),
            ("the answer kept whole", kept(&whole)),
            ("an answer kept that goes wrong", kept(&garbled)),
            ("another answer kept", kept(&wrong)),
        ] {
            for (eos, most) in [
                (NEVER, 48usize),
                (ending, 48),
                (NEVER, 7),
                (NEVER, 1),
                (NEVER, 0),
            ] {
                let want = a_token_at_a_time(
                    &mut model(&cfg, &weights, Some(serve), true)?,
                    &PROMPT,
                    eos,
                    most,
                )?;
                // Under the odds a served model counts, whatever they come
                // to on this machine today: the same tokens.
                let mut counted = model(&cfg, &weights, Some(serve), true)?;
                let (mut tokens, mut reading) = (PROMPT.to_vec(), ahead::Reading::default());
                let read = ahead::decode(
                    &mut counted,
                    &mut tokens,
                    PROMPT.len(),
                    ahead::Ahead {
                        eos,
                        most,
                        reading: &mut reading,
                        far: &far,
                        odds: &mut odds,
                    },
                    &mut GenTiming::default(),
                )?;
                assert!(read.is_some(), "{what}: the generation was not read ahead");
                assert_eq!(tokens, want, "{what}, counted, at {eos} or {most} tokens");
                // And with every guess offered carried, so that passes of
                // every width are read.
                let mut ahead = model(&cfg, &weights, Some(serve), true)?;
                let (mut tokens, mut reading) = (PROMPT.to_vec(), ahead::Reading::default());
                let mut timing = GenTiming::default();
                let Some(passes) = ahead::decode(
                    &mut ahead,
                    &mut tokens,
                    PROMPT.len(),
                    ahead::Ahead {
                        eos,
                        most,
                        reading: &mut reading,
                        far: &far,
                        odds: &mut ahead::Odds::eager(),
                    },
                    &mut timing,
                )?
                else {
                    bail!("{what}: the generation was not read ahead")
                };
                assert_eq!(tokens, want, "{what}, ending at {eos} or {most} tokens");
                assert_eq!(timing.passes, passes.passes, "{what}");
                assert!(passes.kept <= passes.guessed, "{what}");
                if what == "the text itself" && eos == NEVER && most == 48 {
                    assert!(
                        passes.passes * 2 < written && passes.kept * 2 > written,
                        "{what}: {} passes and {} guesses kept for {written} tokens",
                        passes.passes,
                        passes.kept
                    );
                }
                if what == "the answer kept whole" && eos == NEVER && most == 48 {
                    // Sixteen tokens a pass from the first.
                    assert!(
                        passes.passes <= written.div_ceil(16) && passes.kept + 1 >= written - 1,
                        "{what}: {} passes and {} guesses kept for {written} tokens",
                        passes.passes,
                        passes.kept
                    );
                }
                if what == "nothing remembered" && most == 48 {
                    // Its own prompt and tokens are all it can guess from.
                    assert!(passes.passes <= tokens.len() - PROMPT.len(), "{what}");
                }
            }
        }
        // A model with no sketch of its head is not read ahead, and nothing
        // of the prompt is read.
        let mut dense = model(&cfg, &weights, None, false)?;
        let (mut tokens, mut reading) = (PROMPT.to_vec(), ahead::Reading::default());
        let none = ahead::decode(
            &mut dense,
            &mut tokens,
            PROMPT.len(),
            ahead::Ahead {
                eos: NEVER,
                most: 8,
                reading: &mut reading,
                far: &Drafts::new(),
                odds: &mut ahead::Odds::default(),
            },
            &mut GenTiming::default(),
        )?;
        assert!(none.is_none() && tokens == PROMPT);
        Ok(())
    }

    /// A pass that carries several rows of a sequence beside the rows of
    /// others gives after each row what the row gives read alone after the
    /// rows of its sequence before it, to the bit, whatever the others are;
    /// and a sequence whose cache is cut back to a row reads on from there
    /// as if the rows after it had never been read.
    #[test]
    fn rows_of_sequences_read_together_are_each_read_alone() -> CResult<()> {
        let cfg = config();
        let weights = weights(&cfg)?;
        for serve in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let mut together = model(&cfg, &weights, Some(serve), true)?;
            if together.step(&[token(0)], 0, false)?.is_none() {
                eprintln!("skipped: this CPU reads no rows of {serve:?} as plain values");
                continue;
            }
            together.clear_cache();
            together.sequences(3);
            let prompts: [Vec<u32>; 3] = [
                (0..5).map(token).collect(),
                (10..13).map(token).collect(),
                (20..27).map(token).collect(),
            ];
            for (sequence, prompt) in prompts.iter().enumerate() {
                together.clear_sequence(sequence)?;
                if together.step_in(sequence, 0, prompt, 0, false)?.is_none() {
                    bail!("{serve:?}: sequence {sequence}'s prompt was not read")
                }
            }
            // What is read after each prompt in the pass: three rows of the
            // first sequence, one of the second, two of the third.
            let more: [Vec<u32>; 3] = [
                vec![token(31), token(32), token(33)],
                vec![token(41)],
                vec![token(51), token(52)],
            ];
            let mut each = Vec::new();
            for (sequence, (prompt, rows)) in prompts.iter().zip(&more).enumerate() {
                for (i, row) in rows.iter().enumerate() {
                    each.push(Next {
                        sequence,
                        token: *row,
                        position: prompt.len() + i,
                        blend: 0,
                    });
                }
            }
            let Some(got) = together.step_each(&each, false)? else {
                bail!("{serve:?}: the rows were not read together")
            };
            // Each row alone: its sequence up to it, on a model of its own.
            let alone = |tokens: &[u32]| -> CResult<Vec<f32>> {
                let mut own = model(&cfg, &weights, Some(serve), true)?;
                match stepped(&mut own, tokens, 0)? {
                    Some(logits) => Ok(logits),
                    None => bail!("{serve:?}: a sequence alone was not stepped"),
                }
            };
            let mut at = 0;
            for (sequence, (prompt, rows)) in prompts.iter().zip(&more).enumerate() {
                for i in 0..rows.len() {
                    let mut tokens = prompt.clone();
                    tokens.extend_from_slice(&rows[..=i]);
                    let Stepped::Logits(got) = &got[at] else {
                        bail!("{serve:?}: logits were asked for")
                    };
                    assert_eq!(
                        got,
                        &alone(&tokens)?,
                        "{serve:?}: sequence {sequence}, row {i}"
                    );
                    at += 1;
                }
            }
            // The first sequence keeps its first row of the pass and loses
            // the two after it; the third keeps both of its own.
            together.truncate_sequence(0, prompts[0].len() + 1)?;
            let on = [
                Next {
                    sequence: 0,
                    token: token(61),
                    position: prompts[0].len() + 1,
                    blend: 0,
                },
                Next {
                    sequence: 2,
                    token: token(62),
                    position: prompts[2].len() + 2,
                    blend: 0,
                },
            ];
            let Some(got) = together.step_each(&on, false)? else {
                bail!("{serve:?}: the rows after the cut were not read")
            };
            let mut first = prompts[0].clone();
            first.extend_from_slice(&[more[0][0], token(61)]);
            let mut third = prompts[2].clone();
            third.extend_from_slice(&[more[2][0], more[2][1], token(62)]);
            for (got, tokens, what) in [
                (&got[0], &first, "cut back"),
                (&got[1], &third, "left whole"),
            ] {
                let Stepped::Logits(got) = got else {
                    bail!("{serve:?}: logits were asked for")
                };
                assert_eq!(got, &alone(tokens)?, "{serve:?}: the sequence {what}");
            }
            assert!(
                together.truncate_sequence(40, 0).is_err(),
                "no such sequence"
            );
        }
        Ok(())
    }

    /// Requests answered together, each carrying its own guesses in the
    /// passes they share, write what each writes a token at a time by
    /// itself: whatever each guesses from, wherever each ends, with every
    /// guess offered carried and under the counted odds, and where the pass
    /// has room for one guess in all.
    #[test]
    fn sequences_read_ahead_together_are_each_a_token_at_a_time() -> CResult<()> {
        use crate::draft::Drafts;
        let cfg = config();
        let weights = weights(&cfg)?;
        let serve = GgmlDType::Q6K;
        if !model(&cfg, &weights, Some(serve), true)?.reads_ahead() {
            eprintln!("skipped: this CPU multiplies no sketch of the head");
            return Ok(());
        }
        // No token is 64: a generation that only its length ends.
        const NEVER: u32 = 64;
        let prompts: [Vec<u32>; 3] = [PROMPT.to_vec(), vec![9, 33, 2], (0..7).map(token).collect()];
        let alone = |prompt: &[u32], eos: u32, most: usize| {
            a_token_at_a_time(
                &mut model(&cfg, &weights, Some(serve), true)?,
                prompt,
                eos,
                most,
            )
        };
        let wholes = [
            alone(&prompts[0], NEVER, 40)?,
            alone(&prompts[1], NEVER, 40)?,
            alone(&prompts[2], NEVER, 40)?,
        ];
        let remembering = |text: &[u32], prompt: usize| {
            let mut memory = Drafts::new();
            memory.learn(text, prompt);
            memory
        };
        let kept = |text: &[u32], prompt: usize| {
            let mut memory = Drafts::new();
            memory.keep(&text[..prompt], &text[prompt..]);
            memory
        };
        let garbled = |text: &[u32], prompt: usize| {
            let mut text = text.to_vec();
            for token in text[prompt..].iter_mut().skip(2).step_by(3) {
                *token = (*token + 1) % 64;
            }
            text
        };
        // What each request's blend wrote before: the first its own text,
        // the second nothing, the third its answer kept whole; then the
        // first its text with every third token wrong, the second the
        // third's text, the third an answer that goes wrong.
        let helped = [
            Some(remembering(&wholes[0], prompts[0].len())),
            None,
            Some(kept(&wholes[2], prompts[2].len())),
        ];
        let misled = [
            Some(remembering(
                &garbled(&wholes[0], prompts[0].len()),
                prompts[0].len(),
            )),
            Some(remembering(&wholes[2], prompts[2].len())),
            Some(kept(
                &garbled(&wholes[2], prompts[2].len()),
                prompts[2].len(),
            )),
        ];
        // An end token the first request writes part of the way through.
        let ending = wholes[0][prompts[0].len() + (wholes[0].len() - prompts[0].len()) / 2];
        let mut fewer = false;
        for (what, fars) in [("helped", &helped), ("misled", &misled)] {
            for (eos, most, rows_most) in [
                (NEVER, 40usize, 32usize),
                (ending, 40, 32),
                (NEVER, 6, 32),
                (NEVER, 2, 32),
                (NEVER, 40, 4),
            ] {
                let want = [
                    alone(&prompts[0], eos, most)?,
                    alone(&prompts[1], eos, most)?,
                    alone(&prompts[2], eos, most)?,
                ];
                for eager in [true, false] {
                    let mut odds = match eager {
                        true => ahead::Odds::eager(),
                        false => ahead::Odds::default(),
                    };
                    let mut together = model(&cfg, &weights, Some(serve), true)?;
                    together.sequences(3);
                    let mut tokens = prompts.clone();
                    let mut readings: [ahead::Reading; 3] = Default::default();
                    let mut ended = [false; 3];
                    // The pass each request ended in.
                    let mut ended_in = [0usize; 3];
                    // Each request is taken in: its prompt, and the first
                    // token after it.
                    for sequence in 0..3 {
                        together.clear_sequence(sequence)?;
                        let first = together.step_in(sequence, 0, &tokens[sequence], 0, true)?;
                        let Some(Stepped::First(next)) = first else {
                            bail!("{what}: sequence {sequence}'s prompt gave no first token")
                        };
                        let said = fars[sequence]
                            .as_ref()
                            .and_then(|far| far.said(&prompts[sequence]));
                        readings[sequence].begin(said);
                        match next == eos {
                            true => ended[sequence] = true,
                            false => tokens[sequence].push(next),
                        }
                    }
                    let mut passes = 0;
                    while ended.iter().any(|ended| !ended) {
                        let mut who = Vec::new();
                        let mut seats: Vec<ahead::Seat> = tokens
                            .iter_mut()
                            .zip(readings.iter_mut())
                            .zip(fars.iter())
                            .enumerate()
                            .filter(|(sequence, _)| !ended[*sequence])
                            .map(|(sequence, ((tokens, reading), far))| {
                                who.push(sequence);
                                ahead::Seat {
                                    sequence,
                                    blend: 0,
                                    tokens,
                                    prompt_len: prompts[sequence].len(),
                                    reading,
                                    far: far.as_ref(),
                                }
                            })
                            .collect();
                        let passed = ahead::pass(
                            &mut together,
                            &mut seats,
                            eos,
                            most,
                            rows_most,
                            &mut odds,
                        )?;
                        drop(seats);
                        passes += 1;
                        for (sequence, done) in who.into_iter().zip(passed.ended) {
                            if done {
                                (ended[sequence], ended_in[sequence]) = (true, passes);
                            }
                        }
                        assert!(passes < 200, "{what}: the requests did not end");
                    }
                    for sequence in 0..3 {
                        assert_eq!(
                            tokens[sequence], want[sequence],
                            "{what}, request {sequence}, ending at {eos} or {most} tokens, {rows_most} rows a pass, every guess carried: {eager}"
                        );
                    }
                    // Guessed right, a request ends in far fewer passes
                    // than it has tokens: the one whose answer was kept
                    // whole in a handful, the one whose text was written
                    // before in under half.
                    if what == "helped" && eager && eos == NEVER && most == 40 && rows_most == 32 {
                        fewer = ended_in[2] <= 5 && ended_in[0] * 2 < 40;
                        assert!(fewer, "the requests ended in passes {ended_in:?}");
                    }
                }
            }
        }
        assert!(fewer, "the helped requests were never read");
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

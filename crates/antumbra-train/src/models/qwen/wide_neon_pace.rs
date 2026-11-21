//! SCRATCH, not for the repository: the pace of each NEON kernel beside the
//! plain loop it replaces, at the sizes a 1.5B base has (heads of 128, six
//! query heads to a cached one, an MLP 8960 wide, adapters of rank 16 and
//! 48). The plain loops are the ones in `attend.rs`, `pack.rs` and
//! `layers/step.rs`, copied.

use super::*;
use crate::models::qwen::slices::dot;
use std::hint::black_box;
use std::time::Instant;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn factors(seed: u64, n: usize) -> Vec<i8> {
    values(seed, n).iter().map(|v| (v * 127.0) as i8).collect()
}

fn plain_gate(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        *o = g / (1.0 + (-g).exp()) * u;
    }
}

unsafe fn plain_scores(
    q: &[f32],
    keys: &[f32],
    dim: usize,
    scale: f32,
    out: *mut f32,
    stride: usize,
    most: &mut [f32],
) {
    for (p, key) in keys.chunks_exact(dim).enumerate() {
        for (r, (query, top)) in q.chunks_exact(dim).zip(most.iter_mut()).enumerate() {
            let score = dot(query, key) * scale;
            *out.add(r * stride + p) = score;
            *top = top.max(score);
        }
    }
}

unsafe fn plain_weigh(
    scores: *const f32,
    stride: usize,
    most: &[f32],
    values: &[f32],
    dim: usize,
    acc: &mut [f32],
    sums: &mut [f32],
) {
    for (r, ((top, row), sum)) in most
        .iter()
        .zip(acc.chunks_exact_mut(dim))
        .zip(sums.iter_mut())
        .enumerate()
    {
        for (p, value) in values.chunks_exact(dim).enumerate() {
            let weight = (*scores.add(r * stride + p) - top).exp();
            *sum += weight;
            for (o, x) in row.iter_mut().zip(value) {
                *o += weight * x;
            }
        }
    }
}

fn plain_through(
    a: &[i8],
    n: usize,
    x: &[f32],
    stride: usize,
    rows: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    let (a, x) = (&a[..n * 16], &x[..(rows - 1) * stride + n]);
    let out = &mut out[..(rows - 1) * out_stride + 16];
    for i in 0..rows {
        let sums = &mut out[i * out_stride..i * out_stride + 16];
        sums.fill(0.0);
        for (column, x) in a.chunks_exact(16).zip(&x[i * stride..]) {
            for (sum, a) in sums.iter_mut().zip(column) {
                *sum += f32::from(*a) * x;
            }
        }
    }
}

unsafe fn plain_add(
    b: &[i8],
    rank: usize,
    sixteens: usize,
    t: &[f32],
    rows: usize,
    out: *mut f32,
    stride: usize,
) {
    let (b, t) = (&b[..sixteens * rank * 16], &t[..rows * rank]);
    for (i, t) in t.chunks_exact(rank).enumerate() {
        for (s, b) in b.chunks_exact(rank * 16).enumerate() {
            let products = std::slice::from_raw_parts_mut(out.add(i * stride + s * 16), 16);
            for (factors, t) in b.chunks_exact(16).zip(t) {
                for (product, factor) in products.iter_mut().zip(factors) {
                    *product += f32::from(*factor) * t;
                }
            }
        }
    }
}

/// The least of several timings of `reps` calls, in nanoseconds a call.
fn least(reps: usize, f: &mut dyn FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(start.elapsed().as_nanos() as f64 / reps as f64);
    }
    best
}

/// Plain and NEON turn and turn about, the least of each.
fn both(reps: usize, plain: &mut dyn FnMut(), neon: &mut dyn FnMut()) -> (f64, f64) {
    let (mut p, mut n) = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        p = p.min(least(reps, plain));
        n = n.min(least(reps, neon));
    }
    (p, n)
}

fn line(what: &str, (plain, neon): (f64, f64)) {
    eprintln!(
        "PACE {what}: plain {plain:.0} ns, neon {neon:.0} ns, plain/neon {:.2}",
        plain / neon
    );
}

#[test]
#[ignore = "scratch: timing"]
fn pace_of_the_plain_rows_old_and_new() {
    // The gate: a block of the MLP, and a whole row of it.
    for n in [256usize, 8960] {
        let gate: Vec<f32> = values(1, n).iter().map(|v| v * 9.0).collect();
        let up = values(2, n);
        let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
        let reps = 2_000_000 / n;
        let got = both(
            reps,
            &mut || plain_gate(black_box(&gate), black_box(&up), black_box(&mut a)),
            // SAFETY: every aarch64 CPU has NEON.
            &mut || unsafe { silu_times(black_box(&gate), black_box(&up), black_box(&mut b)) },
        );
        line(&format!("gate {n}"), got);
    }

    // Attention: six query heads against a run of one cached head's
    // positions, as a unit of work has them.
    let (dim, rep) = (128usize, 6usize);
    for positions in [32usize, 256] {
        let q = values(3, rep * dim);
        let keys = values(4, positions * dim);
        let vals = values(5, positions * dim);
        let scale = 1.0 / (dim as f32).sqrt();
        let mut scored = vec![0f32; rep * positions];
        let mut most = vec![f32::NEG_INFINITY; rep];
        let at = scored.as_mut_ptr();
        let reps = 40_000 / positions;
        let got = both(
            reps,
            // SAFETY: the room is what the kernels are told of.
            &mut || unsafe {
                let mut most = [f32::NEG_INFINITY; 6];
                plain_scores(
                    black_box(&q),
                    black_box(&keys),
                    dim,
                    scale,
                    at,
                    positions,
                    &mut most,
                );
                black_box(&most);
            },
            &mut || unsafe {
                let mut most = [f32::NEG_INFINITY; 6];
                scores(
                    black_box(&q),
                    black_box(&keys),
                    dim,
                    scale,
                    at,
                    positions,
                    &mut most,
                );
                black_box(&most);
            },
        );
        line(&format!("scores {positions} positions"), got);
        // SAFETY: as above.
        unsafe { scores(&q, &keys, dim, scale, at, positions, &mut most) };
        let scored = scored.as_ptr();
        let (mut acc, mut sums) = (vec![0f32; rep * dim], vec![0f32; rep]);
        let (mut acc_n, mut sums_n) = (vec![0f32; rep * dim], vec![0f32; rep]);
        let got = both(
            reps,
            &mut || unsafe {
                plain_weigh(
                    black_box(scored),
                    positions,
                    black_box(&most),
                    black_box(&vals),
                    dim,
                    &mut acc,
                    &mut sums,
                );
            },
            &mut || unsafe {
                weigh(
                    black_box(scored),
                    positions,
                    black_box(&most),
                    black_box(&vals),
                    dim,
                    &mut acc_n,
                    &mut sums_n,
                );
            },
        );
        black_box((&acc, &sums, &acc_n, &sums_n));
        line(&format!("weigh {positions} positions"), got);
    }

    // An adapter's A: sixteen ranks over a projection's columns, and over a
    // block of the MLP's.
    for (n, rows) in [
        (1536usize, 1usize),
        (1536, 4),
        (1536, 8),
        (1536, 32),
        (256, 1),
        (256, 8),
    ] {
        let a = factors(6, 16 * n);
        let x = values(7, rows * n);
        let (mut one, mut other) = (vec![0f32; rows * 16], vec![0f32; rows * 16]);
        let reps = 600_000 / (n * rows);
        let got = both(
            reps,
            &mut || {
                plain_through(
                    black_box(&a),
                    n,
                    black_box(&x),
                    n,
                    rows,
                    black_box(&mut one),
                    16,
                )
            },
            // SAFETY: the slices hold what the kernel is told of.
            &mut || unsafe {
                through_cols(
                    black_box(a.as_ptr()),
                    n,
                    black_box(x.as_ptr()),
                    n,
                    rows,
                    black_box(other.as_mut_ptr()),
                    16,
                )
            },
        );
        line(&format!("A {n} columns, {rows} rows"), got);
    }

    // An adapter's B: a block's sixteens, and a projection's.
    for (rank, sixteens, rows) in [
        (16usize, 16usize, 1usize),
        (16, 16, 8),
        (16, 96, 1),
        (16, 96, 4),
        (16, 96, 32),
        (48, 16, 1),
        (48, 16, 8),
        (48, 96, 1),
    ] {
        let b = factors(8, 16 * rank * sixteens);
        let t = values(9, rows * rank);
        let stride = 16 * sixteens;
        let (mut one, mut other) = (vec![0f32; rows * stride], vec![0f32; rows * stride]);
        let reps = (2_000_000 / (rank * sixteens * rows * 16)).max(8);
        let got = both(
            reps,
            // SAFETY: the room is what the kernels are told of.
            &mut || unsafe {
                plain_add(
                    black_box(&b),
                    rank,
                    sixteens,
                    black_box(&t),
                    rows,
                    black_box(one.as_mut_ptr()),
                    stride,
                )
            },
            &mut || unsafe {
                add_sixteens(
                    black_box(b.as_ptr()),
                    rank,
                    sixteens,
                    black_box(t.as_ptr()),
                    rows,
                    black_box(other.as_mut_ptr()),
                    stride,
                )
            },
        );
        line(&format!("B rank {rank}, {sixteens} sixteens, {rows} rows"), got);
    }
}

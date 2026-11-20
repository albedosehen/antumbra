//! A generation's keys and values on the CPU, as plain rows with room to
//! grow.
//!
//! As tensors a cache is joined to each decoded token's keys and values, which
//! copies every position so far in every layer for every token, or it is
//! written into a larger tensor through tensor ops and read back as a view.
//! Here each KV head's positions are one run of f32 that a token's rows are
//! written onto the end of, and attention reads them where they lie.

use candle_core::{bail, Device, Result, Storage, Tensor};

use super::slices::values;

/// Positions a store has room for when it is first written, and at least as
/// many again each time it fills.
const ROOM: usize = 512;

/// The keys and values of the positions read so far, for one sequence:
/// `[head][position][dim]`, each head's positions a run with room after it.
pub(super) struct KvStore {
    heads: usize,
    dim: usize,
    len: usize,
    room: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl KvStore {
    /// An empty store for `heads` KV heads of `dim` values a position.
    pub(super) fn new(heads: usize, dim: usize) -> Self {
        Self {
            heads,
            dim,
            len: 0,
            room: 0,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Positions held.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn heads(&self) -> usize {
        self.heads
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    /// Forget the positions from `len` on: rows that were read on text that
    /// was never written. A store that holds no more is left as it is.
    pub(super) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Forget every position; the room is kept for the next sequence.
    pub(super) fn reset(&mut self) {
        self.len = 0;
    }

    /// Make room for `more` positions after those held.
    fn reserve(&mut self, more: usize) {
        if self.len + more <= self.room {
            return;
        }
        let room = (self.len + more).max(self.room * 2).max(ROOM);
        let per = room * self.dim;
        let mut keys = vec![0f32; self.heads * per];
        let mut values = vec![0f32; self.heads * per];
        let (old, held) = (self.room * self.dim, self.len * self.dim);
        for head in 0..self.heads {
            keys[head * per..head * per + held]
                .copy_from_slice(&self.keys[head * old..head * old + held]);
            values[head * per..head * per + held]
                .copy_from_slice(&self.values[head * old..head * old + held]);
        }
        (self.keys, self.values, self.room) = (keys, values, room);
    }

    /// Write the keys and values of the next positions after those held.
    /// Each is `[head][position][dim]` for the same number of positions.
    pub(super) fn append(&mut self, keys: &[f32], values: &[f32]) -> Result<()> {
        let per_position = self.heads * self.dim;
        if per_position == 0
            || keys.len() != values.len()
            || !keys.len().is_multiple_of(per_position)
        {
            bail!(
                "a cache of {} heads of {}: {} key values and {} value values do not fit",
                self.heads,
                self.dim,
                keys.len(),
                values.len()
            )
        }
        let more = keys.len() / per_position;
        self.reserve(more);
        let (per, at, n) = (self.room * self.dim, self.len * self.dim, more * self.dim);
        for head in 0..self.heads {
            self.keys[head * per + at..head * per + at + n]
                .copy_from_slice(&keys[head * n..(head + 1) * n]);
            self.values[head * per + at..head * per + at + n]
                .copy_from_slice(&values[head * n..(head + 1) * n]);
        }
        self.len += more;
        Ok(())
    }

    /// [`KvStore::append`] from tensors of `(1, heads, positions, dim)`, f32
    /// in host memory.
    pub(super) fn append_tensors(&mut self, keys: &Tensor, values: &Tensor) -> Result<()> {
        let fits = |t: &Tensor| matches!(*t.dims(), [1, heads, _, dim] if heads == self.heads && dim == self.dim);
        if !fits(keys) || keys.dims() != values.dims() {
            bail!(
                "a cache of {} heads of {}: keys of {:?} and values of {:?} do not fit",
                self.heads,
                self.dim,
                keys.dims(),
                values.dims()
            )
        }
        let (keys, values) = (keys.contiguous()?, values.contiguous()?);
        let (k_storage, k_layout) = keys.storage_and_layout();
        let (v_storage, v_layout) = values.storage_and_layout();
        let (Storage::Cpu(k_storage), Storage::Cpu(v_storage)) = (&*k_storage, &*v_storage) else {
            bail!("a cache of plain rows holds keys and values in host memory")
        };
        self.append(
            self::values(k_storage, k_layout, "the keys to cache")?,
            self::values(v_storage, v_layout, "the values to cache")?,
        )
    }

    /// One head's keys: `len` positions of `dim`.
    pub(super) fn keys(&self, head: usize) -> &[f32] {
        let per = self.room * self.dim;
        &self.keys[head * per..head * per + self.len * self.dim]
    }

    /// One head's values: `len` positions of `dim`.
    pub(super) fn values(&self, head: usize) -> &[f32] {
        let per = self.room * self.dim;
        &self.values[head * per..head * per + self.len * self.dim]
    }

    /// Everything held as tensors of `(1, heads, len, dim)`: a copy, for the
    /// pass that reads several positions at once.
    pub(super) fn tensors(&self) -> Result<(Tensor, Tensor)> {
        Ok((self.gathered(Self::keys)?, self.gathered(Self::values)?))
    }

    /// Every head's rows, read through `of`, as one tensor.
    fn gathered(&self, of: fn(&Self, usize) -> &[f32]) -> Result<Tensor> {
        let mut all = Vec::with_capacity(self.heads * self.len * self.dim);
        for head in 0..self.heads {
            all.extend_from_slice(of(self, head));
        }
        Tensor::from_vec(all, (1, self.heads, self.len, self.dim), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(heads: usize, positions: usize, dim: usize, from: f32) -> Vec<f32> {
        (0..heads * positions * dim)
            .map(|i| from + i as f32)
            .collect()
    }

    /// Positions written a few at a time, past the room twice over, come
    /// back head by head in the order they were written, and as tensors.
    #[test]
    fn a_store_keeps_what_is_written_as_it_grows() -> Result<()> {
        let (heads, dim) = (2, 4);
        let mut store = KvStore::new(heads, dim);
        assert!(store.is_empty());
        let mut want_k: Vec<Vec<f32>> = vec![Vec::new(); heads];
        let mut want_v: Vec<Vec<f32>> = vec![Vec::new(); heads];
        let mut held = 0;
        for (step, positions) in [3usize, 1, ROOM, 1, 2 * ROOM, 5].into_iter().enumerate() {
            let k = rows(heads, positions, dim, step as f32 * 1e6);
            let v = rows(heads, positions, dim, -(step as f32) * 1e6 - 1.0);
            store.append(&k, &v)?;
            held += positions;
            for head in 0..heads {
                let n = positions * dim;
                want_k[head].extend_from_slice(&k[head * n..(head + 1) * n]);
                want_v[head].extend_from_slice(&v[head * n..(head + 1) * n]);
            }
            assert_eq!(store.len(), held);
            for head in 0..heads {
                assert_eq!(store.keys(head), want_k[head].as_slice(), "step {step}");
                assert_eq!(store.values(head), want_v[head].as_slice(), "step {step}");
            }
        }
        let (k, v) = store.tensors()?;
        assert_eq!(k.dims(), &[1, heads, held, dim]);
        assert_eq!(k.flatten_all()?.to_vec1::<f32>()?, want_k.concat());
        assert_eq!(v.flatten_all()?.to_vec1::<f32>()?, want_v.concat());

        store.reset();
        assert!(store.is_empty());
        assert!(store.keys(1).is_empty());
        Ok(())
    }

    /// Tensors are written as their rows are, and what does not fit the
    /// store is refused.
    #[test]
    fn a_store_takes_tensors_and_refuses_what_does_not_fit() -> Result<()> {
        let (heads, dim) = (2, 4);
        let mut store = KvStore::new(heads, dim);
        let k = Tensor::from_vec(rows(heads, 3, dim, 0.0), (1, heads, 3, dim), &Device::Cpu)?;
        let v = Tensor::from_vec(rows(heads, 3, dim, 100.0), (1, heads, 3, dim), &Device::Cpu)?;
        store.append_tensors(&k, &v)?;
        let (back_k, back_v) = store.tensors()?;
        assert_eq!(
            back_k.flatten_all()?.to_vec1::<f32>()?,
            rows(heads, 3, dim, 0.0)
        );
        assert_eq!(
            back_v.flatten_all()?.to_vec1::<f32>()?,
            rows(heads, 3, dim, 100.0)
        );

        let wide = Tensor::zeros(
            (1, heads, 1, dim + 1),
            candle_core::DType::F32,
            &Device::Cpu,
        )?;
        assert!(store.append_tensors(&wide, &wide).is_err(), "another width");
        let two = Tensor::zeros((2, heads, 1, dim), candle_core::DType::F32, &Device::Cpu)?;
        assert!(store.append_tensors(&two, &two).is_err(), "two sequences");
        assert!(
            store.append(&[0.0; 8], &[0.0; 16]).is_err(),
            "keys without values"
        );
        assert!(
            store.append(&[0.0; 7], &[0.0; 7]).is_err(),
            "part of a position"
        );
        assert_eq!(store.len(), 3);
        Ok(())
    }
}

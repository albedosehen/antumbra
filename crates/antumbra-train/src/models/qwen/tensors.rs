//! Rows read ahead as tensors.
//!
//! A model on a GPU keeps its weights there and reads its rows as tensors:
//! nothing of it is stepped through as plain values, so the kernels that
//! carry guessed rows on a CPU ([`super::layers`]) have nothing to carry them
//! in. The guesses are worth as much there. A pass on a GPU is a few
//! thousand small products, most of its time is the reading of the weights
//! and the waiting between them, and the rows of a pass share both.
//!
//! But a GPU's library does not compute a row among others as it computes
//! the row alone. Measured on an RTX 3090 Ti with the base in bf16, a row of
//! the MLP's way down read alone and the same row read among sixteen differ
//! in two values out of five, by a last place each; and in sixteen bits a
//! last place is enough to change which of two all-but-tied tokens is first.
//! Rows guessed ahead and simply read beside the token in hand wrote another
//! answer than a token a pass for 29 of one expert's 78 held-out tasks, and
//! for 33 when the tasks were asked again.
//!
//! What that library does keep to, measured on the same card in bf16, f16
//! and f32, is this: among rows of one count, a row's values are the same
//! to the bit wherever the row is and whatever the others are; and a row's
//! attention is the same to the bit whatever lies at the positions it may
//! not see, among positions of one count. So every pass read here is of one
//! shape. It is [`WIDE`] rows: the token in hand, the tokens guessed to
//! follow it, and rows of padding, cut from the cache when the pass is over.
//! And it attends over as many positions as its generation may come to
//! ([`QwenLora::read_within`]), the ones not yet written masked. A pass with
//! nothing guessed is the same pass with more padding. A guess then decides
//! nothing here either: over one expert's 78 held-out tasks a reader that
//! had answered nothing, one that had answered the tasks the expert was
//! taught, and that one asked the same tasks again wrote the same 78
//! answers, at 1.4, 1.8 and 12.5 tokens a pass.
//!
//! That is a property of a library that is not ours, so it is asked of the
//! device a model is on before anything is read this way
//! ([`QwenLora::keeps_rows_apart`]), and a device that does not keep to it
//! decodes a token a pass as it always has.
//!
//! A model that is not set to read ahead decodes a row a pass there too, as
//! it always has. That is the other rounding: its answer and a reader's to
//! the same prompt can differ in a word where two tokens are all but tied
//! (34 of those 78 did), so a model reads ahead or does not for as long as
//! its answers are to be the same answers, and whatever admits an expert
//! decodes as whatever serves it.

use candle_core::{bail, DType, Result as CResult, Tensor, D};

use super::QwenLora;
use crate::frozen::frozen_matmul_t;

/// Rows in every pass read ahead as tensors: the token in hand and the most
/// guesses a pass carries after it ([`super::ahead`]).
pub(in crate::models::qwen) const WIDE: usize = 16;

impl QwenLora {
    /// Whether this model reads rows ahead as tensors: one on a CUDA device
    /// whose library keeps a row apart from the rows beside it, asked once
    /// ([`QwenLora::keeps_rows_apart`]). A device that could not be asked is
    /// taken not to.
    pub(in crate::models::qwen) fn reads_ahead_as_tensors(&self) -> bool {
        #[cfg(test)]
        if self.ahead_as_tensors {
            return true;
        }
        self.device.is_cuda()
            && *self
                .rows_apart
                .get_or_init(|| self.keeps_rows_apart().unwrap_or(false))
    }

    /// The most positions the generation about to be read may come to: its
    /// prompt and the most tokens it may write. Every pass read ahead as
    /// tensors attends over that many positions and a pass's width more,
    /// whatever the cache holds by then. `false` where the rotary table does
    /// not reach that far: such a generation is not read this way.
    pub(in crate::models::qwen) fn read_within(&mut self, positions: usize) -> bool {
        let reach = self
            .layers
            .first()
            .map_or(0, |layer| layer.positions_most());
        self.room = positions;
        positions + WIDE <= reach
    }

    /// The token the head ranks first after the last of `tokens`, the first
    /// of which is at `position`, read in one pass as tensors with the cache
    /// on: a prompt, of which the last row alone goes through the head. A
    /// prompt is read once whatever is guessed after it, so its pass is as
    /// wide as it is long. `None`, with nothing read or written, where rows
    /// are not read ahead as tensors.
    pub(in crate::models::qwen) fn first_as_tensors(
        &mut self,
        tokens: &[u32],
        position: usize,
    ) -> CResult<Option<u32>> {
        if tokens.is_empty() || !self.reads_ahead_as_tensors() {
            return Ok(None);
        }
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let hidden = self.hidden(&input, position, true)?;
        let last = hidden.narrow(1, tokens.len() - 1, 1)?;
        let firsts = first_places(&self.logits(&last)?.squeeze(0)?)?;
        match firsts.first() {
            Some(first) => Ok(Some(*first)),
            None => bail!("the head gave no row for the last of a prompt"),
        }
    }

    /// The token the head ranks first after every one of `tokens`, the first
    /// of which is at `position`, read in one pass of [`WIDE`] rows as
    /// tensors with the cache on: for each of them what this gives the token
    /// read in a pass of its own after the ones before it. The rows the
    /// tokens do not fill are padding, cut from the cache again before this
    /// returns. The rows after a token whose first is not the next of
    /// `tokens` were read on text that is not the model's, and their places
    /// in the cache are the caller's to cut back ([`QwenLora::truncate`]).
    ///
    /// `None`, with nothing read or written, where rows are not read ahead
    /// as tensors or there are more tokens than a pass is wide.
    pub(in crate::models::qwen) fn firsts_as_tensors(
        &mut self,
        tokens: &[u32],
        position: usize,
    ) -> CResult<Option<Vec<u32>>> {
        let Some(last) = tokens.last() else {
            return Ok(None);
        };
        if tokens.len() > WIDE || !self.reads_ahead_as_tensors() {
            return Ok(None);
        }
        // Padding is the last token again: any token would do, since no row
        // that is kept attends to a row after it.
        let mut rows = tokens.to_vec();
        rows.resize(WIDE, *last);
        // Against as many positions as the generation may come to, and the
        // padding of a pass at its end.
        let room = self.room + WIDE;
        if position + WIDE > room {
            bail!(
                "a pass at {position} is past the {} positions its generation was to come to",
                self.room
            )
        }
        let mask = self.causal_mask_within(WIDE, position, room)?;
        let input = Tensor::new(rows.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden = self.hidden_under(&input, Some(&mask), position, true)?;
        // The head multiplies every row of the pass, as it does in every
        // pass; the first is looked for after the rows that are tokens.
        let logits = self.logits(&hidden)?.squeeze(0)?;
        let firsts = first_places(&logits.narrow(0, 0, tokens.len())?)?;
        if firsts.len() != tokens.len() {
            bail!(
                "the head gave {} rows for {} tokens",
                firsts.len(),
                tokens.len()
            )
        }
        self.truncate(position + tokens.len())?;
        Ok(Some(firsts))
    }

    /// Whether the device this model is on computes a row of a pass the same
    /// way wherever the row is and whatever is beside it, and a row's
    /// attention the same way whatever lies where it may not look: asked of
    /// the device with this model's own weights, rows of its own embeddings
    /// and passes of the width read here. Every kind of product a pass is
    /// made of is asked once: each of the first layer's projections, the
    /// head, and attention's two.
    ///
    /// It is what reading ahead as tensors rests on, and it is the
    /// library's to keep or not. A yes here is a few dozen products that
    /// came out the same, not a proof; a no is certain.
    pub(in crate::models::qwen) fn keeps_rows_apart(&self) -> CResult<bool> {
        let table = self.embed_tokens.embeddings();
        let (vocabulary, hidden) = table.dims2()?;
        let Some(layer) = self.layers.first() else {
            return Ok(false);
        };
        // A base that is not held dense is multiplied some other way, which
        // is not asked here and so is not read this way.
        let bases = layer.dense_bases();
        if vocabulary < 4 * WIDE || bases.len() != 7 {
            return Ok(false);
        }
        // Four texts of a pass's width, from the table's own rows.
        let text = |from: usize| table.narrow(0, from * WIDE, WIDE)?.unsqueeze(0);
        let (one, other) = (text(0)?, text(1)?);
        // A row of `one` where it is, and among the rows of `other` in
        // another place.
        let apart = |one: &Tensor, other: &Tensor, weight: &Tensor| -> CResult<bool> {
            let alone = frozen_matmul_t(one, weight)?;
            for (row, place) in [(0, 9), (7, 0), (WIDE - 1, 4)] {
                let among = Tensor::cat(
                    &[
                        &other.narrow(1, 0, place)?,
                        &one.narrow(1, row, 1)?,
                        &other.narrow(1, place + 1, WIDE - place - 1)?,
                    ],
                    1,
                )?;
                let among = frozen_matmul_t(&among, weight)?;
                if !same(&alone.narrow(1, row, 1)?, &among.narrow(1, place, 1)?)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        for weight in bases {
            let (_, inn) = weight.dims2()?;
            let kept = match inn == hidden {
                true => apart(&one, &other, weight)?,
                // The way down reads rows as wide as the gate's: the same
                // two texts, over and over to that width.
                false => {
                    let widen = |text: &Tensor| -> CResult<Tensor> {
                        Tensor::cat(&vec![text; inn.div_ceil(hidden)], 2)?
                            .narrow(2, 0, inn)?
                            .contiguous()
                    };
                    apart(&widen(&one)?, &widen(&other)?, weight)?
                }
            };
            if !kept {
                return Ok(false);
            }
        }
        if !apart(&one, &other, &self.lm_head_w)? {
            return Ok(false);
        }
        // Attention: the rows of a pass against positions of one count,
        // whose last half no row may see and holds one thing or another.
        let (heads, kv_heads, head_dim) = layer.attention_shape();
        if heads * head_dim != hidden || kv_heads == 0 || heads % kv_heads != 0 {
            return Ok(false);
        }
        let stacked = |text: &Tensor| -> CResult<Tensor> {
            text.reshape((1, WIDE, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, kv_heads, (heads / kv_heads) * WIDE, head_dim))
        };
        let (q, seen) = (stacked(&one)?, stacked(&other)?);
        let (unseen, unseen_other) = (stacked(&text(2)?)?, stacked(&text(3)?)?);
        let positions = seen.dim(2)?;
        let keys = Tensor::cat(&[&seen, &unseen], 2)?;
        let keys_other = Tensor::cat(&[&seen, &unseen_other], 2)?;
        let scores = q.matmul(&keys.t()?)?;
        let scores_other = q.matmul(&keys_other.t()?)?;
        if !same(
            &scores.narrow(3, 0, positions)?,
            &scores_other.narrow(3, 0, positions)?,
        )? {
            return Ok(false);
        }
        let none = Tensor::zeros(
            (1, kv_heads, q.dim(2)?, positions),
            scores.dtype(),
            scores.device(),
        )?;
        let weights = Tensor::cat(
            &[
                &candle_nn::ops::softmax_last_dim(&scores.narrow(3, 0, positions)?.contiguous()?)?,
                &none,
            ],
            3,
        )?;
        same(&weights.matmul(&keys)?, &weights.matmul(&keys_other)?)
    }
}

/// Whether two tensors of one shape hold the same values, every one.
fn same(a: &Tensor, b: &Tensor) -> CResult<bool> {
    let differ = a
        .ne(b)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    Ok(differ == 0.0)
}

/// The first place the largest value of each row of `logits` is at, found
/// where the logits are: `(rows, vocabulary)` in, a place for each row out.
/// The first place of all for a row with no value more than nothing.
///
/// It is the token [`crate::decode::pick_token_in`] takes from the same row
/// under a policy that takes the first, by the same rule: the largest value,
/// then where it first is. A base held in sixteen bits gives logits in
/// sixteen bits, where two tokens tie exactly often enough to matter, and a
/// device's own argmax settles a tie by which of its threads saw it.
fn first_places(logits: &Tensor) -> CResult<Vec<u32>> {
    let (rows, vocabulary) = logits.dims2()?;
    // A place is held as an f32, which counts every whole number this far.
    if vocabulary >= 1 << 24 {
        bail!("a vocabulary of {vocabulary} has places an f32 does not count")
    }
    let logits = logits.to_dtype(DType::F32)?;
    let device = logits.device();
    let largest = logits.max_keepdim(D::Minus1)?;
    let places =
        Tensor::arange(0f32, vocabulary as f32, device)?.broadcast_as((rows, vocabulary))?;
    let past = Tensor::new(vocabulary as f32, device)?.broadcast_as((rows, vocabulary))?;
    let first = logits
        .broadcast_eq(&largest)?
        .where_cond(&places, &past)?
        .min(D::Minus1)?
        .to_vec1::<f32>()?;
    Ok(first
        .into_iter()
        // Nothing equals a largest value that is not a number.
        .map(|at| match at >= 0.0 && (at as usize) < vocabulary {
            true => at as u32,
            false => 0,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// The first place of the largest value, row by row: of equals the
    /// first, and the first of all where nothing is more than nothing.
    #[test]
    fn the_first_place_of_the_largest_is_found_row_by_row() -> CResult<()> {
        let none = f32::NEG_INFINITY;
        let rows: [[f32; 6]; 5] = [
            [0.5, 2.0, -1.0, 2.0, 1.5, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [none, none, none, none, none, none],
            [-4.0, -9.0, -2.5, -2.5, -7.0, none],
            [0.0, -0.0, -1.0, none, -3.0, 0.0],
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits = Tensor::from_vec(flat, (5, 6), &Device::Cpu)?;
        assert_eq!(first_places(&logits)?, [1, 0, 0, 2, 0]);
        // In half the bits, where values that were apart are the same.
        let near = Tensor::from_vec(vec![1.0f32, 20.001, 20.002, 19.0], (1, 4), &Device::Cpu)?;
        assert_eq!(first_places(&near)?, [2]);
        assert_eq!(first_places(&near.to_dtype(DType::F16)?)?, [1]);
        Ok(())
    }

    /// Tensors are the same where every value is, and not where one is not.
    #[test]
    fn the_same_values_are_told_from_others() -> CResult<()> {
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let c = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.000001], (2, 2), &Device::Cpu)?;
        assert!(same(&a, &b)? && !same(&a, &c)?);
        Ok(())
    }
}

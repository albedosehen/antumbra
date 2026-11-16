//! The small arithmetic more than one crate needs, written once.

/// The mean of `values`, `None` of none: a threshold applied to the mean of
/// nothing should see no number rather than a zero.
pub fn mean(values: impl IntoIterator<Item = f32>) -> Option<f32> {
    let (n, sum) = values
        .into_iter()
        .fold((0u32, 0.0f32), |(n, s), v| (n + 1, s + v));
    (n > 0).then(|| sum / n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mean_of_nothing_is_none_and_of_something_its_average() {
        assert_eq!(mean(std::iter::empty()), None);
        assert_eq!(mean([2.0]), Some(2.0));
        assert_eq!(mean([1.0, 2.0, 3.0, 6.0]), Some(3.0));
        assert_eq!(mean(vec![0.5, 1.5].into_iter()), Some(1.0));
    }
}

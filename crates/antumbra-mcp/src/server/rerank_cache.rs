//! The reranker's result cache, kept by the server between recalls.

/// Bounded cache of `(query, sorted candidate ids) -> reranked id order`. A plain
/// insertion-ordered map capped at `CAP`; on overflow the oldest entry is
/// evicted. A miss merely recomputes, so eviction is always safe.
pub(super) struct RerankCache {
    map: std::collections::HashMap<String, Vec<String>>,
    order: std::collections::VecDeque<String>,
}

impl RerankCache {
    const CAP: usize = 1024;

    pub(super) fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    pub(super) fn get(&self, key: &str) -> Option<Vec<String>> {
        self.map.get(key).cloned()
    }

    pub(super) fn put(&mut self, key: String, value: Vec<String>) {
        if self.map.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
            while self.order.len() > Self::CAP {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
    }
}

//! The learned gate as the loop keeps it (ADR-0022 S-5): retrained by the
//! trainer whenever the population it routes over changes, so a graduate is
//! judged under the gate that would serve it, and serves under that gate once
//! it is admitted.
//!
//! The heuristic gate's top-two margin is what a population of one falls back
//! to without a learned router. On the GPU it turned away every warm-started
//! region specialist: with the generalist and a specialist trained from it
//! side by side, the margin between them collapsed, and the tasks between the
//! two escalated to the base model. The learned router is the gate for any
//! population of two or more, so the first specialist is judged under it too.

use antumbra_core::ports::Embedder;
use antumbra_core::{AntumbraError, Expert, ExpertId, LearnedRouter, Result};
use antumbra_store::repo::{lifecycle, router};

use crate::GenerationLoop;

/// The capability exemplars the learned router trains on, embedded and
/// labeled with the expert each describes. `None` when fewer than two experts
/// are given or fewer than two exemplars between them: too few to need a
/// router, so the heuristic gate routes.
pub async fn gate_exemplars(
    experts: &[Expert],
    embedder: &dyn Embedder,
) -> Result<Option<Vec<(ExpertId, Vec<f32>)>>> {
    if experts.len() < 2 {
        return Ok(None);
    }
    let mut exemplars = Vec::new();
    for e in experts {
        for text in e.exemplars() {
            exemplars.push((e.id.clone(), embedder.embed(&text).await?));
        }
    }
    Ok((exemplars.len() >= 2).then_some(exemplars))
}

impl GenerationLoop<'_> {
    /// A learned router retrained over `experts`. `None` when they are too few
    /// to need one, or when the trainer cannot train one.
    pub(crate) async fn retrained_gate(&self, experts: &[Expert]) -> Result<Option<LearnedRouter>> {
        let Some(exemplars) = gate_exemplars(experts, self.embedder).await? else {
            return Ok(None);
        };
        match self.trainer.train_router(&exemplars).await {
            Ok(learned) => Ok(Some(learned)),
            Err(AntumbraError::Unimplemented(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Keep the stored gate current after the population changes, as the
    /// CLI's refresh does at the end of a run: retrained over the routable
    /// experts unless it already routes over exactly those with exemplars.
    /// Returns the router it stored. Too few experts to need one, or a trainer
    /// that cannot train one, leave the gate as it is.
    pub(crate) async fn refresh_gate(&self) -> Result<Option<LearnedRouter>> {
        let routable = lifecycle::routable(self.store).await?;
        let covered = routable
            .iter()
            .filter(|e| !e.exemplars().is_empty())
            .map(|e| &e.id);
        if let Some(current) = router::load(self.store).await? {
            if current.trained_over(covered) {
                return Ok(None);
            }
        }
        let Some(learned) = self.retrained_gate(&routable).await? else {
            return Ok(None);
        };
        router::save(self.store, &learned).await?;
        Ok(Some(learned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antumbra_core::Generation;
    use async_trait::async_trait;
    use chrono::Utc;

    struct Axis;

    #[async_trait]
    impl Embedder for Axis {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32, 1.0])
        }
        fn dim(&self) -> usize {
            2
        }
    }

    fn expert(id: &str, exemplars: &[&str]) -> Expert {
        Expert {
            id: ExpertId::new(id),
            name: id.into(),
            base_model: "base".into(),
            artifact_uri: format!("adapters/{id}"),
            capability_card: serde_json::json!({ "exemplars": exemplars }),
            capability_vec: None,
            fitness: 1.0,
            frozen_at: Some(Utc::now()),
            generation: Generation::ZERO,
            owner: None,
            compartment: None,
            placed_on: None,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn exemplars_are_embedded_and_labeled_by_expert() -> Result<()> {
        let got = gate_exemplars(&[expert("a", &["x", "yy"]), expert("b", &["zzz"])], &Axis)
            .await?
            .expect("two experts, three exemplars");
        let labels: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(labels, ["a", "a", "b"]);
        assert_eq!(got[2].1, vec![3.0, 1.0]);
        Ok(())
    }

    #[tokio::test]
    async fn too_few_to_need_a_router() -> Result<()> {
        assert!(gate_exemplars(&[expert("a", &["x", "y"])], &Axis)
            .await?
            .is_none());
        assert!(
            gate_exemplars(&[expert("a", &["x"]), expert("b", &[])], &Axis)
                .await?
                .is_none()
        );
        Ok(())
    }
}

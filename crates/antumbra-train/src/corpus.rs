//! A file-backed corpus of verifiable tasks (your selected repos, the source of plasticity).
//!
//! Each task is `{ "id", "prompt", "verify" }`, where `verify` is the spec the
//! verifier consumes (e.g. `CommandVerifier`'s `{program,args,cwd}`). The loop
//! passes a `TrainRequest`'s `corpus_task_ids` to [`JsonCorpus::tasks`]; an
//! empty id list selects the whole corpus.

use antumbra_core::{AntumbraError, Result};

use crate::model::{Corpus, CorpusTask, TaskScope};

pub struct JsonCorpus {
    tasks: Vec<CorpusTask>,
}

/// Read a contrastive scope from a task object, when it carries both
/// `fail_context` and `near_ok_context` -- so a file-based correction can assert
/// *where* it applies and become an actionable boundary of competence once verified.
/// `governing_feature` is optional: absent, it is inferred from the one key that
/// differs between the two contexts. Missing either context -> a plain correction.
fn parse_scope(task: &serde_json::Value) -> Option<TaskScope> {
    Some(TaskScope {
        governing_feature: task
            .get("governing_feature")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        fail_context: task.get("fail_context")?.clone(),
        near_ok_context: task.get("near_ok_context")?.clone(),
    })
}

impl JsonCorpus {
    pub fn from_tasks(tasks: Vec<CorpusTask>) -> Self {
        Self { tasks }
    }

    /// Load tasks from a JSON array of `{id, prompt, verify}` objects.
    pub fn from_file(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| AntumbraError::other(e.to_string()))?;
        let raw: Vec<serde_json::Value> = serde_json::from_slice(&bytes)?;
        let tasks = raw
            .iter()
            .map(|t| CorpusTask {
                id: t
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                prompt: t
                    .get("prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                verify: t.get("verify").cloned().unwrap_or(serde_json::Value::Null),
                completion: t
                    .get("completion")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                skill: t.get("skill").and_then(|v| v.as_str()).map(str::to_string),
                scope: parse_scope(t),
                impossible: t
                    .get("impossible")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            })
            .collect();
        Ok(Self { tasks })
    }
}

impl Corpus for JsonCorpus {
    fn tasks(&self, task_ids: &[String]) -> Vec<CorpusTask> {
        if task_ids.is_empty() {
            self.tasks.clone()
        } else {
            self.tasks
                .iter()
                .filter(|t| task_ids.iter().any(|id| id == &t.id))
                .cloned()
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ids_select_all_and_filter_works() {
        let corpus = JsonCorpus::from_tasks(vec![
            CorpusTask::new("a", "pa"),
            CorpusTask::new("b", "pb").with_verify(serde_json::json!({ "program": "true" })),
        ]);
        assert_eq!(corpus.tasks(&[]).len(), 2);
        let one = corpus.tasks(&["b".to_string()]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].id, "b");
        assert_eq!(one[0].verify["program"], "true");
    }

    #[test]
    fn an_impossible_marker_is_read_and_defaults_to_false() {
        let path = std::env::temp_dir().join("antumbra_corpus_impossible_test.json");
        std::fs::write(
            &path,
            r#"[{"id":"a","prompt":"p"},{"id":"b","prompt":"p","impossible":true}]"#,
        )
        .unwrap();
        let tasks = JsonCorpus::from_file(path.to_str().unwrap())
            .unwrap()
            .tasks(&[]);
        std::fs::remove_file(&path).ok();
        assert!(!tasks[0].impossible);
        assert!(tasks[1].impossible);
    }

    #[test]
    fn loads_from_json_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("antumbra_corpus_test.json");
        std::fs::write(
            &path,
            r#"[{"id":"t1","prompt":"do","verify":{"program":"sh","args":["-c","exit 0"]}}]"#,
        )
        .unwrap();
        let corpus = JsonCorpus::from_file(path.to_str().unwrap()).unwrap();
        let tasks = corpus.tasks(&[]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "t1");
        assert_eq!(tasks[0].verify["program"], "sh");
        std::fs::remove_file(&path).ok();
    }
}

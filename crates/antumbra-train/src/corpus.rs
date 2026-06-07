//! A file-backed corpus of verifiable tasks (ADR-0002: your selected repos).
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

/// Read a contrastive scope from a task object, when it carries all three of
/// `governing_feature`, `fail_context`, `near_ok_context` -- so a file-based
/// correction can assert *where* it applies and become an actionable boundary
/// once verified (ADR-0004). Absent or partial -> a plain correction.
fn parse_scope(task: &serde_json::Value) -> Option<TaskScope> {
    let governing_feature = task.get("governing_feature")?.as_str()?.to_string();
    Some(TaskScope {
        governing_feature,
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

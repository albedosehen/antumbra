//! An embedded store that SurrealDB 3.x wrote, opened by this build, which
//! runs SurrealDB 4: refused by default, with the way forward named and the
//! store left as it was, and converted only when asked (the
//! `ANTUMBRA_UPGRADE_STORE_FROM_3X` opt-in), after which it opens plainly.
//!
//! The fixture is a SurrealKV directory written by SurrealDB 3.3.5, namespace
//! and database `fixture`, made for surql-rs's own test of the same path.

use std::fs;
use std::path::{Path, PathBuf};

use antumbra_store::{ConnectionConfig, Store, UPGRADE_FROM_3X_ENV};

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/surrealkv-3.3.5");

/// A fresh copy of the 3.x store, removed again when dropped.
struct Copy(PathBuf);

impl Copy {
    fn of_fixture(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "antumbra-store-3x-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        copy_dir(Path::new(FIXTURE), &dir);
        Copy(dir)
    }

    fn config(&self) -> ConnectionConfig {
        ConnectionConfig::builder()
            .url(format!("surrealkv://{}", self.0.display()))
            .namespace("fixture")
            .database("fixture")
            .build()
            .expect("a valid config")
    }
}

impl Drop for Copy {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create the copy");
    for entry in fs::read_dir(from).expect("read the fixture") {
        let entry = entry.expect("a fixture entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("an entry type").is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).expect("copy a fixture file");
        }
    }
}

/// Every file under `dir` with its bytes, by relative path, leaving out the
/// lock file an open attempt may create.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        for entry in fs::read_dir(&next).expect("read a directory") {
            let entry = entry.expect("an entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if entry.file_name() != "LOCK" {
                let relative = path.strip_prefix(dir).expect("under dir").to_path_buf();
                out.push((relative, fs::read(&path).expect("read a file")));
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn a_3x_store_is_refused_with_the_way_forward_and_left_as_it_was() {
    let copy = Copy::of_fixture("refused");
    let before = snapshot(&copy.0);

    let err = match Store::open(copy.config(), 8, false).await {
        Ok(_) => panic!("a 3.x store opened without the opt-in"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("written by SurrealDB 3.x"), "{err}");
    assert!(err.contains(&format!("{UPGRADE_FROM_3X_ENV}=true")), "{err}");
    assert!(err.contains("back up"), "{err}");
    assert_eq!(snapshot(&copy.0), before, "the refusal changed the store");
}

#[tokio::test]
async fn a_3x_store_converts_when_asked_and_then_opens_without_the_opt_in() {
    let copy = Copy::of_fixture("converted");

    let store = Store::open(copy.config(), 8, true)
        .await
        .expect("the opt-in converts the store");
    drop(store);

    Store::open(copy.config(), 8, false)
        .await
        .expect("a converted store opens with no opt-in");
}

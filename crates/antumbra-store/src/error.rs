//! Maps surql-rs errors into the storage-agnostic core error. The orphan rule
//! forbids a `From` impl between two foreign types, so this is a free function.

use antumbra_core::AntumbraError;
use surql::SurqlError;

pub(crate) fn map(err: SurqlError) -> AntumbraError {
    AntumbraError::store(err.to_string())
}

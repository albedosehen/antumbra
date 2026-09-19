//! Webhook delivery signatures. GitHub signs each delivery's raw body with the
//! App's webhook secret (HMAC-SHA256) and sends the hex digest in
//! `X-Hub-Signature-256: sha256=<hex>`. A receiver that does not verify it
//! would let anyone who finds the URL re-anchor or orphan memories.

use hmac::{Hmac, Mac};
use sha2::Sha256;

/// The header carrying the HMAC-SHA256 signature of the body.
pub const SIGNATURE_HEADER: &str = "x-hub-signature-256";
/// The header naming the event kind (`pull_request`, `delete`, `ping`, ...).
pub const EVENT_HEADER: &str = "x-github-event";
/// The header carrying the delivery's unique id (for logs and redelivery).
pub const DELIVERY_HEADER: &str = "x-github-delivery";

const PREFIX: &str = "sha256=";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SignatureError {
    #[error("missing the {SIGNATURE_HEADER} header")]
    Missing,
    #[error("malformed {SIGNATURE_HEADER} header: expected sha256=<64 hex digits>")]
    Malformed,
    #[error("the signature does not match the body")]
    Mismatch,
}

/// Verify that `header` is the HMAC-SHA256 of `body` under `secret`. The
/// comparison is constant-time, so a forged signature learns nothing from how
/// long the rejection took.
pub fn verify(secret: &[u8], body: &[u8], header: Option<&str>) -> Result<(), SignatureError> {
    let header = header.ok_or(SignatureError::Missing)?;
    let digest = header
        .trim()
        .strip_prefix(PREFIX)
        .ok_or(SignatureError::Malformed)?;
    let expected = hex::decode(digest).map_err(|_| SignatureError::Malformed)?;
    if expected.len() != 32 {
        return Err(SignatureError::Malformed);
    }
    let mut mac = mac(secret);
    mac.update(body);
    mac.verify_slice(&expected)
        .map_err(|_| SignatureError::Mismatch)
}

/// The header value GitHub would send for `body` under `secret`
/// (`sha256=<hex>`): what a test, a replay tool, or a local simulator signs
/// with.
pub fn sign(secret: &[u8], body: &[u8]) -> String {
    let mut mac = mac(secret);
    mac.update(body);
    format!("{PREFIX}{}", hex::encode(mac.finalize().into_bytes()))
}

fn mac(secret: &[u8]) -> Hmac<Sha256> {
    // HMAC accepts a key of any length, so this cannot fail.
    Hmac::<Sha256>::new_from_slice(secret).expect("HMAC accepts any key length")
}

#[cfg(test)]
mod tests {
    use super::*;

    // The worked example from GitHub's webhook documentation.
    const SECRET: &[u8] = b"It's a Secret to Everybody";
    const BODY: &[u8] = b"Hello, World!";
    const SIGNATURE: &str =
        "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17";

    #[test]
    fn the_documented_vector_verifies_and_signs() {
        assert_eq!(sign(SECRET, BODY), SIGNATURE);
        assert_eq!(verify(SECRET, BODY, Some(SIGNATURE)), Ok(()));
        assert_eq!(
            verify(SECRET, BODY, Some(&format!("  {SIGNATURE} "))),
            Ok(()),
            "surrounding whitespace is tolerated"
        );
    }

    #[test]
    fn a_missing_malformed_or_wrong_signature_is_refused() {
        assert_eq!(verify(SECRET, BODY, None), Err(SignatureError::Missing));
        for bad in [
            "",
            "757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            "sha1=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            "sha256=zz7107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17",
            "sha256=757107ea",
        ] {
            assert_eq!(
                verify(SECRET, BODY, Some(bad)),
                Err(SignatureError::Malformed),
                "{bad}"
            );
        }
        let other_secret = sign(b"another secret", BODY);
        assert_eq!(
            verify(SECRET, BODY, Some(&other_secret)),
            Err(SignatureError::Mismatch)
        );
        assert_eq!(
            verify(SECRET, b"Hello, World?", Some(SIGNATURE)),
            Err(SignatureError::Mismatch),
            "a changed body no longer matches"
        );
    }
}

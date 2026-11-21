//! A member's profile: the name they go by and their image.
//!
//! A user id (`user:dana`) is what the engine's permissions are written
//! against and is never shown as a person's name where a profile gives one.
//! A profile is its own user's to write and their workspace's members' to
//! read, so one member is drawn for another by name and picture.
//!
//! The image is kept beside the profile, in the store, not in an archive: it
//! is small by rule, and a server with no archive has it all the same.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{AntumbraError, Result, TenantId, UserId};

/// The longest name a profile holds, in characters.
pub const MAX_DISPLAY_CHARS: usize = 64;

/// The largest image a profile holds, in bytes. A 256-pixel square, which is
/// what a client scales an image to before sending it, is a tenth of this in
/// any of the three forms; the room is for a client that did not.
pub const MAX_AVATAR_BYTES: usize = 256 * 1024;

/// A profile's image: its bytes and the media type its first bytes say it is.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Avatar {
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// One member's profile in one workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub tenant_id: TenantId,
    pub user: UserId,
    /// The name the member goes by, when they gave one.
    pub display: Option<String>,
    pub avatar: Option<Avatar>,
    pub updated_at: DateTime<Utc>,
}

/// The media type of an image a profile may hold, read from its first bytes:
/// PNG, JPEG or WebP. `None` for anything else, SVG among them: an SVG is a
/// document that can carry script, and what a member uploads is shown to
/// every other member.
pub fn image_type(bytes: &[u8]) -> Option<&'static str> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if bytes.starts_with(PNG) {
        return Some("image/png");
    }
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    None
}

/// `bytes` as a profile's image, or why it cannot be one. What the image is
/// goes by its bytes, never by what its sender called it.
pub fn avatar_of(bytes: Vec<u8>) -> Result<Avatar> {
    if bytes.is_empty() {
        return Err(AntumbraError::other("the image is empty"));
    }
    if bytes.len() > MAX_AVATAR_BYTES {
        return Err(AntumbraError::other(format!(
            "the image is {} KB; a profile's image is at most {} KB",
            bytes.len().div_ceil(1024),
            MAX_AVATAR_BYTES / 1024
        )));
    }
    let Some(media_type) = image_type(&bytes) else {
        return Err(AntumbraError::other(
            "a profile's image is a PNG, a JPEG or a WebP",
        ));
    };
    Ok(Avatar {
        media_type: media_type.to_string(),
        bytes,
    })
}

/// `raw` as a name to show: trimmed, its runs of whitespace made one space.
/// `None` for a name that is blank, which clears the one held. Refused when
/// it is too long or holds a control character, which a name never does and
/// a line drawn from it should not.
pub fn display_of(raw: &str) -> Result<Option<String>> {
    let name = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if name.is_empty() {
        return Ok(None);
    }
    if name.chars().any(char::is_control) {
        return Err(AntumbraError::other("a name holds no control characters"));
    }
    if name.chars().count() > MAX_DISPLAY_CHARS {
        return Err(AntumbraError::other(format!(
            "a name is at most {MAX_DISPLAY_CHARS} characters"
        )));
    }
    Ok(Some(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_image_is_what_its_first_bytes_say() -> anyhow::Result<()> {
        assert_eq!(image_type(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(image_type(&[0xFF, 0xD8, 0xFF, 0xE0, 0]), Some("image/jpeg"));
        assert_eq!(image_type(b"RIFF\x10\0\0\0WEBPVP8 "), Some("image/webp"));
        // A document that can carry script, a GIF, and nothing at all.
        assert_eq!(
            image_type(b"<svg xmlns='http://www.w3.org/2000/svg'>"),
            None
        );
        assert_eq!(image_type(b"<?xml version='1.0'?><svg>"), None);
        assert_eq!(image_type(b"GIF89a"), None);
        assert_eq!(image_type(b"RIFF\x10\0\0\0WAVEfmt "), None);
        assert_eq!(image_type(b""), None);
        Ok(())
    }

    #[test]
    fn an_image_is_held_within_its_size_and_only_of_a_kind() -> anyhow::Result<()> {
        let png = |len: usize| {
            let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
            bytes.resize(len, 0);
            bytes
        };
        let held = avatar_of(png(MAX_AVATAR_BYTES))?;
        assert_eq!(held.media_type, "image/png");
        assert!(avatar_of(png(MAX_AVATAR_BYTES + 1)).is_err());
        assert!(avatar_of(Vec::new()).is_err());
        assert!(avatar_of(b"<svg onload='x()'>".to_vec()).is_err());
        Ok(())
    }

    #[test]
    fn a_name_is_trimmed_and_a_blank_one_is_none() -> anyhow::Result<()> {
        assert_eq!(
            display_of("  Dana   Whitfield ")?.as_deref(),
            Some("Dana Whitfield")
        );
        assert_eq!(display_of("   ")?, None);
        assert!(display_of(&"a".repeat(MAX_DISPLAY_CHARS + 1)).is_err());
        assert_eq!(
            display_of(&"é".repeat(MAX_DISPLAY_CHARS))?.map(|n| n.chars().count()),
            Some(MAX_DISPLAY_CHARS)
        );
        assert!(display_of("Dana\u{7}").is_err());
        Ok(())
    }
}

//! Profiles: the name a member goes by and their image.
//!
//! Its own router, joined to the others in `engine.rs`. `profiles` is what a
//! page draws a member from; `set_profile` is how a member says who they are.
//! Both act as the caller: the engine lets a session read its workspace's
//! profiles and write only its own user's.

use super::*;

use antumbra_core::profile::{self, Profile};
use antumbra_store::repo::profile as store_profile;
use base64::Engine as _;

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct ProfilesParams {
    /// Send each member's image with their profile, as a `data:` URL. Left
    /// out, a profile says only whether it has one.
    #[serde(default)]
    pub(super) images: bool,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ProfileView {
    pub(super) user: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) display: Option<String>,
    pub(super) has_image: bool,
    /// The image as a `data:` URL, when asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) image: Option<String>,
    pub(super) updated_at: String,
}

#[derive(Serialize, schemars::JsonSchema)]
pub(super) struct ProfilesOut {
    /// The caller's own user id.
    pub(super) you: String,
    pub(super) profiles: Vec<ProfileView>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub(super) struct SetProfileParams {
    /// The name to go by. Blank clears it. Left out, the name is unchanged.
    #[serde(default)]
    pub(super) display: Option<String>,
    /// A PNG, JPEG or WebP of at most 256 KB, as base64 (a `data:` URL is
    /// accepted too). Left out, the image is unchanged.
    #[serde(default)]
    pub(super) image: Option<String>,
    /// Remove the image.
    #[serde(default)]
    pub(super) clear_image: bool,
}

fn view(p: &Profile, images: bool) -> ProfileView {
    ProfileView {
        user: p.user.as_str().to_string(),
        display: p.display.clone(),
        has_image: p.avatar.is_some(),
        image: p.avatar.as_ref().filter(|_| images).map(|a| {
            format!(
                "data:{};base64,{}",
                a.media_type,
                base64::engine::general_purpose::STANDARD.encode(&a.bytes)
            )
        }),
        updated_at: p.updated_at.to_rfc3339(),
    }
}

/// The bytes of an image sent as base64, bare or as a `data:` URL. The media
/// type a URL names is not believed: the bytes say what they are.
fn image_bytes(sent: &str) -> Result<Vec<u8>, ErrorData> {
    let encoded = match sent.trim().strip_prefix("data:") {
        Some(rest) => rest
            .split_once(";base64,")
            .map(|(_, data)| data)
            .ok_or_else(|| {
                ErrorData::invalid_params("an image sent as a data: URL is base64", None)
            })?,
        None => sent.trim(),
    };
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| ErrorData::invalid_params("the image is not base64", None))
}

#[tool_router(router = profile_router, vis = "pub(super)")]
impl McpServer {
    /// The profiles of this workspace's members.
    #[tool(
        description = "The profiles of this workspace's members: the name each goes by and whether they have an image (with `images`, the image itself as a data: URL). A member who has written no profile is not listed. Read-only."
    )]
    pub(super) async fn profiles(
        &self,
        Parameters(p): Parameters<ProfilesParams>,
    ) -> Result<Json<ProfilesOut>, ErrorData> {
        let all = store_profile::list(&self.store, &self.tenant)
            .await
            .map_err(err)?;
        Ok(Json(ProfilesOut {
            you: self.user.as_str().to_string(),
            profiles: all.iter().map(|profile| view(profile, p.images)).collect(),
        }))
    }

    /// Say who you are: the name you go by, and your image.
    #[tool(
        description = "Set your own profile in this workspace: `display`, the name you go by (blank clears it), and `image`, a PNG, JPEG or WebP of at most 256 KB as base64 (`clear_image` removes it). What is left out is unchanged. Other members see it; only you can change it."
    )]
    pub(super) async fn set_profile(
        &self,
        Parameters(p): Parameters<SetProfileParams>,
    ) -> Result<Json<ProfileView>, ErrorData> {
        let invalid =
            |e: antumbra_core::AntumbraError| ErrorData::invalid_params(e.to_string(), None);
        let held = store_profile::get(&self.store, &self.tenant, &self.user)
            .await
            .map_err(err)?;
        let display = match &p.display {
            Some(raw) => profile::display_of(raw).map_err(invalid)?,
            None => held.as_ref().and_then(|h| h.display.clone()),
        };
        let avatar = match (&p.image, p.clear_image) {
            (Some(_), true) => {
                return Err(ErrorData::invalid_params(
                    "send an image or clear it, not both",
                    None,
                ))
            }
            (Some(sent), false) => Some(profile::avatar_of(image_bytes(sent)?).map_err(invalid)?),
            (None, true) => None,
            (None, false) => held.and_then(|h| h.avatar),
        };
        let written = Profile {
            tenant_id: self.tenant.clone(),
            user: self.user.clone(),
            display,
            avatar,
            updated_at: chrono::Utc::now(),
        };
        store_profile::put(&self.store, &written)
            .await
            .map_err(err)?;
        Ok(Json(view(&written, false)))
    }
}

//! Who belongs to a workspace, and as what.
//!
//! A role is a member's standing in one workspace. It is not the `operator`
//! claim a token may carry: that one is whoever runs the server, and reads the
//! substrate's own state across every workspace it holds. A role reaches no
//! further than its workspace.
//!
//! - An **owner** decides who the workspace's admins and owners are. A
//!   workspace always has one.
//! - An **admin** lets people in and removes members.
//! - A **member** uses the workspace.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{AntumbraError, Result, UserId};

/// A member's standing in one workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Owner,
    Admin,
    Member,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Owner => "owner",
            Role::Admin => "admin",
            Role::Member => "member",
        }
    }

    /// Whether this role lets people in and removes members.
    pub fn admits(self) -> bool {
        matches!(self, Role::Owner | Role::Admin)
    }
}

impl std::str::FromStr for Role {
    type Err = AntumbraError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "owner" => Ok(Role::Owner),
            "admin" => Ok(Role::Admin),
            "member" => Ok(Role::Member),
            other => Err(AntumbraError::other(format!(
                "unknown role `{other}` (owner, admin or member)"
            ))),
        }
    }
}

/// One member of a workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    pub user: UserId,
    pub role: Role,
    /// When they joined, where that was recorded: a member from before roles
    /// has no date.
    pub joined_at: Option<DateTime<Utc>>,
}

/// Whether `actor` may give `target` the role `to`, and why not. Only an owner
/// changes roles; nobody is their own judge of a demotion that would leave the
/// workspace with no owner, which the store checks with the count in hand.
pub fn may_set_role(actor: Role, target: Role, to: Role) -> Result<()> {
    if actor != Role::Owner {
        return Err(AntumbraError::rejected(
            "only an owner of this workspace changes a member's role",
        ));
    }
    if target == to {
        return Err(AntumbraError::rejected(format!(
            "they are already {}",
            match to {
                Role::Owner => "an owner",
                Role::Admin => "an admin",
                Role::Member => "a member",
            }
        )));
    }
    Ok(())
}

/// Whether `actor` may remove a member whose role is `target`, and why not.
/// An admin removes members; only an owner removes an admin or an owner.
pub fn may_remove(actor: Role, target: Role) -> Result<()> {
    match (actor, target) {
        (Role::Owner, _) | (Role::Admin, Role::Member) => Ok(()),
        (Role::Admin, _) => Err(AntumbraError::rejected(
            "only an owner removes an admin or an owner",
        )),
        (Role::Member, _) => Err(AntumbraError::rejected(
            "only an admin or an owner of this workspace removes a member",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_role_is_read_as_it_is_written() -> anyhow::Result<()> {
        for role in [Role::Owner, Role::Admin, Role::Member] {
            assert_eq!(role.as_str().parse::<Role>()?, role);
            assert_eq!(serde_json::to_value(role)?, role.as_str());
        }
        assert!("operator".parse::<Role>().is_err());
        assert!(Role::Owner.admits() && Role::Admin.admits() && !Role::Member.admits());
        Ok(())
    }

    #[test]
    fn only_an_owner_changes_roles_and_an_admin_removes_only_members() -> anyhow::Result<()> {
        assert!(may_set_role(Role::Owner, Role::Member, Role::Admin).is_ok());
        assert!(may_set_role(Role::Owner, Role::Owner, Role::Member).is_ok());
        assert!(may_set_role(Role::Admin, Role::Member, Role::Admin).is_err());
        assert!(may_set_role(Role::Member, Role::Member, Role::Owner).is_err());
        assert!(may_set_role(Role::Owner, Role::Admin, Role::Admin).is_err());

        assert!(may_remove(Role::Owner, Role::Owner).is_ok());
        assert!(may_remove(Role::Admin, Role::Member).is_ok());
        assert!(may_remove(Role::Admin, Role::Admin).is_err());
        assert!(may_remove(Role::Admin, Role::Owner).is_err());
        assert!(may_remove(Role::Member, Role::Member).is_err());
        Ok(())
    }
}

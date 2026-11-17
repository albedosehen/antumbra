//! Magic-link delivery: the SMTP mailer behind `--features smtp`, selected by
//! `SMTP_*` at startup, and the dev mailer that logs the link to stderr, which
//! the operator must ask for with `--dev-mailer`. A sign-in link is a live
//! credential for a quarter of an hour, so nothing here falls back to logging
//! one: SMTP that was asked for and cannot be built refuses to start.

use std::sync::Arc;

use antumbra_control::Mailer;

/// The dev mailer: log the link to stderr so the passwordless flow is usable
/// without an email provider, for development. Anyone who reads the logs can
/// sign in, so it is chosen only by `--dev-mailer`.
pub struct StderrMailer;

impl Mailer for StderrMailer {
    fn send_link(&self, to: &str, link: &str) -> antumbra_core::Result<()> {
        eprintln!("[control-server dev mailer] magic link for {to}:\n  {link}");
        Ok(())
    }
}

#[cfg(feature = "smtp")]
pub struct SmtpMailer {
    transport: lettre::SmtpTransport,
    from: String,
}

#[cfg(feature = "smtp")]
impl SmtpMailer {
    fn from_env() -> anyhow::Result<Self> {
        use lettre::transport::smtp::authentication::Credentials;
        let host = std::env::var("SMTP_HOST")?;
        let from = std::env::var("SMTP_FROM")?;
        let mut relay = lettre::SmtpTransport::relay(&host)?;
        if let Ok(port) = std::env::var("SMTP_PORT") {
            relay = relay.port(port.parse()?);
        }
        if let (Ok(u), Ok(p)) = (std::env::var("SMTP_USER"), std::env::var("SMTP_PASS")) {
            relay = relay.credentials(Credentials::new(u, p));
        }
        Ok(Self {
            transport: relay.build(),
            from,
        })
    }
}

#[cfg(feature = "smtp")]
impl Mailer for SmtpMailer {
    fn send_link(&self, to: &str, link: &str) -> antumbra_core::Result<()> {
        use lettre::Transport;
        let oops = |e: String| antumbra_core::AntumbraError::other(format!("smtp: {e}"));
        let email = lettre::Message::builder()
            .from(self.from.parse().map_err(|e| oops(format!("from: {e}")))?)
            .to(to.parse().map_err(|e| oops(format!("to: {e}")))?)
            .subject("Your Antumbra sign-in link")
            .body(format!(
                "Sign in to Antumbra:\n\n{link}\n\nThe link expires shortly."
            ))
            .map_err(|e| oops(format!("build: {e}")))?;
        self.transport
            .send(&email)
            .map_err(|e| oops(format!("send: {e}")))?;
        Ok(())
    }
}

/// Whether the operator asked for SMTP delivery: a **non-empty** `SMTP_HOST`.
/// An empty value counts as unset, so a compose file can pass the `SMTP_*` seam
/// through with empty defaults without forcing SMTP on.
fn smtp_requested(host: Option<&str>) -> bool {
    host.is_some_and(|h| !h.is_empty())
}

/// The mailer this process sends sign-in links with: SMTP when `SMTP_HOST` is
/// set, and only SMTP, since a configuration that cannot be built must not
/// turn into links in the logs; otherwise the stderr mailer, and only when the
/// operator asked for it with `dev_mailer`. Without either there is no mailer,
/// and the error says what to set.
pub fn build_mailer(dev_mailer: bool) -> anyhow::Result<Arc<dyn Mailer>> {
    let host = std::env::var("SMTP_HOST").ok();
    mailer_for(host.as_deref(), dev_mailer, smtp_from_env)
}

/// The SMTP mailer this build makes from `SMTP_*`: one under `--features smtp`,
/// and the reason there is none otherwise.
fn smtp_from_env() -> anyhow::Result<Arc<dyn Mailer>> {
    #[cfg(feature = "smtp")]
    {
        Ok(Arc::new(SmtpMailer::from_env()?))
    }
    #[cfg(not(feature = "smtp"))]
    {
        anyhow::bail!("this build lacks --features smtp")
    }
}

/// [`build_mailer`]'s choice, given what was asked for and what SMTP can make.
fn mailer_for(
    host: Option<&str>,
    dev_mailer: bool,
    smtp: impl FnOnce() -> anyhow::Result<Arc<dyn Mailer>>,
) -> anyhow::Result<Arc<dyn Mailer>> {
    if smtp_requested(host) {
        return match smtp() {
            Ok(mailer) => {
                eprintln!("antumbra-control-server: delivering magic links via SMTP");
                Ok(mailer)
            }
            Err(e) => Err(anyhow::anyhow!(
                "SMTP_HOST is set but the SMTP mailer could not be built: {e}. Fix the \
                 SMTP_* settings, or unset SMTP_HOST and pass --dev-mailer to log \
                 sign-in links to stderr instead (for development only)"
            )),
        };
    }
    if dev_mailer {
        eprintln!(
            "antumbra-control-server: DEV mailer -- sign-in links are logged to stderr, \
             and anyone who reads the logs can sign in (set SMTP_* for real email)"
        );
        return Ok(Arc::new(StderrMailer));
    }
    anyhow::bail!(
        "no way to deliver sign-in links: set SMTP_HOST (with SMTP_FROM, and SMTP_PORT, \
         SMTP_USER, SMTP_PASS as needed) for email, or pass --dev-mailer \
         (ANTUMBRA_DEV_MAILER=1) to log links to stderr for development"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smtp_is_requested_only_by_a_non_empty_host() -> anyhow::Result<()> {
        assert!(!smtp_requested(None), "unset = not requested");
        assert!(!smtp_requested(Some("")), "empty (compose default) = unset");
        assert!(smtp_requested(Some("smtp.example.com")));
        Ok(())
    }

    /// An SMTP mailer that cannot be built, as a bad `SMTP_PORT` leaves it.
    fn broken_smtp() -> anyhow::Result<Arc<dyn Mailer>> {
        anyhow::bail!("port: invalid digit found in string")
    }

    /// An SMTP mailer that builds; what it is does not matter here.
    fn working_smtp() -> anyhow::Result<Arc<dyn Mailer>> {
        Ok(Arc::new(StderrMailer))
    }

    #[test]
    fn a_requested_smtp_that_fails_to_build_is_fatal() -> anyhow::Result<()> {
        let Err(e) = mailer_for(Some("smtp.example"), false, broken_smtp) else {
            anyhow::bail!("a mailer was built from a broken SMTP configuration");
        };
        let said = e.to_string();
        anyhow::ensure!(
            said.contains("SMTP_HOST is set") && said.contains("invalid digit"),
            "{said}"
        );
        Ok(())
    }

    #[test]
    fn the_dev_flag_does_not_rescue_a_broken_smtp() -> anyhow::Result<()> {
        anyhow::ensure!(
            mailer_for(Some("smtp.example"), true, broken_smtp).is_err(),
            "SMTP was asked for; the flag must not turn it into logged links"
        );
        Ok(())
    }

    #[test]
    fn a_requested_smtp_that_builds_is_used() -> anyhow::Result<()> {
        mailer_for(Some("smtp.example"), false, working_smtp)?;
        mailer_for(Some("smtp.example"), true, working_smtp)?;
        Ok(())
    }

    #[test]
    fn without_smtp_the_stderr_mailer_needs_the_flag() -> anyhow::Result<()> {
        let Err(e) = mailer_for(None, false, broken_smtp) else {
            anyhow::bail!("a mailer was chosen with neither SMTP nor the flag");
        };
        let said = e.to_string();
        anyhow::ensure!(
            said.contains("--dev-mailer") && said.contains("SMTP_HOST"),
            "{said}"
        );
        mailer_for(None, true, broken_smtp)?;
        mailer_for(Some(""), true, broken_smtp)?;
        Ok(())
    }
}

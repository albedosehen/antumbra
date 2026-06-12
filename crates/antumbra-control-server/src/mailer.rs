//! Magic-link delivery: the dev mailer that logs the link to stderr (the
//! default, so the passwordless flow works without an email provider), and the
//! real SMTP mailer behind `--features smtp`, selected by `SMTP_*` at startup.

use std::sync::Arc;

use antumbra_control::Mailer;

/// The dev mailer: log the link to stderr so the passwordless flow is usable
/// without an email provider. Swap in [`SmtpMailer`] (under `--features smtp`).
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

pub fn build_mailer() -> Arc<dyn Mailer> {
    let host = std::env::var("SMTP_HOST").ok();
    if smtp_requested(host.as_deref()) {
        #[cfg(feature = "smtp")]
        match SmtpMailer::from_env() {
            Ok(m) => {
                eprintln!("antumbra-control-server: delivering magic links via SMTP");
                return Arc::new(m);
            }
            Err(e) => {
                eprintln!("antumbra-control-server: SMTP config failed ({e}); using dev mailer")
            }
        }
        #[cfg(not(feature = "smtp"))]
        eprintln!(
            "antumbra-control-server: SMTP_HOST is set but this build lacks \
             --features smtp; using dev mailer"
        );
    }
    eprintln!(
        "antumbra-control-server: DEV mailer -- links are logged to stderr \
         (build --features smtp and set SMTP_* for real email)"
    );
    Arc::new(StderrMailer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smtp_is_requested_only_by_a_non_empty_host() {
        assert!(!smtp_requested(None), "unset = dev mailer");
        assert!(!smtp_requested(Some("")), "empty (compose default) = unset");
        assert!(smtp_requested(Some("smtp.example.com")));
    }
}

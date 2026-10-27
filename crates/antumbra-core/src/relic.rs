//! Relics: the pages a user keeps, found and viewed in the dashboard.
//!
//! A relic is a rendered document, today an HTML page an agent wrote, kept in
//! copal and listed here. Its row holds what the gallery shows and what a view
//! needs: a stable slug (the `/relic/<slug>` address), a title, who owns it,
//! the compartment that decides who else may see it, whether it is public, and
//! every version's copal file. The bytes live only in copal; the text is
//! ingested as a document titled `relic:<slug>`, so recall finds relics by what
//! they say.
//!
//! Private is the default, and a private relic's audience is its compartment's
//! audience (ADR-0020): its owner, and whoever the compartment is granted to.
//! Public means anyone who can reach the server, with no sign-in.

use chrono::{DateTime, Utc};

use crate::{CompartmentId, TenantId, UserId};

mod markdown;

pub use markdown::html_to_markdown;

/// Who may see a relic beyond its compartment's audience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Its compartment's audience: the owner and any grantee.
    Private,
    /// Anyone who can reach the server, signed in or not.
    Public,
}

impl Visibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Private => "private",
            Visibility::Public => "public",
        }
    }
}

impl std::str::FromStr for Visibility {
    type Err = crate::AntumbraError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "private" => Ok(Visibility::Private),
            "public" => Ok(Visibility::Public),
            other => Err(crate::AntumbraError::other(format!(
                "unknown visibility `{other}` (use private | public)"
            ))),
        }
    }
}

/// One published version of a relic: its own copal file, never overwritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelicVersion {
    /// 1 for the first version, counting up.
    pub number: u32,
    pub copal_file: String,
    pub copal_digest: String,
    pub created_at: DateTime<Utc>,
    /// The machine the version was published from, when the call named one.
    pub author_host: Option<String>,
}

/// A relic as the gallery and the viewer see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relic {
    /// The address: `/relic/<slug>`. Unique across the server, because a
    /// public relic is found by its slug alone.
    pub slug: String,
    pub tenant: TenantId,
    pub owner: UserId,
    /// Decides who besides the owner may see it while it is private.
    pub compartment: CompartmentId,
    pub title: String,
    pub visibility: Visibility,
    /// Oldest first; never empty.
    pub versions: Vec<RelicVersion>,
    /// Where it came from, when the publisher said.
    pub source: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Relic {
    /// The version a view shows: the newest.
    pub fn current(&self) -> Option<&RelicVersion> {
        self.versions.last()
    }

    /// The number the next published version takes.
    pub fn next_version(&self) -> u32 {
        self.current().map_or(1, |v| v.number + 1)
    }
}

/// The longest slug, so an address stays something a person can type.
pub const MAX_SLUG_LEN: usize = 64;

/// Whether `slug` can be a relic's address: lowercase letters, digits and
/// hyphens, starting and ending with a letter or digit, at most
/// [`MAX_SLUG_LEN`].
pub fn is_valid_slug(slug: &str) -> bool {
    let bytes = slug.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_SLUG_LEN
        && bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-')
        && bytes.first() != Some(&b'-')
        && bytes.last() != Some(&b'-')
}

/// A slug drawn from a title: lowercase words joined by hyphens, cut to 48
/// characters so a suffix still fits. `relic` when the title has no letters
/// or digits to draw from.
pub fn slugify(title: &str) -> String {
    let mut slug = String::new();
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
        if slug.len() >= 48 {
            break;
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "relic".to_string()
    } else {
        slug.to_string()
    }
}

/// The title a relic's text is ingested under, so recall can find relics and
/// tell their chunks from other documents'.
pub fn document_title(slug: &str) -> String {
    format!("relic:{slug}")
}

/// The slug a document title names, when it is a relic's.
pub fn slug_of_document(title: &str) -> Option<&str> {
    title.strip_prefix("relic:")
}

/// Elements whose contents are never shown as text.
const HIDDEN: &[&str] = &["script", "style", "template", "noscript"];

/// Elements that start a new block, rendered as a paragraph break.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "caption",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "td",
    "th",
    "title",
    "tr",
    "ul",
];

/// The words a reader sees in an HTML page: tags, comments and the contents of
/// `script`, `style`, `template` and `noscript` dropped, block elements turned
/// into paragraph breaks. A reader, not a parser: it never fails, and malformed
/// markup yields whatever text it can find.
pub fn html_to_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let mut raw = String::with_capacity(html.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            if lower[i..].starts_with("<!--") {
                i = lower[i..]
                    .find("-->")
                    .map_or(bytes.len(), |end| i + end + 3);
                continue;
            }
            let (name, closing, end) = read_tag(&lower, i);
            if !closing && HIDDEN.contains(&name.as_str()) {
                let close = format!("</{name}");
                i = match lower[end..].find(&close) {
                    Some(at) => read_tag(&lower, end + at).2,
                    None => bytes.len(),
                };
                continue;
            }
            if BLOCKS.contains(&name.as_str()) {
                raw.push_str("\n\n");
            }
            i = end;
        } else if bytes[i] == b'&' {
            let (text, next) = read_entity(html, i);
            raw.push_str(&text);
            i = next;
        } else {
            let next = html[i..].find(['<', '&']).map_or(bytes.len(), |at| i + at);
            raw.push_str(&html[i..next]);
            i = next;
        }
    }
    tidy(&raw)
}

/// The tag starting at `start` (a `<`): its lowercase name, whether it closes
/// an element, and the index just past its `>`, honoring quoted attributes.
fn read_tag(lower: &str, start: usize) -> (String, bool, usize) {
    let bytes = lower.as_bytes();
    let mut i = start + 1;
    let closing = bytes.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    let name_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-') {
        i += 1;
    }
    let name = lower[name_start..i].to_owned();
    let mut quote = None;
    while i < bytes.len() {
        match (quote, bytes[i]) {
            (None, b'"' | b'\'') => quote = Some(bytes[i]),
            (Some(q), b) if b == q => quote = None,
            (None, b'>') => return (name, closing, i + 1),
            _ => {}
        }
        i += 1;
    }
    (name, closing, bytes.len())
}

/// Decode the character reference at `start` (a `&`); an unknown or malformed
/// one is kept as written.
fn read_entity(html: &str, start: usize) -> (String, usize) {
    let rest = &html[start + 1..];
    let Some(semi) = rest.find(';').filter(|&at| at <= 10) else {
        return ("&".to_owned(), start + 1);
    };
    let name = &rest[..semi];
    let decoded = if let Some(code) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
        u32::from_str_radix(code, 16).ok().and_then(char::from_u32)
    } else if let Some(code) = name.strip_prefix('#') {
        code.parse().ok().and_then(char::from_u32)
    } else {
        named(name)
    };
    match decoded {
        Some(c) => (c.to_string(), start + semi + 2),
        None => ("&".to_owned(), start + 1),
    }
}

fn named(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "middot" => '·',
        "hellip" => '…',
        "mdash" => '—',
        "ndash" => '–',
        "lsquo" => '‘',
        "rsquo" => '’',
        "ldquo" => '“',
        "rdquo" => '”',
        "copy" => '©',
        "times" => '×',
        "rarr" => '→',
        "larr" => '←',
        "eacute" => 'é',
        "oslash" => 'ø',
        _ => return None,
    })
}

/// Collapse runs of spaces within lines and of blank lines between them.
fn tidy(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut blank_run = 0;
    for line in raw.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.is_empty() {
            blank_run += 1;
            continue;
        }
        if !out.is_empty() {
            out.push_str(if blank_run > 0 { "\n\n" } else { "\n" });
        }
        out.push_str(&words.join(" "));
        blank_run = 0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_are_typeable_addresses() {
        for good in ["ceiling", "the-ceiling-v4", "a", "x1", &"a".repeat(64)] {
            assert!(is_valid_slug(good), "{good:?}");
        }
        for bad in [
            "",
            "-ceiling",
            "ceiling-",
            "Ceiling",
            "two words",
            "relic/x",
            &"a".repeat(65),
        ] {
            assert!(!is_valid_slug(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_title_becomes_a_valid_slug() {
        assert_eq!(
            slugify("The Ceiling Pays for the Floor"),
            "the-ceiling-pays-for-the-floor"
        );
        assert_eq!(
            slugify("  DP-450: Event Gateway!  "),
            "dp-450-event-gateway"
        );
        assert_eq!(slugify("¡¿!"), "relic");
        let long = slugify(&"word ".repeat(40));
        assert!(long.len() <= 48 && is_valid_slug(&long), "{long}");
    }

    #[test]
    fn visibility_reads_its_own_words_only() {
        assert_eq!("Public".parse::<Visibility>().unwrap(), Visibility::Public);
        assert_eq!(
            " private ".parse::<Visibility>().unwrap(),
            Visibility::Private
        );
        assert!("unlisted".parse::<Visibility>().is_err());
        assert_eq!(Visibility::Public.as_str(), "public");
    }

    #[test]
    fn document_titles_name_their_relic() {
        assert_eq!(document_title("ceiling"), "relic:ceiling");
        assert_eq!(slug_of_document("relic:ceiling"), Some("ceiling"));
        assert_eq!(slug_of_document("handbook"), None);
    }

    #[test]
    fn html_text_keeps_the_words_and_drops_the_machinery() {
        let page = r#"<!doctype html><html><head><title>The Ceiling</title>
            <style>.lead { color: red }</style></head><body>
            <!-- a note -->
            <h1>The ceiling pays</h1><p class="lead">People are <em>hurting</em> today.</p>
            <script>var DATA = {"labor":[[1947,65.2]]};</script>
            <noscript>Turn on scripts</noscript>
            <p>G&oslash;sta &amp; AT&T &#8212; done.</p></body></html>"#;
        assert_eq!(
            html_to_text(page),
            "The Ceiling\n\nThe ceiling pays\n\nPeople are hurting today.\n\nGøsta & AT&T — done."
        );
        assert_eq!(html_to_text("before <script>never closed"), "before");
        assert_eq!(html_to_text(r#"<a title="a > b">kept</a>"#), "kept");
    }
}

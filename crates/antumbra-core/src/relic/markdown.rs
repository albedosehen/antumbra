//! Markdown from a relic's HTML, for the copy a person takes away: what a
//! reader sees on the page, in the shapes Markdown has (headings, paragraphs,
//! emphasis, links, lists, quotes, code and tables). Like
//! [`super::html_to_text`] it is a reader, not a parser: it never fails, and
//! malformed markup yields whatever structure it can find. Scripts, styles,
//! form controls and drawings (SVG, canvas) have no Markdown and are left out.

use super::read_entity;

/// Elements whose contents are raw text rather than markup: never parsed, and
/// never shown, except a `title`, which names the document.
const RAW: &[&str] = &[
    "noscript", "script", "style", "template", "textarea", "title",
];

/// Elements that never have contents or a closing tag.
const VOID: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// Elements left out with everything inside them.
const SKIPPED: &[&str] = &[
    "audio", "base", "button", "canvas", "datalist", "dialog", "embed", "head", "iframe", "input",
    "link", "map", "meta", "object", "option", "select", "svg", "video",
];

/// Elements that stand as blocks of their own rather than run in a line.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "caption",
    "center",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hgroup",
    "hr",
    "html",
    "legend",
    "li",
    "main",
    "menu",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
];

/// Stands for a `<br>` inside a line until the line is tidied, when it becomes
/// a Markdown hard break. A private-use character no page means as text.
const BREAK: char = '\u{E000}';

/// The page's Markdown: its title as the top heading when the page has no
/// `h1` of its own, then its blocks separated by blank lines.
pub fn html_to_markdown(html: &str) -> String {
    let tree = Tree::parse(html);
    let blocks = tree.blocks(ROOT);
    let mut out = String::new();
    let title = tidy(&tree.title.replace(BREAK, " "));
    if !title.is_empty() && !blocks.iter().any(|b| b.starts_with("# ")) {
        out.push_str("# ");
        out.push_str(&title);
        out.push_str("\n\n");
    }
    out.push_str(&blocks.join("\n\n"));
    let trimmed = out.trim_end().len();
    out.truncate(trimmed);
    out.push('\n');
    out
}

const ROOT: usize = 0;

enum Node {
    Element {
        name: String,
        attrs: Vec<(String, String)>,
        kids: Vec<usize>,
    },
    Text(String),
}

/// The page as elements and text, built the forgiving way a browser builds
/// it: void elements never open, a closing tag with nothing to close is
/// dropped, and the tags HTML lets a page leave open (`p`, `li`, `tr`, `td`
/// and their kin) close when the next one starts.
struct Tree {
    nodes: Vec<Node>,
    title: String,
}

struct Tag {
    name: String,
    closing: bool,
    attrs: Vec<(String, String)>,
    end: usize,
}

impl Tree {
    fn parse(html: &str) -> Tree {
        let lower = html.to_ascii_lowercase();
        let bytes = html.as_bytes();
        let mut tree = Tree {
            nodes: vec![Node::Element {
                name: "#root".into(),
                attrs: Vec::new(),
                kids: Vec::new(),
            }],
            title: String::new(),
        };
        let mut open = vec![ROOT];
        let mut i = 0;
        while i < bytes.len() {
            let at = *open.last().unwrap_or(&ROOT);
            if bytes[i] == b'<' {
                if lower[i..].starts_with("<!--") {
                    i = lower[i..]
                        .find("-->")
                        .map_or(bytes.len(), |end| i + end + 3);
                    continue;
                }
                match bytes.get(i + 1) {
                    Some(b'!' | b'?') => {
                        i = lower[i..].find('>').map_or(bytes.len(), |end| i + end + 1);
                        continue;
                    }
                    Some(b) if b.is_ascii_alphabetic() || *b == b'/' => {}
                    // A `<` that starts no tag is text.
                    _ => {
                        tree.text(at, "<");
                        i += 1;
                        continue;
                    }
                }
                let tag = read_tag(html, &lower, i);
                i = tag.end;
                if tag.name.is_empty() {
                    continue;
                }
                if tag.closing {
                    close(&mut open, &tree, &tag.name);
                    continue;
                }
                if RAW.contains(&tag.name.as_str()) {
                    let close = format!("</{}", tag.name);
                    let (content_end, after) = match lower[i..].find(&close) {
                        Some(found) => (i + found, read_tag(html, &lower, i + found).end),
                        None => (bytes.len(), bytes.len()),
                    };
                    if tag.name == "title" && tree.title.is_empty() {
                        tree.title = decode(&html[i..content_end]);
                    }
                    i = after;
                    continue;
                }
                implied_closes(&mut open, &tree, &tag.name);
                let parent = *open.last().unwrap_or(&ROOT);
                let void = VOID.contains(&tag.name.as_str());
                let index = tree.push(
                    parent,
                    Node::Element {
                        name: tag.name,
                        attrs: tag.attrs,
                        kids: Vec::new(),
                    },
                );
                if !void {
                    open.push(index);
                }
            } else if bytes[i] == b'&' {
                let (text, next) = read_entity(html, i);
                tree.text(at, &text);
                i = next;
            } else {
                let next = html[i..].find(['<', '&']).map_or(bytes.len(), |n| i + n);
                tree.text(at, &html[i..next]);
                i = next;
            }
        }
        tree
    }

    fn push(&mut self, parent: usize, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        if let Node::Element { kids, .. } = &mut self.nodes[parent] {
            kids.push(index);
        }
        index
    }

    /// Append text to `parent`, joining the text node it ends with.
    fn text(&mut self, parent: usize, text: &str) {
        if text.is_empty() {
            return;
        }
        let last = match &self.nodes[parent] {
            Node::Element { kids, .. } => kids.last().copied(),
            Node::Text(_) => None,
        };
        if let Some(last) = last {
            if let Node::Text(existing) = &mut self.nodes[last] {
                existing.push_str(text);
                return;
            }
        }
        self.push(parent, Node::Text(text.to_string()));
    }

    fn name(&self, index: usize) -> Option<&str> {
        match &self.nodes[index] {
            Node::Element { name, .. } => Some(name),
            Node::Text(_) => None,
        }
    }

    fn kids(&self, index: usize) -> &[usize] {
        match &self.nodes[index] {
            Node::Element { kids, .. } => kids,
            Node::Text(_) => &[],
        }
    }

    fn attr(&self, index: usize, key: &str) -> Option<&str> {
        match &self.nodes[index] {
            Node::Element { attrs, .. } => attrs
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str()),
            Node::Text(_) => None,
        }
    }

    /// Whether `index` is left out: an element with no Markdown, or one the
    /// page itself hides.
    fn skipped(&self, index: usize) -> bool {
        self.name(index)
            .is_some_and(|name| SKIPPED.contains(&name) || self.attr(index, "hidden").is_some())
    }

    /// The blocks inside `index`: runs of inline content become paragraphs,
    /// and each block element its own blocks.
    fn blocks(&self, index: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut line = String::new();
        let mut after_element = false;
        for &kid in self.kids(index) {
            if self.skipped(kid) {
                continue;
            }
            match self.name(kid) {
                Some(name) if BLOCKS.contains(&name) => {
                    flush(&mut line, &mut out);
                    after_element = false;
                    out.extend(self.block(kid));
                }
                _ => self.append_inline(&mut line, &mut after_element, kid),
            }
        }
        flush(&mut line, &mut out);
        out
    }

    /// Append `kid`'s inline Markdown to `line`. Two elements written back to
    /// back with no space between them were set apart by the page's styles
    /// (a label above its value, a title above its note), so their words get
    /// a space; a superscript, a subscript and punctuation stay attached.
    fn append_inline(&self, line: &mut String, after_element: &mut bool, kid: usize) {
        let piece = self.inline(kid);
        let element = self.name(kid);
        if piece.is_empty() {
            return;
        }
        let ends_word = line
            .chars()
            .last()
            .is_some_and(|c| !c.is_whitespace() && c != BREAK);
        let starts_word = piece
            .chars()
            .next()
            .is_some_and(|c| !c.is_whitespace() && c != BREAK && !".,;:!?)".contains(c));
        if *after_element
            && ends_word
            && starts_word
            && !matches!(element, None | Some("sup" | "sub" | "br" | "wbr"))
        {
            line.push(' ');
        }
        line.push_str(&piece);
        *after_element = element.is_some();
    }

    fn block(&self, index: usize) -> Vec<String> {
        let name = self.name(index).unwrap_or_default();
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = usize::from(name.as_bytes()[1] - b'0');
                let text = one_line(&self.inline_kids(index));
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![format!("{} {text}", "#".repeat(level))]
                }
            }
            "hr" => vec!["---".into()],
            "pre" => vec![self.fence(index)],
            "blockquote" => {
                let inner = self.blocks(index).join("\n\n");
                if inner.is_empty() {
                    Vec::new()
                } else {
                    vec![inner
                        .lines()
                        .map(|l| {
                            if l.is_empty() {
                                ">".to_string()
                            } else {
                                format!("> {l}")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")]
                }
            }
            "ul" | "ol" | "menu" => {
                let list = self.list(index);
                if list.is_empty() {
                    Vec::new()
                } else {
                    vec![list]
                }
            }
            "table" => self.table(index),
            "dt" => {
                let text = one_line(&self.inline_kids(index));
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![format!("**{text}**")]
                }
            }
            _ => self.blocks(index),
        }
    }

    /// The Markdown of one node inside a line, untidied.
    fn inline(&self, index: usize) -> String {
        let Some(name) = self.name(index) else {
            return match &self.nodes[index] {
                Node::Text(text) => escape(text),
                Node::Element { .. } => String::new(),
            };
        };
        if self.skipped(index) {
            return String::new();
        }
        match name {
            "br" => BREAK.to_string(),
            "strong" | "b" => wrap(&self.inline_kids(index), "**"),
            "em" | "i" | "cite" | "dfn" => wrap(&self.inline_kids(index), "*"),
            "del" | "s" | "strike" => wrap(&self.inline_kids(index), "~~"),
            "code" | "kbd" | "samp" | "tt" => code_span(&self.raw_text(index)),
            "q" => format!("“{}”", self.inline_kids(index)),
            "a" => {
                let text = one_line(&self.inline_kids(index));
                let href = self.attr(index, "href").unwrap_or_default().trim();
                if href.is_empty() || href.starts_with('#') || href.starts_with("javascript:") {
                    text
                } else if text.is_empty() {
                    format!("<{href}>")
                } else if href.contains([' ', '(', ')']) {
                    format!("[{text}](<{href}>)")
                } else {
                    format!("[{text}]({href})")
                }
            }
            "img" => {
                let alt = one_line(&escape(self.attr(index, "alt").unwrap_or_default()));
                match self.attr(index, "src").map(str::trim) {
                    // An image carried inline would be pages of base64.
                    Some(src) if !src.is_empty() && !src.starts_with("data:") => {
                        format!("![{alt}]({src})")
                    }
                    _ => alt,
                }
            }
            // A block met inside a line keeps its words apart from the next.
            _ if BLOCKS.contains(&name) => format!(" {} ", self.inline_kids(index)),
            _ => self.inline_kids(index),
        }
    }

    fn inline_kids(&self, index: usize) -> String {
        let mut line = String::new();
        let mut after_element = false;
        for &kid in self.kids(index) {
            self.append_inline(&mut line, &mut after_element, kid);
        }
        line
    }

    /// Every character of text inside `index`, as written: what code shows.
    fn raw_text(&self, index: usize) -> String {
        match &self.nodes[index] {
            Node::Text(text) => text.clone(),
            Node::Element { name, kids, .. } => {
                if name == "br" {
                    "\n".into()
                } else if self.skipped(index) {
                    String::new()
                } else {
                    kids.iter().map(|&k| self.raw_text(k)).collect()
                }
            }
        }
    }

    /// A `pre` as a fenced code block, with its language when a
    /// `language-<name>` class names one.
    fn fence(&self, index: usize) -> String {
        let raw = self.raw_text(index);
        let text = raw.strip_prefix('\n').unwrap_or(&raw).trim_end();
        let language = std::iter::once(index)
            .chain(self.kids(index).iter().copied())
            .filter_map(|k| self.attr(k, "class"))
            .flat_map(str::split_whitespace)
            .find_map(|c| {
                c.strip_prefix("language-")
                    .or_else(|| c.strip_prefix("lang-"))
            })
            .unwrap_or_default();
        let mut fence = "```".to_string();
        while text.contains(fence.as_str()) {
            fence.push('`');
        }
        format!("{fence}{language}\n{text}\n{fence}")
    }

    fn list(&self, index: usize) -> String {
        let ordered = self.name(index) == Some("ol");
        let mut number: i64 = self
            .attr(index, "start")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1);
        let mut items = Vec::new();
        let mut loose = false;
        for &kid in self.kids(index) {
            if self.name(kid) != Some("li") || self.skipped(kid) {
                continue;
            }
            let marker = if ordered {
                format!("{number}. ")
            } else {
                "- ".to_string()
            };
            number += 1;
            let blocks = self.blocks(kid);
            if blocks.iter().skip(1).any(|b| !is_list(b)) {
                loose = true;
            }
            // The first line follows the marker; every later line is indented
            // under it, so a nested list or a second paragraph stays inside
            // the item.
            let indent = " ".repeat(marker.len());
            let mut item = marker.clone();
            for (j, block) in blocks.iter().enumerate() {
                if j > 0 {
                    item.push_str(if is_list(block) { "\n" } else { "\n\n" });
                }
                for (k, line) in block.lines().enumerate() {
                    if k > 0 {
                        item.push('\n');
                    }
                    if j == 0 && k == 0 {
                        item.push_str(line);
                    } else if !line.is_empty() {
                        item.push_str(&indent);
                        item.push_str(line);
                    }
                }
            }
            items.push(item.trim_end().to_string());
        }
        items.join(if loose { "\n\n" } else { "\n" })
    }

    /// A table as a GitHub table: its caption above it, the first row as the
    /// header, and a column aligned right when its header cell is numeric
    /// (`align="right"` or a `num` class).
    fn table(&self, index: usize) -> Vec<String> {
        let mut caption = String::new();
        let mut rows: Vec<usize> = Vec::new();
        for &kid in self.kids(index) {
            match self.name(kid) {
                Some("caption") => caption = one_line(&self.inline_kids(kid)),
                Some("thead" | "tbody" | "tfoot") => rows.extend(
                    self.kids(kid)
                        .iter()
                        .copied()
                        .filter(|&r| self.name(r) == Some("tr")),
                ),
                Some("tr") => rows.push(kid),
                _ => {}
            }
        }
        let mut grid: Vec<Vec<String>> = Vec::new();
        let mut right: Vec<bool> = Vec::new();
        for (r, &row) in rows.iter().enumerate() {
            let mut cells = Vec::new();
            for &cell in self.kids(row) {
                if !matches!(self.name(cell), Some("td" | "th")) || self.skipped(cell) {
                    continue;
                }
                let text = one_line(&self.inline_kids(cell)).replace('|', "\\|");
                let span = self
                    .attr(cell, "colspan")
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .unwrap_or(1)
                    .clamp(1, 64);
                if r == 0 {
                    let numeric = self.attr(cell, "align") == Some("right")
                        || self
                            .attr(cell, "class")
                            .is_some_and(|c| c.split_whitespace().any(|c| c == "num"));
                    right.extend(std::iter::repeat_n(numeric, span));
                }
                cells.push(text);
                cells.extend(std::iter::repeat_n(String::new(), span - 1));
            }
            grid.push(cells);
        }
        let columns = grid.iter().map(Vec::len).max().unwrap_or(0);
        let mut out = Vec::new();
        if !caption.is_empty() {
            out.push(caption);
        }
        if columns == 0 {
            return out;
        }
        right.resize(columns, false);
        let row = |cells: &[String]| {
            let mut line = String::from("|");
            for c in 0..columns {
                line.push(' ');
                line.push_str(cells.get(c).map_or("", String::as_str));
                line.push_str(" |");
            }
            line
        };
        let mut lines = vec![row(&grid[0])];
        lines.push(
            std::iter::once("|".to_string())
                .chain(
                    right
                        .iter()
                        .map(|&r| if r { " ---: |" } else { " --- |" }.to_string()),
                )
                .collect(),
        );
        lines.extend(grid[1..].iter().map(|cells| row(cells)));
        out.push(lines.join("\n"));
        out
    }
}

/// The tag starting at `start` (a `<`): its lowercase name, whether it closes
/// an element, its attributes (names lowercase, values decoded) and the index
/// just past its `>`.
fn read_tag(html: &str, lower: &str, start: usize) -> Tag {
    let bytes = lower.as_bytes();
    let len = bytes.len();
    let mut i = start + 1;
    let closing = bytes.get(i) == Some(&b'/');
    if closing {
        i += 1;
    }
    let name_start = i;
    while i < len && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'-' | b':')) {
        i += 1;
    }
    let name = lower[name_start..i].to_string();
    let mut attrs = Vec::new();
    loop {
        while i < len && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
            i += 1;
        }
        if i >= len {
            return Tag {
                name,
                closing,
                attrs,
                end: len,
            };
        }
        if bytes[i] == b'>' {
            return Tag {
                name,
                closing,
                attrs,
                end: i + 1,
            };
        }
        let key_start = i;
        while i < len && !bytes[i].is_ascii_whitespace() && !matches!(bytes[i], b'=' | b'>' | b'/')
        {
            i += 1;
        }
        let key = lower[key_start..i].to_string();
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut value = String::new();
        if i < len && bytes[i] == b'=' {
            i += 1;
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < len && matches!(bytes[i], b'"' | b'\'') {
                let quote = bytes[i] as char;
                let end = lower[i + 1..].find(quote).map_or(len, |at| i + 1 + at);
                value = decode(&html[i + 1..end]);
                i = (end + 1).min(len);
            } else {
                let value_start = i;
                while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' {
                    i += 1;
                }
                value = decode(&html[value_start..i]);
            }
        } else if key_start == i {
            // Nothing a tag can hold: step over it.
            i += 1;
        }
        if !key.is_empty() {
            attrs.push((key, value));
        }
    }
}

/// Text with its character references decoded.
fn decode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        match text[i..].find('&') {
            Some(at) => {
                out.push_str(&text[i..i + at]);
                let (decoded, next) = read_entity(text, i + at);
                out.push_str(&decoded);
                i = next;
            }
            None => {
                out.push_str(&text[i..]);
                break;
            }
        }
    }
    out
}

/// Close the nearest open `name`, and everything opened inside it. A closing
/// tag with nothing to close is dropped.
fn close(open: &mut Vec<usize>, tree: &Tree, name: &str) {
    if let Some(at) = open.iter().rposition(|&i| tree.name(i) == Some(name)) {
        if at > 0 {
            open.truncate(at);
        }
    }
}

/// Close what HTML lets a page leave open when `next` starts: a paragraph at
/// any block, a list item at the next item, a row at the next row, a cell at
/// the next cell.
fn implied_closes(open: &mut Vec<usize>, tree: &Tree, next: &str) {
    let (closes, within): (&[&str], &[&str]) = match next {
        "li" => (&["li"], &["ul", "ol", "menu"]),
        "dt" | "dd" => (&["dt", "dd"], &["dl"]),
        "tr" => (&["tr"], &["table", "thead", "tbody", "tfoot"]),
        "td" | "th" => (&["td", "th"], &["tr", "table"]),
        "thead" | "tbody" | "tfoot" => (&["thead", "tbody", "tfoot"], &["table"]),
        _ => (&[], &[]),
    };
    for (depth, &i) in open.iter().enumerate().rev() {
        let Some(name) = tree.name(i) else { break };
        if within.contains(&name) {
            break;
        }
        if closes.contains(&name) {
            open.truncate(depth);
            break;
        }
    }
    if BLOCKS.contains(&next) && open.last().and_then(|&i| tree.name(i)) == Some("p") {
        open.pop();
    }
}

/// A run of inline Markdown as a paragraph, when it has any words.
fn flush(line: &mut String, out: &mut Vec<String>) {
    let paragraph = tidy(line);
    if !paragraph.is_empty() {
        out.push(paragraph);
    }
    line.clear();
}

/// Spaces collapsed, ends trimmed, and each `<br>` a hard break.
fn tidy(line: &str) -> String {
    line.split(BREAK)
        .map(|part| part.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("  \n")
}

/// Inline Markdown on one line: for headings and table cells.
fn one_line(line: &str) -> String {
    tidy(&line.replace(BREAK, " "))
}

fn wrap(inner: &str, marker: &str) -> String {
    let words = inner.trim();
    if words.is_empty() {
        return inner.to_string();
    }
    let before = if inner.starts_with(char::is_whitespace) {
        " "
    } else {
        ""
    };
    let after = if inner.ends_with(char::is_whitespace) {
        " "
    } else {
        ""
    };
    format!("{before}{marker}{words}{marker}{after}")
}

fn code_span(text: &str) -> String {
    let words = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if words.is_empty() {
        return String::new();
    }
    if words.contains('`') {
        format!("`` {words} ``")
    } else {
        format!("`{words}`")
    }
}

/// Text with the characters that would start Markdown of their own escaped.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\\' | '*' | '`' | '<') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

fn is_list(block: &str) -> bool {
    block.starts_with("- ")
        || block
            .split_once(". ")
            .is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_reads_as_its_markdown() {
        let page = r##"<!doctype html><html><head><title>Plan</title>
            <style>h1 { color: red }</style></head><body>
            <h1>ACE right-sizing</h1>
            <p>The end state is <strong>0 VPC</strong>, from <em>101</em> today.
               See <a href="https://www.ibm.com/docs">IBM's terms</a> and <a href="#p1">below</a>.</p>
            <ol start="3"><li>Install License Service<li>Get the contract rate</ol>
            <ul><li>prd<ul><li>large1</li></ul></li><li>tst</li></ul>
            <blockquote><p>Charged for all cores.</p></blockquote>
            <p>Run <code>kubectl get ir</code> first.<br>Then wait.</p>
            <hr>
            <script>var x = "<h2>never</h2>";</script>
            <button>Copy link</button><svg><text>chart label</text></svg>
            <p hidden>Not shown</p>
            </body></html>"##;
        assert_eq!(
            html_to_markdown(page),
            "# ACE right-sizing\n\n\
             The end state is **0 VPC**, from *101* today. See [IBM's terms](https://www.ibm.com/docs) and below.\n\n\
             3. Install License Service\n4. Get the contract rate\n\n\
             - prd\n  - large1\n- tst\n\n\
             > Charged for all cores.\n\n\
             Run `kubectl get ir` first.  \nThen wait.\n\n\
             ---\n"
        );
    }

    #[test]
    fn a_table_keeps_its_rows_and_numeric_columns_align_right() {
        let page = r#"<table><caption>Prod runtimes</caption>
            <thead><tr><th>Runtime</th><th class="num">Limit</th></tr></thead>
            <tbody><tr><td>ir-01-bidtools<td class="num">4000m
            <tr><td>a|b</td><td colspan="2">wide</td></tr></tbody></table>"#;
        assert_eq!(
            html_to_markdown(page),
            "Prod runtimes\n\n\
             | Runtime | Limit |  |\n\
             | --- | ---: | --- |\n\
             | ir-01-bidtools | 4000m |  |\n\
             | a\\|b | wide |  |\n"
        );
    }

    #[test]
    fn code_keeps_its_spacing_and_names_its_language() {
        let page = "<pre class=\"language-yaml\">\nspec:\n  limits: { cpu: 500m }\n</pre>\
                    <pre><code>a &lt; b &amp;&amp; ```fence```</code></pre>";
        assert_eq!(
            html_to_markdown(page),
            "```yaml\nspec:\n  limits: { cpu: 500m }\n```\n\n\
             ````\na < b && ```fence```\n````\n"
        );
    }

    #[test]
    fn the_title_heads_a_page_without_an_h1() {
        assert_eq!(
            html_to_markdown("<title>The &amp; Plan</title><h2>Summary</h2><p>Text</p>"),
            "# The & Plan\n\n## Summary\n\nText\n"
        );
    }

    #[test]
    fn elements_set_apart_by_styles_keep_their_words_apart() {
        assert_eq!(
            html_to_markdown(
                "<ul><li><span>Kevin</span><span>Approve the limits.</span></li></ul>\
                 <p><b>Phase 1</b><span>Weeks 1–2</span>, x<sup>2</sup> and <a href=\"/a\">a</a><span>.</span></p>"
            ),
            "- Kevin Approve the limits.\n\n**Phase 1** Weeks 1–2, x2 and [a](/a).\n"
        );
    }

    #[test]
    fn broken_markup_still_reads() {
        assert_eq!(
            html_to_markdown("<p>one<p>two</div></span><div>three < four * 2"),
            "one\n\ntwo\n\nthree \\< four \\* 2\n"
        );
        assert_eq!(html_to_markdown(""), "\n");
        assert_eq!(
            html_to_markdown("<img src=\"data:image/png;base64,AAA\" alt=\"chart\">"),
            "chart\n"
        );
        assert_eq!(
            html_to_markdown(
                "<a href=\"https://x.test/a b\">link</a> <a href=\"https://x.test\"></a>"
            ),
            "[link](<https://x.test/a b>) <https://x.test>\n"
        );
    }
}

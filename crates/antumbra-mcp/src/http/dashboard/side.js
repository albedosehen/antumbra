// The sidebar every page of the dashboard shares, the relics pages among
// them: collapsing it to its icons and back, and saying who is signed in at
// its foot. Whether it is collapsed is kept with the preferences, and put on
// the page before it is drawn by theme.js, whose key and `applyTheme` this
// uses. Loaded deferred, ahead of each page's own script.

function showSide(side) {
  const toggle = document.getElementById("side-toggle");
  if (!toggle) return;
  const collapsed = side === "collapsed";
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.setAttribute("aria-label", collapsed ? "Expand the sidebar" : "Collapse the sidebar");
  toggle.title = collapsed ? "Expand the sidebar ( [ )" : "Collapse the sidebar ( [ )";
}

/** Collapse the sidebar, or expand it, and keep which: the other preferences
 *  stay as they were kept. */
function toggleSide() {
  const root = document.documentElement;
  const side = root.dataset.side === "collapsed" ? "open" : "collapsed";
  if (side === "collapsed") root.dataset.side = "collapsed";
  else delete root.dataset.side;
  try {
    const kept = JSON.parse(localStorage.getItem(THEME_KEY) || "{}") || {};
    localStorage.setItem(THEME_KEY, JSON.stringify({ ...kept, side }));
  } catch {
    // Refused: this load only.
  }
  showSide(side);
}

/** A name without its kind: `user:dana` is dana. */
function bareName(id) {
  return String(id || "").replace(/^[a-z]+:/, "");
}

function initialsOf(name) {
  const parts = bareName(name).split(/[^A-Za-z0-9]+/).filter(Boolean);
  const letters = parts.length > 1 ? parts[0][0] + parts[1][0] : (parts[0] || "?").slice(0, 2);
  return letters.toUpperCase();
}

/** Say who is signed in wherever the page names them, from a token's claims
 *  (for display only), or no one when there are none. */
function showSideUser(who) {
  const user = who && who.user ? bareName(who.user) : who ? "Signed in" : "";
  const workspace = who && who.tenant ? bareName(who.tenant) : "";
  for (const node of document.querySelectorAll("[data-user-name]")) node.textContent = user;
  for (const node of document.querySelectorAll("[data-workspace-name]")) node.textContent = workspace;
  for (const node of document.querySelectorAll("[data-initials]")) node.textContent = user ? initialsOf(user) : "";
}

/** Draw the signed-in member as their profile has them: the name they go
 *  by in place of their user id, and their image in place of their initials.
 *  `image` is a `data:` URL the server made from bytes it checked; it is set
 *  as an image's source and nothing else. Either may be absent. */
function showSideProfile(display, image) {
  if (display) {
    for (const node of document.querySelectorAll("[data-user-name]")) node.textContent = display;
  }
  for (const node of document.querySelectorAll("[data-initials]")) {
    const held = node.querySelector("img");
    if (!image) {
      // No image: the initials, of the name shown.
      if (held) held.remove();
      node.classList.remove("has-image");
      if (display) node.textContent = initialsOf(display);
      continue;
    }
    const picture = held || document.createElement("img");
    picture.alt = "";
    picture.src = image;
    node.textContent = "";
    node.append(picture);
    node.classList.add("has-image");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("side-toggle");
  if (!toggle) return;
  showSide(document.documentElement.dataset.side);
  toggle.addEventListener("click", toggleSide);
  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable;
    const plain = !event.metaKey && !event.ctrlKey && !event.altKey;
    // Only while the sidebar is on the page: not at the sign-in.
    const shown = toggle.offsetParent !== null;
    if (event.key === "[" && plain && !typing && shown) {
      event.preventDefault();
      toggleSide();
    }
  });
});

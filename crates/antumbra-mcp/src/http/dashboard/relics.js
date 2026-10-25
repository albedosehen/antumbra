// The relics gallery (`/relics`) and one relic's viewer (`/relic/<slug>`).
// Everything comes from `POST mcp/call` with the token the dashboard keeps,
// except a public relic, which `GET /relic/<slug>/public` describes to anyone.
// A relic itself is only ever shown in an <iframe> from copal's render origin,
// sandboxed there; nothing a relic holds is parsed by this page. Text is
// written as text, never as markup.
"use strict";

const TOKEN_KEY = "antumbra.dashboard.token";
// Where this page's assets and the tool endpoint are, relative to it: the
// gallery sits beside `/dashboard`, the viewer one level down.
const BASE = document.body.dataset.base || "";
const PAGE = document.body.dataset.page;
// Previews of private relics each need a render link; ask for a few at once.
const PREVIEW_CONCURRENCY = 4;
const REFUSED =
  "The server refused the token: it has expired, been revoked, or was signed with another key.";

const stamps = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

const state = { token: null };

class SignedOut extends Error {}

const byId = (id) => document.getElementById(id);

/** An element whose children are elements or strings; a string becomes a
 *  text node, so nothing passed here is ever parsed as markup. */
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "class") node.className = value;
    else node.setAttribute(name, value === true ? "" : String(value));
  }
  node.append(...children.filter((c) => c !== undefined && c !== null && c !== false));
  return node;
}

/** A small icon drawn with DOM calls: a padlock for private, a globe for
 *  public. Shape repeats what the label beside it says. */
function icon(kind) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute(
    "d",
    kind === "public"
      ? "M8 1.5a6.5 6.5 0 1 0 0 13a6.5 6.5 0 1 0 0-13Zm0 0c-1.8 1.8-2.7 4-2.7 6.5s.9 4.7 2.7 6.5m0-13c1.8 1.8 2.7 4 2.7 6.5s-.9 4.7-2.7 6.5M1.5 8h13"
      : "M4.5 7V5a3.5 3.5 0 0 1 7 0v2M3.5 7h9v7h-9z",
  );
  svg.append(path);
  return svg;
}

function visibilityTag(visibility) {
  return h("span", { class: `visibility visibility-${visibility}` }, icon(visibility), visibility === "public" ? "Public" : "Private");
}

function when(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : stamps.format(date);
}

// The token, as the dashboard keeps it: in this tab's session storage.

function storedToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function keepToken(token) {
  try {
    sessionStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Refused: this load only.
  }
}

function forgetToken() {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // Nothing was kept.
  }
}

/** The token's claims, for display only: the server is what checks them. */
function claims(token) {
  const part = token.split(".")[1];
  if (!part) return null;
  try {
    const base64 = part.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const bytes = Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
}

/** One tool call over the REST shim, as the signed-in identity. */
async function call(tool, args = {}) {
  const res = await fetch(`${BASE}mcp/call`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${state.token}`,
    },
    body: JSON.stringify({ tool, arguments: args }),
    cache: "no-store",
    credentials: "omit",
  });
  if (res.status === 401) throw new SignedOut();
  if (!res.ok) {
    const text = await res.text();
    let message = text || `${res.status} ${res.statusText}`;
    try {
      message = JSON.parse(text).error || message;
    } catch {
      // Not JSON: the text is the message.
    }
    throw new Error(message);
  }
  return res.json();
}

// Signing in and out

function showGate(message) {
  byId("app").hidden = true;
  byId("gate").hidden = false;
  const error = byId("gate-error");
  error.textContent = message || "";
  error.hidden = !message;
  const input = byId("token");
  input.value = "";
  input.focus();
}

function showApp() {
  byId("gate").hidden = true;
  byId("app").hidden = false;
  const who = byId("who");
  if (who) {
    const c = state.token ? claims(state.token) : null;
    who.textContent = c ? [c.sub, c.tenant].filter(Boolean).join(" · ") : "";
  }
}

function signOut(message) {
  state.token = null;
  forgetToken();
  if (PAGE === "gallery") byId("relics").replaceChildren();
  showGate(message);
}

function wireGate(start) {
  byId("gate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const token = byId("token").value.trim();
    if (!token) return;
    state.token = token;
    keepToken(token);
    start();
  });
  const out = byId("signout");
  if (out) out.addEventListener("click", () => signOut());
}

// The gallery

const gallery = {
  ticket: 0,
  queue: [],
  running: 0,
  observer: null,
};

function scope() {
  return document.querySelector('input[name="scope"]:checked')?.value || "all";
}

async function loadRelics() {
  const ticket = ++gallery.ticket;
  gallery.queue = [];
  const list = byId("relics");
  const summary = byId("relics-summary");
  const note = byId("relics-note");
  const query = byId("relic-query").value.trim();
  summary.textContent = "Loading…";
  note.hidden = true;
  let out;
  try {
    out = await call("list_relics", query ? { scope: scope(), query } : { scope: scope() });
  } catch (error) {
    if (error instanceof SignedOut) return signOut(REFUSED);
    if (ticket !== gallery.ticket) return;
    list.replaceChildren();
    summary.textContent = "";
    note.textContent = `Could not list relics: ${error.message}`;
    note.hidden = false;
    return;
  }
  if (ticket !== gallery.ticket) return;
  const relics = out.relics || [];
  summary.textContent = relics.length === 1 ? "1 relic" : `${relics.length} relics`;
  if (out.note) {
    note.textContent = out.note;
    note.hidden = false;
  }
  list.replaceChildren(...relics.map(card));
}

function card(relic) {
  const preview = h("div", { class: "relic-preview" }, h("span", { class: "relic-placeholder" }, relic.title));
  preview.dataset.slug = relic.slug;
  if (relic.preview_url) preview.dataset.src = relic.preview_url;
  const byline = [];
  byline.push(visibilityTag(relic.visibility));
  byline.push(h("span", {}, `Updated ${when(relic.updated_at)}`));
  if (relic.version > 1) byline.push(h("span", {}, `v${relic.version}`));
  if (!relic.yours) byline.push(h("span", { class: "owner" }, relic.owner));
  const item = h(
    "li",
    { class: "relic-card" },
    h(
      "a",
      { class: "relic-link", href: `relic/${encodeURIComponent(relic.slug)}` },
      preview,
      h("span", { class: "relic-title" }, relic.title),
      h("span", { class: "relic-byline" }, ...byline),
    ),
  );
  watch(preview);
  return item;
}

/** Start a card's preview once it nears the screen, so a long gallery does not
 *  load every page at once. */
function watch(preview) {
  if (!gallery.observer) {
    gallery.observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          gallery.observer.unobserve(entry.target);
          enqueue(entry.target);
        }
      },
      { rootMargin: "200px" },
    );
  }
  gallery.observer.observe(preview);
}

function enqueue(preview) {
  gallery.queue.push({ preview, ticket: gallery.ticket });
  drain();
}

async function drain() {
  while (gallery.running < PREVIEW_CONCURRENCY && gallery.queue.length) {
    const { preview, ticket } = gallery.queue.shift();
    gallery.running += 1;
    showPreview(preview, ticket).finally(() => {
      gallery.running -= 1;
      drain();
    });
  }
}

async function showPreview(preview, ticket) {
  let src = preview.dataset.src;
  if (!src) {
    try {
      src = (await call("get_relic", { slug: preview.dataset.slug })).view_url;
    } catch (error) {
      if (error instanceof SignedOut) return signOut(REFUSED);
      return;
    }
  }
  if (ticket !== gallery.ticket || !src) return;
  const frame = h("iframe", {
    src,
    sandbox: "allow-scripts",
    referrerpolicy: "no-referrer",
    loading: "lazy",
    tabindex: "-1",
    "aria-hidden": "true",
    title: "",
  });
  preview.replaceChildren(frame);
}

function startGallery() {
  showApp();
  loadRelics();
}

function wireGallery() {
  byId("relic-search").addEventListener("submit", (event) => {
    event.preventDefault();
    loadRelics();
  });
  for (const input of document.querySelectorAll('input[name="scope"]')) {
    input.addEventListener("change", loadRelics);
  }
  for (const input of document.querySelectorAll('input[name="layout"]')) {
    input.addEventListener("change", () => {
      byId("relics").classList.toggle("relic-list", input.value === "list" && input.checked);
    });
  }
}

// The viewer

function slugOfPage() {
  const last = location.pathname.split("/").filter(Boolean).pop() || "";
  return decodeURIComponent(last);
}

const viewer = { relic: null };

function showRelic(relic, url) {
  showApp();
  document.title = `${relic.title} · Relics`;
  byId("relic-title").textContent = relic.title;
  const meta = [visibilityTag(relic.visibility), h("span", {}, `Updated ${when(relic.updated_at)}`)];
  if (relic.version > 1) meta.push(h("span", {}, `Version ${relic.version}`));
  if (relic.owner && !relic.yours) meta.push(h("span", { class: "owner" }, relic.owner));
  byId("relic-meta").replaceChildren(...meta);
  byId("relic-frame").src = url;
  const raw = byId("open-raw");
  raw.href = url;
  raw.hidden = false;
  const toggle = byId("visibility");
  toggle.hidden = !relic.yours;
  toggle.textContent = relic.visibility === "public" ? "Make private" : "Make public";
  byId("signin").hidden = Boolean(state.token);
  byId("signout").hidden = !state.token;
  byId("viewer-error").hidden = true;
}

async function loadSignedIn(slug) {
  const relic = await call("get_relic", { slug });
  viewer.relic = relic;
  showRelic(relic, relic.view_url);
}

async function loadPublic(slug) {
  const res = await fetch(`${encodeURIComponent(slug)}/public`, {
    cache: "no-store",
    credentials: "omit",
  });
  if (!res.ok) return false;
  const relic = await res.json();
  viewer.relic = null;
  showRelic({ ...relic, visibility: "public", yours: false }, relic.iframe_src);
  return true;
}

async function startViewer() {
  const slug = slugOfPage();
  let refused = "";
  if (state.token) {
    try {
      await loadSignedIn(slug);
      return;
    } catch (error) {
      if (error instanceof SignedOut) {
        state.token = null;
        forgetToken();
        refused = REFUSED;
      } else {
        // Not one the token can see: it may still be public.
        refused = error.message;
      }
    }
  }
  if (await loadPublic(slug)) return;
  showGate(refused);
}

async function toggleVisibility() {
  const relic = viewer.relic;
  if (!relic) return;
  const button = byId("visibility");
  button.disabled = true;
  try {
    await call("set_relic_visibility", {
      slug: relic.slug,
      visibility: relic.visibility === "public" ? "private" : "public",
    });
    await loadSignedIn(relic.slug);
  } catch (error) {
    if (error instanceof SignedOut) return signOut(REFUSED);
    const failed = byId("viewer-error");
    failed.textContent = `Could not change who sees it: ${error.message}`;
    failed.hidden = false;
  } finally {
    button.disabled = false;
  }
}

/** Copy the relic's address. The clipboard API needs a secure context, which
 *  a LAN address over plain HTTP is not; then the address is shown to copy. */
async function copyLink() {
  const address = location.href;
  try {
    await navigator.clipboard.writeText(address);
    const button = byId("copy-link");
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = "Copy link";
    }, 1500);
  } catch {
    const fallback = byId("copy-fallback");
    const input = byId("link-text");
    input.value = address;
    fallback.hidden = false;
    input.focus();
    input.select();
  }
}

function wireViewer() {
  byId("visibility").addEventListener("click", toggleVisibility);
  byId("copy-link").addEventListener("click", copyLink);
  byId("signin").addEventListener("click", () => showGate());
}

// Start

function start() {
  if (PAGE === "viewer") startViewer();
  else startGallery();
}

document.addEventListener("DOMContentLoaded", () => {
  wireGate(start);
  if (PAGE === "viewer") wireViewer();
  else wireGallery();
  state.token = storedToken();
  if (state.token || PAGE === "viewer") start();
  else showGate();
});

// The read-only dashboard (P-2). Everything on the page comes from
// `POST mcp/call` with the token its user pasted in, so the page sees what that
// token's (tenant, user) sees and no more. Stored text is written as text,
// never parsed as markup.
"use strict";

const TOKEN_KEY = "antumbra.dashboard.token";
const PAGE_SIZE = 25;
const DOCUMENTS_SHOWN = 8;
const RECALL_TOP_K = 20;
const STATUS_ORDER = ["active", "dormant", "archived", "deleted"];
const STATS = [
  ["memories", "Memories"],
  ["documents", "Documents"],
  ["experts", "Experts"],
  ["boundaries", "Boundaries"],
  ["compartments", "Compartments"],
];
const REFUSED =
  "The server refused the token: it has expired, been revoked, or was signed with another key.";

const numbers = new Intl.NumberFormat();
const dates = new Intl.DateTimeFormat(undefined, { dateStyle: "medium" });
const stamps = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const clock = new Intl.DateTimeFormat(undefined, { timeStyle: "medium" });

const state = {
  token: null,
  memories: null,
  memory: { view: null, rows: [], shown: 0, more: false, ticket: 0 },
};

/** Thrown when the server answers 401: the token no longer opens anything. */
class SignedOut extends Error {}

/** Thrown when an answer arrives for a request something newer replaced: a
 *  sign-out, or a later memory search. It is dropped, never shown. */
class Superseded extends Error {}

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

// The token: kept in this tab's session storage, which the browser clears
// with the tab. Storage can be refused (a private window); then the token
// lasts this load only.

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
  const token = state.token;
  const res = await fetch("mcp/call", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
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
  const body = await res.json();
  // Signed out (or in as someone else) while this was in flight: what it
  // fetched belongs to no one on the page now.
  if (state.token !== token) throw new Superseded();
  return body;
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

function signOut(message) {
  state.token = null;
  state.memories = null;
  state.memory = { view: null, rows: [], shown: 0, more: false, ticket: state.memory.ticket + 1 };
  forgetToken();
  clearSections();
  showGate(message);
}

/** Empty every section, so nothing one token loaded is left in the page for
 *  the next sign-in to show. */
function clearSections() {
  for (const id of ["stats", "population-rows", "compartments", "documents", "memories"]) {
    byId(id).replaceChildren();
  }
  for (const id of ["who", "expires", "updated", "population-summary", "documents-summary", "memory-summary", "memory-note"]) {
    byId(id).textContent = "";
  }
  byId("memory-note").hidden = true;
  byId("more").hidden = true;
  byId("documents-more").hidden = true;
  byId("browse").textContent = "List all";
  byId("recall").reset();
}

async function open(token) {
  const who = claims(token);
  if (who && typeof who.exp === "number" && who.exp * 1000 <= Date.now()) {
    signOut("That token has expired. Mint a new one.");
    return;
  }
  state.token = token;
  byId("gate").hidden = true;
  byId("app").hidden = false;
  showIdentity(who);
  if (await refresh()) keepToken(token);
}

function showIdentity(who) {
  byId("who").textContent =
    who && who.tenant ? `${who.tenant} · ${who.user || "unknown user"}` : "Signed in";
  const expires = byId("expires");
  if (who && typeof who.exp === "number") {
    const at = new Date(who.exp * 1000);
    expires.textContent = `· token expires ${stamps.format(at)}`;
  } else {
    expires.textContent = "";
  }
}

// Loading

/** Load every section. False when the token was refused. */
async function refresh() {
  const button = byId("refresh");
  button.disabled = true;
  try {
    const loads = [
      section(byId("stats"), loadStats),
      section(byId("population-rows"), loadPopulation),
      section(byId("compartments"), loadCompartments),
      section(byId("documents"), loadDocuments),
    ];
    if (state.memory.view) loads.push(section(byId("memories"), () => runMemory(state.memory.view)));
    const results = await Promise.allSettled(loads);
    const failed = (kind) => results.some((r) => r.status === "rejected" && r.reason instanceof kind);
    if (failed(Superseded)) return false;
    if (failed(SignedOut)) {
      signOut(REFUSED);
      return false;
    }
    byId("updated").textContent = `Updated ${clock.format(new Date())}`;
    return true;
  } finally {
    button.disabled = false;
  }
}

/** Run one section's load; a failure other than a refused token is shown in
 *  the section, so one tool the server does not offer leaves the rest. */
async function section(target, load) {
  try {
    await load();
  } catch (error) {
    if (error instanceof SignedOut || error instanceof Superseded) throw error;
    const failed = h("p", { class: "failed", role: "alert" }, `Could not load: ${error.message}`);
    target.replaceChildren(target.tagName === "TBODY" ? h("tr", {}, h("td", { colspan: 5 }, failed)) : failed);
  }
}

async function loadStats() {
  const stats = await call("workspace_stats");
  byId("stats").replaceChildren(
    ...STATS.map(([key, label]) =>
      h("div", { class: "stat" }, h("dt", {}, label), h("dd", {}, numbers.format(stats[key] ?? 0))),
    ),
  );
  state.memories = stats.memories ?? null;
  byId("browse").textContent =
    state.memories === null ? "List all" : `List all ${numbers.format(state.memories)}`;
}

function statusRank(status) {
  const rank = STATUS_ORDER.indexOf(status);
  return rank < 0 ? STATUS_ORDER.length : rank;
}

async function loadPopulation() {
  const { experts } = await call("population");
  const rows = byId("population-rows");
  const summary = byId("population-summary");
  if (!experts.length) {
    summary.textContent = "";
    rows.replaceChildren(
      h("tr", {}, h("td", { colspan: 5, class: "empty" }, "No experts yet: the population has not been seeded.")),
    );
    return;
  }
  const sorted = [...experts].sort(
    (a, b) =>
      statusRank(a.status) - statusRank(b.status) ||
      b.fitness - a.fitness ||
      a.name.localeCompare(b.name),
  );
  rows.replaceChildren(...sorted.map(expertRow));
  const tally = new Map();
  for (const e of sorted) tally.set(e.status, (tally.get(e.status) || 0) + 1);
  summary.textContent = [...tally].map(([status, n]) => `${numbers.format(n)} ${status}`).join(" · ");
}

function expertRow(e) {
  const fill = h("span");
  const fitness = typeof e.fitness === "number" ? Math.min(1, Math.max(0, e.fitness)) : 0;
  fill.style.setProperty("--value", String(fitness));
  const known = STATUS_ORDER.includes(e.status) ? e.status : "other";
  return h(
    "tr",
    {},
    h("td", {}, h("span", { class: "name" }, e.name), h("span", { class: "id" }, e.id)),
    h("td", { class: "num" }, String(e.generation)),
    h(
      "td",
      { class: "num" },
      h("span", { class: "fitness" }, h("span", { class: "meter", "aria-hidden": "true" }, fill), fixed(e.fitness, 3)),
    ),
    h("td", {}, h("span", { class: `status status-${known}` }, e.status)),
    h("td", {}, e.private ? "Private" : "Shared"),
  );
}

async function loadCompartments() {
  const { compartments } = await call("list_compartments");
  const list = byId("compartments");
  if (!compartments.length) {
    list.replaceChildren(h("li", { class: "empty" }, "None yet."));
    return;
  }
  list.replaceChildren(
    ...compartments.map((c) =>
      h(
        "li",
        {},
        h("div", {}, h("span", { class: "name" }, c.name), h("span", { class: "id" }, c.id)),
        h("span", { class: "muted small" }, c.origin),
      ),
    ),
  );
}

/** The knowledge documents, by title: the first few, and the rest on asking. */
async function loadDocuments() {
  const { documents } = await call("list_documents");
  const list = byId("documents");
  const more = byId("documents-more");
  byId("documents-summary").textContent = documents.length
    ? `${numbers.format(documents.length)} ingested`
    : "";
  more.hidden = true;
  if (!documents.length) {
    list.replaceChildren(h("li", { class: "empty" }, "None yet."));
    return;
  }
  const row = (d) =>
    h(
      "li",
      {},
      h("span", { class: "name" }, d.title),
      h(
        "span",
        { class: "muted small nowrap" },
        `${numbers.format(d.chunks)} ${d.chunks === 1 ? "chunk" : "chunks"}${d.archived ? " · archived" : ""}`,
      ),
    );
  list.replaceChildren(...documents.slice(0, DOCUMENTS_SHOWN).map(row));
  const rest = documents.slice(DOCUMENTS_SHOWN);
  if (rest.length) {
    more.textContent = `Show ${numbers.format(rest.length)} more`;
    more.hidden = false;
    more.onclick = () => {
      list.append(...rest.map(row));
      more.hidden = true;
    };
  }
}

// Memory: recalled by meaning, or listed most recently updated first. A list
// comes from the server a page at a time, so a large store costs one page per
// "Show more"; a recall is one answer, shown a page at a time.

function memoryForm() {
  const form = byId("recall");
  return { query: form.elements.query.value.trim(), network: form.elements.network.value };
}

async function runMemory(view) {
  const ticket = ++state.memory.ticket;
  state.memory.view = view;
  const summary = byId("memory-summary");
  const note = byId("memory-note");
  const list = byId("memories");
  summary.textContent = "Loading";
  const filter = view.network ? { network: view.network } : {};
  const where = view.network ? ` in ${view.network}` : "";
  const request =
    view.mode === "recall"
      ? call("recall_memories", { ...filter, query: view.query, top_k: RECALL_TOP_K })
      : call("list_memories", { ...filter, limit: PAGE_SIZE });
  // A later search started while this one was in flight; it owns the list,
  // whether this one answered or failed.
  const out = await request.finally(() => {
    if (ticket !== state.memory.ticket) throw new Superseded();
  });
  const rows = out.memories;
  note.hidden = true;
  byId("more").hidden = true;
  if (view.mode === "recall") {
    summary.textContent = `${numbers.format(rows.length)} recalled${where}, nearest first`;
    if (out.nothing_cleared_the_floor) {
      note.textContent = "Nothing recalled cleared the relevance floor: no stored memory answers this.";
      note.hidden = false;
    }
  }
  state.memory.rows = rows;
  state.memory.more = Boolean(out.more);
  state.memory.shown = 0;
  list.replaceChildren();
  if (!rows.length && note.hidden) {
    list.replaceChildren(h("li", { class: "empty" }, "Nothing here."));
  }
  showMore();
}

/** Show the next page: the rest of what is loaded, or, in a list with every
 *  loaded row shown, the server's next page. */
async function showMore() {
  const memory = state.memory;
  const more = byId("more");
  if (memory.shown >= memory.rows.length && memory.more) {
    const ticket = memory.ticket;
    const filter = memory.view.network ? { network: memory.view.network } : {};
    more.disabled = true;
    try {
      const out = await call("list_memories", { ...filter, limit: PAGE_SIZE, offset: memory.rows.length });
      // A newer search owns the list now.
      if (ticket !== state.memory.ticket) return;
      memory.rows.push(...out.memories);
      memory.more = Boolean(out.more);
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
      if (!(error instanceof SignedOut || error instanceof Superseded)) {
        const note = byId("memory-note");
        note.textContent = `Could not load more: ${error.message}`;
        note.hidden = false;
      }
      return;
    } finally {
      more.disabled = false;
    }
  }
  const next = memory.rows.slice(memory.shown, memory.shown + PAGE_SIZE);
  byId("memories").append(...next.map(memoryItem));
  memory.shown += next.length;
  const left = memory.rows.length - memory.shown;
  more.hidden = left <= 0 && !memory.more;
  more.textContent =
    left > 0 ? `Show ${numbers.format(Math.min(PAGE_SIZE, left))} more of ${numbers.format(left)}` : "Show more";
  if (memory.view && memory.view.mode === "list") byId("memory-summary").textContent = listSummary();
  // Only a memory long enough to be clamped gets a toggle, which takes layout
  // to know.
  requestAnimationFrame(offerToggles);
}

/** How much of a list is showing: of how many, when the count is known. */
function listSummary() {
  const { view, shown, more } = state.memory;
  const total = !view.network && state.memories !== null ? state.memories : null;
  const where = view.network ? ` in ${view.network}` : "";
  const count =
    total !== null
      ? `${numbers.format(shown)} of ${numbers.format(total)}`
      : `${numbers.format(shown)}${more ? " so far" : ""}${where}`;
  return `${count}, most recently updated first`;
}

function offerToggles() {
  for (const item of byId("memories").querySelectorAll(".memory:not([data-measured])")) {
    item.dataset.measured = "";
    const content = item.querySelector(".content");
    if (content.scrollHeight > content.clientHeight + 1) item.querySelector(".toggle").hidden = false;
  }
}

function memoryItem(m) {
  const toggle = h("button", { type: "button", class: "link toggle", "aria-expanded": "false", hidden: true }, "Show all");
  const [relate, relations] = relationsOf(m);
  const item = h(
    "li",
    { class: "memory" },
    h("p", { class: "content" }, m.content),
    memoryMeta(m),
    h("div", { class: "actions-row" }, toggle, relate),
    relations,
  );
  toggle.addEventListener("click", () => {
    const open = item.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? "Show less" : "Show all";
  });
  return item;
}

/** A number to `digits` places, or "unknown" when the value is not one. */
function fixed(value, digits) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "unknown";
}

/** A date, with the full time on hover. */
function when(iso) {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  return h("time", { datetime: iso, title: stamps.format(at) }, dates.format(at));
}

function anchor(p) {
  const text = `${p.repo} @ ${p.commit.slice(0, 7)}${p.branch ? ` on ${p.branch}` : ""}${p.path ? ` · ${p.path}` : ""}`;
  return h("span", { class: "mono", title: p.commit }, text);
}

function memoryMeta(m) {
  return h(
    "div",
    { class: "meta" },
    h("span", { class: "tag" }, m.network),
    h("span", {}, `confidence ${fixed(m.confidence, 2)}`),
    m.reinforcement > 0 && h("span", {}, `reinforced ${numbers.format(m.reinforcement)}×`),
    h("span", {}, "updated ", when(m.updated_at)),
    m.provenance && anchor(m.provenance),
    m.truncated && h("span", {}, `the start of ${numbers.format(m.content_chars)} characters`),
    m.orphaned_at && h("span", { class: "flag" }, "orphaned ", when(m.orphaned_at)),
    h("span", { class: "id" }, m.id),
  );
}

/** A memory's relations to others (the `memory_edge` graph, one step out):
 *  a toggle, and the panel it opens, fetched the first time it opens. */
function relationsOf(m) {
  const button = h("button", { type: "button", class: "link", "aria-expanded": "false" }, "Relations");
  const panel = h("div", { class: "relations", hidden: true });
  let loaded = false;
  button.addEventListener("click", async () => {
    const open = panel.hidden;
    panel.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    if (!open || loaded) return;
    loaded = true;
    panel.replaceChildren(h("p", { class: "empty" }, "Loading"));
    try {
      const { neighbors } = await call("get_neighbors", { memory_id: m.id });
      panel.replaceChildren(
        neighbors.length
          ? h("ul", { class: "relation-list" }, ...neighbors.map(relationRow))
          : h("p", { class: "empty" }, "No relations from this memory."),
      );
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
      if (error instanceof SignedOut || error instanceof Superseded) return;
      loaded = false;
      panel.replaceChildren(h("p", { class: "failed", role: "alert" }, `Could not load: ${error.message}`));
    }
  });
  return [button, panel];
}

/** One relation: its type and weight, then the memory it leads to. */
function relationRow(n) {
  return h(
    "li",
    {},
    h(
      "div",
      { class: "meta" },
      h("span", { class: "tag" }, n.edge_type),
      h("span", {}, `weight ${fixed(n.weight, 2)}`),
      h("span", {}, n.memory.network),
      h("span", { class: "id" }, n.memory.id),
    ),
    h("p", { class: "relation-content" }, n.memory.content),
  );
}

/** Run a memory load from a control, showing a failure in the section. */
async function memoryAction(view) {
  try {
    await section(byId("memories"), () => runMemory(view));
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
  }
}

function start() {
  byId("gate-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const token = byId("token").value.trim().replace(/^Bearer\s+/i, "");
    if (token) open(token);
  });
  byId("signout").addEventListener("click", () => signOut(""));
  byId("refresh").addEventListener("click", () => refresh());
  byId("recall").addEventListener("submit", (event) => {
    event.preventDefault();
    const { query, network } = memoryForm();
    if (!query) {
      byId("query").focus();
      return;
    }
    memoryAction({ mode: "recall", query, network });
  });
  byId("browse").addEventListener("click", () => {
    memoryAction({ mode: "list", network: memoryForm().network });
  });
  // A different network re-runs what is showing, under the new filter.
  byId("recall").addEventListener("change", (event) => {
    if (event.target.name === "network" && state.memory.view) {
      memoryAction({ ...state.memory.view, network: event.target.value });
    }
  });
  byId("more").addEventListener("click", showMore);

  const token = storedToken();
  if (token) open(token);
  else showGate("");
}

start();

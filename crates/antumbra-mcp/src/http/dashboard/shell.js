// The dashboard's shell: which page is showing, the signed-in user's own
// pages (Account, Devices and handoffs), the workspace's policy, and what the
// Overview says is waiting. Loaded after app.js and behaviors.js and sharing
// their helpers (h, byId, call, section, claims, signOut, numbers, stamps),
// side.js's (showSide, showSideUser, bareName) and
// their errors (SignedOut, Superseded).
//
// Stored text is written with `textContent` through `h`, never as markup.

const VIEWS = {
  overview: "Overview",
  memory: "Memory",
  behaviors: "Behaviors",
  experts: "Experts",
  dependencies: "Dependencies",
  knowledge: "Knowledge",
  account: "Account",
  agent: "Agent",
  share: "Share",
  devices: "Devices and handoffs",
  members: "Members",
  settings: "Policy",
};

/** Where the preferences are kept: the key theme.js reads the theme from. */
const PREFERENCES_KEY = THEME_KEY;
const PREFERENCES = {
  theme: ["system", "light", "dark"],
  density: ["comfortable", "compact"],
  home: ["overview", "memory", "behaviors"],
  side: ["open", "collapsed"],
};
/** A token with fewer days left than this is said to be expiring. */
const EXPIRING_DAYS = 30;
const RECENT_SHOWN = 5;
const DAY = 24 * 60 * 60 * 1000;

/** The workspace's policy, in the groups a person thinks of it in: each knob's
 *  key in the snapshot, its name, how its value reads, and what it does. */
const POLICY = [
  [
    "How long memory lasts",
    "A memory nobody recalls or reinforces fades; this is how long each kind takes to lose half its strength.",
    [
      ["world_half_life_days", "Facts (world)", days, "days"],
      ["bank_half_life_days", "Experiences (bank)", days, "days"],
      ["opinion_half_life_days", "Judgments (opinion)", days, "days"],
      ["volatile_half_life_days", "Marked volatile", days, "days"],
    ],
  ],
  [
    "How memories are linked and counted",
    "What the server does by itself as memories are stored and recalled.",
    [
      ["auto", "Automatic upkeep", onOff],
      ["link_threshold", "Similarity that links two memories", ratio],
      ["max_links", "Links made for one memory, at most", whole],
      ["recurrence_threshold", "Similarity that counts as the same thing said again", ratio],
      ["use_threshold", "Similarity that counts a recalled memory as used", ratio],
      ["strength_weight", "How much strength counts in ranking", ratio],
      ["pass_over_weight", "What a pass-over weighs, next to one use", ratio],
    ],
  ],
  [
    "How behaviors are taught",
    "What happens to a behavior between being proposed and an expert learning it.",
    [
      ["accept_after_hours", "A proposed behavior is accepted after", hoursOrNever, "hours"],
      ["behaviors_per_expert", "Behaviors one expert is taught, at most", whole],
      ["standing_training", "Training of standing experts", onOff],
    ],
  ],
];

function days(value) {
  return `${numbers.format(value)} days`;
}

function ratio(value) {
  return Number(value).toFixed(2);
}

function whole(value) {
  return numbers.format(Math.round(value));
}

function onOff(value) {
  return value ? "On" : "Paused";
}

function hoursOrNever(value) {
  return value > 0 ? `${numbers.format(value)} hours` : "Never: you accept each one";
}

// Pages

/** The page the address names, or the one the user opens on. */
function viewOf(hash) {
  const name = hash.replace(/^#\/?/, "");
  return name in VIEWS ? name : null;
}

function showView(name) {
  for (const view of document.querySelectorAll("[data-view]")) view.hidden = view.dataset.view !== name;
  for (const link of document.querySelectorAll("[data-nav]")) {
    if (link.dataset.nav === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  byId("crumb-page").textContent = VIEWS[name];
  document.title = `${VIEWS[name]} · Antumbra`;
  closeMenu();
}

function route() {
  const name = viewOf(location.hash) || preferences().home;
  showView(name);
  window.scrollTo(0, 0);
}

// The user menu

function closeMenu() {
  byId("user-menu").hidden = true;
  byId("user-button").setAttribute("aria-expanded", "false");
}

function toggleMenu() {
  const menu = byId("user-menu");
  menu.hidden = !menu.hidden;
  byId("user-button").setAttribute("aria-expanded", String(!menu.hidden));
}

// Preferences: how the dashboard looks for this browser's user. Nothing of
// the workspace's is kept here, and a browser that refuses storage keeps
// them for this load only.

function preferences() {
  let kept = {};
  try {
    kept = JSON.parse(localStorage.getItem(PREFERENCES_KEY) || "{}") || {};
  } catch {
    // Refused, or not what was written: the defaults.
  }
  const out = {};
  for (const [key, allowed] of Object.entries(PREFERENCES)) {
    out[key] = allowed.includes(kept[key]) ? kept[key] : allowed[0];
  }
  return out;
}

function applyPreferences(chosen) {
  // theme.js put the kept theme on the page before it was first drawn; this
  // is the same call, for a change made while the page is open.
  applyTheme(chosen.theme, chosen.density, chosen.side);
  showSide(chosen.side);
  for (const [key, value] of Object.entries(chosen)) {
    const input = byId("preferences").querySelector(`input[name=${key}][value=${value}]`);
    if (input) input.checked = true;
  }
  const inMenu = byId("menu-theme").querySelector(`input[value=${chosen.theme}]`);
  if (inMenu) inMenu.checked = true;
}

function keepPreferences() {
  const form = new FormData(byId("preferences"));
  // The sidebar's width is not in the form: it is as the page has it.
  const side = document.documentElement.dataset.side === "collapsed" ? "collapsed" : "open";
  const chosen = Object.fromEntries(Object.keys(PREFERENCES).map((key) => [key, key === "side" ? side : form.get(key)]));
  try {
    localStorage.setItem(PREFERENCES_KEY, JSON.stringify(chosen));
  } catch {
    // Refused: this load only.
  }
  applyPreferences(chosen);
}

// Who is signed in

function fact(name, value, mono) {
  return h("div", { class: "fact" }, h("dt", {}, name), h("dd", mono ? { class: "mono" } : {}, value));
}

/** The signed-in identity, wherever the page says it: the bar, the menu, the
 *  sidebar and the Account page. From the token's claims, for display only. */
function showAccount(who) {
  const workspace = who && who.tenant ? bareName(who.tenant) : "";
  // The sidebar's foot, the menu and the Account page's card (side.js).
  showSideUser(who || {});
  byId("who").textContent = workspace || "Workspace";

  const operator = Boolean(who && who.operator);
  byId("role").replaceChildren(
    h("span", { class: "tag" }, operator ? "operator" : "member"),
    h("span", { class: "small muted" }, operator
      ? "May read the server's own state and act on it from the console."
      : "Reads and writes this workspace as you."),
  );
  byId("identity").replaceChildren(
    fact("User", who && who.user ? who.user : "unknown", true),
    fact("Workspace", who && who.tenant ? who.tenant : "unknown", true),
    fact("Server", location.host, true),
  );

  const issued = who && typeof who.iat === "number" ? new Date(who.iat * 1000) : null;
  const expires = who && typeof who.exp === "number" ? new Date(who.exp * 1000) : null;
  byId("token-facts").replaceChildren(
    fact("Issued", issued ? stamps.format(issued) : "not said"),
    fact("Expires", expires ? stamps.format(expires) : "never"),
    fact("Token id", who && who.jti ? who.jti : "not said", true),
  );
  const left = expires ? Math.max(0, Math.ceil((expires - Date.now()) / DAY)) : null;
  const bar = byId("token-bar");
  bar.hidden = !(issued && expires);
  if (issued && expires) {
    const spent = Math.min(1, Math.max(0, (Date.now() - issued) / (expires - issued)));
    byId("token-spent").style.width = `${(spent * 100).toFixed(1)}%`;
    bar.classList.toggle("ending", left <= EXPIRING_DAYS);
    bar.setAttribute("aria-label", `${Math.round(spent * 100)}% of this token's life has passed`);
  }
  byId("token-left").textContent =
    left === null ? "This token does not expire." : left === 0 ? "Expires today." : `${numbers.format(left)} ${left === 1 ? "day" : "days"} left.`;
  state.shell.tokenDaysLeft = left;
}

// Devices and handoffs

function ago(iso) {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "unknown";
  const minutes = Math.max(0, Math.round((Date.now() - at) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 48 * 60) return `${Math.round(minutes / 60)} h ago`;
  return stamps.format(at);
}

async function loadDevices() {
  const out = await call("devices");
  const devices = out.devices ?? [];
  state.shell.devices = devices;
  offerDevices();
  byId("devices-summary").textContent = devices.length
    ? `${numbers.format(devices.length)} registered`
    : "";
  const rows = devices.map((d) =>
    h(
      "tr",
      {},
      h("th", { scope: "row", class: "mono" }, d.host),
      h("td", {}, d.role === "memory" ? "Memory" : d.role === "train" ? "Trains experts" : d.role || "unknown"),
      h("td", { class: "mono" }, d.backend || "unknown"),
      h("td", { title: d.last_seen || "" }, d.last_seen ? ago(d.last_seen) : "never"),
    ),
  );
  byId("devices").replaceChildren(
    ...(rows.length ? rows : [h("tr", {}, h("td", { colspan: 4, class: "empty" }, "No device has registered yet: an agent's session registers the machine it runs on."))]),
  );
  byId("devices-brief").replaceChildren(
    ...(devices.length
      ? devices.map((d) =>
          h(
            "li",
            { class: "brief" },
            h("span", { class: "mono" }, d.host),
            h("span", { class: "small muted", title: d.last_seen || "" }, d.last_seen ? `seen ${ago(d.last_seen)}` : "never seen"),
          ),
        )
      : [h("li", { class: "empty" }, "No device has registered yet.")]),
  );
  await loadHandoffs(devices);
}

/** The handoffs waiting for any of the user's devices, each listed once. */
async function loadHandoffs(devices) {
  const hosts = devices.map((d) => d.host);
  const lists = await Promise.all(
    (hosts.length ? hosts : [undefined]).map((host) => call("handoffs", host ? { host, full: true } : { full: true })),
  );
  const seen = new Map();
  for (const out of lists) for (const note of out.handoffs ?? []) seen.set(note.id, note);
  const notes = [...seen.values()].sort((a, b) => String(b.left_at).localeCompare(String(a.left_at)));
  state.shell.handoffs = notes.length;
  byId("handoffs-summary").textContent = notes.length ? `${numbers.format(notes.length)} waiting` : "";
  byId("handoffs").replaceChildren(
    ...(notes.length ? notes.map(handoffItem) : [h("li", { class: "empty" }, "Nothing is waiting: a session leaves a handoff for another machine with leave_handoff.")]),
  );
  showCounts();
}

function handoffItem(note) {
  const [title, ...rest] = String(note.content || note.title || "").split("\n");
  const body = rest.join("\n").trim();
  const done = h("button", { type: "button", class: "quiet" }, "Mark done");
  const item = h(
    "li",
    { class: "handoff" },
    h("p", { class: "handoff-title" }, title || "Untitled"),
    body ? rich(body) : null,
    h(
      "p",
      { class: "meta small muted" },
      h("span", {}, `from ${note.from_host || "unknown"}`),
      h("span", {}, `for ${note.for_host || "any device"}`),
      h("span", { title: note.left_at || "" }, note.left_at ? ago(note.left_at) : ""),
      h("span", { class: "mono" }, note.id),
    ),
    h("div", { class: "actions-row" }, done),
  );
  done.addEventListener("click", async () => {
    done.disabled = true;
    try {
      await call("complete_handoff", { handoff_id: note.id });
      await section(byId("handoffs"), () => loadHandoffs(state.shell.devices));
      renderAttention();
    } catch (error) {
      done.disabled = false;
      if (error instanceof SignedOut) signOut(REFUSED);
      else if (!(error instanceof Superseded)) done.textContent = "Could not mark it done";
    }
  });
  return item;
}

// The workspace's policy: read from what the console reads, and for an
// operator a form over it, saved through the console's own action. The
// server names each knob's range and keeps what it stores inside it.

/** `path` on this server as the signed-in identity, with the dashboard's
 *  errors: the answer's JSON, or nothing for an answer with no body. */
async function consoleFetch(path, body) {
  const token = state.token;
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body
      ? { authorization: `Bearer ${token}`, "content-type": "application/json" }
      : { authorization: `Bearer ${token}` },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
    credentials: "omit",
  });
  if (res.status === 401) throw new SignedOut();
  const text = await res.text();
  if (!res.ok) {
    let message = text || `${res.status} ${res.statusText}`;
    try {
      message = JSON.parse(text).error || message;
    } catch {
      // Not JSON: the text is the message.
    }
    throw new Error(message);
  }
  if (state.token !== token) throw new Superseded();
  return text ? JSON.parse(text) : null;
}

/** A knob that is on or off, by its range. */
function isSwitch(knob) {
  return knob && knob.min === 0 && knob.max === 1 && knob.step === 1;
}

/** A knob's value as a number, whatever the policy holds it as. */
function knobNumber(value) {
  return typeof value === "boolean" ? Number(value) : Number(value);
}

function sentence(text) {
  return text ? text[0].toUpperCase() + text.slice(1) + "." : "";
}

async function loadPolicy() {
  const [snapshot, named] = await Promise.all([consoleFetch("console/snapshot"), consoleFetch("console/knobs")]);
  // The server's numbers are single precision: 0.1 arrives as 0.100000001.
  const exact = (value) => Number(Number(value).toFixed(4));
  const knobs = new Map(
    (named.knobs ?? []).map((k) => [k.key, { ...k, min: exact(k.min), max: exact(k.max), step: exact(k.step) }]),
  );
  state.shell.policy = { stored: snapshot.policy || {}, knobs, operator: Boolean(snapshot.operator) };
  drawPolicy("");
}

/** Draw the policy from what was last read: a form for an operator, the
 *  values for anyone else. `said` is what the last save came to. */
function drawPolicy(said) {
  const { stored, knobs, operator } = state.shell.policy;
  byId("policy-summary").textContent = operator
    ? "You are an operator: a change applies at the next recall."
    : "Read only: an operator's token changes these.";
  const groups = POLICY.map(([name, about, rows]) =>
    h(
      "section",
      { class: "panel policy-group" },
      h("div", { class: "policy-head" }, h("h4", {}, name), h("p", { class: "muted small" }, about)),
      h(
        "div",
        { class: "knobs" },
        ...rows.filter(([key]) => key in stored).map((row) => knobRow(row, stored, knobs.get(row[0]), operator)),
      ),
    ),
  );
  const status = h("p", { id: "policy-status", class: "small muted", role: "status" }, said);
  const save = h("button", { type: "submit", disabled: true }, "Save changes");
  const discard = h("button", { type: "button", class: "quiet", disabled: true }, "Discard");
  const form = h(
    "form",
    { id: "policy-form", class: "policy", novalidate: "" },
    ...groups,
    operator ? h("div", { class: "save-bar" }, status, h("div", { class: "save-actions" }, discard, save)) : null,
  );
  byId("policy").replaceChildren(form);
  if (!operator) return;

  const changed = () => policyChanges(form);
  const settle = () => {
    const changes = changed();
    const valid = form.checkValidity();
    save.disabled = !changes.length || !valid;
    discard.disabled = !changes.length;
    for (const row of form.querySelectorAll(".knob")) row.classList.toggle("changed", changes.includes(row.dataset.key));
    status.textContent = !valid
      ? "A value is outside its range."
      : changes.length
        ? `${numbers.format(changes.length)} ${changes.length === 1 ? "change" : "changes"} not saved`
        : said;
  };
  form.addEventListener("input", settle);
  discard.addEventListener("click", () => drawPolicy(""));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!changed().length || !form.checkValidity()) return;
    save.disabled = true;
    discard.disabled = true;
    status.textContent = "Saving";
    try {
      await consoleFetch("console/action", { action: "set_memory_policy", policy: policyOf(form) });
      const snapshot = await consoleFetch("console/snapshot");
      state.shell.policy.stored = snapshot.policy || {};
      drawPolicy(`Saved at ${clock.format(new Date())}.`);
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
      else if (!(error instanceof Superseded)) {
        status.textContent = `Not saved: ${error.message}`;
        save.disabled = false;
        discard.disabled = false;
      }
    }
  });
}

function knobRow([key, label, says, unit], stored, knob, operator) {
  const help = h("p", { class: "muted small" }, sentence(knob ? knob.help : ""));
  const text = h("div", { class: "knob-text" }, h("p", { class: "knob-name", id: `knob-${key}` }, label), help);
  let control;
  if (!operator || !knob) {
    control = h("p", { class: "knob-value" }, says(stored[key]));
  } else if (isSwitch(knob)) {
    const on = knobNumber(stored[key]) >= 0.5;
    const radio = (value, name, checked) => {
      const id = `knob-${key}-${value}`;
      return [h("input", { type: "radio", id, name: key, value: String(value), checked: checked ? "" : null }), h("label", { for: id }, name)];
    };
    control = h("fieldset", { class: "segmented", "aria-labelledby": `knob-${key}` }, ...radio(1, "On", on), ...radio(0, key === "auto" ? "Off" : "Paused", !on));
  } else {
    const input = h("input", {
      type: "number",
      name: key,
      min: String(knob.min),
      max: String(knob.max),
      step: knob.step >= 1 ? "1" : "0.01",
      value: String(Number(Number(stored[key]).toFixed(2))),
      required: "",
      inputmode: "decimal",
      "aria-labelledby": `knob-${key}`,
    });
    control = h(
      "div",
      { class: "knob-input" },
      // The unit's room is the same on every row, named or not, so the
      // fields line up down the page.
      h("div", { class: "knob-field" }, input, h("span", { class: "unit small muted" }, unit || "")),
      h("p", { class: "small muted" }, `${numbers.format(knob.min)} to ${numbers.format(knob.max)}`),
    );
  }
  return h("div", { class: "knob", "data-key": key }, text, control);
}

/** The form's value for `key`, as a number. */
function knobValue(form, key) {
  const field = form.elements[key];
  if (!field) return null;
  return Number(field.value);
}

/** The knobs whose value in the form is not the stored one. */
function policyChanges(form) {
  const { stored } = state.shell.policy;
  return Object.keys(stored).filter((key) => {
    const value = knobValue(form, key);
    return value !== null && Math.abs(value - Number(Number(knobNumber(stored[key])).toFixed(2))) > 1e-9;
  });
}

/** The whole policy as the form has it, each value in the kind the stored
 *  one is: what the action replaces the workspace's knobs with. */
function policyOf(form) {
  const { stored } = state.shell.policy;
  const policy = {};
  for (const [key, was] of Object.entries(stored)) {
    const value = knobValue(form, key);
    if (value === null || !policyChanges(form).includes(key)) policy[key] = was;
    else policy[key] = typeof was === "boolean" ? value >= 0.5 : Number.isInteger(was) && key === "max_links" ? Math.round(value) : value;
  }
  return policy;
}

// The Overview: what is waiting, and the latest memories.

async function loadRecent() {
  const out = await call("list_memories", { limit: RECENT_SHOWN });
  const rows = out.memories ?? [];
  byId("recent").replaceChildren(
    ...(rows.length
      ? rows.map((m) =>
          h(
            "li",
            { class: "recent" },
            h("p", { class: "recent-text" }, plain(m.content)),
            h(
              "p",
              { class: "meta small muted" },
              h("span", { class: "tag" }, m.network || "memory"),
              m.created_at ? h("span", { title: m.created_at }, ago(m.created_at)) : null,
            ),
          ),
        )
      : [h("li", { class: "empty" }, "No memories yet: an agent stores one with store_memory.")]),
  );
}

/** Proposed behaviors, counted from what the Behaviors page drew. */
function proposedCount() {
  const text = byId("behaviors-summary").textContent;
  const found = /([\d,.\s]+) proposed/.exec(text);
  return found ? Number(found[1].replace(/[^\d]/g, "")) : 0;
}

function showCounts() {
  const badge = (id, count) => {
    const node = byId(id);
    node.hidden = !count;
    node.textContent = count ? numbers.format(count) : "";
  };
  badge("nav-proposed", proposedCount());
  badge("nav-handoffs", state.shell.handoffs);
  badge("nav-asking", state.shell.asking || 0);
}

function attentionItem(text, detail, href) {
  return h("li", {}, h("a", { class: "attention", href }, h("span", { class: "attention-text" }, text), h("span", { class: "small muted" }, detail)));
}

/** What needs the user: drawn from what the other pages loaded. */
function renderAttention() {
  showCounts();
  const items = [];
  const proposed = proposedCount();
  if (proposed) {
    items.push(attentionItem(`${numbers.format(proposed)} proposed ${proposed === 1 ? "behavior" : "behaviors"}`, "Accept or retire", "#/behaviors"));
  }
  const handoffs = state.shell.handoffs;
  if (handoffs) {
    items.push(attentionItem(`${numbers.format(handoffs)} ${handoffs === 1 ? "handoff" : "handoffs"} waiting`, "Read and mark done", "#/devices"));
  }
  const offered = state.shell.offered || 0;
  const rulesPool = state.shell.pool && ["owner", "admin"].includes(state.shell.pool.your_role);
  if (offered && rulesPool) {
    items.push(attentionItem(`${numbers.format(offered)} offered to the workspace`, "Accept or decline", "#/knowledge"));
  }
  const asking = state.shell.asking || 0;
  if (asking) {
    items.push(attentionItem(`${numbers.format(asking)} asking to join`, "Approve or refuse", "#/members"));
  }
  const left = state.shell.tokenDaysLeft;
  if (left !== null && left <= EXPIRING_DAYS) {
    items.push(attentionItem(left === 0 ? "Your token expires today" : `Your token expires in ${left} ${left === 1 ? "day" : "days"}`, "Mint a new one", "#/account"));
  }
  byId("attention").replaceChildren(...(items.length ? items : [h("li", { class: "empty" }, "Nothing is waiting for you.")]));
}

/** The shell's own loads, run beside the sections' by `refresh`. */
function shellLoads() {
  return [
    section(byId("devices"), loadDevices),
    section(byId("members"), loadMembers),
    section(byId("pool-offers"), loadPool),
    // The member's name and image (manage.js), drawn after the identity the
    // token gave, which it then stands in for. A server from before profiles
    // does not have the tool: the initials stay.
    loadProfile().catch((error) => {
      if (error instanceof SignedOut || error instanceof Superseded) throw error;
    }),
    section(byId("policy"), loadPolicy),
    section(byId("recent"), loadRecent),
  ];
}

/** Empty what the shell drew, at sign-out. */
function clearShell() {
  for (const id of ["devices", "devices-brief", "handoffs", "policy", "recent", "attention", "identity", "token-facts", "role"]) {
    byId(id).replaceChildren();
  }
  for (const id of ["devices-summary", "handoffs-summary", "policy-summary", "token-left"]) byId(id).textContent = "";
  for (const node of document.querySelectorAll("[data-user-name], [data-workspace-name], [data-initials]")) node.textContent = "";
  state.shell = { devices: [], handoffs: 0, tokenDaysLeft: null, policy: null };
  showCounts();
  closeMenu();
}

function startShell() {
  state.shell = { devices: [], handoffs: 0, tokenDaysLeft: null, policy: null };
  applyPreferences(preferences());
  byId("preferences").addEventListener("change", keepPreferences);
  // The menu's theme is the Account page's, a click nearer.
  byId("menu-theme").addEventListener("change", (event) => {
    const same = byId("preferences").querySelector(`input[name=theme][value=${event.target.value}]`);
    if (!same) return;
    same.checked = true;
    keepPreferences();
  });
  // Changed in another tab (theme.js has already put it on the page).
  document.addEventListener("antumbra:preferences", () => applyPreferences(preferences()));
  window.addEventListener("hashchange", route);
  route();

  byId("user-button").addEventListener("click", (event) => {
    event.stopPropagation();
    toggleMenu();
  });
  document.addEventListener("click", (event) => {
    if (!byId("user-menu").hidden && !event.target.closest(".menu")) closeMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenu();
    // "/" goes to the search, unless something is being typed into.
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName) || event.target.isContentEditable;
    if (event.key === "/" && !typing && !byId("app").hidden) {
      event.preventDefault();
      byId("jump-query").focus();
    }
  });
  byId("signout-here").addEventListener("click", () => signOut(""));
  // The bar's search is the Memory page's recall, from any page.
  byId("jump").addEventListener("submit", (event) => {
    event.preventDefault();
    const query = byId("jump-query").value.trim();
    if (!query) return;
    byId("query").value = query;
    location.hash = "#/memory";
    memoryAction({ mode: "recall", query, network: memoryForm().network });
    byId("jump-query").value = "";
  });
}

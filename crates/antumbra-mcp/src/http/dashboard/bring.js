// The Share page: choosing which of your own memories go to a workspace you
// joined. It is built for a store of thousands. Every narrowing is done by
// the server (`bring_candidates`), which also counts what each way of
// narrowing would leave, and a whole filter's worth is decided in one call
// (`bring_choose`), so nothing here depends on how many rows are on the page.
//
// Loaded after app.js, shell.js and manage.js and sharing their helpers (h,
// byId, call, section, attempt, signOut, numbers, dates) and their errors.
// A memory's text is written with `textContent` through `h`, never as markup.

const BRING_PAGE = 50;
/** The ways of narrowing, each a select the server's counts fill. */
const BRING_FACETS = [
  ["network", "Any kind", { world: "Facts", bank: "Experiences", opinion: "Judgments" }],
  ["repo", "Any repository", { none: "No repository" }],
  ["compartment", "Any compartment", { none: "No compartment" }],
  ["host", "Any machine", { none: "Not said" }],
  ["month", "Any month", {}],
];
/** What each state offers to do with everything that matches. */
const BRING_BULK = {
  undecided: [["chosen", "Choose all"], ["declined", "Leave all out"]],
  chosen: [["clear", "Undo all"], null],
  declined: [["clear", "Undo all"], ["chosen", "Choose all"]],
  never: [["clear", "Undo all"], null],
  brought: [null, null],
};
const BRING_HELD = {
  housekeeping: "kept by your agent's hooks, not a memory of your work",
  looks_secret: "looks like it holds a key, a token or a password",
};

const bring = { ticket: 0, shown: 0, total: 0 };

/** The filter the form holds, as the server takes it. */
function bringFilter() {
  const form = new FormData(byId("bring-filter"));
  const filter = { state: form.get("state") || "undecided" };
  for (const [name] of BRING_FACETS) {
    if (form.get(name)) filter[name] = form.get(name);
  }
  const contains = (form.get("contains") || "").trim();
  if (contains) filter.contains = contains;
  if (byId("bring-used").checked) filter.used = true;
  if (byId("bring-held").checked) filter.include_held = true;
  return filter;
}

/** Fill a select from the server's counts, keeping what was chosen in it. */
function fillFacet(name, any, names, values) {
  const select = byId(`bring-${name}`);
  const chosen = select.value;
  select.replaceChildren(
    h("option", { value: "" }, any),
    ...values.map(([value, count]) => h("option", { value }, `${names[value] || value} (${numbers.format(count)})`)),
  );
  // A value the new state does not have is still what was asked for.
  if (chosen && !values.some(([value]) => value === chosen)) {
    select.append(h("option", { value: chosen }, `${names[chosen] || chosen} (0)`));
  }
  select.value = chosen;
}

function bringRow(m, state) {
  const status = h("span", { class: "small muted", role: "status" });
  const item = h("li", { class: "bring-row" });
  const decide = (choice, label, quiet) => {
    const button = h("button", { type: "button", class: quiet ? "link" : "link strong" }, label);
    button.addEventListener("click", async () => {
      button.disabled = true;
      const ok = await attempt(status, "", "", () => call("bring_choose", { choice, ids: [m.id] }));
      if (!ok) {
        button.disabled = false;
        return;
      }
      // Decided: it leaves this list, and the counts follow.
      item.remove();
      bring.shown -= 1;
      bring.total -= 1;
      drawBringTotal(state);
      refreshBringCounts();
    });
    return button;
  };
  const acts =
    state === "undecided"
      ? [decide("chosen", "Choose"), decide("declined", "Leave out", true), decide("never", "Never", true)]
      : state === "brought"
        ? []
        : [decide("clear", "Undo", true), state === "declined" ? decide("chosen", "Choose") : null];
  item.append(
    rich(m.truncated ? `${m.content} \u2026` : m.content),
    h(
      "p",
      { class: "meta small muted" },
      h("span", { class: "tag" }, m.network),
      m.repo ? h("span", { class: "mono" }, m.repo) : null,
      m.compartment ? h("span", {}, m.compartment) : null,
      m.host ? h("span", {}, `on ${m.host}`) : null,
      h("span", { title: m.created_at }, dates.format(new Date(m.created_at))),
      m.reinforcement ? h("span", {}, `used ${numbers.format(m.reinforcement)}\u00d7`) : null,
      m.held ? h("span", { class: "held" }, `Held back: ${BRING_HELD[m.held] || m.held}`) : null,
    ),
    h("div", { class: "actions-row" }, ...acts, status),
  );
  return item;
}

function drawBringTotal(state) {
  const total = bring.total;
  byId("bring-total").textContent = `${numbers.format(total)} ${total === 1 ? "memory matches" : "memories match"}`;
  const offers = BRING_BULK[state] || [null, null];
  for (const [id, offer] of [["bring-all", offers[0]], ["bring-all-second", offers[1]]]) {
    const button = byId(id);
    button.hidden = !offer || total === 0;
    if (offer) {
      button.dataset.choice = offer[0];
      button.textContent = `${offer[1]} ${numbers.format(total)}`;
    }
  }
  const more = byId("bring-more");
  more.hidden = bring.shown >= total;
  more.textContent = `Show ${numbers.format(Math.min(BRING_PAGE, total - bring.shown))} more`;
}

/** The states' counts and the sidebar's, from the server. */
function drawBringStates(out) {
  const states = out.states || {};
  const count = (name) => numbers.format(states[name] || 0);
  byId("bring-states").textContent = `${count("undecided")} to decide \u00b7 ${count("chosen")} chosen \u00b7 ${count("brought")} brought`;
  byId("bring-held-count").textContent = out.held_back ? `(${numbers.format(out.held_back)})` : "";
  const badge = byId("nav-chosen");
  badge.hidden = !states.chosen;
  badge.textContent = states.chosen ? numbers.format(states.chosen) : "";
  byId("bring-send").hidden = !states.chosen;
}

async function refreshBringCounts() {
  try {
    drawBringStates(await call("bring_candidates", { filter: bringFilter(), limit: 1 }));
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
  }
}

/** Read the first page for the filter as it stands and draw it. */
async function loadBring() {
  const ticket = ++bring.ticket;
  const filter = bringFilter();
  const out = await call("bring_candidates", { filter, limit: BRING_PAGE });
  // A later change of the filter owns the page.
  if (ticket !== bring.ticket) return;
  for (const [name, any, names] of BRING_FACETS) fillFacet(name, any, names, (out.facets || {})[name] || []);
  drawBringStates(out);
  bring.total = out.total;
  bring.shown = out.memories.length;
  const empty =
    filter.state === "undecided"
      ? "Nothing here to decide. Loosen the filters, or look at what is held back."
      : "None.";
  byId("bring-list").replaceChildren(
    ...(out.memories.length ? out.memories.map((m) => bringRow(m, filter.state)) : [h("li", { class: "empty" }, empty)]),
  );
  drawBringTotal(filter.state);
}

async function showBring() {
  try {
    await section(byId("bring-list"), loadBring);
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
  }
}

/** The page reads every memory to count them, so it is read when looked at. */
function showBringWhenAsked() {
  if (location.hash.replace(/^#\/?/, "") === "share" && state.token) showBring();
}

function startBring() {
  window.addEventListener("hashchange", showBringWhenAsked);
  const form = byId("bring-filter");
  form.addEventListener("submit", (event) => event.preventDefault());
  form.addEventListener("change", showBring);
  // Typing narrows once it pauses.
  let typing = 0;
  byId("bring-contains").addEventListener("input", () => {
    clearTimeout(typing);
    typing = setTimeout(showBring, 350);
  });
  byId("bring-reset").addEventListener("click", () => {
    const state = new FormData(form).get("state");
    form.reset();
    const kept = form.querySelector(`input[name=state][value=${state}]`);
    if (kept) kept.checked = true;
    showBring();
  });
  byId("bring-more").addEventListener("click", async () => {
    const filter = bringFilter();
    try {
      const out = await call("bring_candidates", { filter, limit: BRING_PAGE, offset: bring.shown });
      byId("bring-list").append(...out.memories.map((m) => bringRow(m, filter.state)));
      bring.shown += out.memories.length;
      bring.total = out.total;
      drawBringTotal(filter.state);
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
    }
  });
  for (const id of ["bring-all", "bring-all-second"]) {
    byId(id).addEventListener("click", async () => {
      const button = byId(id);
      const choice = button.dataset.choice;
      const said = button.textContent;
      button.disabled = true;
      // Every memory the filter matches, decided by the server in one call.
      let changed = 0;
      const ok = await attempt(byId("bring-status"), "Deciding", "", async () => {
        changed = (await call("bring_choose", { choice, filter: bringFilter() })).changed;
      });
      button.disabled = false;
      if (ok) {
        await showBring();
        byId("bring-status").textContent = `${said.replace(/\s[\d,.\s]+$/, "")}: done for ${numbers.format(changed)}.`;
      }
    });
  }
}

/** Empty what this page drew, at sign-out. */
function clearBring() {
  bring.ticket += 1;
  byId("bring-list").replaceChildren();
  byId("bring-filter").reset();
  for (const id of ["bring-states", "bring-total", "bring-status", "bring-held-count"]) byId(id).textContent = "";
  byId("nav-chosen").hidden = true;
}

// The Behaviors section: the user's behaviors by status, and the two decisions
// the dashboard makes for them. A behavior drawn from the store is proposed
// until the user accepts it here (or with `accept_behavior`); a retired one
// trains nothing. Loaded after app.js and sharing its helpers (h, byId, call,
// section, signOut) and its errors (SignedOut, Superseded).
//
// Stored text is written with `textContent` through `h`, never as markup.

const BEHAVIOR_STATUSES = ["proposed", "accepted", "trained", "retired"];

/** The status the filter shows: one of BEHAVIOR_STATUSES, or "" for all. */
function behaviorsFilter() {
  const checked = byId("behaviors-filter").querySelector("input[name=status]:checked");
  return checked ? checked.value : "proposed";
}

/** The user's behaviors, newest first, cut to the filter's status. */
async function loadBehaviors() {
  const { behaviors } = await call("list_behaviors");
  const counts = Object.fromEntries(BEHAVIOR_STATUSES.map((s) => [s, 0]));
  for (const b of behaviors) counts[b.status] = (counts[b.status] ?? 0) + 1;
  byId("behaviors-summary").textContent = behaviors.length
    ? BEHAVIOR_STATUSES.map((s) => `${numbers.format(counts[s])} ${s}`).join(" · ")
    : "";
  const status = behaviorsFilter();
  const shown = status ? behaviors.filter((b) => b.status === status) : behaviors;
  const list = byId("behaviors");
  if (!shown.length) {
    const empty = !behaviors.length
      ? "None yet: an agent records one when you state or correct how it should act."
      : status === "proposed"
        ? "Nothing is waiting for you."
        : `None ${status}.`;
    list.replaceChildren(h("li", { class: "empty" }, empty));
    return;
  }
  list.replaceChildren(...shown.map(behaviorItem));
}

function plural(count, one, many) {
  return `${numbers.format(count)} ${count === 1 ? one : many}`;
}

function behaviorItem(b) {
  const sources = b.sources ?? [];
  const meta = [
    h("span", { class: "tag" }, b.status),
    h("span", {}, b.scope),
    h("span", {}, `${plural(b.examples, "example", "examples")} · ${plural(b.violations, "violation", "violations")}`),
    sources.length ? h("span", {}, `from ${plural(sources.length, "memory", "memories")}`) : null,
    b.supersedes ? h("span", {}, "replaces an earlier one") : null,
    b.last_training ? h("span", { class: "muted" }, b.last_training) : null,
    b.last_audit ? h("span", { class: "muted" }, b.last_audit) : null,
    ...(b.problems ?? []).map((p) => h("span", { class: "flag" }, p)),
    h("span", { class: "id" }, b.id),
  ];
  const acts = [];
  if (b.status === "proposed") acts.push(behaviorButton("Accept", "accept_behavior", b));
  if (b.status !== "retired") acts.push(behaviorButton("Retire", "retire_behavior", b, "quiet"));
  return h(
    "li",
    { class: "behavior" },
    h("p", { class: "rule" }, b.rule),
    h("div", { class: "meta" }, ...meta),
    acts.length ? h("div", { class: "actions-row" }, ...acts) : null,
  );
}

function behaviorButton(label, tool, b, kind) {
  const button = h("button", { type: "button", class: kind }, label);
  button.addEventListener("click", () => behaviorAction(button, label, tool, b));
  return button;
}

/** One decision on one behavior, as the signed-in identity; the list reloads
 *  so what is shown is what is stored. A refusal is shown on the item. */
async function behaviorAction(button, label, tool, b) {
  const item = button.closest("li");
  const buttons = item.querySelectorAll("button");
  for (const el of buttons) el.disabled = true;
  try {
    const out = await call(tool, { behavior_id: b.id });
    if (!out.found) throw new Error("it is not one of your behaviors any more");
    await section(byId("behaviors"), loadBehaviors);
  } catch (error) {
    if (error instanceof SignedOut) {
      signOut(REFUSED);
      return;
    }
    if (error instanceof Superseded) return;
    for (const el of buttons) el.disabled = false;
    const shown = item.querySelector(".failed");
    if (shown) shown.remove();
    item.append(h("p", { class: "failed", role: "alert" }, `Could not ${label.toLowerCase()}: ${error.message}`));
  }
}

// A different status re-reads the list under the new filter.
byId("behaviors-filter").addEventListener("change", async (event) => {
  if (event.target.name !== "status" || !state.token) return;
  try {
    await section(byId("behaviors"), loadBehaviors);
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
  }
});

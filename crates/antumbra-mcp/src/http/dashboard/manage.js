// What the dashboard changes, and the Agent page. Until now it read, but for
// a behavior's Accept and Retire and a handoff's Mark done. Here are the
// rest of what is a user's own to do: write a memory, reinforce one, judge
// one wrong, forget one; leave a handoff for an agent on one of their
// machines; add a document; make a compartment. Each is the tool an agent
// would call, through the same `/mcp/call`, as the signed-in identity.
//
// The Agent page shows what a coding agent's hooks count and what it was
// told: which skills are used, and the rules it is given. Both are keyed
// memories, read with the same call the CLI's `antumbra claude skills` makes.
//
// Loaded after app.js, behaviors.js and shell.js and sharing their helpers
// (h, byId, call, section, signOut, numbers, stamps, ago) and their errors.
// Stored text is written with `textContent` through `h`, never as markup.

/** The keys a counter of a skill's use and a rule for a coding agent open
 *  with (antumbra_core::keyed). */
const SKILL_USE = "[skill-use:";
const AGENT_RULE = "[claude-code:";
/** A rule is written at full confidence and one penalty takes it clearly
 *  below this: the line the console draws for "retired". */
const RETIRED_BELOW = 0.85;

/** The name a keyed memory's text opens with, for `family`, or null. */
function keyedName(content, family) {
  if (typeof content !== "string" || !content.startsWith(family)) return null;
  const end = content.indexOf("]");
  const name = end > family.length ? content.slice(family.length, end) : "";
  return name || null;
}

/** Run a write and say how it went in `status`; true when it was done. A
 *  refused token signs out, as anywhere. */
async function attempt(status, doing, done, write) {
  status.classList.remove("error");
  status.textContent = doing;
  try {
    await write();
    status.textContent = done;
    return true;
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
    else if (!(error instanceof Superseded)) {
      status.classList.add("error");
      status.textContent = `Not done: ${error.message}`;
    }
    return false;
  }
}

// The Agent page

async function loadAgent() {
  const out = await call("list_memories", { network: "world" });
  const skills = new Map();
  const rules = new Map();
  for (const m of out.memories ?? []) {
    const skill = keyedName(m.content, SKILL_USE);
    if (skill) {
      // A counter is made by its first use and reinforced by each later one;
      // two counters for one skill are one count.
      const row = skills.get(skill) ?? { skill, uses: 0, last: "" };
      row.uses += 1 + (m.reinforcement ?? 0);
      if (String(m.updated_at ?? "") > row.last) row.last = String(m.updated_at ?? "");
      skills.set(skill, row);
      continue;
    }
    const rule = keyedName(m.content, AGENT_RULE);
    if (rule) {
      const text = m.content.slice(m.content.indexOf("]") + 1).trim();
      const current = (m.confidence ?? 1) >= RETIRED_BELOW;
      const row = rules.get(rule);
      // A rule's current text is the one shown; a retired one only if no
      // current one is held.
      if (!row || (current && !row.current)) rules.set(rule, { rule, text, current });
    }
  }
  // Stalest first: the question the page answers is what is no longer used.
  const used = [...skills.values()].sort((a, b) => a.last.localeCompare(b.last));
  byId("skills-summary").textContent = used.length
    ? `${numbers.format(used.length)} used, the longest unused first`
    : "";
  byId("skills").replaceChildren(
    ...(used.length
      ? used.map((s) =>
          h(
            "tr",
            {},
            h("th", { scope: "row", class: "mono" }, s.skill),
            h("td", { class: "num" }, numbers.format(s.uses)),
            h("td", { title: s.last }, s.last ? ago(s.last) : "not said"),
          ),
        )
      : [
          h(
            "tr",
            {},
            h(
              "td",
              { colspan: 3, class: "empty" },
              "No skill use has been counted. Setup wires the count for Claude Code; a skill is counted when an agent uses it.",
            ),
          ),
        ]),
  );
  const told = [...rules.values()].sort((a, b) => a.rule.localeCompare(b.rule));
  byId("rules-summary").textContent = told.length
    ? `${numbers.format(told.filter((r) => r.current).length)} current of ${numbers.format(told.length)}`
    : "";
  byId("rules").replaceChildren(
    ...(told.length
      ? told.map((r) =>
          h(
            "li",
            { class: "rule-row" },
            h(
              "p",
              { class: "meta" },
              h("span", { class: "mono" }, r.rule),
              h("span", { class: "tag" }, r.current ? "current" : "retired"),
            ),
            h("p", { class: "content" }, r.text),
          ),
        )
      : [h("li", { class: "empty" }, "No rules are kept: `antumbra claude rules` writes the ones for an agent run with its feature flags off.")]),
  );
}

/** The Agent page reads every fact in the workspace, so it is read when it is
 *  looked at, not with every refresh of the rest. */
async function showAgentWhenAsked() {
  if (location.hash.replace(/^#\/?/, "") !== "agent" || !state.token) return;
  try {
    await section(byId("skills"), loadAgent);
  } catch (error) {
    if (error instanceof SignedOut) signOut(REFUSED);
  }
}

// Memory

/** The buttons under a memory: count it as right, judge it wrong, forget it. */
function memoryActions(m) {
  const status = h("span", { class: "small muted", role: "status" });
  const act = (label, tool, doing, done) => {
    const button = h("button", { type: "button", class: "link" }, label);
    button.addEventListener("click", async () => {
      button.disabled = true;
      await attempt(status, doing, done, () => call(tool, { memory_id: m.id }));
      button.disabled = false;
    });
    return button;
  };
  const forget = h("button", { type: "button", class: "link danger" }, "Forget");
  let asked = false;
  forget.addEventListener("click", async () => {
    if (!asked) {
      // Forgetting is not undone from here: asked for twice.
      asked = true;
      forget.textContent = "Forget it for good?";
      status.textContent = "Press again to forget this memory.";
      return;
    }
    forget.disabled = true;
    const gone = await attempt(status, "Forgetting", "Forgotten.", () => call("forget_memory", { memory_id: m.id }));
    if (gone && state.memory.view) {
      try {
        await section(byId("memories"), () => runMemory(state.memory.view));
      } catch (error) {
        if (error instanceof SignedOut) signOut(REFUSED);
      }
    } else {
      forget.disabled = false;
    }
  });
  return [
    act("Reinforce", "reinforce_memory", "Reinforcing", "Reinforced."),
    act("Penalize", "penalize_memory", "Penalizing", "Penalized: it counts for less."),
    forget,
    status,
  ];
}

/** Wire a form that writes: `write` is given the form's values and returns
 *  when the write is done; `after` redraws what it changed. */
function wireForm(id, doing, done, write, after) {
  const form = byId(id);
  const status = form.querySelector("[role=status]");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    const values = Object.fromEntries(new FormData(form));
    const ok = await attempt(status, doing, done, () => write(values));
    button.disabled = false;
    if (!ok) return;
    // What was typed is kept until it is written, and cleared once it is.
    for (const field of form.querySelectorAll("textarea, input[type=text]")) field.value = "";
    try {
      await after();
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
    }
  });
}

/** The device a handoff may be left for: the user's own, or any of them. */
function offerDevices() {
  const select = byId("handoff-for");
  const chosen = select.value;
  select.replaceChildren(
    h("option", { value: "" }, "Whichever starts a session next"),
    ...state.shell.devices.map((d) => h("option", { value: d.host }, d.host)),
  );
  if ([...select.options].some((o) => o.value === chosen)) select.value = chosen;
}

// The member's own profile: the name they go by, and their image.

/** The side of the square a profile's image is made, in pixels. */
const AVATAR_SIDE = 256;

/** Read the workspace's profiles and draw the signed-in member from theirs. */
async function loadProfile() {
  const out = await call("profiles", { images: true });
  const mine = (out.profiles ?? []).find((p) => p.user === out.you) || {};
  state.shell.profile = mine;
  showSideProfile(mine.display || "", mine.image || "");
  byId("profile-display").value = mine.display || "";
  byId("profile-image-clear").hidden = !mine.image;
}

/** `file` cut to its middle square, made AVATAR_SIDE across, as base64: a
 *  WebP where the browser writes one, else a PNG. The server checks what it
 *  is given all the same. */
async function squared(file) {
  const picture = await createImageBitmap(file);
  const side = Math.min(picture.width, picture.height);
  const canvas = document.createElement("canvas");
  canvas.width = AVATAR_SIDE;
  canvas.height = AVATAR_SIDE;
  canvas
    .getContext("2d")
    .drawImage(picture, (picture.width - side) / 2, (picture.height - side) / 2, side, side, 0, 0, AVATAR_SIDE, AVATAR_SIDE);
  picture.close();
  const made = (type, quality) => new Promise((done) => canvas.toBlob(done, type, quality));
  let blob = await made("image/webp", 0.9);
  if (!blob || blob.type !== "image/webp") blob = await made("image/png");
  if (!blob) throw new Error("this browser could not prepare the image");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

function startProfile() {
  const form = byId("profile");
  const status = form.querySelector("[role=status]");
  const redraw = async () => {
    try {
      await loadProfile();
    } catch (error) {
      if (error instanceof SignedOut) signOut(REFUSED);
    }
  };
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    const file = byId("profile-image").files[0];
    const ok = await attempt(status, "Saving", "Saved.", async () => {
      const sent = { display: byId("profile-display").value };
      if (file) sent.image = await squared(file);
      await call("set_profile", sent);
    });
    button.disabled = false;
    if (!ok) return;
    byId("profile-image").value = "";
    await redraw();
  });
  byId("profile-image-clear").addEventListener("click", async () => {
    const ok = await attempt(status, "Removing", "Removed.", () => call("set_profile", { clear_image: true }));
    if (ok) await redraw();
  });
}

function startManage() {
  startProfile();
  window.addEventListener("hashchange", showAgentWhenAsked);

  wireForm(
    "memory-new",
    "Storing",
    "Stored.",
    (v) => call("store_memory", { content: v.content.trim(), network: v.network }),
    async () => {
      await section(byId("stats"), loadStats);
      if (state.memory.view) await section(byId("memories"), () => runMemory(state.memory.view));
    },
  );
  wireForm(
    "handoff-new",
    "Leaving it",
    "Left: the agent there is told when it next starts a session.",
    (v) => call("leave_handoff", { content: v.content.trim(), for_host: v.for_host || "any", from_host: "dashboard" }),
    () => section(byId("handoffs"), () => loadHandoffs(state.shell.devices)),
  );
  wireForm(
    "document-new",
    "Adding",
    "Added: it is cut into pieces and searchable.",
    (v) => {
      const source = v.source.trim();
      return call("ingest_document", { title: v.title.trim(), content: v.content, ...(source ? { source } : {}) });
    },
    async () => {
      await section(byId("documents"), loadDocuments);
      await section(byId("stats"), loadStats);
    },
  );
  wireForm(
    "compartment-new",
    "Making it",
    "Made.",
    (v) => call("create_compartment", { name: v.name.trim() }),
    async () => {
      await section(byId("compartments"), loadCompartments);
      await section(byId("stats"), loadStats);
    },
  );
}

/** Empty what this file drew, at sign-out. */
function clearManage() {
  showSideProfile("", "");
  byId("profile-image-clear").hidden = true;
  for (const id of ["skills", "rules"]) byId(id).replaceChildren();
  for (const id of ["skills-summary", "rules-summary"]) byId(id).textContent = "";
  for (const id of ["memory-new", "handoff-new", "document-new", "compartment-new", "profile"]) {
    byId(id).reset();
    byId(id).querySelector("[role=status]").textContent = "";
  }
}

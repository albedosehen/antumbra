// Antumbra for OpenCode. Written to ~/.config/opencode/plugins by
// `antumbra setup`, which replaces it whole; run setup again to update it.
//
// OpenCode has no hooks that run a command. It has plugins, and two release
// lines whose plugins are written differently, so this one file is both: its
// default export has `setup` for the 2.x line and `server` for the 1.x line,
// and each line takes the half it knows.
//
// It does three things, each by running `antumbra hook opencode <event>`,
// which answers with the same hook scripts Claude Code runs:
//
// - a session's first prompt: what Antumbra remembers that matters here;
// - every prompt: what is recalled for it. Both are put in the system prompt
//   for that session, where the model reads them with every call;
// - a turn's end: how it ended, and now and then a reminder to store what
//   was learned, sent to the session as a prompt.
//
// It also registers the MCP server (the bridge, `antumbra mcp bridge`), so
// nothing is added to opencode.json.
//
// Like the hooks, it never fails the session: whatever goes wrong, the
// session goes on without what this would have added.

import { spawn } from "node:child_process"

// How this machine runs `antumbra`. Setup writes this line.
const ANTUMBRA = ["antumbra"]

const NAME = "antumbra"
const SERVER = { type: "local", command: [...ANTUMBRA, "mcp", "bridge", "--agent", "opencode"] }

// Longer than the hook's own patience with its script, so it is the hook
// that gives up first, and says why.
const PATIENCE_MS = 30000

/** Run one hook with `input` on its standard input; what it answered, or null. */
function hook(event, input) {
  return new Promise((resolve) => {
    let out = ""
    let settled = false
    const done = (value) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve(value)
    }
    let child
    try {
      child = spawn(ANTUMBRA[0], [...ANTUMBRA.slice(1), "hook", "opencode", event], {
        stdio: ["pipe", "pipe", "ignore"],
        windowsHide: true,
      })
    } catch {
      return resolve(null)
    }
    const timer = setTimeout(() => {
      try {
        child.kill()
      } catch {}
      done(null)
    }, PATIENCE_MS)
    child.on("error", () => done(null))
    child.stdout.on("data", (chunk) => {
      out += chunk
    })
    child.on("close", () => {
      try {
        done(out.trim() ? JSON.parse(out) : null)
      } catch {
        done(null)
      }
    })
    child.stdin.on("error", () => {})
    child.stdin.end(JSON.stringify(input))
  })
}

// What each session is told: what it opened with, and what was recalled for
// its latest prompt. `reminding` marks a prompt this plugin sent itself.
const sessions = new Map()
const told = (id) => {
  if (!sessions.has(id)) sessions.set(id, { opened: null, recalled: "", reminding: false })
  return sessions.get(id)
}

/** A prompt arrived: open the session if this is its first, and recall for it. */
async function prompted(sessionID, cwd, prompt) {
  if (!sessionID) return
  const session = told(sessionID)
  if (session.reminding) {
    // The reminder is this plugin's own prompt: nothing is recalled for it.
    session.reminding = false
    return
  }
  if (session.opened === null) {
    session.opened = ""
    const start = await hook("session-start", { session_id: sessionID, cwd })
    session.opened = typeof start?.context === "string" ? start.context : ""
  }
  const recall = await hook("prompt", { session_id: sessionID, cwd, prompt })
  session.recalled = typeof recall?.context === "string" ? recall.context : ""
}

/** What a session's model is to read, oldest first. */
function context(sessionID) {
  const session = sessions.get(sessionID)
  return session ? [session.opened, session.recalled].filter(Boolean) : []
}

/** A turn ended with `last`: report it, and hand back a reminder when one is due. */
async function ended(sessionID, cwd, last) {
  // A session this plugin never saw a prompt in is not one to speak into.
  if (!sessionID || !sessions.has(sessionID)) return null
  const stop = await hook("stop", { session_id: sessionID, cwd, last_assistant_message: last })
  if (typeof stop?.reminder !== "string" || !stop.reminder) return null
  told(sessionID).reminding = true
  return stop.reminder
}

/** The text of a list of parts, the ones a person or the model wrote. */
const text = (parts) =>
  (Array.isArray(parts) ? parts : [])
    .filter((part) => part?.type === "text" && typeof part.text === "string" && !part.synthetic)
    .map((part) => part.text)
    .join("\n")

export default {
  id: NAME,

  // OpenCode 2.x.
  async setup(ctx) {
    const cwd = ctx?.location?.directory
    try {
      ctx.mcp.transform((servers) => {
        if (!servers.get(NAME)) servers.set(NAME, SERVER)
      })
    } catch {}
    ctx.session.hook("prompt", async (event) => {
      try {
        await prompted(event.sessionID, cwd, event.prompt?.text ?? "")
      } catch {}
    })
    ctx.session.hook("context", (event) => {
      for (const part of context(event.sessionID)) event.system.push({ type: "text", text: part })
    })
    ctx.session.hook("experimental.idle", async (event) => {
      try {
        const messages = await ctx.session.context({ sessionID: event.sessionID })
        const answer = (Array.isArray(messages) ? messages : []).filter((m) => m?.type === "assistant").pop()
        const reminder = await ended(event.sessionID, cwd, text(answer?.content))
        if (reminder) await ctx.session.prompt({ sessionID: event.sessionID, text: reminder })
      } catch {}
    })
  },

  // OpenCode 1.x.
  async server({ client, directory }) {
    return {
      config: async (config) => {
        config.mcp = { ...(config.mcp ?? {}) }
        if (!config.mcp[NAME]) config.mcp[NAME] = { ...SERVER, enabled: true }
      },
      "chat.message": async (input, output) => {
        try {
          await prompted(input?.sessionID, directory, text(output?.parts))
        } catch {}
      },
      "experimental.chat.system.transform": async (input, output) => {
        for (const part of context(input?.sessionID)) output.system.push(part)
      },
      event: async ({ event }) => {
        if (event?.type !== "session.idle") return
        const sessionID = event.properties?.sessionID
        try {
          const listed = await client.session.messages({ path: { id: sessionID } })
          const messages = Array.isArray(listed) ? listed : (listed?.data ?? [])
          const answer = messages.filter((m) => m?.info?.role === "assistant").pop()
          const reminder = await ended(sessionID, directory, text(answer?.parts))
          if (reminder) {
            await client.session.prompt({
              path: { id: sessionID },
              body: { parts: [{ type: "text", text: reminder }] },
            })
          }
        } catch {}
      },
    }
  },
}

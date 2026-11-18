// Tests of the OpenCode plugin (scripts/hooks/antumbra-opencode.js), both
// halves: `server` as OpenCode 1.x calls it and `setup` as 2.x does.
//
//   node --test scripts/hooks/tests/opencode-plugin.test.mjs
//
// The plugin runs `antumbra hook opencode <event>`. Here `antumbra` is a
// stand-in that records what it was run with and handed, and answers as the
// hook does, so no server and no OpenCode is needed.

import assert from "node:assert/strict"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { test } from "node:test"
import { fileURLToPath, pathToFileURL } from "node:url"

const here = path.dirname(fileURLToPath(import.meta.url))
const source = fs.readFileSync(path.join(here, "..", "antumbra-opencode.js"), "utf8")
const RUNS = 'const ANTUMBRA = ["antumbra"]'

// The stand-in: appends {event, input} to the log named by its first
// argument, then answers. `stop` reminds once a session, as the capture hook
// does; a prompt of "break" answers with something that is not JSON.
const STAND_IN = `
import fs from "node:fs"
const [log, , , event] = process.argv.slice(2)
let raw = ""
process.stdin.on("data", (d) => (raw += d))
process.stdin.on("end", () => {
  const input = JSON.parse(raw)
  fs.appendFileSync(log, JSON.stringify({ event, input }) + "\\n")
  if (event === "session-start") process.stdout.write(JSON.stringify({ context: "OPENED " + input.session_id }))
  if (event === "prompt") {
    if (input.prompt === "break") process.stdout.write("not json")
    else if (input.prompt !== "nothing") process.stdout.write(JSON.stringify({ context: "RECALLED " + input.prompt }))
  }
  if (event === "stop") {
    const seen = fs.readFileSync(log, "utf8").split("\\n").filter((l) => l.includes('"event":"stop"') && l.includes(input.session_id)).length
    if (seen === 1) process.stdout.write(JSON.stringify({ reminder: "REMINDER" }))
  }
})
`

/** A copy of the plugin whose `antumbra` is the stand-in, and the log it writes. */
async function plugin() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "antumbra-opencode-"))
  const log = path.join(dir, "ran.jsonl")
  fs.writeFileSync(log, "")
  const standIn = path.join(dir, "antumbra.mjs")
  fs.writeFileSync(standIn, STAND_IN)
  assert.equal(source.split(RUNS).length, 2, "the line setup rewrites is there once")
  const file = path.join(dir, "plugin.mjs")
  fs.writeFileSync(file, source.replace(RUNS, `const ANTUMBRA = ${JSON.stringify([process.execPath, standIn, log])}`))
  const loaded = (await import(pathToFileURL(file).href)).default
  const ran = () =>
    fs
      .readFileSync(log, "utf8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line))
  return { loaded, ran, standIn, log }
}

test("1.x: the server is registered beside the user's own, and never over one of that name", async () => {
  const { loaded, standIn, log } = await plugin()
  const hooks = await loaded.server({ client: {}, directory: "/work" })
  const config = { mcp: { context7: { type: "local", command: ["npx", "x"] } } }
  await hooks.config(config)
  assert.deepEqual(config.mcp.antumbra, {
    type: "local",
    command: [process.execPath, standIn, log, "mcp", "bridge", "--agent", "opencode"],
    enabled: true,
  })
  assert.deepEqual(config.mcp.context7, { type: "local", command: ["npx", "x"] })

  const theirs = { mcp: { antumbra: { type: "remote", url: "https://memory.example/mcp" } } }
  await hooks.config(theirs)
  assert.deepEqual(theirs.mcp.antumbra, { type: "remote", url: "https://memory.example/mcp" })

  const none = {}
  await hooks.config(none)
  assert.equal(none.mcp.antumbra.type, "local")
})

test("1.x: a session opens once, each prompt is recalled for, and the model is told both", async () => {
  const { loaded, ran } = await plugin()
  const hooks = await loaded.server({ client: {}, directory: "/work" })
  const system = async (sessionID) => {
    const output = { system: ["OpenCode's own"] }
    await hooks["experimental.chat.system.transform"]({ sessionID }, output)
    return output.system
  }
  // Before its first prompt a session is told nothing.
  assert.deepEqual(await system("s1"), ["OpenCode's own"])

  await hooks["chat.message"](
    { sessionID: "s1" },
    { parts: [{ type: "text", text: "first" }, { type: "text", text: "added by a tool", synthetic: true }, { type: "file" }] },
  )
  assert.deepEqual(await system("s1"), ["OpenCode's own", "OPENED s1", "RECALLED first"])
  assert.deepEqual(
    ran().map((r) => [r.event, r.input.session_id, r.input.cwd, r.input.prompt]),
    [
      ["session-start", "s1", "/work", undefined],
      ["prompt", "s1", "/work", "first"],
    ],
  )

  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "second" }] })
  assert.deepEqual(await system("s1"), ["OpenCode's own", "OPENED s1", "RECALLED second"])
  assert.equal(ran().filter((r) => r.event === "session-start").length, 1)

  // A prompt nothing is recalled for keeps what the session opened with.
  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "nothing" }] })
  assert.deepEqual(await system("s1"), ["OpenCode's own", "OPENED s1"])

  // A hook that answers with something unreadable costs the session nothing.
  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "break" }] })
  assert.deepEqual(await system("s1"), ["OpenCode's own", "OPENED s1"])

  // Another session is its own.
  await hooks["chat.message"]({ sessionID: "s2" }, { parts: [{ type: "text", text: "other" }] })
  assert.deepEqual(await system("s2"), ["OpenCode's own", "OPENED s2", "RECALLED other"])
})

test("1.x: a turn's end is reported with the answer, and the reminder is sent once and not recalled for", async () => {
  const { loaded, ran } = await plugin()
  const prompts = []
  const client = {
    session: {
      messages: async ({ path: { id } }) => ({
        data: [
          { info: { role: "user" }, parts: [{ type: "text", text: "question" }] },
          { info: { role: "assistant" }, parts: [{ type: "text", text: "an earlier answer" }] },
          { info: { role: "assistant" }, parts: [{ type: "tool" }, { type: "text", text: `the answer in ${id}` }] },
        ],
      }),
      prompt: async (request) => prompts.push(request),
    },
  }
  const hooks = await loaded.server({ client, directory: "/work" })
  const idle = (sessionID) => hooks.event({ event: { type: "session.idle", properties: { sessionID } } })

  // A session this plugin saw no prompt in is not reported or spoken into.
  await idle("stranger")
  assert.equal(ran().length, 0)

  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "question" }] })
  await idle("s1")
  const stop = ran().filter((r) => r.event === "stop")
  assert.equal(stop.length, 1)
  assert.equal(stop[0].input.last_assistant_message, "the answer in s1")
  assert.deepEqual(prompts, [{ path: { id: "s1" }, body: { parts: [{ type: "text", text: "REMINDER" }] } }])

  // The reminder arrives as a message: nothing is recalled for it.
  const before = ran().filter((r) => r.event === "prompt").length
  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "REMINDER" }] })
  assert.equal(ran().filter((r) => r.event === "prompt").length, before)

  // The turn the reminder caused ends, and is reported; nothing more is sent.
  await idle("s1")
  assert.equal(ran().filter((r) => r.event === "stop").length, 2)
  assert.equal(prompts.length, 1)

  // Events that are not a turn's end are not this plugin's.
  await hooks.event({ event: { type: "session.updated", properties: { sessionID: "s1" } } })
  assert.equal(ran().filter((r) => r.event === "stop").length, 2)
})

/** What 2.x hands a plugin's `setup`, recording what the plugin does with it. */
function context2() {
  const hooks = {}
  const servers = new Map([["context7", { type: "local", command: ["npx", "x"] }]])
  const prompts = []
  const ctx = {
    location: { directory: "/work" },
    mcp: { transform: (change) => change({ get: (name) => servers.get(name), set: (name, server) => servers.set(name, server) }) },
    session: {
      hook: (name, run) => (hooks[name] = run),
      context: async ({ sessionID }) => [
        { type: "user", text: "question" },
        { type: "assistant", content: [{ type: "text", text: `the answer in ${sessionID}` }], finish: "stop" },
      ],
      prompt: async (request) => prompts.push(request),
    },
  }
  return { ctx, hooks, servers, prompts }
}

test("2.x: the same three moments, through its own hooks", async () => {
  const { loaded, ran, standIn, log } = await plugin()
  const { ctx, hooks, servers, prompts } = context2()
  await loaded.setup(ctx)
  assert.equal(loaded.id, "antumbra")
  assert.deepEqual(servers.get("antumbra"), {
    type: "local",
    command: [process.execPath, standIn, log, "mcp", "bridge", "--agent", "opencode"],
  })
  assert.ok(servers.has("context7"))

  await hooks.prompt({ sessionID: "s1", prompt: { text: "first", files: [] } })
  const event = { sessionID: "s1", system: [{ type: "text", text: "OpenCode's own" }] }
  hooks.context(event)
  assert.deepEqual(event.system, [
    { type: "text", text: "OpenCode's own" },
    { type: "text", text: "OPENED s1" },
    { type: "text", text: "RECALLED first" },
  ])
  assert.deepEqual(
    ran().map((r) => [r.event, r.input.cwd]),
    [
      ["session-start", "/work"],
      ["prompt", "/work"],
    ],
  )

  await hooks["experimental.idle"]({ sessionID: "s1", background: [] })
  const stop = ran().filter((r) => r.event === "stop")
  assert.equal(stop[0].input.last_assistant_message, "the answer in s1")
  assert.deepEqual(prompts, [{ sessionID: "s1", text: "REMINDER" }])

  // The reminder comes back through the prompt hook, and is not recalled for.
  const before = ran().filter((r) => r.event === "prompt").length
  await hooks.prompt({ sessionID: "s1", prompt: { text: "REMINDER" } })
  assert.equal(ran().filter((r) => r.event === "prompt").length, before)

  // A session with no prompt seen is left alone.
  await hooks["experimental.idle"]({ sessionID: "stranger", background: [] })
  assert.equal(prompts.length, 1)
})

test("2.x: a server the user already has under that name is left theirs", async () => {
  const { loaded } = await plugin()
  const { ctx, servers } = context2()
  servers.set("antumbra", { type: "remote", url: "https://memory.example/mcp" })
  await loaded.setup(ctx)
  assert.deepEqual(servers.get("antumbra"), { type: "remote", url: "https://memory.example/mcp" })
})

test("an antumbra that cannot be run costs the session nothing", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "antumbra-opencode-"))
  const file = path.join(dir, "plugin.mjs")
  fs.writeFileSync(file, source.replace(RUNS, `const ANTUMBRA = ${JSON.stringify([path.join(dir, "is-not-there")])}`))
  const loaded = (await import(pathToFileURL(file).href)).default
  const hooks = await loaded.server({ client: {}, directory: "/work" })
  await hooks["chat.message"]({ sessionID: "s1" }, { parts: [{ type: "text", text: "first" }] })
  const output = { system: [] }
  await hooks["experimental.chat.system.transform"]({ sessionID: "s1" }, output)
  assert.deepEqual(output.system, [])
})

"""R-3 networked-MCP end-to-end validation (live, multi-tenant, over ws://).

Proves engine-enforced tenant isolation against a *real* SurrealDB v3 server
reached over the network -- not the embedded engine. Stdlib only.

Run it:

  # 1. a real ws:// SurrealDB v3 (matches the v3 client), root-authenticated:
  docker run -d --name antumbra-surreal -p 8000:8000 surrealdb/surrealdb:v3.0.5 \
      start --user root --pass root --bind 0.0.0.0:8000 memory

  # 2. the networked MCP surface against it (root login + a JWT secret):
  ANTUMBRA_JWT_SECRET=test-secret antumbra-mcp \
      --http 127.0.0.1:8081 --url ws://127.0.0.1:8000/rpc --db-user root --db-pass root

  # 3. this probe:
  python docs/r3_isolation_probe.py

Expected: "RESULT: PASS - multi-tenant isolation holds over ws://".
"""
import base64
import hashlib
import hmac
import json
import time
import urllib.request

SECRET = b"test-secret"
BASE = "http://127.0.0.1:8081/mcp"


def b64url(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def mint(tenant, user):
    h = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64url(json.dumps({"tenant": tenant, "user": user, "exp": int(time.time()) + 3600}).encode())
    sig = b64url(hmac.new(SECRET, f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def post(jwt, body, sid=None):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", f"Bearer {jwt}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    req.add_header("Host", "localhost")
    if sid:
        req.add_header("mcp-session-id", sid)
    resp = urllib.request.urlopen(req, timeout=20)
    return resp.status, resp.headers.get("mcp-session-id"), resp.read().decode()


def parse(text):
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


def session(jwt):
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "r3", "version": "0"}}}
    status, sid, text = post(jwt, init)
    assert status == 200 and sid, (status, text)
    post(jwt, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid


def call(jwt, sid, name, args, _id):
    body = {"jsonrpc": "2.0", "id": _id, "method": "tools/call",
            "params": {"name": name, "arguments": args}}
    _, _, text = post(jwt, body, sid)
    return parse(text)


a, b = mint("ws:a", "user:a"), mint("ws:b", "user:b")
sa, sb = session(a), session(b)
print("[ok] both tenants initialized over ws:// (sessions assigned)")

stored = call(a, sa, "store_memory", {"content": "alice top secret recipe", "network": "world"}, 2)
print("[A] store_memory ->", json.dumps(stored.get("result", stored))[:160])

la = json.dumps(call(a, sa, "list_memories", {}, 3))
lb = json.dumps(call(b, sb, "list_memories", {}, 3))
a_sees = "alice top secret recipe" in la
b_sees = "alice top secret recipe" in lb
print(f"[A] list_memories sees A's secret: {a_sees}")
print(f"[B] list_memories sees A's secret: {b_sees}")

rb = json.dumps(call(b, sb, "recall_memories", {"query": "alice top secret recipe"}, 4))
b_recalls = "alice top secret recipe" in rb
print(f"[B] recall_memories reaches A's secret: {b_recalls}")

ok = a_sees and not b_sees and not b_recalls
print("RESULT:", "PASS - multi-tenant isolation holds over ws://" if ok else "FAIL")

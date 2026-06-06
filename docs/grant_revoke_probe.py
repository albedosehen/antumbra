"""Grant-revoke security validation (live, over ws://). Proves the engine ACL
fails CLOSED on a real SurrealDB v3 server: a grantee sees a shared memory while
granted, and STOPS seeing it the instant the grant is revoked (a tombstone the
ACL subquery excludes) -- not a stale grant that keeps them in.

Reuses the R-3 setup (docker surrealdb v3 + antumbra-mcp --http ... --url ws://...
--db-user root --db-pass root, JWT secret 'test-secret'). Stdlib only.

  python docs/grant_revoke_probe.py

NOTE (2026-06-06): this currently FAILS over ws:// because of R-6 (the root
connection bypasses the engine ACL on a remote — see docs/roadmap.md). The
grant-revoke tombstone logic itself is correct (the embedded grant-ACL test fails
closed after revoke); this probe will pass once R-6 lands a non-root serving
connection. It is kept as the R-6 reproduction.
"""
import base64, hashlib, hmac, json, time, urllib.request

SECRET = b"test-secret"
BASE = "http://127.0.0.1:8081/mcp"
TENANT = "ws:t"  # same tenant: grants are intra-tenant


def b64url(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def mint(user):
    h = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64url(json.dumps({"tenant": TENANT, "user": user, "exp": int(time.time()) + 3600}).encode())
    sig = b64url(hmac.new(SECRET, f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def post(jwt, body, sid=None):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), method="POST")
    for k, v in {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream", "Host": "localhost"}.items():
        req.add_header(k, v)
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
                       "clientInfo": {"name": "g", "version": "0"}}}
    status, sid, text = post(jwt, init)
    assert status == 200 and sid, (status, text)
    post(jwt, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid


def call(jwt, sid, name, args, _id):
    body = {"jsonrpc": "2.0", "id": _id, "method": "tools/call",
            "params": {"name": name, "arguments": args}}
    _, _, text = post(jwt, body, sid)
    return parse(text)


a, b = mint("user:a"), mint("user:b")
sa, sb = session(a), session(b)
a_comp = "comp:ws:t:user:a:default"  # A's provisioned default compartment
SECRET_TXT = "alice shared secret"

call(a, sa, "store_memory", {"content": SECRET_TXT, "network": "world"}, 2)
before = SECRET_TXT in json.dumps(call(b, sb, "list_memories", {}, 3))

call(a, sa, "share_compartment", {"compartment_id": a_comp, "grantee": "user:b", "capability": "reference"}, 4)
granted = SECRET_TXT in json.dumps(call(b, sb, "list_memories", {}, 5))

call(a, sa, "revoke_compartment", {"compartment_id": a_comp, "grantee": "user:b"}, 6)
after = SECRET_TXT in json.dumps(call(b, sb, "list_memories", {}, 7))

print(f"[B] sees A's secret before grant: {before}  (want False)")
print(f"[B] sees A's secret while granted: {granted}  (want True)")
print(f"[B] sees A's secret after revoke: {after}  (want False)")
ok = (not before) and granted and (not after)
print("RESULT:", "PASS - revoke fails closed over ws://" if ok else "FAIL")

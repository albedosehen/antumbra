# Import a kushtaka-export.json into a RUNNING Antumbra MCP server, one memory per
# `store_memory` call (which embeds each through the server's configured embedder,
# so the result is genuinely recallable). The mirror of kushtaka-export.ps1;
# together they migrate Kushtakas -> Antumbra.
#
# Antumbra MCP must already be serving with a real embedder, e.g.:
#   antumbra-mcp --http 127.0.0.1:8081 --url surrealkv://./data/antumbra.skv `
#     --embedder-url http://127.0.0.1:11434/v1/embeddings --embed-model all-minilm `
#     --jwt-secret $env:ANTUMBRA_JWT_SECRET
# and a bearer token minted for the target (tenant, user):
#   antumbra-mcp --mint-token --tenant ws:default --user user:default `
#     --jwt-secret $env:ANTUMBRA_JWT_SECRET --token-ttl-days 365
#
# Env:
#   ANTUMBRA_URL    Antumbra MCP base (default http://127.0.0.1:8081)
#   ANTUMBRA_TOKEN  bearer JWT for the target (tenant, user) (required)

param(
    [string]$In = "kushtaka-export.json",
    [int]$ThrottleMs = 0
)

$ErrorActionPreference = 'Stop'
$apiUrl = if ($env:ANTUMBRA_URL) { $env:ANTUMBRA_URL } else { 'http://127.0.0.1:8081' }
$token = $env:ANTUMBRA_TOKEN
if (-not $token) {
    Write-Error "ANTUMBRA_TOKEN is not set (import needs a bearer token for the target tenant/user)."
    exit 1
}
$headers = @{ 'Content-Type' = 'application/json'; 'Authorization' = "Bearer $token" }

$data = @(Get-Content $In -Raw | ConvertFrom-Json)
$total = $data.Count
$ok = 0; $fail = 0; $i = 0

foreach ($m in $data) {
    $i++
    $mcpArgs = @{ content = [string]$m.content; network = [string]$m.network }
    if ($null -ne $m.confidence) { $mcpArgs['confidence'] = [double]$m.confidence }
    $body = @{ tool = 'store_memory'; arguments = $mcpArgs } | ConvertTo-Json -Compress -Depth 6
    try {
        Invoke-RestMethod -Method Post -Uri "$apiUrl/mcp/call" -Headers $headers -Body $body -TimeoutSec 60 | Out-Null
        $ok++
    } catch {
        $fail++
        if ($fail -le 5) { Write-Host ("  fail #{0} at record {1}: {2}" -f $fail, $i, $_.Exception.Message) }
    }
    if ($i % 200 -eq 0) { Write-Host ("  {0}/{1}  ({2} ok, {3} fail)" -f $i, $total, $ok, $fail) }
    if ($ThrottleMs -gt 0) { Start-Sleep -Milliseconds $ThrottleMs }
}

Write-Host ("done: {0} stored, {1} failed, of {2}" -f $ok, $fail, $total)

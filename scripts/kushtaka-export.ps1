# Export every Kushtakas memory in a workspace to the JSON shape that
# `antumbra memory-import --source <file>` expects (a JSON array of normalized
# MemoryRecord objects). One half of the Kushtakas -> Antumbra migration; the
# other half is `antumbra memory-import`.
#
# Talks to the Kushtakas MCP HTTP surface (POST /mcp/call), the same endpoint the
# SessionStart hook uses. Credentials come from the environment, so no secret is
# baked into the file:
#   KUSHTAKA_API_URL       MCP engine (default http://10.0.0.110:8081)
#   KUSHTAKA_API_KEY       sent as X-API-Key (required; the key scopes the workspace)
#   KUSHTAKA_WORKSPACE_ID  informational, recorded in the output header comment
#
# Usage:
#   pwsh scripts/kushtaka-export.ps1 -Out kushtaka-export.json
#   doppler run -- pwsh scripts/kushtaka-export.ps1   # when the key lives in Doppler
#
# Field mapping (Kushtakas -> MemoryRecord): content->content, network->network,
# strength->confidence, access_count->reinforcement, id->id. The remaining
# MemoryRecord fields (prompt/scope/marker/forbid/volatile/verifiable) are left
# absent so memory-import applies its documented defaults.

param(
    [string]$Out = "kushtaka-export.json",
    [int]$Limit = 100000,
    [string[]]$Networks = @('world', 'bank', 'opinion')
)

$ErrorActionPreference = 'Stop'

$apiUrl = if ($env:KUSHTAKA_API_URL) { $env:KUSHTAKA_API_URL } else { 'http://10.0.0.110:8081' }
$apiKey = $env:KUSHTAKA_API_KEY
if (-not $apiKey) {
    Write-Error "KUSHTAKA_API_KEY is not set (export needs the Kushtakas API key)."
    exit 1
}
$headers = @{ 'Content-Type' = 'application/json'; 'X-API-Key' = $apiKey }

$records = [System.Collections.Generic.List[object]]::new()
$seen = [System.Collections.Generic.HashSet[string]]::new()

foreach ($net in $Networks) {
    # A 1000-year window with a large limit pulls the whole network; ids are
    # de-duplicated defensively in case the server pages or overlaps.
    $payload = @{
        name      = 'get_recent_memories'
        arguments = @{ network = $net; hours = 8760000; limit = $Limit }
    } | ConvertTo-Json -Compress -Depth 5

    $resp = Invoke-RestMethod -Method Post -Uri "$apiUrl/mcp/call" `
        -Headers $headers -Body $payload -TimeoutSec 300
    $mems = @($resp.result.memories)

    $added = 0
    foreach ($m in $mems) {
        if ($m.id -and -not $seen.Add([string]$m.id)) { continue }
        $rec = [ordered]@{ content = [string]$m.content; network = [string]$m.network }
        if ($null -ne $m.strength) { $rec['confidence'] = [double]$m.strength }
        if ($null -ne $m.access_count) { $rec['reinforcement'] = [int]$m.access_count }
        if ($m.id) { $rec['id'] = [string]$m.id }
        $records.Add([pscustomobject]$rec)
        $added++
    }
    Write-Host ("  {0,-8} {1} memories" -f $net, $added)
}

# Serialize as a JSON array and write UTF-8 without a BOM (serde_json rejects a
# leading BOM), regardless of the PowerShell edition.
$json = ConvertTo-Json -InputObject @($records) -Depth 6
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($Out), $json)
Write-Host ("wrote {0} memories -> {1}" -f $records.Count, $Out)

# Antumbra SessionStart hook (Windows / PowerShell).
# Bootstraps a coding-agent session: pulls the agent's standing conventions and
# the memory most relevant to this project from a running antumbra-mcp surface,
# and returns it as `additionalContext`. Never blocks session start -- every
# failure degrades to an empty bootstrap.
#
# Targets a REST convenience endpoint POST {ANTUMBRA_URL}/mcp/call {tool,arguments}
# (roadmap P-1; see ../../docs/product.md). Antumbra's current networked surface is
# JSON-RPC at /mcp, so until P-1 ships either run a local shim or skip this script
# and have the agent call recall_memories at the top of its first turn.
param()

$url       = if ($env:ANTUMBRA_URL)          { $env:ANTUMBRA_URL }          else { 'http://127.0.0.1:8081' }
$workspace = if ($env:ANTUMBRA_WORKSPACE_ID) { $env:ANTUMBRA_WORKSPACE_ID } else { '' }
$token     = if ($env:ANTUMBRA_TOKEN)        { $env:ANTUMBRA_TOKEN }        else { '' }
$hostId    = if ($env:ANTUMBRA_HOST_ID)      { $env:ANTUMBRA_HOST_ID }      else { 'local' }

$headers = @{ 'Content-Type' = 'application/json' }
if ($token) { $headers['Authorization'] = "Bearer $token" }

# Recall the agent's standing conventions + project memory (semantic search).
$memText = ''
try {
    $payload = @{
        tool      = 'recall_memories'
        arguments = @{ query = 'standing conventions, project context, and active tasks for this agent'; limit = 12 }
    } | ConvertTo-Json -Compress -Depth 5
    $resp = Invoke-RestMethod -Method Post -Uri "$url/mcp/call" -Headers $headers -Body $payload -TimeoutSec 5 -ErrorAction Stop
    $mems = $resp.result.memories
    if ($mems -and $mems.Count -gt 0) {
        $memText = ($mems | ForEach-Object { $_.content }) -join "`n`n---`n`n"
    }
} catch { }

if (-not $memText) { $memText = '[Antumbra bootstrap empty / unreachable -- starting cold.]' }

$additionalContext = "# Antumbra session bootstrap (host=$hostId)`n`n$memText"

@{
    hookSpecificOutput = @{
        hookEventName     = 'SessionStart'
        additionalContext = $additionalContext
    }
} | ConvertTo-Json -Compress -Depth 10 | Write-Output

# UserPromptSubmit: recall the memories most relevant to what the user just
# typed and inject them as additionalContext, so recall happens on every message
# rather than only at session start.
#
# The bootstrap (antumbra-session-start) runs once and cannot know what the
# session will turn out to be about. This runs per prompt and does, which is why
# both exist.
#
# Env (see scripts/hooks/README.md):
#   ANTUMBRA_URL         the antumbra-mcp engine (default http://127.0.0.1:8081)
#   ANTUMBRA_TOKEN       bearer JWT, or
#   ANTUMBRA_TOKEN_FILE  a file holding it (default ~/.antumbra/token.txt)
#
# It never fails a prompt. Every failure path exits 0 with no output: a recall
# that cannot answer must not stop the user from talking to the agent.

$ErrorActionPreference = 'Stop'

$apiUrl = if ($env:ANTUMBRA_URL) { $env:ANTUMBRA_URL } else { 'http://127.0.0.1:8081' }
$token = if ($env:ANTUMBRA_TOKEN) { $env:ANTUMBRA_TOKEN } else {
    $file = if ($env:ANTUMBRA_TOKEN_FILE) { $env:ANTUMBRA_TOKEN_FILE }
            else { Join-Path $HOME '.antumbra/token.txt' }
    try { if (Test-Path $file) { (Get-Content $file -Raw).Trim() } else { '' } } catch { '' }
}
if (-not $token) { exit 0 }

try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $prompt = [string]$payload.prompt
    $cwd = [string]$payload.cwd
} catch { exit 0 }

if (-not $prompt -or $prompt.Trim().Length -lt 2) { exit 0 }

# Bound the query; the embedder has a 512-token window and a novel would be
# truncated into noise anyway.
$query = $prompt.Trim()
if ($query.Length -gt 500) { $query = $query.Substring(0, 500) }

# Where the caller is. The server scopes hits against this and, since scope
# reaches RETRIEVAL rather than only ordering, it also widens the candidate pool
# so a repo-anchored memory can actually compete. Absent git, it is simply
# omitted and recall stays global.
$repo = ''
$branch = ''
if ($cwd -and (Test-Path $cwd)) {
    try {
        Push-Location $cwd
        $origin = (git config --get remote.origin.url 2>$null)
        if ($origin) {
            $repo = ($origin -replace '^git@([^:]+):', 'https://$1/' -replace '\.git$', '') -replace '^https?://', ''
        }
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
    } catch { } finally { Pop-Location }
}

$memories = @()
try {
    $arguments = @{ query = $query; top_k = 3 }
    if ($repo) { $arguments['repo'] = [string]$repo }
    if ($branch -and $branch -ne 'HEAD') { $arguments['branch'] = [string]$branch }
    $body = @{ tool = 'recall_memories'; arguments = $arguments } | ConvertTo-Json -Compress -Depth 6
    $resp = Invoke-RestMethod -Method Post -Uri "$apiUrl/mcp/call" `
        -Headers @{ 'Content-Type' = 'application/json'; 'Authorization' = "Bearer $token" } `
        -Body $body -TimeoutSec 10 -ErrorAction Stop
    # recall_memories answers {"memories":[...]} at the TOP LEVEL -- there is no
    # `result` wrapper. Reading $resp.result.memories yields one $null element and
    # then throws on .content, which looks like a server fault and is not one.
    # Both shapes are accepted so a future wrapper would not silently empty this.
    if ($resp.memories) { $memories = @($resp.memories) }
    elseif ($resp.result.memories) { $memories = @($resp.result.memories) }
} catch { exit 0 }

if ($memories.Count -eq 0) { exit 0 }

$lines = foreach ($m in $memories) {
    $content = [string]$m.content
    if (-not $content) { continue }
    if ($content.Length -gt 900) {
        $content = $content.Substring(0, 900) + " [truncated - full entry: $($m.id)]"
    }
    # The scope tag says whether this was learned here: in_scope, other_branch,
    # other_repo, or absent when the caller gave no git context.
    $tag = if ($m.scope) { " [$($m.scope)]" } else { '' }
    "- ($($m.network))$tag $content"
}

if (-not $lines) { exit 0 }

$context = "## Antumbra recall (top matches for this prompt)`n`n" +
    ($lines -join "`n`n") +
    "`n`nThese are automatic; call recall_memories for a deeper or differently-phrased search."

@{
    hookSpecificOutput = @{
        hookEventName     = 'UserPromptSubmit'
        additionalContext = $context
    }
} | ConvertTo-Json -Compress -Depth 10

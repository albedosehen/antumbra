# Antumbra capture hook (Windows / PowerShell). Wire to BOTH Stop and PreCompact.
# Nudges the agent to deposit non-obvious observations into Antumbra via the
# `store_memory` MCP tool before the turn ends / context is compacted. A sentinel
# file makes it fire once per turn (no infinite loop). The verified traces it
# leaves are what `antumbra metabolize` later turns into a trained expert.
#
# Git-aware: inside a repository the nudge carries the exact provenance to stamp
# on memories about this code (repo slug, commit, branch), so a later session
# can tell whether each one still applies.
#
# At the end of a turn it also tells the server how the turn ended
# (`observe_turn`, with the turn's last answer), so the system can judge which
# of the memories it showed this session the answer drew on. Nobody has to
# reinforce or penalize by hand. Brief and fail-open: a server that is down or
# slower than two seconds leaves the turn unjudged and the stop unaffected.
$hookInput   = [System.Console]::In.ReadToEnd()
$json        = $hookInput | ConvertFrom-Json -ErrorAction SilentlyContinue
$sessionId   = if ($json.session_id) { $json.session_id } else { 'default' }
$event       = if ($json.hook_event_name) { $json.hook_event_name } else { 'Stop' }
$workspace   = if ($env:ANTUMBRA_WORKSPACE_ID) { $env:ANTUMBRA_WORKSPACE_ID } else { '<workspace>' }

$tmp = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
$sentinel = Join-Path $tmp "antumbra-capture-$event-$sessionId"
if (Test-Path $sentinel) { Remove-Item $sentinel -Force; exit 0 }
New-Item -ItemType File -Path $sentinel -Force | Out-Null

if ($event -eq 'Stop' -and $json.last_assistant_message) {
    $apiUrl = if ($env:ANTUMBRA_URL) { $env:ANTUMBRA_URL } else { 'http://127.0.0.1:8081' }
    $token = if ($env:ANTUMBRA_TOKEN) { $env:ANTUMBRA_TOKEN } else {
        $file = if ($env:ANTUMBRA_TOKEN_FILE) { $env:ANTUMBRA_TOKEN_FILE }
                else { Join-Path $HOME '.antumbra/token.txt' }
        try { if (Test-Path $file) { (Get-Content $file -Raw).Trim() } else { '' } } catch { '' }
    }
    if ($token) {
        try {
            Add-Type -AssemblyName System.Net.Http
            $text = [string]$json.last_assistant_message
            if ($text.Length -gt 4000) { $text = $text.Substring(0, 4000) }
            $arguments = @{ session = [string]$sessionId; text = $text }
            if ($json.prompt_id) { $arguments['turn'] = [string]$json.prompt_id }
            $body = @{ tool = 'observe_turn'; arguments = $arguments } | ConvertTo-Json -Compress -Depth 4
            $handler = New-Object System.Net.Http.HttpClientHandler
            $handler.UseProxy = $false
            $client = New-Object System.Net.Http.HttpClient($handler)
            try {
                $client.Timeout = [TimeSpan]::FromSeconds(2)
                $client.DefaultRequestHeaders.Authorization =
                    New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $token)
                # The machine, named as the MCP client names it
                # (antumbra-mcp-headers), so what this machine's agent recalled
                # for itself is judged against this turn.
                $hostId = if ($env:ANTUMBRA_HOST_ID) { $env:ANTUMBRA_HOST_ID.Trim() } else { '' }
                if ($hostId) { [void]$client.DefaultRequestHeaders.TryAddWithoutValidation('X-Antumbra-Host', $hostId) }
                $content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
                [void]$client.PostAsync("$apiUrl/mcp/call", $content).GetAwaiter().GetResult()
            } finally { $client.Dispose() }
        } catch { }
    }
}

# Where the session is, in git terms (fail-open: no repository, no anchor).
function Invoke-Git([string[]]$GitArgs) {
    try {
        $out = & git @GitArgs 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $text = ($out | Out-String).Trim()
        if ($text) { return $text } else { return $null }
    } catch { return $null }
}
function Get-RepoSlug([string]$remote) {
    if (-not $remote) { return '' }
    $s = $remote.Trim()
    $s = $s -replace '^[a-z+]+://', ''
    $s = $s -replace '^[^/@]*@', ''
    $s = $s -replace '^([^/:]+):', '$1/'
    $s = $s -replace '\.git/?$', ''
    $s = $s.TrimEnd('/').ToLowerInvariant()
    if ($s -notmatch '/' -or $s -match '\\' -or $s.StartsWith('/')) { return '' }
    return $s
}
$gitNote = ''
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Invoke-Git @('rev-parse', '--is-inside-work-tree'))) {
    $commit = Invoke-Git @('rev-parse', 'HEAD')
    $branch = Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')
    if ($branch -eq 'HEAD' -or -not $branch) { $branch = '' }
    $repo = Get-RepoSlug (Invoke-Git @('remote', 'get-url', 'origin'))
    if ($commit -and $repo) {
        $gitNote = " For any memory about this code, pass provenance {repo: `"$repo`", commit: `"$commit`", branch: `"$branch`"} (add path when it is about one file): that anchor is how a later session tells whether the memory still applies."
    }
}

$reason = "Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. " +
          "Call the store_memory MCP tool (workspace `"$workspace`") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). " +
          "Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. " +
          "Where a result was verified (a test passed, a command worked), note it. When the user states or corrects how an agent should act (a convention, a preference, a correction), also call record_behavior with rule (one sentence), must (regular expressions every answer that follows it matches), examples (at least four {task, answer} objects that follow it) and violations (plain-string answers that break it): behaviors are what the user's own expert learns." +
          $gitNote +
          " If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

@{ decision = 'block'; reason = $reason } | ConvertTo-Json -Compress -Depth 3 | Write-Output

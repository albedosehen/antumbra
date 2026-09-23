# Antumbra SessionStart hook (Windows / PowerShell).
# Bootstraps a coding-agent session: pulls the agent's standing conventions and
# the memory most relevant to this project from a running antumbra-mcp surface,
# and returns it as `additionalContext`. Never blocks session start -- every
# failure degrades to an empty bootstrap.
#
# Git-aware. Inside a repository it tells Antumbra where the session is (repo
# slug + branch) so recall scopes its hits, then checks each hit's git anchor
# with git itself -- is the commit still on HEAD, does the branch still exist --
# and tags it [live], [not-on-head], or [orphaned]. Nothing is re-extracted and
# nothing is garbage-collected: a stale memory is visible instead of silently
# wrong. Set ANTUMBRA_PENALIZE_ORPHANS=1 to also penalize orphaned memories.
#
# Talks to the REST convenience endpoint POST {ANTUMBRA_URL}/mcp/call
# {tool, arguments}, which dispatches the same tools as the JSON-RPC /mcp
# router under the same auth.
param()

$url       = if ($env:ANTUMBRA_URL)              { $env:ANTUMBRA_URL }              else { 'http://127.0.0.1:8081' }
# The bearer, from the environment or from a file. The file is the better home
# for it: a hook's environment usually comes from the agent's own settings file,
# which is shared, diffed and backed up, and a long-lived credential does not
# belong there. ANTUMBRA_TOKEN_FILE names it; ~/.antumbra/token.txt is the
# default so the common case needs no configuration at all. Read failures are
# swallowed deliberately -- a bootstrap that cannot authenticate says so by
# starting cold, and must never break the session it is opening.
$token = if ($env:ANTUMBRA_TOKEN) { $env:ANTUMBRA_TOKEN } else {
    $file = if ($env:ANTUMBRA_TOKEN_FILE) { $env:ANTUMBRA_TOKEN_FILE }
            else { Join-Path $HOME '.antumbra/token.txt' }
    try { if (Test-Path $file) { (Get-Content $file -Raw).Trim() } else { '' } } catch { '' }
}
$hostId    = if ($env:ANTUMBRA_HOST_ID)          { $env:ANTUMBRA_HOST_ID }          else { 'local' }
$penalize  = if ($env:ANTUMBRA_PENALIZE_ORPHANS) { $env:ANTUMBRA_PENALIZE_ORPHANS } else { '0' }

$headers = @{ 'Content-Type' = 'application/json' }
if ($token) { $headers['Authorization'] = "Bearer $token" }

# --- where the session is, in git terms (fail-open) -------------------------
function Invoke-Git([string[]]$GitArgs) {
    try {
        $out = & git @GitArgs 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $text = ($out | Out-String).Trim()
        if ($text) { return $text } else { return $null }
    } catch { return $null }
}
function Get-RepoSlug([string]$remote) {
    # The ssh, scp, and https spellings of one remote become one slug: host/org/name.
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

$repo = ''; $commit = ''; $branch = ''
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Invoke-Git @('rev-parse', '--is-inside-work-tree'))) {
    $commit = Invoke-Git @('rev-parse', 'HEAD')
    $branch = Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')
    if ($branch -eq 'HEAD') { $branch = '' }
    $repo = Get-RepoSlug (Invoke-Git @('remote', 'get-url', 'origin'))
    if (-not $commit) { $commit = '' }
    if (-not $branch) { $branch = '' }
}

# --- report this repository's recent merges (fail-open, detached) ------------
# A server GitHub cannot reach never hears that a branch merged, so what was
# learned on it reads other_branch from the base branch until someone says so.
# `antumbra claude reanchor` asks GitHub (with `gh`) for the last few days'
# merges and reports each one. It runs detached, so it never delays the session,
# and reporting a merge again moves nothing, so every session can run it. Off
# with ANTUMBRA_REANCHOR=0. The last run's output is in ~/.antumbra/reanchor.log,
# or wherever ANTUMBRA_REANCHOR_LOG names.
$bin = if ($env:ANTUMBRA_BIN) { $env:ANTUMBRA_BIN } else { 'antumbra' }
if ($repo -and $env:ANTUMBRA_REANCHOR -ne '0' -and (Get-Command $bin -ErrorAction SilentlyContinue)) {
    try {
        $env:ANTUMBRA_URL = $url
        if ($token) { $env:ANTUMBRA_TOKEN = $token }
        $log = if ($env:ANTUMBRA_REANCHOR_LOG) { $env:ANTUMBRA_REANCHOR_LOG } else { Join-Path $HOME '.antumbra/reanchor.log' }
        $logDir = Split-Path -Parent $log
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        # Started so that it holds none of this hook's handles. A process started
        # with its output redirected inherits the hook's own output pipe as well,
        # and the agent waits on that pipe, so the session would wait for the
        # report after all. On Windows cmd does the redirecting and the shell
        # starts cmd, which passes no handles; elsewhere sh backgrounds it with
        # every stream pointed at the log and returns at once.
        if ($env:OS -eq 'Windows_NT') {
            Start-Process -FilePath 'cmd.exe' -WindowStyle Hidden `
                -ArgumentList "/d /s /c `"`"$bin`" claude reanchor --days 3 >`"$log`" 2>&1`"" | Out-Null
        } else {
            & sh -c '"$0" claude reanchor --days 3 </dev/null >"$1" 2>&1 &' $bin $log
        }
    } catch { }
}

# --- recall, scoped to here when known --------------------------------------
$mems = @()
try {
    $arguments = @{ query = 'standing conventions, project context, and active tasks for this agent'; top_k = 12 }
    if ($repo)   { $arguments['repo']   = $repo }
    if ($branch) { $arguments['branch'] = $branch }
    $payload = @{ tool = 'recall_memories'; arguments = $arguments } | ConvertTo-Json -Compress -Depth 5
    $resp = Invoke-RestMethod -Method Post -Uri "$url/mcp/call" -Headers $headers -Body $payload -TimeoutSec 5 -ErrorAction Stop
    # The live /mcp/call answers with the tool's value at the TOP level
    # ({memories: [...]}); the .result envelope is tolerated for older shims.
    $mems = if ($resp.memories) { @($resp.memories) } elseif ($resp.result.memories) { @($resp.result.memories) } else { @() }
} catch { $mems = @() }

# --- judge each hit's anchor with git in hand --------------------------------
# One status per memory id: live | not-on-head | orphaned. Memories with no
# anchor, or from another repository, get none (the server's `scope` still shows).
$statuses = @{}
if ($commit) {
    foreach ($m in $mems) {
        $p = $m.provenance
        if (-not $p -or -not $p.commit -or $p.repo -ne $repo) { continue }
        $st = 'live'
        & git merge-base --is-ancestor $p.commit HEAD 2>$null
        if ($LASTEXITCODE -ne 0) { $st = 'not-on-head' }
        if ($p.branch -and $p.branch -ne $branch) {
            & git show-ref --verify --quiet "refs/heads/$($p.branch)" 2>$null
            $local = ($LASTEXITCODE -eq 0)
            & git show-ref --verify --quiet "refs/remotes/origin/$($p.branch)" 2>$null
            $remote = ($LASTEXITCODE -eq 0)
            if (-not $local -and -not $remote) { $st = 'orphaned' }
        }
        # The server already knows when GitHub deleted the branch (the App's delete
        # event marks the memory), even if this clone still has a stale local ref.
        if ($m.orphaned_at) { $st = 'orphaned' }
        $statuses[[string]$m.id] = $st
    }
}

# Optionally push the judgment back: an orphaned memory loses standing now,
# instead of waiting for someone to notice it was about a branch that is gone.
if ($penalize -eq '1') {
    foreach ($id in ($statuses.Keys | Where-Object { $statuses[$_] -eq 'orphaned' })) {
        try {
            $body = @{ tool = 'penalize_memory'; arguments = @{ memory_id = $id } } | ConvertTo-Json -Compress -Depth 4
            Invoke-RestMethod -Method Post -Uri "$url/mcp/call" -Headers $headers -Body $body -TimeoutSec 3 -ErrorAction Stop | Out-Null
        } catch { }
    }
}

# --- what is different about this session (fail-open) ---------------------------
# With its telemetry off the agent has also lost its feature flags, and the
# features gated on them, and nothing tells it (ADR-0021). `antumbra claude brief`
# prints a few lines when that is so and nothing when it is not. No antumbra on
# the path, or any failure: no lines.
$brief = ''
if (Get-Command $bin -ErrorAction SilentlyContinue) {
    try {
        $said = & $bin claude brief 2>$null
        if ($LASTEXITCODE -eq 0 -and $said) { $brief = ($said | Out-String).Trim() }
    } catch { $brief = '' }
}

# --- render ---------------------------------------------------------------------
$entries = @()
if ($mems.Count -gt 0) {
    $entries = @(foreach ($m in $mems) {
        $tags = @()
        if ($statuses.ContainsKey([string]$m.id)) { $tags += $statuses[[string]$m.id] }
        if ($m.scope) { $tags += [string]$m.scope }
        $prefix = if ($tags.Count -gt 0) { '[' + ($tags -join ', ') + '] ' } else { '' }
        $anchor = ''
        if ($m.provenance) {
            $anchor = "`n  (learned at $($m.provenance.repo)@$($m.provenance.commit)"
            if ($m.provenance.branch) { $anchor += "#$($m.provenance.branch)" }
            $anchor += ')'
        }
        "$prefix$($m.content)$anchor"
    })
}

$gitLine = ''
if ($commit) {
    $shownBranch = if ($branch) { $branch } else { '(detached)' }
    $shownRepo   = if ($repo)   { $repo }   else { '?' }
    $gitLine = "Git context: repo=$shownRepo branch=$shownBranch commit=$commit. " +
               "When storing a memory about this code, pass provenance {repo: `"$repo`", commit: `"$commit`", branch: `"$branch`"} to store_memory (add path for a single file) so a later session can tell whether it still applies. " +
               "Tags: [live] the anchor is on HEAD; [not-on-head] learned on a commit this HEAD does not contain; [orphaned] its branch no longer exists here or on origin, or GitHub reported it deleted -- verify before relying on it, and penalize_memory if it is wrong.`n`n"
}

# The agent caps a hook's context at 10,000 characters. Past the cap it is handed
# a file path and a 2,000-character preview it is never asked to open, so an
# oversized bootstrap is a truncated one that says nothing about it. What must
# survive goes first; memories follow, best first, while they fit, and the rest
# are counted so the agent knows to recall them.
$limit = 9500
$sep   = "`n`n---`n`n"
$head  = "# Antumbra session bootstrap (host=$hostId)`n`n"
if ($brief) { $head += "$brief`n`n" }
$head += $gitLine

$room = $limit - 160
$used = $head.Length
$kept = @()
$omitted = 0
foreach ($entry in $entries) {
    $cost = $entry.Length + $sep.Length
    if ($used + $cost -le $room) { $kept += $entry; $used += $cost } else { $omitted++ }
}
$memText = if ($entries.Count -eq 0) { '[Antumbra bootstrap empty / unreachable -- starting cold.]' } else { $kept -join $sep }
if ($omitted -gt 0) {
    $memText += "${sep}[$omitted more recalled but left out to stay under the 10,000-character limit on hook context; use recall_memories for them.]"
}

$additionalContext = "$head$memText"

@{
    hookSpecificOutput = @{
        hookEventName     = 'SessionStart'
        additionalContext = $additionalContext
    }
} | ConvertTo-Json -Compress -Depth 10 | Write-Output

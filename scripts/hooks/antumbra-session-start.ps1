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
#
# One wall-clock budget, ANTUMBRA_SESSION_BUDGET_SEC (default 8), measured from
# process start, bounds the whole hook and must stay below the hook's "timeout"
# in settings.json. Past the timeout the agent kills the hook and the session
# starts cold; telemetry showed exactly that when the hook made its three calls
# one after another with 5 + 3 + 5 seconds of timeouts between them, while
# every other session-starting process competed for the machine. The calls now
# go out together and whatever has not answered by the budget is left out.
param()

$clock = [System.Diagnostics.Stopwatch]::StartNew()
$budgetSec = 8
if ($env:ANTUMBRA_SESSION_BUDGET_SEC) { [void][int]::TryParse($env:ANTUMBRA_SESSION_BUDGET_SEC, [ref]$budgetSec) }
# What is kept back from the budget to judge anchors, render and print.
$reserveMs = 1200
function Get-RemainingMs { [Math]::Max(0, ($budgetSec * 1000) - $reserveMs - $clock.ElapsedMilliseconds) }

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
$bin       = if ($env:ANTUMBRA_BIN)              { $env:ANTUMBRA_BIN }              else { 'antumbra' }

# --- the calls to Antumbra (fail-open) ----------------------------------------
# HttpClient rather than Invoke-RestMethod. The server sends application/json
# with no charset, which Windows PowerShell's Invoke-RestMethod decodes as
# ISO-8859-1, so every em dash in a recalled memory reached the agent as
# mojibake; here the body is decoded as UTF-8. The proxy is bypassed because the
# engine is a LAN address and proxy discovery only spends the budget, and the
# connection limit is raised because .NET Framework allows two per host, which
# queued the third call behind the other two.
$client = $null
try {
    Add-Type -AssemblyName System.Net.Http
    [System.Net.ServicePointManager]::DefaultConnectionLimit = 8
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromMilliseconds([Math]::Max(1000, (Get-RemainingMs)))
    if ($token) {
        $client.DefaultRequestHeaders.Authorization =
            New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $token)
    }
} catch { $client = $null }

# Sends one call and returns its pending answer, or $null when it cannot.
function Start-Call([string]$Tool, [hashtable]$Arguments) {
    if (-not $client) { return $null }
    try {
        $body = @{ tool = $Tool; arguments = $Arguments } | ConvertTo-Json -Compress -Depth 5
        $content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
        return $client.PostAsync("$url/mcp/call", $content)
    } catch { return $null }
}

# A sent call's answer, parsed, if it lands inside the budget; $null for no
# call, no answer in time, a failure status, or a body that is not JSON.
function Receive-Call($Pending) {
    if (-not $Pending) { return $null }
    try {
        if (-not $Pending.Wait([int](Get-RemainingMs))) { return $null }
        $response = $Pending.Result
        if (-not $response.IsSuccessStatusCode) { return $null }
        $read = $response.Content.ReadAsByteArrayAsync()
        if (-not $read.Wait([int](Get-RemainingMs))) { return $null }
        return ([System.Text.Encoding]::UTF8.GetString($read.Result) | ConvertFrom-Json)
    } catch { return $null }
}

# --- handoffs waiting for this machine ---------------------------------
# What a session on another of the user's machines left for this one, announced
# until a session marks it done. The server writes the lines; this only places
# them. No answer, no line: it fails open like everything else here.
$handoffsCall = Start-Call 'handoffs' @{ host = $hostId }

# --- this machine, in the user's fabric --------------------------------------
# A server registers the machine it runs on, and a laptop talking to a hosted
# hub runs none, so the session names it: that lists it among the user's devices
# and marks when it was last seen. Skipped for `local`, the name of a machine
# nobody named. The answer is not used, and no answer changes nothing.
$registerCall = $null
if ($hostId.Trim() -ne 'local') { $registerCall = Start-Call 'register_device' @{ host = $hostId } }

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

# --- recall, scoped to here when known --------------------------------------
$arguments = @{ query = 'standing conventions, project context, and active tasks for this agent'; top_k = 12 }
if ($repo)   { $arguments['repo']   = $repo }
if ($branch) { $arguments['branch'] = $branch }
$recallCall = Start-Call 'recall_memories' $arguments

# --- the same, of each workspace this machine joined ------------------------
# A workspace (`antumbra join`) is asked with the token kept for it and never
# this machine's own: its calls go through a client of their own that carries
# no token, each request given the workspace's. At most three workspaces, at
# once with everything else and inside the one budget; one this machine already
# points at is the first ask, not a second. ANTUMBRA_RECALL_WORKSPACES=0 asks
# only your own.
$wsCalls = @()
$wsClient = $null
if ($client -and $env:ANTUMBRA_RECALL_WORKSPACES -ne '0') {
    try {
        $wsDir = if ($env:ANTUMBRA_WORKSPACES_DIR) { $env:ANTUMBRA_WORKSPACES_DIR }
                 else { Join-Path $HOME '.antumbra/workspaces' }
        if (Test-Path $wsDir) {
            $wsArguments = $arguments.Clone()
            $wsArguments.top_k = 5
            $wsBody = @{ tool = 'recall_memories'; arguments = $wsArguments } | ConvertTo-Json -Compress -Depth 5
            foreach ($dir in (Get-ChildItem -Path $wsDir -Directory | Sort-Object Name)) {
                if ($wsCalls.Count -ge 3) { break }
                $file = Join-Path $dir.FullName 'workspace.json'
                if (-not (Test-Path $file)) { continue }
                $ws = Get-Content $file -Raw | ConvertFrom-Json
                $wsUrl = ([string]$ws.url).TrimEnd('/')
                if (-not $wsUrl -or -not $ws.workspace -or $wsUrl -eq $url.TrimEnd('/')) { continue }
                if (-not $ws.token_file -or -not (Test-Path ([string]$ws.token_file))) { continue }
                $wsToken = (Get-Content ([string]$ws.token_file) -Raw).Trim()
                if (-not $wsToken) { continue }
                if (-not $wsClient) {
                    $wsHandler = New-Object System.Net.Http.HttpClientHandler
                    $wsHandler.UseProxy = $false
                    $wsClient = New-Object System.Net.Http.HttpClient($wsHandler)
                    $wsClient.Timeout = [TimeSpan]::FromMilliseconds([Math]::Max(1000, (Get-RemainingMs)))
                }
                $request = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$wsUrl/mcp/call")
                $request.Headers.Authorization = New-Object System.Net.Http.Headers.AuthenticationHeaderValue('Bearer', $wsToken)
                $request.Content = New-Object System.Net.Http.StringContent($wsBody, [System.Text.Encoding]::UTF8, 'application/json')
                $wsCalls += @{ name = [string]$ws.workspace; call = $wsClient.SendAsync($request) }
            }
        }
    } catch { }
}

# --- report this repository's recent merges (fail-open, detached) ------------
# A server GitHub cannot reach never hears that a branch merged, so what was
# learned on it reads other_branch from the base branch until someone says so.
# `antumbra claude reanchor` asks GitHub (with `gh`) for the last few days'
# merges and reports each one. It runs detached, so it never delays the session,
# and reporting a merge again moves nothing, so every session can run it. Off
# with ANTUMBRA_REANCHOR=0. The last run's output is in ~/.antumbra/reanchor.log,
# or wherever ANTUMBRA_REANCHOR_LOG names.
$binPath = (Get-Command $bin -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if ($repo -and $env:ANTUMBRA_REANCHOR -ne '0' -and $binPath) {
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

# --- what is different about this session (fail-open) ---------------------------
# With its telemetry off the agent has also lost its feature flags, and the
# features gated on them, and nothing tells it. `antumbra claude brief`
# prints a few lines when that is so and nothing when it is not. No antumbra on
# the path, any failure, or no answer inside the budget: no lines. It runs while
# the calls above are in flight, and its output is read as UTF-8. Off with
# ANTUMBRA_BRIEF=0, which is how the hook is run for an agent the brief is not
# about (`antumbra hook`).
$brief = ''
if ($binPath -and $env:ANTUMBRA_BRIEF -ne '0') {
    try {
        $start = New-Object System.Diagnostics.ProcessStartInfo
        $start.FileName = $binPath
        $start.Arguments = 'claude brief'
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        $start.RedirectStandardOutput = $true
        $start.RedirectStandardError = $true
        $start.StandardOutputEncoding = [System.Text.Encoding]::UTF8
        $proc = [System.Diagnostics.Process]::Start($start)
        $said = $proc.StandardOutput.ReadToEndAsync()
        $null = $proc.StandardError.ReadToEndAsync()
        if ($proc.WaitForExit([int](Get-RemainingMs)) -and $proc.ExitCode -eq 0) { $brief = $said.Result.Trim() }
        else { try { $proc.Kill() } catch { } }
    } catch { $brief = '' }
}

# --- keep the rules this mode runs by (fail-open, detached) ----------------------
# What `antumbra claude brief` just said is also kept as memories, in a
# `claude-code` compartment of the user's own: the console's Agent page lists
# them, and an agent can recall why a feature is missing. `antumbra claude
# remember` does that, and was a command for a person to run, which nobody
# did. It is safe every session: a rule already kept is left, and a changed one
# is stored and the old text penalized. Only in the mode the rules are for,
# which is when the brief says anything. Detached, like the merges, so it never
# delays the session. Off with ANTUMBRA_REMEMBER=0. Its last output is in
# ~/.antumbra/remember.log, or wherever ANTUMBRA_REMEMBER_LOG names.
if ($brief -and $env:ANTUMBRA_REMEMBER -ne '0' -and $binPath) {
    try {
        $env:ANTUMBRA_URL = $url
        if ($token) { $env:ANTUMBRA_TOKEN = $token }
        $log = if ($env:ANTUMBRA_REMEMBER_LOG) { $env:ANTUMBRA_REMEMBER_LOG } else { Join-Path $HOME '.antumbra/remember.log' }
        $logDir = Split-Path -Parent $log
        if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
        # Started as the merges' report is, holding none of this hook's handles.
        if ($env:OS -eq 'Windows_NT') {
            Start-Process -FilePath 'cmd.exe' -WindowStyle Hidden `
                -ArgumentList "/d /s /c `"`"$bin`" claude remember >`"$log`" 2>&1`"" | Out-Null
        } else {
            & sh -c '"$0" claude remember </dev/null >"$1" 2>&1 &' $bin $log
        }
    } catch { }
}

# --- the answers -------------------------------------------------------------------
$handoffs = ''
$resp = Receive-Call $handoffsCall
if ($resp) {
    $said = if ($resp.announcement) { $resp.announcement } elseif ($resp.result.announcement) { $resp.result.announcement } else { '' }
    if ($said) { $handoffs = "$said`n`n" }
}
# The live /mcp/call answers with the tool's value at the TOP level
# ({memories: [...]}); the .result envelope is tolerated for older shims. The
# @() is around the whole choice: an `if` unrolls what it yields, and Windows
# PowerShell gives the one memory left of a one-memory recall no Count, so that
# recall rendered as starting cold.
$mems = @()
$resp = Receive-Call $recallCall
if ($resp) {
    $mems = @(if ($resp.memories) { $resp.memories } elseif ($resp.result.memories) { $resp.result.memories })
}
# Not used, but waited for: a hook that exits first takes the call with it.
$null = Receive-Call $registerCall

# --- judge each hit's anchor with git in hand --------------------------------
# One status per memory id: live | not-on-head | orphaned. Memories with no
# anchor, or from another repository, get none (the server's `scope` still shows).
# Three git processes judge every anchor at once, where asking per memory took
# up to three each (36 for a full recall), and a session start is when each
# process costs the most.
$statuses = @{}
if ($commit) {
    $anchored = @($mems | Where-Object { $_.provenance -and $_.provenance.commit -and $_.provenance.repo -eq $repo })
    if ($anchored.Count -gt 0) {
        # Which anchors name a commit this clone has, by its full id.
        $wanted = @($anchored | ForEach-Object { [string]$_.provenance.commit } | Sort-Object -Unique)
        $full = @{}
        $checked = @($wanted | & git cat-file --batch-check 2>$null)
        for ($i = 0; $i -lt [Math]::Min($wanted.Count, $checked.Count); $i++) {
            $parts = ([string]$checked[$i]).Split(' ')
            if ($parts.Count -ge 2 -and $parts[1] -eq 'commit') { $full[$wanted[$i]] = $parts[0] }
        }
        # Which of those HEAD does not contain: listing what they reach that HEAD
        # does not names each one that is not an ancestor of HEAD.
        $offHead = @{}
        if ($full.Count -gt 0) {
            $reached = @(& git rev-list @($full.Values) --not HEAD 2>$null)
            if ($LASTEXITCODE -eq 0) {
                foreach ($c in $reached) { $offHead[[string]$c] = $true }
            } else {
                foreach ($c in $full.Values) {
                    & git merge-base --is-ancestor $c HEAD 2>$null
                    if ($LASTEXITCODE -ne 0) { $offHead[$c] = $true }
                }
            }
        }
        # Every branch this clone knows, here and on origin.
        $refs = @{}
        foreach ($r in @(& git for-each-ref '--format=%(refname)' refs/heads refs/remotes/origin 2>$null)) { $refs[[string]$r] = $true }
        foreach ($m in $anchored) {
            $p = $m.provenance
            $c = [string]$p.commit
            $st = if ($full.ContainsKey($c) -and -not $offHead.ContainsKey($full[$c])) { 'live' } else { 'not-on-head' }
            if ($p.branch -and $p.branch -ne $branch -and
                -not $refs.ContainsKey("refs/heads/$($p.branch)") -and
                -not $refs.ContainsKey("refs/remotes/origin/$($p.branch)")) { $st = 'orphaned' }
            # The server already knows when GitHub deleted the branch (the App's delete
            # event marks the memory), even if this clone still has a stale local ref.
            if ($m.orphaned_at) { $st = 'orphaned' }
            $statuses[[string]$m.id] = $st
        }
    }
}

# Optionally push the judgment back: an orphaned memory loses standing now,
# instead of waiting for someone to notice it was about a branch that is gone.
if ($penalize -eq '1') {
    $pending = @(foreach ($id in ($statuses.Keys | Where-Object { $statuses[$_] -eq 'orphaned' })) {
        Start-Call 'penalize_memory' @{ memory_id = $id }
    })
    foreach ($call in $pending) { $null = Receive-Call $call }
}
if ($client) { $client.Dispose() }

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

# What each workspace answered, after your own and marked as the workspace's.
# No anchor is judged for these: they were learned in someone else's clone.
foreach ($asked in $wsCalls) {
    $answer = Receive-Call $asked.call
    if (-not $answer) { continue }
    $theirs = if ($answer.memories) { @($answer.memories) } elseif ($answer.result.memories) { @($answer.result.memories) } else { @() }
    foreach ($m in $theirs) {
        $anchor = ''
        if ($m.provenance) {
            $anchor = "`n  (learned at $($m.provenance.repo)@$($m.provenance.commit)"
            if ($m.provenance.branch) { $anchor += "#$($m.provenance.branch)" }
            $anchor += ')'
        }
        $entries += "[workspace $($asked.name), not your own store] $($m.content)$anchor"
    }
}
if ($wsClient) { $wsClient.Dispose() }

$gitLine = ''
if ($commit) {
    $shownBranch = if ($branch) { $branch } else { '(detached)' }
    $shownRepo   = if ($repo)   { $repo }   else { '?' }
    $gitLine = "Git context: repo=$shownRepo branch=$shownBranch commit=$commit. " +
               "When storing a memory about this code, pass provenance {repo: `"$repo`", commit: `"$commit`", branch: `"$branch`"} to store_memory (add path for a single file) so a later session can tell whether it still applies. " +
               "Tags: [live] the anchor is on HEAD; [not-on-head] learned on a commit this HEAD does not contain; [orphaned] its branch no longer exists here or on origin, or GitHub reported it deleted -- verify before relying on it, and penalize_memory if it is wrong.`n`n"
}

# The workspaces this machine joined (`antumbra join`), so the agent knows
# what the labelled memories in its prompts are, whether it has a workspace's
# tools, and where what it learns is to go. Read from the files join keeps;
# no server is asked.
$joined = ''
try {
    $joinedDir = if ($env:ANTUMBRA_WORKSPACES_DIR) { $env:ANTUMBRA_WORKSPACES_DIR }
                 else { Join-Path $HOME '.antumbra/workspaces' }
    $joinedLines = @()
    if (Test-Path $joinedDir) {
        foreach ($dir in (Get-ChildItem -Path $joinedDir -Directory | Sort-Object Name)) {
            $file = Join-Path $dir.FullName 'workspace.json'
            if (-not (Test-Path $file)) { continue }
            $ws = Get-Content $file -Raw | ConvertFrom-Json
            if (-not $ws.url -or -not $ws.workspace) { continue }
            $server = [string]$ws.server
            $line = "- $($ws.workspace) at $($ws.url)"
            if ($ws.user) { $line += ", as $($ws.user)" }
            $line += '. '
            $line += if ($server) { "Its tools are the MCP server ``$server``: use them to search it further or to write to it. " }
                     else { 'You have no tools for it: you read what your prompts are given. ' }
            $line += if ($ws.write -eq 'workspace' -and $server) { "Store what you learn about this team's work there, with its store_memory; what is the user's own goes to your own store." }
                     elseif ($ws.write -eq 'ask') { "New memories go to your own store. When something you learn is the team's and not only the user's, ask the user whether to store it in the workspace too." }
                     elseif ($server) { 'New memories go to your own store. Its tools read and judge; a write to it is refused on this machine until the user turns that on (antumbra join --write ask, or workspace).' }
                     else { 'New memories go to your own store. Write to the workspace only when the user asks.' }
            $joinedLines += $line
        }
    }
    if ($joinedLines.Count -gt 0) {
        $joined = "Workspaces this machine joined. What each holds for a prompt comes with the prompt, labelled as the workspace's and not your own.`n" + ($joinedLines -join "`n")
    }
} catch { $joined = '' }

# The agent caps a hook's context at 10,000 characters. Past the cap it is handed
# a file path and a 2,000-character preview it is never asked to open, so an
# oversized bootstrap is a truncated one that says nothing about it. What must
# survive goes first; memories follow, best first, while they fit, and the rest
# are counted so the agent knows to recall them.
# ANTUMBRA_CONTEXT_LIMIT sets it lower, for an agent that takes less: Codex
# moves a hook's context to a file well before 10,000 characters.
$limit = 9500
if ($env:ANTUMBRA_CONTEXT_LIMIT -match '^\d+$') { $limit = [Math]::Max(1000, [int]$env:ANTUMBRA_CONTEXT_LIMIT) }
$sep   = "`n`n---`n`n"
$head  = "# Antumbra session bootstrap (host=$hostId)`n`n"
$head += $handoffs
if ($joined) { $head += "$joined`n`n" }
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
    $memText += "${sep}[$omitted more recalled but left out to stay under the limit on hook context; use recall_memories for them.]"
}

$additionalContext = "$head$memText"

$json = @{
    hookSpecificOutput = @{
        hookEventName     = 'SessionStart'
        additionalContext = $additionalContext
    }
} | ConvertTo-Json -Compress -Depth 10

# Written as UTF-8 bytes. Plain output goes through the console code page, which
# turns anything outside it into '?' before the agent ever reads it.
$out = [System.Text.UTF8Encoding]::new($false).GetBytes($json)
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($out, 0, $out.Length)
$stdout.Flush()

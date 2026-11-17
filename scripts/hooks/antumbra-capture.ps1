# Antumbra capture hook (Windows / PowerShell). Wire to BOTH Stop and PreCompact.
# Reminds the agent to deposit non-obvious observations into Antumbra via the
# `store_memory` MCP tool before what it learned is lost. The verified traces
# it leaves are what `antumbra metabolize` later turns into a trained expert.
#
# At the end of a turn (Stop) the reminder goes to the agent as feedback that
# keeps the turn going, not as a block: an agent reports a block to its user as
# a hook that failed, each time this one does what it is meant to. A sentinel
# file makes it fire once per turn (no infinite loop), and a turn's end is
# reminded only when a reminder is due: not when the agent has saved since the
# hook last looked, and not more often than every ANTUMBRA_CAPTURE_EVERY_SECS
# seconds. ANTUMBRA_CAPTURE_NUDGE=block answers with the block instead, for an
# agent that does not read the feedback (Claude Code before 2.1.163).
#
# Before a compaction (PreCompact) it blocks nothing. A block there stops the
# compaction, and one the agent needed to get under its context limit fails the
# request. It prints what the summary is to keep, which is what the agent uses
# a PreCompact hook's output for, and the turn's end after it is reminded.
#
# On any other event it says nothing.
#
# Git-aware: inside a repository the reminder carries the exact provenance to
# stamp on memories about this code (repo slug, commit, branch), so a later
# session can tell whether each one still applies.
#
# At the end of a turn it also tells the server how the turn ended
# (`observe_turn`, with the turn's last answer), so the system can judge which
# of the memories it showed this session the answer drew on. Nobody has to
# reinforce or penalize by hand. Brief and fail-open: a server that is down or
# slower than two seconds leaves the turn unjudged and the stop unaffected.
$hookInput   = [System.Console]::In.ReadToEnd()
$json        = $hookInput | ConvertFrom-Json -ErrorAction SilentlyContinue
$sessionId   = if ($json.session_id) { $json.session_id } else { 'default' }
$event       = if ($json.hook_event_name) { [string]$json.hook_event_name } else { 'Stop' }
$workspace   = if ($env:ANTUMBRA_WORKSPACE_ID) { $env:ANTUMBRA_WORKSPACE_ID } else { '<workspace>' }
if ($event -ne 'Stop' -and $event -ne 'PreCompact') { exit 0 }

# Written as UTF-8 bytes. Plain output goes through the console code page, which
# turns anything outside it into '?' before the agent ever reads it.
function Write-Answer([string]$Text) {
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($Text)
    $stdout = [Console]::OpenStandardOutput()
    $stdout.Write($bytes, 0, $bytes.Length)
    $stdout.Flush()
}

# The names of the files kept here are built from the hook's input, so the
# session id is cut to [A-Za-z0-9_-] before it names one, and they live under a
# directory only this user writes (the session runtime directory, else the
# per-user local data, else one under the home), not a shared temp directory.
$safeSession = [string]$sessionId -replace '[^A-Za-z0-9_-]', ''
if (-not $safeSession) { $safeSession = 'default' }
# One Join-Path inside another: Windows PowerShell 5.1 has no form that takes
# three parts, and with it the directory came out empty, no sentinel could be
# written, and every stop was nudged.
$runDir = if ($env:XDG_RUNTIME_DIR) { $env:XDG_RUNTIME_DIR }
          elseif ($env:LOCALAPPDATA) { Join-Path (Join-Path $env:LOCALAPPDATA 'antumbra') 'run' }
          else { Join-Path (Join-Path $HOME '.antumbra') 'run' }
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

# What is kept between two runs for one session: when it was last reminded or
# last saved (epoch seconds), and how far into the transcript this hook read.
$state = Join-Path $runDir "antumbra-capture-state-$safeSession"
$last = 0L; $mark = 0L
try {
    $kept = ((Get-Content -LiteralPath $state -Raw -ErrorAction Stop).Trim()) -split '\s+'
    $last = [long]$kept[0]; $mark = [long]$kept[1]
} catch { $last = 0L; $mark = 0L }

if ($event -eq 'PreCompact') {
    # The turn's end after this is reminded however lately the last one was:
    # by then the summary is where what was not stored is kept.
    try { Set-Content -LiteralPath $state -Value "0 $mark" -NoNewline } catch { }
    Write-Answer ("Antumbra capture, for this summary. Keep a section titled `"Not yet stored in Antumbra`". " +
        "List in it what this session learned that has not been stored with the store_memory or record_behavior tools: " +
        "decisions and the reasons for them, the user's corrections and preferences, conventions, bugs and incidents, deadlines, and results that were verified. " +
        "Give each enough detail to be stored after the compaction: what it is, the repository, branch and commit it concerns, and whether it was verified. " +
        "If everything was stored, say so in one line.")
    exit 0
}

$sentinel = Join-Path $runDir "antumbra-capture-Stop-$safeSession"
if (Test-Path $sentinel) { Remove-Item $sentinel -Force; exit 0 }
# A stop this hook's own reminder caused goes through, whatever became of the
# sentinel.
if ($json.stop_hook_active) { exit 0 }

if ($json.last_assistant_message) {
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

# Whether this stop is reminded. The end of a turn is not when the agent has
# saved since this hook last looked (the transcript holds a store_memory or
# record_behavior call past the mark kept from then), or when the session was
# reminded, or saved, within the last ANTUMBRA_CAPTURE_EVERY_SECS seconds (600
# unless set; 0 reminds every turn). A long job stops many times while it waits
# on something, and each reminder costs it a turn and says nothing new.
$every = 600
if ($env:ANTUMBRA_CAPTURE_EVERY_SECS -match '^\d+$') { $every = [long]$env:ANTUMBRA_CAPTURE_EVERY_SECS }
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$size = 0L; $saved = $false
$transcript = [string]$json.transcript_path
if ($transcript -and (Test-Path -LiteralPath $transcript)) {
    try {
        $stream = [System.IO.File]::Open($transcript, 'Open', 'Read', 'ReadWrite')
        try {
            $size = $stream.Length
            # Another transcript, or this one rewritten: read it from the start.
            if ($mark -gt $size) { $mark = 0L }
            # No further back than four megabytes, however long the session.
            if (($size - $mark) -gt 4MB) { $mark = $size - 4MB }
            [void]$stream.Seek($mark, 'Begin')
            $reader = New-Object System.IO.StreamReader($stream)
            # A tool call's own name, not one quoted inside a result.
            $saved = $reader.ReadToEnd() -match '"name"\s*:\s*"[^"]*(store_memory|record_behavior)"'
        } finally { $stream.Dispose() }
    } catch { }
}
$due = -not $saved -and ($every -eq 0 -or ($now - $last) -ge $every)
$stamp = if ($due -or $saved) { $now } else { $last }
try { Set-Content -LiteralPath $state -Value "$stamp $size" -NoNewline } catch { }
if (-not $due) { exit 0 }
New-Item -ItemType File -Path $sentinel -Force | Out-Null

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
        $gitNote = " For any memory about this code, pass provenance {repo: `"$repo`", commit: `"$commit`", branch: `"$branch`"} (add path when it is about one file): that anchor is how a later session tells whether the memory still applies. That is the repository this session started in; for work done in another, pass that one's own."
    }
}

$reminder = "Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. " +
            "Call the store_memory MCP tool (workspace `"$workspace`") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). " +
            "Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. " +
            "Where a result was verified (a test passed, a command worked), note it. When the user states or corrects how an agent should act (a convention, a preference, a correction), also call record_behavior with rule (one sentence), must (regular expressions every answer that follows it matches), examples (at least four {task, answer} objects that follow it) and violations (plain-string answers that break it): behaviors are what the user's own expert learns." +
            $gitNote +
            " If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

$answer = if ($env:ANTUMBRA_CAPTURE_NUDGE -eq 'block') {
    @{ decision = 'block'; reason = $reminder }
} else {
    @{ hookSpecificOutput = @{ hookEventName = 'Stop'; additionalContext = $reminder } }
}
Write-Answer ($answer | ConvertTo-Json -Compress -Depth 4)

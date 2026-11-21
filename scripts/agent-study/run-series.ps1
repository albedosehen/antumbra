#requires -Version 7
<#
.SYNOPSIS
Run one arm of the agent study (docs/agent-study.md) through one series.

.DESCRIPTION
A fresh checkout of the series' repository at its commit; the arm's settings,
and for the antumbra arm a throwaway Antumbra stack; then each session in
order: the prompt to `claude -p`, the series' conventions checked by program,
one correction turn for whatever was broken, the checks again, and the hidden
tests, which the agent never sees. Everything a session produced is written
under the run's folder, with a summary in run.json.

Nothing of the operator's own Claude Code reaches the arm: user settings and
hooks are not loaded (--setting-sources project,local), nor MCP servers
(--strict-mcp-config), nor instruction files under ~/.claude
(claudeMdExcludes), and Claude Code's own memory is off.

.EXAMPLE
pwsh scripts/agent-study/run-series.ps1 -Series scratch/agent-study/petrel-risk/series.json -Arm none -Run 1 -Out C:\study\runs -Base C:\study\petrel-base -TargetDir C:\study\target
#>
param(
    [Parameter(Mandatory)] [string] $Series,
    [Parameter(Mandatory)] [ValidateSet('none', 'antumbra', 'instructions')] [string] $Arm,
    [Parameter(Mandatory)] [int] $Run,
    [Parameter(Mandatory)] [string] $Out,
    # A local clone to copy from, so a run does not fetch the repository again.
    [string] $Base,
    # One cargo target directory shared by the runs, run one after another.
    [string] $TargetDir,
    [string] $Model = 'sonnet',
    [double] $MaxBudgetUsd = 3,
    # The image the antumbra arm's throwaway stack runs.
    [string] $Image = 'antumbra-mcp:local',
    # Where the antumbra arm's throwaway stack runs: an ssh alias, and the
    # address it answers on there. Without a host it runs here, in Docker.
    [string] $StudyHost,
    [string] $StudyAddress,
    [int] $StudyPort = 18081,
    # Only these session ids, for trying the harness out.
    [string[]] $Only
)

$ErrorActionPreference = 'Stop'
# The operator's own Antumbra, its address, token and workspace, reaches every
# process started from here through the environment, and the hooks take
# ANTUMBRA_TOKEN before the token file the arm names: the antumbra arm's hooks
# presented the operator's token to the throwaway stack and were refused. No
# arm inherits any of it; the antumbra arm is given its own in its settings.
Get-ChildItem Env: | Where-Object { $_.Name -like 'ANTUMBRA_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
# `pwsh -File` hands `-Only s1,s2` over as one string.
if ($Only) { $Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
$here = $PSScriptRoot
$antumbraRepo = (Resolve-Path (Join-Path $here '../..')).Path
$spec = Get-Content -Raw $Series | ConvertFrom-Json
$specDir = Split-Path -Parent (Resolve-Path $Series)
$runDir = Join-Path $Out "$($spec.name)/$Arm-r$Run"
if (Test-Path $runDir) { throw "run folder exists already: $runDir" }
New-Item -ItemType Directory -Force $runDir | Out-Null
$repo = Join-Path $runDir 'repo'
$today = Get-Date -Format 'yyyy-MM-dd'
if ($TargetDir) { $env:CARGO_TARGET_DIR = $TargetDir }

function Write-Step([string] $text) { Write-Host "[$($spec.name) $Arm r$Run] $text" }

function Save-Json($value, [string] $path) {
    $value | ConvertTo-Json -Depth 12 | Set-Content -Path $path -Encoding utf8
}

function Invoke-Native([string] $exe, [string[]] $arguments, [string] $cwd) {
    Push-Location $cwd
    try {
        $output = & $exe @arguments 2>&1 | Out-String
        return [pscustomobject]@{ code = $LASTEXITCODE; output = $output }
    } finally { Pop-Location }
}

# --- the checkout --------------------------------------------------------
Write-Step 'checking out'
git clone --quiet ($Base ? $Base : $spec.repo) $repo
git -C $repo checkout --quiet $spec.commit
git -C $repo remote remove origin
git -C $repo config user.name 'Study Operator'
git -C $repo config user.email 'study@example.invalid'

# --- the arm -------------------------------------------------------------
$home_ = $HOME
$settings = [ordered]@{
    autoMemoryEnabled = $false
    claudeMdExcludes  = @("$($home_ -replace '\\', '/')/.claude/**", "$home_\.claude\**")
    env               = [ordered]@{
        CLAUDE_CODE_USE_POWERSHELL_TOOL = '1'
        DISABLE_TELEMETRY               = '1'
    }
}
if ($TargetDir) { $settings.env.CARGO_TARGET_DIR = $TargetDir }
# cargo and git, and the read-only cmdlets and tools an agent pipes their
# output through; files change through Edit and Write alone. The same in
# every arm, so what one arm is refused, every arm is.
$readOnly = 'Select-String', 'Select-Object', 'Where-Object', 'ForEach-Object', 'Out-String',
    'Get-Content', 'Get-ChildItem', 'Measure-Object', 'Sort-Object', 'Format-Table', 'Test-Path'
$allowed = @('Read', 'Edit', 'Write', 'Glob', 'Grep',
    'PowerShell(cargo:*)', 'PowerShell(git:*)', 'Bash(cargo:*)', 'Bash(git:*)',
    'Bash(grep:*)', 'Bash(head:*)', 'Bash(tail:*)', 'Bash(cat:*)', 'Bash(ls:*)', 'Bash(wc:*)') +
    @($readOnly | ForEach-Object { "PowerShell($($_):*)" })
$mcpPath = $null
$stack = $null

if ($Arm -eq 'instructions') {
    Set-Content -Path (Join-Path $repo 'CLAUDE.md') -Value $spec.instructions -Encoding utf8 -NoNewline
    git -C $repo add CLAUDE.md
    git -C $repo commit --quiet -m 'docs: how to work in this repository'
}

if ($Arm -eq 'antumbra') {
    Write-Step 'starting a throwaway Antumbra stack'
    $a = Join-Path $runDir 'antumbra'
    New-Item -ItemType Directory -Force (Join-Path $a 'hooks'), (Join-Path $a 'workspaces'), (Join-Path $a 'surreal') | Out-Null
    $rand = { $b = [byte[]]::new(36); [Security.Cryptography.RandomNumberGenerator]::Fill($b); [Convert]::ToBase64String($b) -replace '[+/=]', 'x' }
    if ($StudyHost) {
        # On the study host, by remote-stack.sh: its own names, port and
        # in-memory store, so nothing the host runs is touched.
        $apiUrl = "http://$($StudyAddress):$StudyPort"
        $preamble = "export ACTION=up SURREAL_PASS='$(& $rand)' ANTUMBRA_JWT_SECRET='$(& $rand)$(& $rand)' IMAGE='$Image' BIND='$($StudyAddress):$StudyPort'`n"
        $said = ($preamble + (Get-Content -Raw (Join-Path $here 'remote-stack.sh'))) -replace "`r`n", "`n" | ssh -o BatchMode=yes $StudyHost bash -s 2>&1 | Out-String
        Write-Step "  $($said.Trim())"
        $stack = 'remote'
    } else {
        $apiUrl = 'http://127.0.0.1:8081'
        $envFile = Join-Path $a 'stack.env'
        @(
            "SURREAL_PASS=$(& $rand)",
            "ANTUMBRA_JWT_SECRET=$(& $rand)$(& $rand)",
            "ANTUMBRA_SURREAL_DATA=$((Join-Path $a 'surreal') -replace '\\', '/')",
            "ANTUMBRA_IMAGE=$Image"
        ) | Set-Content -Path $envFile -Encoding ascii
        $stack = @('compose', '-p', 'agentstudy', '--env-file', $envFile, '-f', (Join-Path $antumbraRepo 'docker/docker-compose.yml'))
        & docker @stack up -d --no-build surrealdb antumbra-mcp 2>&1 | Out-Null
    }
    $up = $false
    foreach ($i in 1..90) {
        try { $up = (Invoke-WebRequest "$apiUrl/dashboard" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200 } catch { Start-Sleep -Seconds 2 }
        if ($up) { break }
    }
    if (-not $up) { throw "the throwaway Antumbra stack did not answer at $apiUrl" }
    $token = if ($StudyHost) {
        (ssh -o BatchMode=yes $StudyHost docker exec agentstudy-mcp /usr/local/bin/antumbra-mcp --mint-token | Select-Object -Last 1).Trim()
    } else {
        (& docker exec antumbra-mcp /usr/local/bin/antumbra-mcp --mint-token | Select-Object -Last 1).Trim()
    }
    $tokenFile = Join-Path $a 'token.txt'
    Set-Content -Path $tokenFile -Value $token -NoNewline -Encoding ascii
    foreach ($h in 'antumbra-session-start.ps1', 'antumbra-prompt-recall.ps1', 'antumbra-capture.ps1') {
        Copy-Item (Join-Path $antumbraRepo "scripts/hooks/$h") (Join-Path $a "hooks/$h")
    }
    # The leading comma keeps the one-element list a list: returned bare, it is
    # unrolled to its element, written as an object, and Claude Code skips an
    # event whose hooks are not a list, without a word.
    $hook = { param($name, $timeout) , @(@{ matcher = ''; hooks = @(@{ type = 'command'; timeout = $timeout; command = "powershell -NoProfile -NonInteractive -File `"$(Join-Path $a "hooks/$name")`"" }) }) }
    $settings.hooks = [ordered]@{
        SessionStart     = & $hook 'antumbra-session-start.ps1' 10
        UserPromptSubmit = & $hook 'antumbra-prompt-recall.ps1' 20
        Stop             = & $hook 'antumbra-capture.ps1' 10
        PreCompact       = & $hook 'antumbra-capture.ps1' 10
    }
    $settings.env.ANTUMBRA_URL = $apiUrl
    $settings.env.ANTUMBRA_TOKEN_FILE = $tokenFile
    $settings.env.ANTUMBRA_WORKSPACE_ID = 'ws:default'
    $settings.env.ANTUMBRA_HOST_ID = 'study'
    $settings.env.ANTUMBRA_WORKSPACES_DIR = Join-Path $a 'workspaces'
    $settings.env.ANTUMBRA_REANCHOR = '0'
    # `claude -p` ends a turn on a Stop hook's feedback where an interactive
    # session goes on with it, so the reminder to store what was learned never
    # reached the agent and nothing was stored. The block answer keeps the
    # headless turn going, as the feedback keeps an interactive one going.
    $settings.env.ANTUMBRA_CAPTURE_NUDGE = 'block'
    $mcpPath = Join-Path $a 'mcp.json'
    Save-Json @{ mcpServers = @{ antumbra = @{ type = 'http'; url = "$apiUrl/mcp"; headers = @{ Authorization = "Bearer $token" } } } } $mcpPath
    $allowed += 'mcp__antumbra'
}

$settingsPath = Join-Path $runDir 'settings.json'
Save-Json $settings $settingsPath

function Invoke-Agent([string] $prompt, [string] $resume, [string] $label) {
    $arguments = @('-p', '--model', $Model, '--output-format', 'stream-json', '--verbose',
        '--setting-sources', 'project,local', '--settings', $settingsPath, '--strict-mcp-config',
        '--permission-mode', 'acceptEdits', '--max-budget-usd', "$MaxBudgetUsd",
        '--allowedTools') + $allowed
    if ($mcpPath) { $arguments += @('--mcp-config', $mcpPath) }
    if ($resume) { $arguments += @('--resume', $resume) }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    Push-Location $repo
    try { $raw = $prompt | & claude @arguments 2>$null | Out-String } finally { Pop-Location }
    $clock.Stop()
    # Every event of the session, one JSON object a line: the init (with each
    # MCP server's status), every tool call, and the result last.
    Set-Content -Path (Join-Path $runDir "$label.jsonl") -Value $raw -Encoding utf8
    $events = @($raw -split "`r?`n" | Where-Object { $_.Trim().StartsWith('{') } | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } | Where-Object { $_ })
    $init = $events | Where-Object { $_.type -eq 'system' -and $_.subtype -eq 'init' } | Select-Object -First 1
    $j = $events | Where-Object { $_.type -eq 'result' } | Select-Object -Last 1
    $tools = [ordered]@{}
    foreach ($e in $events | Where-Object { $_.type -eq 'assistant' }) {
        foreach ($part in @($e.message.content) | Where-Object { $_.type -eq 'tool_use' }) {
            $tools[$part.name] = 1 + ($tools[$part.name] ?? 0)
        }
    }
    [pscustomobject]@{
        session_id   = $j.session_id
        result       = $j.result
        is_error     = $j.is_error
        turns        = $j.num_turns
        cost_usd     = $j.total_cost_usd
        usage        = $j.usage
        denials      = @($j.permission_denials).Count
        mcp          = @($init.mcp_servers | ForEach-Object { "$($_.name):$($_.status)" })
        tools        = $tools
        wall_seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    }
}

function Test-Conventions([string] $start) {
    foreach ($c in $spec.conventions) {
        $ok = $true; $detail = ''
        switch ($c.check) {
            'ledger' {
                # The date is read from the file: a session after one that set
                # it to today has nothing to change there, and a diff alone
                # would call that broken. The row is a line added beside it.
                $rows = @(git -C $repo diff $start -- $c.path | Where-Object {
                        $_ -match '^\+(?!\+\+)' -and $_ -notmatch '^\+\*\*Last updated:\*\*' })
                $file = Join-Path $repo $c.path
                $dated = (Test-Path $file) -and ((Get-Content -Raw $file) -match "\*\*Last updated:\*\* $today")
                $ok = $dated -and $rows.Count -ge 1
                $detail = "$($rows.Count) lines added besides the date, dated today: $dated"
            }
            'commit' {
                $shas = @(git -C $repo rev-list "$start..HEAD")
                if ($shas.Count -eq 0) { $ok = $false; $detail = 'no commit' }
                foreach ($sha in $shas) {
                    $subject = git -C $repo log -1 --format=%s $sha
                    $message = (git -C $repo log -1 --format=%B $sha) -join "`n"
                    if ($subject -notmatch $c.subject) { $ok = $false; $detail += "subject: $subject; " }
                    if ($message -match $c.must_not) { $ok = $false; $detail += "trailer in $($sha.Substring(0, 7)); " }
                }
            }
            'commands' {
                foreach ($command in $c.commands) {
                    $parts = $command -split ' '
                    $r = Invoke-Native $parts[0] $parts[1..($parts.Count - 1)] $repo
                    if ($r.code -ne 0) { $ok = $false; $detail += "fails: $command; " }
                }
            }
            'diff_must_not' {
                $hits = @(git -C $repo diff $start -- $c.paths | Where-Object { $_ -match $c.pattern })
                $ok = $hits.Count -eq 0
                $detail = ($hits | Select-Object -First 3) -join ' | '
            }
            default { throw "unknown check $($c.check)" }
        }
        [pscustomobject]@{ id = $c.id; ok = $ok; detail = $detail.Trim(); correction = $c.correction }
    }
}

function Test-Hidden($session) {
    $tests = Join-Path $repo "crates/$($spec.crate)/tests"
    $made = -not (Test-Path $tests)
    New-Item -ItemType Directory -Force $tests | Out-Null
    $file = Join-Path $tests $session.hidden_test
    Copy-Item (Join-Path $specDir $session.hidden_test) $file
    $name = [IO.Path]::GetFileNameWithoutExtension($session.hidden_test)
    $r = Invoke-Native 'cargo' @('test', '-p', $spec.crate, '--test', $name) $repo
    Remove-Item $file
    if ($made) { Remove-Item $tests }
    Set-Content -Path (Join-Path $runDir "$($session.id)-hidden.log") -Value $r.output -Encoding utf8
    $r.code -eq 0
}

# --- the sessions --------------------------------------------------------
$claudeVersion = (& claude --version | Select-Object -First 1).Trim()
$results = @()
try {
    foreach ($s in $spec.sessions) {
        if ($Only -and $s.id -notin $Only) { continue }
        Write-Step "session $($s.id)"
        $start = (git -C $repo rev-parse HEAD).Trim()
        $session = Invoke-Agent $s.prompt $null $s.id
        $first = @(Test-Conventions $start)
        $broken = @($first | Where-Object { -not $_.ok })
        $correction = $null
        $after = $first
        if ($broken.Count -gt 0) {
            Write-Step "  correcting: $(($broken | ForEach-Object { $_.id }) -join ', ')"
            $message = ($broken | ForEach-Object { $_.correction }) -join "`n`n"
            $correction = Invoke-Agent $message $session.session_id "$($s.id)-correction"
            $after = @(Test-Conventions $start)
        }
        $passed = Test-Hidden $s
        $results += [pscustomobject]@{
            session          = $s.id
            broken_first     = @($broken | ForEach-Object { $_.id })
            broken_after     = @($after | Where-Object { -not $_.ok } | ForEach-Object { $_.id })
            checks_first     = $first | Select-Object id, ok, detail
            checks_after     = $after | Select-Object id, ok, detail
            hidden_tests     = $passed
            agent            = $session
            correction       = $correction
        }
        Write-Step "  broken $(@($broken).Count) of $(@($first).Count), after correction $(@($after | Where-Object { -not $_.ok }).Count); hidden tests: $passed"
        Save-Json $results (Join-Path $runDir 'sessions.json')
    }
} finally {
    if ($Arm -eq 'antumbra' -and $stack) {
        try {
            $call = { param($tool, $arguments) Invoke-RestMethod -Method Post "$apiUrl/mcp/call" -Headers @{ Authorization = "Bearer $token" } -ContentType 'application/json' -Body (@{ tool = $tool; arguments = $arguments } | ConvertTo-Json -Depth 5) -TimeoutSec 30 }
            Save-Json (& $call 'list_memories' @{ limit = 200 }) (Join-Path $runDir 'antumbra-memories.json')
            Save-Json (& $call 'list_behaviors' @{}) (Join-Path $runDir 'antumbra-behaviors.json')
        } catch { Write-Step "could not read the store back: $($_.Exception.Message)" }
        if ($stack -eq 'remote') {
            $said = ("export ACTION=down`n" + (Get-Content -Raw (Join-Path $here 'remote-stack.sh'))) -replace "`r`n", "`n" | ssh -o BatchMode=yes $StudyHost bash -s 2>&1 | Out-String
            Write-Step "  $($said.Trim())"
        } else {
            & docker @stack down -v 2>&1 | Out-Null
        }
    }
}

$sum = { param($property) ($results | ForEach-Object { $_.agent.$property; $_.correction.$property } | Where-Object { $_ } | Measure-Object -Sum).Sum }
$summary = [ordered]@{
    series                   = $spec.name
    commit                   = $spec.commit
    arm                      = $Arm
    run                      = $Run
    model                    = $Model
    claude                   = $claudeVersion
    finished                 = (Get-Date).ToString('o')
    sessions                 = $results.Count
    broken_first_total       = ($results | ForEach-Object { $_.broken_first.Count } | Measure-Object -Sum).Sum
    broken_first_after_first = ($results | Select-Object -Skip 1 | ForEach-Object { $_.broken_first.Count } | Measure-Object -Sum).Sum
    broken_after_correction  = ($results | ForEach-Object { $_.broken_after.Count } | Measure-Object -Sum).Sum
    hidden_tests_passed      = @($results | Where-Object { $_.hidden_tests }).Count
    cost_usd                 = [math]::Round((& $sum 'cost_usd'), 4)
    turns                    = & $sum 'turns'
    wall_seconds             = [math]::Round((& $sum 'wall_seconds'), 1)
    permission_denials       = & $sum 'denials'
}
Save-Json $summary (Join-Path $runDir 'run.json')
Write-Step "done: $($summary | ConvertTo-Json -Compress)"

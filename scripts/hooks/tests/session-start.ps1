# Tests for antumbra-session-start.ps1. No Antumbra needed: the surface is a
# listener in this process answering from a file, and `antumbra` is a stub.
# On Windows every case runs twice: under pwsh, and under Windows PowerShell,
# which is what the agent's settings run the hook with.
#
#   pwsh -NoProfile -File scripts/hooks/tests/session-start.ps1
$ErrorActionPreference = 'Stop'
# The hook writes UTF-8; read it as UTF-8, so what the agent would see is what
# the checks see.
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch { }
$hook = Join-Path $PSScriptRoot '..' 'antumbra-session-start.ps1'
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("antumbra-hook-test-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$script:failed = 0

function Check([string]$What, [bool]$Holds) {
    if ($Holds) { Write-Output "  ok    $What" } else { Write-Output "  FAIL  $What"; $script:failed = 1 }
}

# --- a stub `antumbra` that knows `claude brief` and `claude reanchor` -------------
# `reanchor` writes its arguments, and whether it was handed the token, to
# ANTUMBRA_TEST_MARKER, then takes five seconds, so a hook that waited for it
# would be caught.
$marker = Join-Path $work 'reanchored.txt'
if ($IsWindows) {
    $stub = Join-Path $work 'antumbra.cmd'
    Set-Content -Path $stub -Value (@(
        '@echo off'
        'if "%1 %2"=="claude brief" ('
        'echo ## Sovereign mode'
        'echo.'
        'echo - a line the agent must see'
        ')'
        'if "%1 %2"=="claude reanchor" ('
        '>"%ANTUMBRA_TEST_MARKER%" echo %*'
        'if defined ANTUMBRA_TOKEN >>"%ANTUMBRA_TEST_MARKER%" echo token handed on'
        'ping -n 6 127.0.0.1 >nul'
        ')'
    ) -join "`r`n")
} else {
    $stub = Join-Path $work 'antumbra'
    Set-Content -Path $stub -Value (@(
        '#!/usr/bin/env bash'
        'if [ "$1 $2" = "claude brief" ]; then printf ''%s\n'' ''## Sovereign mode'' '''' ''- a line the agent must see''; fi'
        'if [ "$1 $2" = "claude reanchor" ]; then echo "$*" > "$ANTUMBRA_TEST_MARKER"; [ -n "$ANTUMBRA_TOKEN" ] && echo "token handed on" >> "$ANTUMBRA_TEST_MARKER"; sleep 5; fi'
    ) -join "`n")
    & chmod +x $stub
}
$env:ANTUMBRA_TEST_MARKER = $marker
$env:ANTUMBRA_TOKEN = 'test-token'
$env:ANTUMBRA_REANCHOR_LOG = Join-Path $work 'reanchor.log'

# A clone with a GitHub origin, for the cases that need a repository.
$clone = Join-Path $work 'clone'
New-Item -ItemType Directory -Path $clone | Out-Null
& git -C $clone init -q
& git -C $clone remote add origin https://github.com/acme/orders.git

# --- a surface that answers every call with the current response file ------------
# It also writes each request's body to the calls file, one line a call, so a
# test can say which tools the hook asked for.
$response = Join-Path $work 'response.json'
Set-Content -Path $response -Value '{}'
$calls = Join-Path $work 'calls.txt'
$port = Get-Random -Minimum 20000 -Maximum 40000
$url = "http://127.0.0.1:$port"
# The listener is made here and handed to the thread, so this script can stop it:
# a thread left waiting for a call would keep the process from ever exiting.
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("$url/")
$listener.Start()
$null = Start-ThreadJob -ArgumentList $listener, $response, $calls -ScriptBlock {
    param($listener, $responseFile, $callsFile)
    try {
        while ($listener.IsListening) {
            $call = $listener.GetContext()
            $body = [System.IO.StreamReader]::new($call.Request.InputStream).ReadToEnd()
            [System.IO.File]::AppendAllText($callsFile, ($body -replace '\s+', ' ') + "`n")
            $bytes = [System.IO.File]::ReadAllBytes($responseFile)
            $call.Response.ContentType = 'application/json'
            $call.Response.OutputStream.Write($bytes, 0, $bytes.Length)
            $call.Response.Close()
        }
    } catch { }
}
$up = $false
foreach ($attempt in 1..50) {
    try { Invoke-RestMethod -Method Post -Uri "$url/mcp/call" -Body '{}' -TimeoutSec 1 | Out-Null; $up = $true; break }
    catch { Start-Sleep -Milliseconds 100 }
}
if (-not $up) { Write-Output 'the test surface did not start'; exit 2 }

function Get-Context([string]$ResponseJson, [string]$Bin, [string]$Dir = $work) {
    Set-Content -Path $response -Value $ResponseJson
    $env:ANTUMBRA_URL = $url
    $env:ANTUMBRA_BIN = $Bin
    Push-Location $Dir
    try { $out = & $script:shell -NoProfile -NonInteractive -File $hook | Out-String } finally { Pop-Location }
    $answer = $out | ConvertFrom-Json
    if ($answer.hookSpecificOutput.hookEventName -ne 'SessionStart') { throw "not a SessionStart answer: $out" }
    return [string]$answer.hookSpecificOutput.additionalContext
}

# Twelve memories of 1,500 characters: 18,000 in all, nearly twice the limit.
$large = @{ memories = @(1..12 | ForEach-Object { @{ id = "memory:$_"; content = "M${_}:" + ('x' * 1500) } }) } | ConvertTo-Json -Depth 5 -Compress
$small = @{ memories = @(@{ id = 'memory:a'; content = 'first small' }, @{ id = 'memory:b'; content = 'second small' }) } | ConvertTo-Json -Depth 5 -Compress
# Text outside ASCII and Latin-1, as memories hold it. Built from code points so
# this file's own encoding cannot decide the case.
$wide = "em dash " + [char]0x2014 + " u-umlaut " + [char]0x00FC + " kanji " + [char]0x65E5 + [char]0x672C
$unicode = @{ memories = @(@{ id = 'memory:u'; content = $wide }) } | ConvertTo-Json -Depth 5 -Compress

# A repository with history, for judging anchors: `main` is a, then c; `feature`
# branched at a and holds b, unmerged.
$history = Join-Path $work 'history'
New-Item -ItemType Directory -Path $history | Out-Null
function Commit([string]$Message) {
    & git -C $history -c user.name=t -c user.email=t@example.com commit -q --allow-empty -m $Message
    (& git -C $history rev-parse HEAD).Trim()
}
& git -C $history init -q
& git -C $history checkout -q -b main
& git -C $history remote add origin https://github.com/acme/orders.git
$onMain = Commit 'a'
& git -C $history checkout -q -b feature
$onFeature = Commit 'b'
& git -C $history checkout -q main
$null = Commit 'c'
function Anchored([string]$Id, [string]$Content, [string]$Commit, [string]$Branch, [string]$Repo = 'github.com/acme/orders') {
    @{ id = $Id; content = $Content; provenance = @{ repo = $Repo; commit = $Commit; branch = $Branch } }
}
$anchors = @{ memories = @(
    (Anchored 'memory:1' 'anchor-live' $onMain 'main'),
    (Anchored 'memory:2' 'anchor-offhead' $onFeature 'feature'),
    (Anchored 'memory:3' 'anchor-orphaned' $onMain 'gone'),
    (Anchored 'memory:4' 'anchor-unknown' '0123456789abcdef0123456789abcdef01234567' 'main'),
    (Anchored 'memory:5' 'anchor-short' $onMain.Substring(0, 7) 'main'),
    (Anchored 'memory:6' 'anchor-elsewhere' $onMain 'main' 'github.com/acme/other')
) } | ConvertTo-Json -Depth 6 -Compress

# A surface that takes the connection and never answers.
$silent = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$silent.Start()
$silentUrl = "http://127.0.0.1:$($silent.LocalEndpoint.Port)"

function Invoke-Cases {
Write-Output 'an oversized recall'
$ctx = Get-Context $large $stub
Check "stays under the agent's 10,000-character limit ($($ctx.Length))" ($ctx.Length -le 10000)
Check 'keeps the best memory' $ctx.Contains('M1:x')
$kept = ([regex]::Matches($ctx, 'M[0-9]+:x')).Count
$found = [regex]::Match($ctx, '\[([0-9]+) more recalled but left out')
$omitted = if ($found.Success) { [int]$found.Groups[1].Value } else { 0 }
Check "says how many it left out, and none go missing (kept $kept, left out $omitted)" (($kept + $omitted) -eq 12)
Check 'leaves some out' ($omitted -gt 0)
$briefAt = $ctx.IndexOf('Sovereign mode')
Check 'says what is different about the session before any memory' (($briefAt -ge 0) -and ($briefAt -lt $ctx.IndexOf('M1:x')))

Write-Output 'a small recall'
$ctx = Get-Context $small $stub
Check 'keeps everything' $ctx.Contains('second small')
Check 'mentions no omission' (-not $ctx.Contains('left out'))

Write-Output 'a handoff waiting for this machine'
$handoff = @{
    memories     = @(@{ id = 'memory:a'; content = 'first small' })
    announcement = "1 handoff waiting for this machine (windows):`n- Rerun the probe (from kuskokwim, 2h ago; id memory:h)"
} | ConvertTo-Json -Depth 5 -Compress
$ctx = Get-Context $handoff $stub
Check 'announces it' $ctx.Contains('1 handoff waiting for this machine (windows):')
Check 'puts it before any memory' (($ctx.IndexOf('handoff waiting') -ge 0) -and ($ctx.IndexOf('handoff waiting') -lt $ctx.IndexOf('first small')))
$ctx = Get-Context $small $stub
Check 'says nothing of handoffs when none wait' (-not $ctx.Contains('handoff'))

Write-Output 'naming this machine'
$hostBefore = $env:ANTUMBRA_HOST_ID
$env:ANTUMBRA_HOST_ID = 'mac'
Set-Content -Path $calls -Value ''
$ctx = Get-Context $small $stub
$asked = @(Get-Content $calls | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
$named = @($asked | Where-Object { $_.tool -eq 'register_device' })
Check 'registers it under its name' (($named.Count -eq 1) -and ($named[0].arguments.host -eq 'mac'))
Check 'still answers' $ctx.Contains('first small')
$env:ANTUMBRA_HOST_ID = ''
Set-Content -Path $calls -Value ''
$ctx = Get-Context $small $stub
$asked = @(Get-Content $calls | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
Check 'registers nothing when the machine has no name' (@($asked | Where-Object { $_.tool -eq 'register_device' }).Count -eq 0)
Check 'still asks for its handoffs' (@($asked | Where-Object { $_.tool -eq 'handoffs' }).Count -eq 1)
$env:ANTUMBRA_HOST_ID = $hostBefore

Write-Output 'no antumbra on the path'
$ctx = Get-Context $small 'antumbra-is-not-installed'
Check 'still answers' $ctx.Contains('first small')
Check 'says nothing about the session' (-not $ctx.Contains('Sovereign mode'))

Write-Output 'no memories at all'
$ctx = Get-Context '{}' $stub
Check 'starts cold and says so' $ctx.Contains('starting cold')
Check 'still says what is different about the session' $ctx.Contains('Sovereign mode')

Write-Output 'outside a repository'
Check 'reports no merges' (-not (Test-Path $marker))

Write-Output 'in a clone'
$clock = [System.Diagnostics.Stopwatch]::StartNew()
$ctx = Get-Context $small $stub $clone
$took = $clock.Elapsed.TotalSeconds
Check "does not wait for the report ($([math]::Round($took, 1)) s, the stub takes 5)" ($took -lt 4)
foreach ($wait in 1..30) { if (Test-Path $marker) { break }; Start-Sleep -Milliseconds 200 }
$said = if (Test-Path $marker) { Get-Content -Raw $marker } else { '' }
Check 'reports the last three days of merges' $said.Contains('claude reanchor --days 3')
Check 'hands the report the token' $said.Contains('token handed on')
Check 'still answers' $ctx.Contains('first small')

Write-Output 'in a clone, turned off'
Remove-Item -Force $marker -ErrorAction SilentlyContinue
$env:ANTUMBRA_REANCHOR = '0'
$ctx = Get-Context $small $stub $clone
Start-Sleep -Seconds 2
Check 'reports no merges' (-not (Test-Path $marker))

Write-Output 'text outside ASCII'
$ctx = Get-Context $unicode $stub
Check 'arrives as it was stored' $ctx.Contains($wide)

Write-Output 'anchors judged against history'
$ctx = Get-Context $anchors $stub $history
Check 'an anchor on HEAD is live' $ctx.Contains('[live] anchor-live')
Check 'an unmerged commit is not on HEAD' $ctx.Contains('[not-on-head] anchor-offhead')
Check 'a deleted branch is orphaned' $ctx.Contains('[orphaned] anchor-orphaned')
Check 'a commit this clone lacks is not on HEAD' $ctx.Contains('[not-on-head] anchor-unknown')
Check 'a short commit id is judged too' $ctx.Contains('[live] anchor-short')
Check 'another repository gets no tag' (-not $ctx.Contains('] anchor-elsewhere'))
Remove-Item Env:ANTUMBRA_REANCHOR

Write-Output 'a surface that never answers'
$env:ANTUMBRA_SESSION_BUDGET_SEC = '3'
$env:ANTUMBRA_URL = $silentUrl
$clock = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $work
try { $out = & $script:shell -NoProfile -NonInteractive -File $hook | Out-String } finally { Pop-Location }
$took = $clock.Elapsed.TotalSeconds
$ctx = [string]($out | ConvertFrom-Json).hookSpecificOutput.additionalContext
Check "answers inside its budget ($([math]::Round($took, 1)) s, budget 3)" ($took -lt 4.5)
Check 'starts cold and says so' $ctx.Contains('starting cold')
Remove-Item Env:ANTUMBRA_SESSION_BUDGET_SEC
}

$shells = @('pwsh')
if ($IsWindows -and (Get-Command powershell -ErrorAction SilentlyContinue)) { $shells += 'powershell' }
foreach ($name in $shells) {
    $script:shell = $name
    Write-Output "== under $name"
    Invoke-Cases
}

$silent.Stop()
$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

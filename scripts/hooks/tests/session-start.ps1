# Tests for antumbra-session-start.ps1. No Antumbra needed: the surface is a
# listener in this process answering from a file, and `antumbra` is a stub.
#
#   pwsh -NoProfile -File scripts/hooks/tests/session-start.ps1
$ErrorActionPreference = 'Stop'
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
$response = Join-Path $work 'response.json'
Set-Content -Path $response -Value '{}'
$port = Get-Random -Minimum 20000 -Maximum 40000
$url = "http://127.0.0.1:$port"
# The listener is made here and handed to the thread, so this script can stop it:
# a thread left waiting for a call would keep the process from ever exiting.
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("$url/")
$listener.Start()
$null = Start-ThreadJob -ArgumentList $listener, $response -ScriptBlock {
    param($listener, $responseFile)
    try {
        while ($listener.IsListening) {
            $call = $listener.GetContext()
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
    try { $out = & pwsh -NoProfile -NonInteractive -File $hook | Out-String } finally { Pop-Location }
    $answer = $out | ConvertFrom-Json
    if ($answer.hookSpecificOutput.hookEventName -ne 'SessionStart') { throw "not a SessionStart answer: $out" }
    return [string]$answer.hookSpecificOutput.additionalContext
}

# Twelve memories of 1,500 characters: 18,000 in all, nearly twice the limit.
$large = @{ memories = @(1..12 | ForEach-Object { @{ id = "memory:$_"; content = "M${_}:" + ('x' * 1500) } }) } | ConvertTo-Json -Depth 5 -Compress
$small = @{ memories = @(@{ id = 'memory:a'; content = 'first small' }, @{ id = 'memory:b'; content = 'second small' }) } | ConvertTo-Json -Depth 5 -Compress

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
Remove-Item Env:ANTUMBRA_REANCHOR

$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

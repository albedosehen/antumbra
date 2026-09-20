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

# --- a stub `antumbra` that only knows `claude brief` ----------------------------
if ($IsWindows) {
    $stub = Join-Path $work 'antumbra.cmd'
    Set-Content -Path $stub -Value "@echo off`r`nif `"%1 %2`"==`"claude brief`" (`r`necho ## Sovereign mode`r`necho.`r`necho - a line the agent must see`r`n)"
} else {
    $stub = Join-Path $work 'antumbra'
    Set-Content -Path $stub -Value "#!/usr/bin/env bash`n[ `"`$1 `$2`" = `"claude brief`" ] && printf '%s\n' '## Sovereign mode' '' '- a line the agent must see'`n"
    & chmod +x $stub
}

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

function Get-Context([string]$ResponseJson, [string]$Bin) {
    Set-Content -Path $response -Value $ResponseJson
    $env:ANTUMBRA_URL = $url
    $env:ANTUMBRA_BIN = $Bin
    Push-Location $work
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

$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

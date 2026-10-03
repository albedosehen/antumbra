# Tests for antumbra-prompt-recall.ps1. No Antumbra needed: the surface is a
# listener in this process answering from a file, after a delay read from
# another, so a slow server can be stood in for.
#
#   pwsh -NoProfile -File scripts/hooks/tests/prompt-recall.ps1
$ErrorActionPreference = 'Stop'
$hook = if ($env:ANTUMBRA_TEST_HOOK) { $env:ANTUMBRA_TEST_HOOK } else { Join-Path $PSScriptRoot '..' 'antumbra-prompt-recall.ps1' }
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("antumbra-hook-test-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$script:failed = 0
# The hook writes UTF-8 bytes; read them as such.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Check([string]$What, [bool]$Holds) {
    if ($Holds) { Write-Output "  ok    $What" } else { Write-Output "  FAIL  $What"; $script:failed = 1 }
}

# --- a surface that answers every call with the response file, after the delay --
$response = Join-Path $work 'response.json'
$delay = Join-Path $work 'delay.txt'
Set-Content -Path $response -Value '{}'
Set-Content -Path $delay -Value '0'
$port = Get-Random -Minimum 20000 -Maximum 40000
$url = "http://127.0.0.1:$port"
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("$url/")
$listener.Start()
$null = Start-ThreadJob -ArgumentList $listener, $response, $delay -ScriptBlock {
    param($listener, $responseFile, $delayFile)
    try {
        while ($listener.IsListening) {
            $call = $listener.GetContext()
            $wait = [int](Get-Content -Raw $delayFile)
            if ($wait -gt 0) { Start-Sleep -Seconds $wait }
            $bytes = [System.IO.File]::ReadAllBytes($responseFile)
            # No charset, as the engine sends it.
            $call.Response.ContentType = 'application/json'
            try {
                $call.Response.OutputStream.Write($bytes, 0, $bytes.Length)
                $call.Response.Close()
            } catch { }
        }
    } catch { }
}
$up = $false
foreach ($attempt in 1..50) {
    try { Invoke-RestMethod -Method Post -Uri "$url/mcp/call" -Body '{}' -TimeoutSec 1 | Out-Null; $up = $true; break }
    catch { Start-Sleep -Milliseconds 100 }
}
if (-not $up) { Write-Output 'the test surface did not start'; exit 2 }

$env:ANTUMBRA_URL = $url
$env:ANTUMBRA_TOKEN = 'test-token'

# Claude Code on Windows runs the hook under Windows PowerShell, which decodes
# a response with no charset as ISO-8859-1 unless the hook reads the bytes as
# UTF-8 itself, so there every check runs under both shells.
$runners = @('pwsh')
if ($IsWindows -and (Get-Command powershell -ErrorAction SilentlyContinue)) { $runners += 'powershell' }

function Invoke-Hook([string]$Prompt) {
    $payload = @{ prompt = $Prompt; cwd = $work } | ConvertTo-Json -Compress
    return ($payload | & $script:runner -NoProfile -NonInteractive -File $hook | Out-String).Trim()
}

function Set-Surface([string]$ResponseJson, [int]$DelaySec = 0) {
    [System.IO.File]::WriteAllText($response, $ResponseJson, [System.Text.UTF8Encoding]::new($false))
    Set-Content -Path $delay -Value "$DelaySec"
}

# An em dash, and a long memory the hook must cut.
$dash = [string][char]0x2014
$recall = @{
    memories = @(
        @{ id = 'memory:a'; network = 'world'; scope = 'in_scope'; content = "keep it plain $dash always" },
        @{ id = 'memory:b'; network = 'bank'; content = ('y' * 1200) }
    )
} | ConvertTo-Json -Depth 5 -Compress

foreach ($script:runner in $runners) {
    Write-Output "--- under $script:runner"
    Write-Output 'a recall'
    Set-Surface $recall
    $out = Invoke-Hook 'how do we word caveats'
    $answer = $out | ConvertFrom-Json
    $ctx = [string]$answer.hookSpecificOutput.additionalContext
    Check 'answers as a UserPromptSubmit hook' ($answer.hookSpecificOutput.hookEventName -eq 'UserPromptSubmit')
    Check 'carries the em dash intact' $ctx.Contains("plain $dash always")
    Check 'tags where the memory was learned' $ctx.Contains('(world) [in_scope]')
    Check 'cuts a long memory and names the whole entry' $ctx.Contains('[truncated - full entry: memory:b]')

    Write-Output 'nothing recalled'
    Set-Surface '{"memories":[]}'
    Check 'says nothing' ((Invoke-Hook 'anything at all') -eq '')

    Write-Output "a background task's notification"
    Set-Surface $recall
    $note = "<task-notification>`n<task-id>b1</task-id>`n<status>completed</status>`n</task-notification>"
    Check 'is not recalled for' ((Invoke-Hook $note) -eq '')
    Check 'a prompt that only mentions one is' ((Invoke-Hook 'why does a <task-notification> arrive twice') -ne '')

    Write-Output 'a one-character prompt'
    Set-Surface $recall
    Check 'is not recalled for' ((Invoke-Hook 'k') -eq '')

    Write-Output 'a surface slower than the budget'
    Set-Surface $recall 6
    $env:ANTUMBRA_RECALL_BUDGET_SEC = '2'
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    $out = Invoke-Hook 'how do we word caveats'
    $took = $clock.Elapsed.TotalSeconds
    Check "gives up inside the budget ($([math]::Round($took, 1)) s, the surface takes 6)" ($took -lt 5)
    Check 'says nothing' ($out -eq '')
    Remove-Item Env:ANTUMBRA_RECALL_BUDGET_SEC
    Set-Surface $recall

    Write-Output 'no token'
    Remove-Item Env:ANTUMBRA_TOKEN
    $env:ANTUMBRA_TOKEN_FILE = Join-Path $work 'no-token-here.txt'
    Check 'says nothing' ((Invoke-Hook 'how do we word caveats') -eq '')
    Remove-Item Env:ANTUMBRA_TOKEN_FILE
    $env:ANTUMBRA_TOKEN = 'test-token'
}

$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

# Tests for antumbra-capture.ps1's report of a turn's end. No Antumbra needed:
# the surface is a listener in this process that records what it was sent.
#
#   pwsh -NoProfile -File scripts/hooks/tests/capture.ps1
$ErrorActionPreference = 'Stop'
$hook = if ($env:ANTUMBRA_TEST_HOOK) { $env:ANTUMBRA_TEST_HOOK } else { Join-Path $PSScriptRoot '..' 'antumbra-capture.ps1' }
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("antumbra-capture-test-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$script:failed = 0

function Check([string]$What, [bool]$Holds) {
    if ($Holds) { Write-Output "  ok    $What" } else { Write-Output "  FAIL  $What"; $script:failed = 1 }
}

# --- a surface that records each request body and answers {} -------------------
$request = Join-Path $work 'request.json'
$port = Get-Random -Minimum 20000 -Maximum 40000
$url = "http://127.0.0.1:$port"
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add("$url/")
$listener.Start()
$null = Start-ThreadJob -ArgumentList $listener, $request -ScriptBlock {
    param($listener, $requestFile)
    try {
        while ($listener.IsListening) {
            $call = $listener.GetContext()
            $reader = [System.IO.StreamReader]::new($call.Request.InputStream)
            [System.IO.File]::WriteAllText($requestFile, $reader.ReadToEnd())
            $bytes = [System.Text.Encoding]::UTF8.GetBytes('{}')
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
$runners = @('pwsh')
if ($IsWindows -and (Get-Command powershell -ErrorAction SilentlyContinue)) { $runners += 'powershell' }

function Invoke-Hook([hashtable]$Payload) {
    Remove-Item $request -ErrorAction SilentlyContinue
    $json = $Payload | ConvertTo-Json -Compress
    return ($json | & $script:runner -NoProfile -NonInteractive -File $hook | Out-String).Trim()
}

foreach ($script:runner in $runners) {
    Write-Output "--- under $script:runner"
    $session = 'capture-test-' + [System.Guid]::NewGuid().ToString('N')

    Write-Output 'the end of a turn'
    $out = Invoke-Hook @{ session_id = $session; prompt_id = 'p1'; hook_event_name = 'Stop'; last_assistant_message = 'deployed it blue green' }
    Check 'still asks for memories before stopping' (($out | ConvertFrom-Json).decision -eq 'block')
    $sent = if (Test-Path $request) { Get-Content -Raw $request | ConvertFrom-Json } else { $null }
    Check 'reports the turn to the server' ($sent -and $sent.tool -eq 'observe_turn')
    Check 'with its session, turn and answer' ($sent -and $sent.arguments.session -eq $session -and $sent.arguments.turn -eq 'p1' -and $sent.arguments.text -eq 'deployed it blue green')

    Write-Output 'the stop after the nudge'
    $out = Invoke-Hook @{ session_id = $session; prompt_id = 'p1'; hook_event_name = 'Stop'; last_assistant_message = 'saved' }
    Check 'is silent and reports nothing again' (($out -eq '') -and -not (Test-Path $request))

    Write-Output 'a compaction'
    $null = Invoke-Hook @{ session_id = "$session-c"; hook_event_name = 'PreCompact' }
    Check 'reports no turn' (-not (Test-Path $request))
    $tmp = if ($env:TEMP) { $env:TEMP } else { [System.IO.Path]::GetTempPath() }
    Remove-Item (Join-Path $tmp "antumbra-capture-PreCompact-$session-c") -ErrorAction SilentlyContinue
}

$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

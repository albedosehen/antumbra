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
            [System.IO.File]::WriteAllText("$requestFile.host", [string]$call.Request.Headers['X-Antumbra-Host'])
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
$run = Join-Path $work 'run'
New-Item -ItemType Directory -Path $run | Out-Null
$env:XDG_RUNTIME_DIR = $run
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

    Write-Output 'the end of a turn on a named machine'
    $env:ANTUMBRA_HOST_ID = 'windows'
    $null = Invoke-Hook @{ session_id = "$session-h"; prompt_id = 'p2'; hook_event_name = 'Stop'; last_assistant_message = 'done' }
    Remove-Item Env:ANTUMBRA_HOST_ID
    Check 'names the machine as the MCP client does' ((Get-Content -Raw "$request.host") -eq 'windows')
    Remove-Item (Join-Path $run "antumbra-capture-Stop-$session-h") -ErrorAction SilentlyContinue

    Write-Output 'the stop after the nudge'
    $out = Invoke-Hook @{ session_id = $session; prompt_id = 'p1'; hook_event_name = 'Stop'; last_assistant_message = 'saved' }
    Check 'is silent and reports nothing again' (($out -eq '') -and -not (Test-Path $request))

    Write-Output 'a session nudged a moment ago'
    $out = Invoke-Hook @{ session_id = $session; prompt_id = 'p3'; hook_event_name = 'Stop'; last_assistant_message = 'waiting on a build' }
    Check 'is let stop without another nudge' ($out -eq '')
    $sent = if (Test-Path $request) { Get-Content -Raw $request | ConvertFrom-Json } else { $null }
    Check 'and the end of its turn is still reported' ($sent -and $sent.arguments.text -eq 'waiting on a build')
    $env:ANTUMBRA_CAPTURE_EVERY_SECS = '0'
    $out = Invoke-Hook @{ session_id = $session; prompt_id = 'p4'; hook_event_name = 'Stop'; last_assistant_message = 'more' }
    Check 'unless the interval is set to none, which nudges every turn' (($out | ConvertFrom-Json).decision -eq 'block')
    Remove-Item (Join-Path $run "antumbra-capture-Stop-$session") -ErrorAction SilentlyContinue

    Write-Output 'a turn that saved as it went'
    $transcript = Join-Path $work "transcript-$session.jsonl"
    Set-Content -LiteralPath $transcript -Value '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"mcp__antumbra__store_memory","input":{}}]}}'
    $out = Invoke-Hook @{ session_id = "$session-s"; hook_event_name = 'Stop'; last_assistant_message = 'saved it'; transcript_path = $transcript }
    Check 'is not stopped to save again' ($out -eq '')
    Check 'and the end of its turn is still reported' (Test-Path $request)
    # With no interval, so only what the transcript holds decides.
    $env:ANTUMBRA_CAPTURE_EVERY_SECS = '0'
    $out = Invoke-Hook @{ session_id = "$session-s"; hook_event_name = 'Stop'; last_assistant_message = 'nothing new'; transcript_path = $transcript }
    Check 'the save is not counted a second time' (($out | ConvertFrom-Json).decision -eq 'block')
    Remove-Item (Join-Path $run "antumbra-capture-Stop-$session-s") -ErrorAction SilentlyContinue
    Add-Content -LiteralPath $transcript -Value '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"mcp__antumbra__record_behavior","input":{}}]}}'
    $out = Invoke-Hook @{ session_id = "$session-s"; hook_event_name = 'Stop'; last_assistant_message = 'recorded one'; transcript_path = $transcript }
    Check 'a recorded behavior counts as a save' ($out -eq '')
    Set-Content -LiteralPath $transcript -Value '{"type":"user","message":{"content":[{"type":"tool_result","content":"{\"name\": \"mcp__antumbra__store_memory\"}"}]}}'
    $out = Invoke-Hook @{ session_id = "$session-q"; hook_event_name = 'Stop'; last_assistant_message = 'looked a tool up'; transcript_path = $transcript }
    Check 'a tool only named inside a result does not' (($out | ConvertFrom-Json).decision -eq 'block')

    Write-Output 'a stop the nudge itself caused'
    $out = Invoke-Hook @{ session_id = "$session-a"; hook_event_name = 'Stop'; stop_hook_active = $true; last_assistant_message = 'saved' }
    Check 'goes through, with no sentinel left to say so' ($out -eq '')
    Remove-Item Env:ANTUMBRA_CAPTURE_EVERY_SECS

    Write-Output 'a compaction'
    $null = Invoke-Hook @{ session_id = "$session-c"; hook_event_name = 'PreCompact' }
    Check 'reports no turn' (-not (Test-Path $request))
    Remove-Item (Join-Path $run "antumbra-capture-PreCompact-$session-c") -ErrorAction SilentlyContinue
    # Nudged a moment ago at the end of a turn: a compaction is nudged all the same.
    $out = Invoke-Hook @{ session_id = $session; hook_event_name = 'PreCompact' }
    Check 'is always nudged' (($out | ConvertFrom-Json).decision -eq 'block')
    Remove-Item (Join-Path $run "antumbra-capture-PreCompact-$session") -ErrorAction SilentlyContinue

    Write-Output 'a hostile session id'
    Get-ChildItem $run -Filter 'antumbra-capture-*' | Remove-Item -Force
    $out = Invoke-Hook @{ session_id = '../../x y'; hook_event_name = 'Stop/../z'; last_assistant_message = 'x' }
    Check 'is handled like any other' (($out | ConvertFrom-Json).decision -eq 'block')
    Check 'names the sentinel from the id cut to [A-Za-z0-9_-], under the runtime directory' (Test-Path (Join-Path $run 'antumbra-capture-Stopz-xy'))
    $others = @(Get-ChildItem $run | Where-Object { $_.Name -ne 'antumbra-capture-Stopz-xy' })
    Check 'and touches nothing else' (($others.Count -eq 0) -and -not (Test-Path (Join-Path $work 'x y')))
    Get-ChildItem $run -Filter 'antumbra-capture-*' | Remove-Item -Force
}

$listener.Stop()
Get-Job | Remove-Job -Force
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

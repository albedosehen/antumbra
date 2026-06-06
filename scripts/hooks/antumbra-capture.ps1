# Antumbra capture hook (Windows / PowerShell). Wire to BOTH Stop and PreCompact.
# Nudges the agent to deposit non-obvious observations into Antumbra via the
# `store_memory` MCP tool before the turn ends / context is compacted. A sentinel
# file makes it fire once per turn (no infinite loop). The verified traces it
# leaves are what `antumbra metabolize` later turns into a trained expert.
$hookInput   = [System.Console]::In.ReadToEnd()
$json        = $hookInput | ConvertFrom-Json -ErrorAction SilentlyContinue
$sessionId   = if ($json.session_id) { $json.session_id } else { 'default' }
$event       = if ($json.hook_event_name) { $json.hook_event_name } else { 'Stop' }
$workspace   = if ($env:ANTUMBRA_WORKSPACE_ID) { $env:ANTUMBRA_WORKSPACE_ID } else { '<workspace>' }

$sentinel = Join-Path $env:TEMP "antumbra-capture-$event-$sessionId"
if (Test-Path $sentinel) { Remove-Item $sentinel -Force; exit 0 }
New-Item -ItemType File -Path $sentinel -Force | Out-Null

$reason = "Before stopping, deposit any non-obvious observations from this phase into Antumbra so they are not lost. " +
          "Call the store_memory MCP tool (workspace `"$workspace`") and pick the network: world (facts), bank (experiences/incidents), opinion (judgments/preferences). " +
          "Save: project context, conventions, bug/incident history, user feedback/corrections, decisions, deadlines. Skip ephemeral chatter. " +
          "Where a result was verified (a test passed, a command worked), note it -- those recurrent, verified traces are what metabolizes into a permanent local expert. " +
          "If nothing is worth saving this turn, just stop again -- the next stop proceeds automatically."

@{ decision = 'block'; reason = $reason } | ConvertTo-Json -Compress -Depth 3 | Write-Output

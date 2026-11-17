# Tests for strip-attribution.ps1 over the cases in strip-attribution.cases,
# the list strip-attribution.sh is held to as well.
#
#   pwsh -NoProfile -File scripts/hooks/tests/strip-attribution.ps1
$ErrorActionPreference = 'Stop'
$hook = if ($env:ANTUMBRA_TEST_HOOK) { $env:ANTUMBRA_TEST_HOOK } else { Join-Path $PSScriptRoot '..' 'strip-attribution.ps1' }
$cases = Join-Path $PSScriptRoot 'strip-attribution.cases'
$script:failed = 0

function Check([string]$What, [bool]$Holds) {
    if ($Holds) { Write-Output "  ok    $What" } else { Write-Output "  FAIL  $What"; $script:failed = 1 }
}

$runners = @('pwsh')
if ($IsWindows -and (Get-Command powershell -ErrorAction SilentlyContinue)) { $runners += 'powershell' }

foreach ($runner in $runners) {
    Write-Output "--- under $runner"
    $seen = 0
    foreach ($line in (Get-Content -Encoding UTF8 $cases)) {
        if ($line -eq '' -or $line.StartsWith('#')) { continue }
        $parts = $line -split "`t", 2
        if ($parts.Count -ne 2) { Check "a verdict, a tab and the input: $line" $false; continue }
        $verdict, $hookInput = $parts
        $seen += 1
        $out = ($hookInput | & $runner -NoProfile -NonInteractive -File $hook | Out-String).Trim()
        switch ($verdict) {
            'deny' {
                $decision = if ($out) { ($out | ConvertFrom-Json).hookSpecificOutput } else { $null }
                Check "denies  $hookInput" ($decision -and
                    $decision.hookEventName -eq 'PreToolUse' -and
                    $decision.permissionDecision -eq 'deny' -and
                    $decision.permissionDecisionReason.StartsWith('Model-vendor attribution is not allowed'))
            }
            'allow' { Check "allows  $hookInput" ($out -eq '') }
            default { Check "a verdict of deny or allow: $line" $false }
        }
    }
    Check "read the cases ($seen)" ($seen -ge 10)
}

if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

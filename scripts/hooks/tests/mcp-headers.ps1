# Tests for antumbra-mcp-headers.ps1: what Claude Code's headersHelper receives.
#
#   pwsh -NoProfile -File scripts/hooks/tests/mcp-headers.ps1
$ErrorActionPreference = 'Stop'
$helper = Join-Path $PSScriptRoot '..' 'antumbra-mcp-headers.ps1'
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("antumbra-headers-test-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$script:failed = 0

function Check([string]$What, [bool]$Holds) {
    if ($Holds) { Write-Output "  ok    $What" } else { Write-Output "  FAIL  $What"; $script:failed = 1 }
}

function Invoke-Helper([string]$TokenFile) {
    $env:ANTUMBRA_TOKEN_FILE = $TokenFile
    $out = & pwsh -NoProfile -NonInteractive -File $helper 2>$null | Out-String
    return @{ Out = $out.Trim(); Code = $LASTEXITCODE }
}

$file = Join-Path $work 'token.txt'

Write-Output 'a token file'
Set-Content -Path $file -Value 'test-token' -NoNewline
$r = Invoke-Helper $file
Check 'exits 0' ($r.Code -eq 0)
Check 'prints one JSON object with the bearer' (($r.Out | ConvertFrom-Json).Authorization -eq 'Bearer test-token')

Write-Output 'a token file with a trailing newline'
Set-Content -Path $file -Value "test-token`r`n" -NoNewline
Check 'sends the token without it' (((Invoke-Helper $file).Out | ConvertFrom-Json).Authorization -eq 'Bearer test-token')

Write-Output 'a token that JSON must escape'
Set-Content -Path $file -Value 'a"b\c' -NoNewline
Check 'still prints valid JSON carrying it' (((Invoke-Helper $file).Out | ConvertFrom-Json).Authorization -eq 'Bearer a"b\c')

Write-Output 'no token file'
$r = Invoke-Helper (Join-Path $work 'missing.txt')
Check 'exits non-zero' ($r.Code -ne 0)
Check 'prints no header' ($r.Out -eq '')

Remove-Item Env:ANTUMBRA_TOKEN_FILE
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
if ($script:failed -eq 0) { Write-Output 'all passed'; exit 0 } else { Write-Output 'FAILED'; exit 1 }

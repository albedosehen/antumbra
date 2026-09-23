# Antumbra MCP auth header for Claude Code's `headersHelper` (Windows / PowerShell).
# Prints {"Authorization": "Bearer <token>"} with the token read from a file, so
# the MCP server entry needs no `${ANTUMBRA_TOKEN}` from the settings file's env
# block: the same file the hooks read, ANTUMBRA_TOKEN_FILE or
# ~/.antumbra/token.txt. A long-lived credential does not belong in a settings
# file that is shared, diffed and backed up.
#
# Claude Code runs the helper when it connects and sends whatever headers it
# prints. A missing or empty token file exits non-zero, which Claude Code reports
# as a failed connection, rather than sending a header that cannot work.
param()

$file = if ($env:ANTUMBRA_TOKEN_FILE) { $env:ANTUMBRA_TOKEN_FILE }
        else { Join-Path $HOME '.antumbra/token.txt' }
$token = ''
try { if (Test-Path $file) { $token = (Get-Content -Raw $file).Trim() } } catch { $token = '' }
if (-not $token) {
    [Console]::Error.WriteLine("antumbra-mcp-headers: no token in $file")
    exit 1
}
@{ Authorization = "Bearer $token" } | ConvertTo-Json -Compress

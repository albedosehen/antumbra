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
#
# With ANTUMBRA_HOST_ID set (setup writes it into the settings' env, which the
# helper inherits), it also prints {"X-Antumbra-Host": "<name>"}, so what this
# machine's agent writes is stamped as written from here rather than from the
# server it reaches. Unset, the server stamps its own name.
param()

$file = if ($env:ANTUMBRA_TOKEN_FILE) { $env:ANTUMBRA_TOKEN_FILE }
        else { Join-Path $HOME '.antumbra/token.txt' }
$token = ''
try { if (Test-Path $file) { $token = (Get-Content -Raw $file).Trim() } } catch { $token = '' }
if (-not $token) {
    [Console]::Error.WriteLine("antumbra-mcp-headers: no token in $file")
    exit 1
}
$headers = [ordered]@{ Authorization = "Bearer $token" }
# Not $host: PowerShell reserves it.
$hostId = if ($env:ANTUMBRA_HOST_ID) { $env:ANTUMBRA_HOST_ID.Trim() } else { '' }
if ($hostId) { $headers['X-Antumbra-Host'] = $hostId }
$headers | ConvertTo-Json -Compress

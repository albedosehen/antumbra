# Antumbra PreToolUse hook (Windows / PowerShell): deny git/gh commands that embed
# model-vendor attribution, so the work in your history is attributed to you, not
# the agent. Reads the hook JSON from stdin; emits a deny decision only on a match.
#
# The same phrases as strip-attribution.sh, held to the same cases by
# scripts/hooks/tests/strip-attribution.cases; a phrase added to one hook is
# added to the other.
$hookInput = [System.Console]::In.ReadToEnd()
$json      = $hookInput | ConvertFrom-Json -ErrorAction SilentlyContinue
$cmd       = if ($json.tool_input.command) { [string]$json.tool_input.command } else { '' }

# Vendor-attribution phrases to keep out of commits/PRs, case-insensitive (as
# -match is). The robot emoji is spelled as its two UTF-16 code units, so this
# file stays ASCII; a command carrying the emoji's JSON escapes as text is
# denied too. Extend per your policy.
$robot   = "`u{1F916}"
$pattern = 'co-authored-by:.*(claude|anthropic\.com|gpt|openai|copilot)|\uD83E\uDD16 generated|\\ud83e\\udd16 generated|generated with \[|\[claude code\]|noreply@anthropic\.com'

if ($cmd -match $pattern) {
    @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = "Model-vendor attribution is not allowed in git/gh commands per policy. Remove the Co-Authored-By trailer, any `"Generated with`" line, and the $robot emoji before retrying."
        }
    } | ConvertTo-Json -Compress -Depth 5 | Write-Output
}

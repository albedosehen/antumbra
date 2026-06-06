# Antumbra PreToolUse hook (Windows / PowerShell): deny git/gh commands that embed
# model-vendor attribution, so the work in your history is attributed to you, not
# the agent. Reads the hook JSON from stdin; emits a deny decision only on a match.
$hookInput = [System.Console]::In.ReadToEnd()
$json      = $hookInput | ConvertFrom-Json -ErrorAction SilentlyContinue
$cmd       = if ($json.tool_input.command) { $json.tool_input.command } else { '' }

# Vendor-attribution phrases to keep out of commits/PRs. Extend per your policy.
$pattern = 'Co-Authored-By:.*?(Claude|anthropic\.com|GPT|OpenAI|Copilot)|Generated with \[|noreply@anthropic\.com'

if ($cmd -match $pattern) {
    @{
        hookSpecificOutput = @{
            hookEventName            = 'PreToolUse'
            permissionDecision       = 'deny'
            permissionDecisionReason = 'Model-vendor attribution is not allowed in git/gh commands per policy. Remove the Co-Authored-By trailer and any "Generated with" reference, then retry.'
        }
    } | ConvertTo-Json -Compress -Depth 5 | Write-Output
}

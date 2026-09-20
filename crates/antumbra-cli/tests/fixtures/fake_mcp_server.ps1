# A stdio MCP server that knows four things: the handshake, `tools/list`,
# `tools/call` (every tool says "ok"), and `ping`. It lists the tools in the JSON
# file given as its first argument (default: fake_tools.json beside it), which
# must be one line. POSIX sibling: fake_mcp_server.sh.
param([string]$ToolsFile = (Join-Path $PSScriptRoot 'fake_tools.json'))

if (-not (Test-Path -LiteralPath $ToolsFile)) { [Console]::Error.WriteLine("fake_mcp_server: cannot read $ToolsFile"); exit 1 }
$tools = (Get-Content -Raw -LiteralPath $ToolsFile) -replace "[\r\n]", ''
while ($null -ne ($line = [Console]::In.ReadLine())) {
    $id = if ($line -match '"id":(\d+)') { $Matches[1] } else { 'null' }
    $answer = $null
    if ($line -match '"method":"initialize"') {
        $answer = "{`"jsonrpc`":`"2.0`",`"id`":$id,`"result`":{`"protocolVersion`":`"2025-06-18`",`"capabilities`":{`"tools`":{}},`"serverInfo`":{`"name`":`"fake`",`"version`":`"0`"}}}"
    } elseif ($line -match '"method":"tools/list"') {
        $answer = "{`"jsonrpc`":`"2.0`",`"id`":$id,`"result`":{`"tools`":$tools}}"
    } elseif ($line -match '"method":"tools/call"') {
        $answer = "{`"jsonrpc`":`"2.0`",`"id`":$id,`"result`":{`"content`":[{`"type`":`"text`",`"text`":`"ok`"}]}}"
    } elseif ($line -match '"method":"ping"') {
        $answer = "{`"jsonrpc`":`"2.0`",`"id`":$id,`"result`":{}}"
    }
    if ($answer) {
        [Console]::Out.WriteLine($answer)
        [Console]::Out.Flush()
    }
}

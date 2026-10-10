param(
    [Parameter(Mandatory)]
    [string]$Prompt
)

Set-Location -LiteralPath $PSScriptRoot
claude $Prompt
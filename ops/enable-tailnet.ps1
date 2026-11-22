param([ValidateRange(1024,65535)][int]$Port = 18765)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$tailnet = & tailscale status --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $tailnet.BackendState -ne 'Running') {
    throw 'Tailscale must be signed in and running on this device.'
}
$dns = $tailnet.Self.DNSName.TrimEnd('.')
if (-not $dns) { throw 'This device has no Tailscale DNS name.' }
$origin = "https://${dns}:$Port"
$serve = & tailscale serve status --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect existing Tailscale Serve configuration.' }
$existing = $null
if ($serve.Web) { $existing = $serve.Web.PSObject.Properties["${dns}:$Port"] }
if ($serve.TCP) {
    $tcp = $serve.TCP.PSObject.Properties["$Port"]
    if ($tcp -and $tcp.Value.HTTPS -ne $true) {
        throw "TCP port $Port already belongs to another Tailscale service."
    }
}
if ($existing -and $existing.Value.Handlers.'/'.Proxy -ne 'http://127.0.0.1:18765') {
    throw "HTTPS port $Port already belongs to another Tailscale service."
}
$configPath = Join-Path $repo 'data\network.json'
New-Item -ItemType Directory -Force (Split-Path -Parent $configPath) | Out-Null
$origins = @()
if (Test-Path $configPath) {
    $origins = @((Get-Content -Raw -Encoding UTF8 $configPath | ConvertFrom-Json).allowed_origins)
}
$origins = @($origins + $origin | Sort-Object -Unique)
@{allowed_origins=$origins} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $configPath
& tailscale serve --bg --yes "--https=$Port" http://127.0.0.1:18765
if ($LASTEXITCODE -ne 0) { throw 'Tailscale Serve failed.' }
Write-Host "Phone URL: $origin/"
Write-Host 'If this origin was newly added, restart the idle backend to load data/network.json:'
Write-Host 'wsl -d Ubuntu-22.04 -- systemctl --user restart soda-voice-dictation.service'

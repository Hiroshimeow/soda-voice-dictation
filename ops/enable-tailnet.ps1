param([ValidateRange(1024,65535)][int]$Port = 53219)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$tailnet = & tailscale status --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $tailnet.BackendState -ne 'Running') {
    throw 'Tailscale must be signed in and running on this device.'
}
$ip = (& tailscale ip -4).Trim()
if (-not $ip) { throw 'This device has no Tailscale IPv4 address.' }

# Remove only the old SODA HTTPS Serve endpoint when it still exists.
# Leave unrelated Tailscale services (for example other TCP forwards or port 443) untouched.
$serve = & tailscale serve status --json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Tailscale Serve configuration.' }
$oldHttps = $null
if ($serve.TCP) { $oldHttps = $serve.TCP.PSObject.Properties['18765'] }
if ($oldHttps -and $oldHttps.Value.HTTPS -eq $true) {
    & tailscale serve --https=18765 off | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not remove the old SODA HTTPS endpoint.' }
}

$configPath = Join-Path $repo 'data\network.json'
New-Item -ItemType Directory -Force (Split-Path -Parent $configPath) | Out-Null
$origin = "http://${ip}:$Port"
$origins = @()
if (Test-Path $configPath) {
    $parsed = Get-Content -Raw -Encoding UTF8 $configPath | ConvertFrom-Json
    $origins = @($parsed.allowed_origins)
}
$origins = @($origins | Where-Object { $_ -notmatch '^https://[^/]+:18765$' })
$origins = @($origins + $origin | Sort-Object -Unique)
@{allowed_origins=$origins} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $configPath

& tailscale serve --bg "--tcp=$Port" tcp://127.0.0.1:18765
if ($LASTEXITCODE -ne 0) { throw 'Tailscale TCP Serve failed.' }

Write-Host "Tailnet URL: http://${ip}:$Port/"
Write-Host 'Funnel is not used. This endpoint is reachable only through the tailnet.'
Write-Warning 'Browsers usually require HTTPS for microphone capture. Direct HTTP Tailscale-IP access is suitable for API/file access, but browser microphone access may be blocked.'
Write-Host 'Restart the idle SODA backend once if this origin was newly added.'

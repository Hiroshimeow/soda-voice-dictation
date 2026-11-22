$ErrorActionPreference = 'Stop'
$started = [DateTime]::UtcNow
$observations = @()
for ($i = 0; $i -le 18; $i++) {
    $processes = @(Get-CimInstance Win32_Process -Filter "name='wsl.exe'")
    $foreign = @($processes | Where-Object { $_.CommandLine -notlike '*google-soda-stt.service*sleep infinity*' })
    $health = $null
    $errorText = $null
    try { $health = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 3 }
    catch { $errorText = $_.Exception.Message }
    $observation = [ordered]@{
        seconds = [Math]::Round(([DateTime]::UtcNow - $started).TotalSeconds, 1)
        healthy = [bool]$health.ok
        active_sessions = $health.active_sessions
        keeper_wsl_processes = $processes.Count - $foreign.Count
        foreign_wsl_processes = $foreign.Count
        error = $errorText
    }
    $observations += $observation
    if (($i % 3) -eq 0) { $observation | ConvertTo-Json -Compress | Write-Output }
    if ($i -lt 18) { Start-Sleep -Seconds 5 }
}
$observations | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $PSScriptRoot '..\data\backend-idle-proof.json')
if (@($observations | Where-Object { -not $_.healthy -or $_.foreign_wsl_processes -ne 0 -or $_.keeper_wsl_processes -eq 0 }).Count) {
    throw 'Idle availability verification failed or was contaminated by another WSL command.'
}
Write-Output 'PASS all Windows health checks across90seconds; only the owned WSL keeper was attached.'

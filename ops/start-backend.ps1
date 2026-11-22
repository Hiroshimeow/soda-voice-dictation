param(
    [switch]$KeepAlive,
    [string]$Distro = $(if ($env:SODA_WSL_DISTRO) { $env:SODA_WSL_DISTRO } else { 'Ubuntu-22.04' })
)
$ErrorActionPreference = 'Stop'
$RepoWindows = Split-Path $PSScriptRoot -Parent

function Get-RepoWslPath {
    if ($RepoWindows -match '^([A-Za-z]):\\(.*)$') {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2].Replace('\', '/')
        return "/mnt/$drive/$rest"
    }
    throw "The repository must be on a Windows drive mounted in WSL (for example C:\\ or D:\\)."
}

if ($KeepAlive) {
    $mutex = [System.Threading.Mutex]::new($false, 'Local\SodaVoiceDictation.BackendKeeper')
    $owned = $false
    try {
        try { $owned = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $owned = $true }
        if (-not $owned) { exit 0 }
        & wsl.exe -d $Distro -- sh -lc 'systemctl --user start soda-voice-dictation.service && exec sleep infinity'
        exit $LASTEXITCODE
    }
    finally {
        if ($owned) { $mutex.ReleaseMutex() }
        $mutex.Dispose()
    }
}

$distros = @(& wsl.exe -l -q 2>$null) | ForEach-Object { ($_ -replace "`0", '').Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
    throw "WSL distro '$Distro' is required. Install it first with: wsl --install -d $Distro"
}

$RepoWsl = Get-RepoWslPath

$bootstrap = @'
set -e
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required inside WSL." >&2
  exit 21
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
cd "$SODA_REPO"
if [ ! -x .venv-wsl/bin/python ]; then
  uv venv --python 3.12 .venv-wsl
fi
uv pip install --python .venv-wsl/bin/python -r requirements.txt
PYTHONPATH="$SODA_REPO" .venv-wsl/bin/python ops/download-models.py
'@
$bootstrap = $bootstrap -replace "`r`n", "`n"
$bootstrap64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bootstrap))
$bootstrapCommand = "echo '$bootstrap64' | base64 -d | SODA_REPO='$RepoWsl' bash"
& wsl.exe -d $Distro -- bash -lc $bootstrapCommand
if ($LASTEXITCODE -ne 0) { throw 'WSL backend bootstrap failed.' }

$service = @"
[Unit]
Description=SODA Voice Dictation backend
After=network.target

[Service]
Type=simple
WorkingDirectory=$RepoWsl
ExecStart=$RepoWsl/.venv-wsl/bin/python -m uvicorn server:app --host 127.0.0.1 --port 18765
Restart=on-failure
RestartSec=2
TimeoutStopSec=10
KillMode=control-group
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=default.target
"@
$service = $service -replace "`r`n", "`n"
$service64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($service))
$install = "mkdir -p ~/.config/systemd/user && systemctl --user disable --now google-soda-stt.service >/dev/null 2>&1 || true; echo '$service64' | base64 -d > ~/.config/systemd/user/soda-voice-dictation.service && systemctl --user daemon-reload && systemctl --user enable --now soda-voice-dictation.service"
& wsl.exe -d $Distro -- bash -lc $install
if ($LASTEXITCODE -ne 0) { throw 'Could not install/start the WSL user service.' }

$null = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -WindowStyle Hidden -PassThru -ArgumentList @(
    '-NoProfile', '-NonInteractive', '-File', ('"' + $PSCommandPath + '"'), '-KeepAlive', '-Distro', $Distro
)

$deadline = [DateTime]::UtcNow.AddSeconds(60)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 2
        if ($health.ok) {
            Write-Output 'SODA Voice Dictation backend is ready at http://127.0.0.1:18765/.'
            exit 0
        }
    }
    catch {}
    Start-Sleep -Milliseconds 500
}
throw 'SODA backend did not become ready within 60 seconds.'

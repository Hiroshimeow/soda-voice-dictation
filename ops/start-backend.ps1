param(
    [switch]$KeepAlive,
    [string]$Distro = $env:SODA_WSL_DISTRO
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'start-backend-helpers.ps1')
$RepoWindows = Split-Path $PSScriptRoot -Parent

function Get-AvailableDistros {
    return @(& wsl.exe -l -q 2>$null) |
        ForEach-Object { ($_ -replace "`0", '').Trim() } |
        Where-Object { $_ }
}

function Resolve-Distro {
    $available = @(Get-AvailableDistros)
    if ($Distro) {
        if ($available -notcontains $Distro) {
            throw "Requested WSL distro '$Distro' is not installed. Available: $($available -join ', ')"
        }
        return $Distro
    }
    foreach ($preferred in @('Ubuntu-22.04', 'Debian', 'Ubuntu')) {
        if ($available -contains $preferred) { return $preferred }
    }
    $candidate = @($available | Where-Object { $_ -notmatch '^docker-desktop(?:-data)?$' } | Select-Object -First 1)
    if ($candidate.Count) { return $candidate[0] }
    throw 'No usable WSL distro found. Install Debian or Ubuntu first (for example: wsl --install -d Debian).'
}

$Distro = Resolve-Distro
if (-not $KeepAlive) { Write-Output "Using WSL distro: $Distro" }

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


$RepoWsl = Get-RepoWslPath

function ConvertTo-BashSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

# Deterministic local smoke check: validate the generated syntax with the same
# parser used by the unit check, including escaped spaces in a path.
$spacePython = ConvertTo-SystemdExecPath '/mnt/c/repo with spaces%100/.venv-wsl/bin/python'
$spaceExec = "ExecStart=$spacePython -m uvicorn server:app --host 127.0.0.1 --port 18765"
Assert-SystemdExecStart -Line $spaceExec -ExpectedPython '/mnt/c/repo with spaces%100/.venv-wsl/bin/python' | Out-Null

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
  rm -rf .venv-wsl
  uv venv --python 3.12 .venv-wsl
fi
uv pip install --python .venv-wsl/bin/python -r requirements.txt
PYTHONPATH="$SODA_REPO" .venv-wsl/bin/python ops/download-models.py
'@
$bootstrap = $bootstrap -replace "`r`n", "`n"
$bootstrap64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bootstrap))
$repoWslShell = ConvertTo-BashSingleQuoted $RepoWsl
$bootstrapCommand = "echo '$bootstrap64' | base64 -d | SODA_REPO=$repoWslShell bash"
& wsl.exe -d $Distro -- bash -lc $bootstrapCommand
if ($LASTEXITCODE -ne 0) { throw 'WSL backend bootstrap failed.' }

$RepoWslSystemd = ConvertTo-SystemdWorkingDirectory $RepoWsl
$PythonWslSystemd = ConvertTo-SystemdExecPath "$RepoWsl/.venv-wsl/bin/python"
$repoWslShell = ConvertTo-BashSingleQuoted $RepoWsl
$identityScriptWsl = "$RepoWsl/ops/checkout_identity.py"
$identityScriptShell = ConvertTo-BashSingleQuoted $identityScriptWsl
$pythonWslShell = ConvertTo-BashSingleQuoted "$RepoWsl/.venv-wsl/bin/python"
# Hash the deterministic launch manifest, not every checkout file. The helper
# is Git-independent for portable ZIPs, excludes runtime/generated artifacts,
# and includes the absolute root so a copied release cannot certify a service
# rooted at a different checkout. Bash arguments are single-quoted so spaces
# and shell metacharacters in the checkout path remain data.
$identityCommand = "$pythonWslShell $identityScriptShell --root $repoWslShell"
$checkoutIdentity = (& wsl.exe -d $Distro -- bash -lc $identityCommand).Trim()
if ($LASTEXITCODE -ne 0 -or $checkoutIdentity -notmatch '^[0-9a-f]{64}$') { throw 'Could not determine content checkout identity.' }

# Do not interrupt a healthy service that already runs this exact launch
# identity. A mismatched or unavailable service is replaced only through the
# owned systemd units below, preserving the strict health identity gate.
try {
    $existingHealth = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 2
    if (Test-BackendHealthIdentity -Health $existingHealth -ExpectedCheckoutId $checkoutIdentity) {
        $null = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -WindowStyle Hidden -PassThru -ArgumentList @(
            '-NoProfile', '-NonInteractive', '-File', ('"' + $PSCommandPath + '"'), '-KeepAlive', '-Distro', $Distro
        )
        Write-Output 'SODA Voice Dictation backend is already ready with the current checkout identity.'
        exit 0
    }
}
catch {}

$service = @"
[Unit]
Description=SODA Voice Dictation backend
After=network.target

[Service]
Type=simple
WorkingDirectory=$RepoWslSystemd
ExecStart=$PythonWslSystemd -m uvicorn server:app --host 127.0.0.1 --port 18765
Restart=on-failure
RestartSec=2
TimeoutStopSec=10
KillMode=control-group
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8
Environment=SODA_CHECKOUT_ID=$checkoutIdentity

[Install]
WantedBy=default.target
"@
$service = $service -replace "`r`n", "`n"
$workingDirectoryLine = @($service -split "`n" | Where-Object { $_ -like 'WorkingDirectory=*' })
$execStartLine = @($service -split "`n" | Where-Object { $_ -like 'ExecStart=*' })
if ($workingDirectoryLine.Count -ne 1 -or $workingDirectoryLine[0] -ne "WorkingDirectory=$RepoWslSystemd") {
    throw 'Generated systemd unit has an invalid WorkingDirectory directive.'
}
if ($execStartLine.Count -ne 1) {
    throw 'Generated systemd unit must contain exactly one ExecStart.'
}
Assert-SystemdExecStart -Line $execStartLine[0] -ExpectedPython "$RepoWsl/.venv-wsl/bin/python" | Out-Null

function Invoke-SystemdVerify {
    param([Parameter(Mandatory)][string]$UnitText)
    $unit64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($UnitText))
    $verify = @'
set -eu
unit=$(mktemp --suffix=.service)
trap 'rm -f "$unit"' EXIT
printf '%s' '__SODA_UNIT_BASE64__' | base64 -d > "$unit"
systemd-analyze verify "$unit"
'@
    $verify = $verify.Replace('__SODA_UNIT_BASE64__', $unit64).Replace("`r`n", "`n")
    $verify64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($verify))
    & wsl.exe -d $Distro -- bash -lc "echo '$verify64' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) {
        throw 'systemd-analyze rejected the generated user unit; the previous service was not stopped.'
    }
}

# Validate both a disposable spaced/% fixture and the real candidate with the
# installed systemd parser before any stop/kill command can run. This catches
# systemd 249 directive semantics rather than relying only on our serializer.
$fixtureId = [Guid]::NewGuid().ToString('N')
# Every invocation owns a private, unique directory, executable, result, and
# unit. This makes concurrent launchers independent and lets the trap remove
# the complete disposable fixture without touching any shared /tmp path.
$fixtureRoot = "/tmp/soda path fixture%100-$fixtureId"
$fixtureWorkingDirectory = ConvertTo-SystemdWorkingDirectory $fixtureRoot
$fixtureExecutable = "$fixtureRoot/bin/python-$fixtureId"
$fixturePython = ConvertTo-SystemdExecPath $fixtureExecutable
$fixtureUnitName = "soda-path-fixture-$fixtureId.service"
$fixtureResult = "$fixtureRoot/result-$fixtureId"
$fixtureService = @"
[Unit]
Description=SODA disposable path fixture

[Service]
Type=oneshot
WorkingDirectory=$fixtureWorkingDirectory
ExecStart=$fixturePython
"@
$fixtureSetup = @'
set -eu
mkdir -p '__FIXTURE_ROOT__/bin'
chmod 700 '__FIXTURE_ROOT__'
printf '%s\n' '#!/bin/sh' "pwd > '__FIXTURE_RESULT__'" > '__FIXTURE_EXECUTABLE__'
chmod +x '__FIXTURE_EXECUTABLE__'
'@
$fixtureSetup = $fixtureSetup.Replace('__FIXTURE_ROOT__', $fixtureRoot).Replace('__FIXTURE_RESULT__', $fixtureResult).Replace('__FIXTURE_EXECUTABLE__', $fixtureExecutable).Replace("`r`n", "`n")
$fixtureSetup64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($fixtureSetup))
$fixtureService64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($fixtureService -replace "`r`n", "`n")))
$fixtureStart = @'
set -eu
unit=$HOME/.config/systemd/user/__FIXTURE_UNIT__
fixture_root='__FIXTURE_ROOT__'
fixture_result='__FIXTURE_RESULT__'
mkdir -p "$(dirname "$unit")"
printf '%s' '__FIXTURE_SERVICE_BASE64__' | base64 -d > "$unit"
cleanup() {
    systemctl --user stop __FIXTURE_UNIT__ >/dev/null 2>&1 || true
    rm -f "$unit"
    rm -rf "$fixture_root"
    systemctl --user daemon-reload >/dev/null 2>&1 || true
}
trap cleanup EXIT
systemctl --user daemon-reload
systemctl --user reset-failed __FIXTURE_UNIT__ >/dev/null 2>&1 || true
systemctl --user start __FIXTURE_UNIT__
[ "$(cat "$fixture_result")" = '__FIXTURE_ROOT__' ]
'@
$fixtureStart = $fixtureStart.Replace('__FIXTURE_UNIT__', $fixtureUnitName).Replace('__FIXTURE_ROOT__', $fixtureRoot).Replace('__FIXTURE_RESULT__', $fixtureResult).Replace('__FIXTURE_SERVICE_BASE64__', $fixtureService64).Replace("`r`n", "`n")
$fixtureStart64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($fixtureStart))
try {
    & wsl.exe -d $Distro -- bash -lc "echo '$fixtureSetup64' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the disposable systemd path fixture.' }
    Invoke-SystemdVerify -UnitText ($fixtureService -replace "`r`n", "`n")

    # Verify the same fixture can really start and chdir under the installed
    # user manager, not merely pass systemd-analyze. The result check is after
    # the real systemctl start, and the inner trap removes the unit plus every
    # private fixture file even when startup or the assertion fails.
    & wsl.exe -d $Distro -- bash -lc "echo '$fixtureStart64' | base64 -d | bash"
    if ($LASTEXITCODE -ne 0) { throw 'Disposable systemd fixture could not start in its spaced WorkingDirectory.' }
}
finally {
    # Also cover failures before the inner script can install its trap (for
    # example systemd-analyze rejection), so no executable/result survives.
    $fixtureCleanup = @'
set +e
systemctl --user stop __FIXTURE_UNIT__ >/dev/null 2>&1 || true
rm -f "$HOME/.config/systemd/user/__FIXTURE_UNIT__"
rm -rf '__FIXTURE_ROOT__'
 systemctl --user daemon-reload >/dev/null 2>&1 || true
'@
    $fixtureCleanup = $fixtureCleanup.Replace('__FIXTURE_UNIT__', $fixtureUnitName).Replace('__FIXTURE_ROOT__', $fixtureRoot).Replace("`r`n", "`n")
    & wsl.exe -d $Distro -- bash -lc $fixtureCleanup
}
Invoke-SystemdVerify -UnitText $service

$service64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($service))
$install = @'
set -eu
unit=$HOME/.config/systemd/user/soda-voice-dictation.service
tmp=$(mktemp)
mkdir -p "$(dirname "$unit")"
# Stop only the two SODA units this launcher owns before replacing the unit.
# systemd then reaps their cgroups; no port-wide or process-name kill can touch
# an unrelated service.
systemctl --user stop soda-voice-dictation.service >/dev/null 2>&1 || true
systemctl --user kill --kill-who=all --signal=TERM soda-voice-dictation.service >/dev/null 2>&1 || true
systemctl --user disable google-soda-stt.service >/dev/null 2>&1 || true
systemctl --user stop google-soda-stt.service >/dev/null 2>&1 || true
systemctl --user kill --kill-who=all --signal=TERM google-soda-stt.service >/dev/null 2>&1 || true
echo '__SODA_SERVICE_BASE64__' | base64 -d > "$tmp"
mv "$tmp" "$unit"
systemctl --user daemon-reload
systemctl --user reset-failed soda-voice-dictation.service >/dev/null 2>&1 || true
systemctl --user enable --now soda-voice-dictation.service
'@
$install = $install.Replace('__SODA_SERVICE_BASE64__', $service64).Replace("`r`n", "`n")
$install64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($install))
$installCommand = "echo '$install64' | base64 -d | bash"
& wsl.exe -d $Distro -- bash -lc $installCommand
if ($LASTEXITCODE -ne 0) { throw 'Could not install/start the WSL user service.' }

$null = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -WindowStyle Hidden -PassThru -ArgumentList @(
    '-NoProfile', '-NonInteractive', '-File', ('"' + $PSCommandPath + '"'), '-KeepAlive', '-Distro', $Distro
)

$deadline = [DateTime]::UtcNow.AddSeconds(60)
while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:18765/health' -TimeoutSec 2
        if (Test-BackendHealthIdentity -Health $health -ExpectedCheckoutId $checkoutIdentity) {
            Write-Output 'SODA Voice Dictation backend is ready at http://127.0.0.1:18765/.'
            exit 0
        }
    }
    catch {}
    Start-Sleep -Milliseconds 500
}
throw 'SODA backend did not become ready within 60 seconds.'

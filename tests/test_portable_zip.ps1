$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) "soda-portable-test-$([Guid]::NewGuid().ToString('N'))"
$fixtureRoot = Join-Path $tempRoot 'fixture'
$output = Join-Path $fixtureRoot 'release\portable.zip'
$scriptPath = Join-Path $repoRoot 'ops\create-portable-zip.ps1'

function Invoke-PortableZip {
    param([string]$Fixture, [string]$Destination)

    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $scriptPath `
        -RepoRoot $Fixture -OutputPath $Destination
    if ($LASTEXITCODE -ne 0) { throw 'Portable ZIP script failed.' }
}

try {
    # Build an isolated checkout fixture so the test can add hostile runtime
    # layouts without ever writing fixtures into the repository.
    New-Item -ItemType Directory -Force -Path $fixtureRoot | Out-Null
    foreach ($relative in @(
        'LICENSE', 'README.md', 'requirements.txt', 'requirements-desktop.txt',
        'server.py', 'native_session.py', 'soda_languages.py', 'run-desktop-windows.bat',
        'run-server-linux.sh', 'run-server-windows.bat', 'web\index.html',
        'ops\checkout_identity.py',
        'training\context_profile.py', 'training\personal_vocabulary.py', 'training\confusion_miner.py', 'training\dataset.py', 'training\storage.py',
        'training\bounded_decode.py', 'training\calibrate.py', 'training\evaluate_samples.py', 'training\release.py',
        'desktop\README.md', 'desktop\app.py', 'desktop\audio.py', 'desktop\file_transcribe.py',
        'desktop\core.py', 'desktop\overlay.py', 'desktop\windows.py',
        'ops\start-backend.ps1', 'ops\start-backend-helpers.ps1', 'ops\download-models.py', 'docs\USER_GUIDE.md',
        'vi_enhance\__init__.py', 'vi_enhance\audio_frontend.py',
        'vi_enhance\dynamic_bias.py', 'vi_enhance\postprocessor.py', 'vi_enhance\long_audio.py',
        'vi_enhance\phonetics.py', 'vi_enhance\postprocessor.py',
        'benchmarks\run_vi_benchmarks.py', 'tests\test_vi_enhance.py',
        'docs\VI_ENHANCEMENT_ARCHITECTURE.md',
        'third_party\google-soda\THIRD_PARTY.md',
        'third_party\google-soda\vi-VN\1.3071.0\SHA256SUMS'
    )) {
        $source = Join-Path $repoRoot $relative
        $destination = Join-Path $fixtureRoot $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination
    }

    # Do not initialize Git or stage anything: packaging must include these
    # dirty/untracked sources from a Git-free extracted checkout.
    if (Test-Path -LiteralPath (Join-Path $fixtureRoot '.git')) {
        throw 'Portable fixture unexpectedly has Git metadata.'
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem

    # Arbitrary SODA_STT_HOME layouts and a model-shaped source subtree must
    # be harmless even if their names differ from today's cache conventions.
    $dangerousFiles = @{
        'SODA_STT_HOME\anything\deep\langpacks\custom\payload.bin' = 'model cache'
        'SODA_STT_HOME\engine\future\SODAFiles\libsoda.so' = 'native model'
        'SODA_STT_HOME\cache\downloaded-language.crx3' = 'archive'
        'third_party\google-soda\future\pack\SODAModels\model.tflite' = 'model payload'
        'desktop\runtime\old-cache.zip' = 'old archive'
        'old-research-checkout.zip' = 'old archive'
        'ops\portable-2025.ZIP' = 'old archive'
        'unexpected-model.bin' = 'model payload'
    }
    foreach ($entry in $dangerousFiles.GetEnumerator()) {
        $path = Join-Path $fixtureRoot $entry.Key
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
        [IO.File]::WriteAllText($path, $entry.Value)
    }
    $archiveSource = Join-Path $tempRoot 'old-archive-content'
    New-Item -ItemType Directory -Force -Path $archiveSource | Out-Null
    [IO.File]::WriteAllText((Join-Path $archiveSource 'payload.txt'), 'old checkout payload')
    foreach ($relative in @('desktop\runtime\old-cache.zip', 'old-research-checkout.zip', 'ops\portable-2025.ZIP')) {
        $archivePath = Join-Path $fixtureRoot $relative
        [IO.File]::Delete($archivePath)
        [IO.Compression.ZipFile]::CreateFromDirectory(
            $archiveSource, $archivePath, [IO.Compression.CompressionLevel]::Optimal, $false
        )
    }

    Invoke-PortableZip -Fixture $fixtureRoot -Destination 'release\portable.zip'
    if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw "Relative output was not resolved under RepoRoot: $output"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($output)
    try {
        $names = @($archive.Entries | ForEach-Object { $_.FullName -replace '\\', '/' })
        foreach ($name in @(
            'README.md', 'server.py', 'desktop/core.py', 'ops/start-backend.ps1',
            'training/confusion_miner.py',
            'ops/checkout_identity.py',
            'requirements.txt', 'requirements-desktop.txt', 'run-desktop-windows.bat',
            'run-server-linux.sh', 'run-server-windows.bat', 'web/index.html',
            'vi_enhance/audio_frontend.py',
            'benchmarks/run_vi_benchmarks.py', 'tests/test_vi_enhance.py',
            'docs/VI_ENHANCEMENT_ARCHITECTURE.md',
            'third_party/google-soda/THIRD_PARTY.md',
            'third_party/google-soda/vi-VN/1.3071.0/SHA256SUMS'
        )) {
            if ($names -notcontains $name) { throw "Required file missing from ZIP: $name" }
        }
        foreach ($name in $names) {
            if ($name -match '(?i)(^|/)(SODA_STT_HOME|SODAModels|SODAFiles|langpacks|engine|cache|models)(/|$)' -or
                $name -match '(?i)\.(zip|crx3|so|dll|tflite|pb|mfar|bin)$' -or
                $name -match '(?i)(^|/)(\.git|\.plan|research)(/|$)') {
                throw "Runtime, model, archive, or research content was included: $name"
            }
        }
    }
    finally {
        $archive.Dispose()
    }

    # Extract into a fresh directory and run a dependency-light source check.
    $extractRoot = Join-Path $tempRoot 'extracted'
    [IO.Compression.ZipFile]::ExtractToDirectory($output, $extractRoot)
    foreach ($relative in @('vi_enhance\postprocessor.py', 'training\context_profile.py',
            'training\dataset.py', 'docs\VI_ENHANCEMENT_ARCHITECTURE.md',
            'benchmarks\run_vi_benchmarks.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $extractRoot $relative) -PathType Leaf)) {
            throw "Extracted portable source missing: $relative"
        }
    }
    & python.exe -c "import ast, pathlib; root=pathlib.Path(r'$extractRoot'); [ast.parse(p.read_text(encoding='utf-8')) for p in root.rglob('*.py')]; print('PASS extracted Python static parse')"
    if ($LASTEXITCODE -ne 0) { throw 'Extracted portable Python static parse failed.' }

    # A linked source directory is a hard failure, never a traversal route.
    # The outside payload must not be copied or appear in an existing archive.
    $outsideRoot = Join-Path $tempRoot 'outside-root'
    $outsideFile = Join-Path $outsideRoot 'escape.txt'
    New-Item -ItemType Directory -Force -Path $outsideRoot | Out-Null
    [IO.File]::WriteAllText($outsideFile, 'must never be packaged')
    $linkedDirectory = Join-Path $fixtureRoot 'desktop\linked-out'
    New-Item -ItemType Junction -Path $linkedDirectory -Target $outsideRoot | Out-Null
    try {
        $linkedStdOut = Join-Path $tempRoot 'linked.stdout'
        $linkedStdErr = Join-Path $tempRoot 'linked.stderr'
        $linkedProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath,
            '-RepoRoot', $fixtureRoot, '-OutputPath', 'release\linked.zip'
        ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $linkedStdOut -RedirectStandardError $linkedStdErr
        if ($linkedProcess.ExitCode -eq 0) { throw 'Packager followed a linked source directory.' }
        if (Test-Path -LiteralPath (Join-Path $fixtureRoot 'release\linked.zip')) {
            throw 'Packager produced an archive after rejecting a linked source directory.'
        }
        if ($names -match 'escape.txt' -or $names -match 'linked-out') {
            throw 'Outside linked-directory content appeared in the ZIP.'
        }
    }
    finally {
        # Remove-Item has a Windows PowerShell junction cleanup bug on some
        # hosts; rmdir removes the link itself without touching its target.
        & cmd.exe /c rmdir /s /q "$linkedDirectory" 2>$null
    }

    # Removing a required launcher helper must fail closed before creating a
    # release. This is deliberately a negative test for runtime dependency
    # drift, not merely a missing optional file.
    $missingRoot = Join-Path $tempRoot 'missing-helper'
    Copy-Item -LiteralPath $fixtureRoot -Destination $missingRoot -Recurse
    Remove-Item -LiteralPath (Join-Path $missingRoot 'ops\start-backend-helpers.ps1') -Force
    $missingStdOut = Join-Path $tempRoot 'missing.stdout'
    $missingStdErr = Join-Path $tempRoot 'missing.stderr'
    $missingProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath,
        '-RepoRoot', $missingRoot, '-OutputPath', 'release\missing.zip'
    ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $missingStdOut -RedirectStandardError $missingStdErr
    if ($missingProcess.ExitCode -eq 0) { throw 'Packager accepted a checkout missing a required launcher helper.' }
    if (Test-Path -LiteralPath (Join-Path $missingRoot 'release\missing.zip')) { throw 'Missing-helper package was produced.' }

    # A root runtime input is also mandatory; the package must not silently
    # degrade into a source tree without its install manifest.
    $missingRuntimeRoot = Join-Path $tempRoot 'missing-runtime'
    Copy-Item -LiteralPath $fixtureRoot -Destination $missingRuntimeRoot -Recurse
    Remove-Item -LiteralPath (Join-Path $missingRuntimeRoot 'web\index.html') -Force
    $missingRuntimeProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath,
        '-RepoRoot', $missingRuntimeRoot, '-OutputPath', 'release\missing-runtime.zip'
    ) -Wait -PassThru -NoNewWindow
    if ($missingRuntimeProcess.ExitCode -eq 0) { throw 'Packager accepted a checkout missing web/index.html.' }
    if (Test-Path -LiteralPath (Join-Path $missingRuntimeRoot 'release\missing-runtime.zip')) { throw 'Missing-runtime package was produced.' }

    # Re-running against the same destination exercises atomic replacement of
    # an existing ZIP without allowing the temporary archive to be packaged.
    Invoke-PortableZip -Fixture $fixtureRoot -Destination 'release\portable.zip'
    if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw 'Existing ZIP was not atomically replaced.'
    }

    # A non-ZIP existing source file must never be replaced, while an existing
    # ZIP is a supported atomic replacement target.
    $sourcePath = Join-Path $fixtureRoot 'keep-me.txt'
    [IO.File]::WriteAllText($sourcePath, 'do not replace')
    $badStdOut = Join-Path $tempRoot 'invalid-output.stdout'
    $badStdErr = Join-Path $tempRoot 'invalid-output.stderr'
    $badProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $scriptPath,
        '-RepoRoot', $fixtureRoot, '-OutputPath', 'keep-me.txt'
    ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $badStdOut -RedirectStandardError $badStdErr
    if ($badProcess.ExitCode -eq 0) { throw 'Script replaced an existing non-ZIP source file.' }
    if ([IO.File]::ReadAllText($sourcePath) -ne 'do not replace') {
        throw 'Existing non-ZIP source file was modified.'
    }

    Write-Output 'PASS portable ZIP contents, fixture exclusions, path safety, and source protection.'
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

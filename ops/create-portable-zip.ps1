[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\', '/')
if (-not $OutputPath) {
    $repoName = Split-Path -Leaf $RepoRoot
    $OutputPath = Join-Path (Split-Path -Parent $RepoRoot) "$repoName-portable.zip"
}

# Relative output paths are deliberately rooted at the checkout, not at the
# process CWD (which can differ when this script is called by another shell).
if ([IO.Path]::IsPathRooted($OutputPath)) {
    $outputCandidate = $OutputPath
}
else {
    $outputCandidate = Join-Path $RepoRoot $OutputPath
}
$OutputPath = [IO.Path]::GetFullPath($outputCandidate)

if ([IO.Path]::GetExtension($OutputPath) -ine '.zip') {
    throw "OutputPath must name a .zip file: $OutputPath"
}
if (Test-Path -LiteralPath $OutputPath -PathType Container) {
    throw "OutputPath must be a file, not a directory: $OutputPath"
}
if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
    $existingOutput = Get-Item -LiteralPath $OutputPath
    if ($existingOutput.Extension -ine '.zip') {
        throw "Refusing to replace an existing non-ZIP source file: $OutputPath"
    }
}

$outputParent = Split-Path -Parent $OutputPath
if (-not $outputParent) {
    throw "OutputPath has no parent directory: $OutputPath"
}
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null

$temporaryRoot = Join-Path ([IO.Path]::GetTempPath()) "soda-portable-$([Guid]::NewGuid().ToString('N'))"
$stagingRoot = Join-Path $temporaryRoot 'content'
$temporaryArchive = Join-Path $outputParent ".soda-portable-$([Guid]::NewGuid().ToString('N')).zip"
$copiedFiles = 0

function Get-RelativeRepoPath {
    param([string]$Path)

    $full = [IO.Path]::GetFullPath($Path)
    $rootPrefix = $RepoRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Source path escapes RepoRoot: $full"
    }
    return $full.Substring($rootPrefix.Length).TrimStart('\', '/')
}

function Assert-SourceDirectory {
    param([string]$Path)

    $full = [IO.Path]::GetFullPath($Path)
    $relative = Get-RelativeRepoPath -Path $full
    $current = $RepoRoot
    foreach ($segment in ($relative -split '[\\/]')) {
        $current = Join-Path $current $segment
        $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to traverse linked source path: $current"
        }
    }
    return Get-Item -LiteralPath $full -Force
}

function Test-ExcludedPath {
    param([System.IO.FileInfo]$File)

    $relative = Get-RelativeRepoPath -Path $File.FullName
    $segments = $relative -split '[\\/]'
    $excludedDirectoryNames = @(
        '.git', '.plan', 'data', 'samples', 'artifacts', 'soda-cache',
        '.cache', '__pycache__', 'research', 'SODA_STT_HOME',
        'SODAModels', 'SODAFiles', 'langpacks', 'engine', 'cache', 'models'
    )
    foreach ($segment in $segments) {
        foreach ($excludedName in $excludedDirectoryNames) {
            if ($segment -ieq $excludedName) {
                return $true
            }
        }
        if ($segment -like '.venv*') {
            return $true
        }
    }

    # Archives and native/model payloads are never portable-package inputs,
    # even when a future checkout puts one under an otherwise allowed source
    # directory. In particular, this covers CRX3 downloads and libsoda.so.
    if ($File.Name -ieq 'libsoda.so' -or $File.Extension.ToLowerInvariant() -in @(
            '.zip', '.crx3', '.so', '.dll', '.tflite', '.pb', '.mfar', '.model',
            '.bin', '.pyc', '.pyo', '.log', '.wav', '.mp3', '.m4a', '.aac',
            '.flac', '.ogg'
        )) {
        return $true
    }

    # This also protects an explicitly requested output located inside the
    # checkout. The source allowlist below independently excludes all zips.
    if ([StringComparer]::OrdinalIgnoreCase.Equals($File.FullName, $OutputPath)) {
        return $true
    }
    return $false
}

function Test-AllowedSourceFile {
    param([System.IO.FileInfo]$File, [System.Collections.Generic.HashSet[string]]$ManifestPaths)

    if (Test-ExcludedPath -File $File) {
        return $false
    }
    if ($File.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        # Do not follow a checkout symlink/junction into a cache or another
        # working tree while creating a release artifact.
        return $false
    }
    $relative = Get-RelativeRepoPath -Path $File.FullName
    if (-not $ManifestPaths.Contains($relative)) { return $false }
    return ($File.Name -ieq 'LICENSE') -or ($File.Name -ieq 'NOTICE') -or $File.Extension.ToLowerInvariant() -in @(
        '.py', '.ps1', '.bat', '.sh', '.md', '.txt', '.html', '.js', '.css',
        '.json', '.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf'
    )
}

function Copy-PortableFile {
    param([System.IO.FileInfo]$File)

    $relative = Get-RelativeRepoPath -Path $File.FullName
    $destination = Join-Path $stagingRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $destination
    $script:copiedFiles++
}

try {
    New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
    # Explicit manifest, independent of Git, so untracked sources and Git-free
    # extracted releases cannot silently produce a broken artifact.
    $manifest = @(
        'LICENSE', 'NOTICE', 'NETWORK.md', 'README.md', 'TRAINING_WORKFLOW.md',
        'native_session.py', 'requirements.txt', 'requirements-desktop.txt',
        'requirements-model-research.txt', 'run-desktop-windows.bat',
        'run-server-linux.sh', 'run-server-windows.bat', 'server.py',
        'soda_languages.py', 'tests/test_vi_enhance.py'
    )
    foreach ($directoryName in @('desktop', 'docs', 'ops', 'training', 'web', 'vi_enhance', 'benchmarks')) {
        $directoryPath = Join-Path $RepoRoot $directoryName
        if (Test-Path -LiteralPath $directoryPath -PathType Container) {
            Assert-SourceDirectory -Path $directoryPath | Out-Null
            $manifest += @(Get-ChildItem -LiteralPath $directoryPath -Force -Recurse -File |
                ForEach-Object { Get-RelativeRepoPath $_.FullName })
        }
    }
    $manifestPaths = [System.Collections.Generic.HashSet[string]]::new(
        [string[]]($manifest | ForEach-Object { $_ -replace '/', '\' }),
        [StringComparer]::OrdinalIgnoreCase)
    $requiredSources = @(
        'requirements.txt', 'requirements-desktop.txt', 'run-desktop-windows.bat',
        'web/index.html', 'server.py', 'native_session.py', 'soda_languages.py',
        'ops/checkout_identity.py', 'ops/start-backend.ps1', 'ops/start-backend-helpers.ps1',
        'ops/download-models.py',
        'training/context_profile.py', 'training/personal_vocabulary.py',
        'training/confusion_miner.py', 'training/dataset.py', 'training/storage.py', 'training/bounded_decode.py',
        'training/calibrate.py', 'training/evaluate_samples.py', 'training/release.py',
        'desktop/app.py', 'desktop/core.py', 'desktop/audio.py', 'desktop/file_transcribe.py',
        'desktop/overlay.py', 'desktop/windows.py',
        'vi_enhance/__init__.py', 'vi_enhance/audio_frontend.py', 'vi_enhance/long_audio.py',
        'vi_enhance/postprocessor.py', 'vi_enhance/dynamic_bias.py', 'vi_enhance/phonetics.py'
    )
    foreach ($required in $requiredSources) {
        $requiredPath = Join-Path $RepoRoot $required
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required portable source is missing: $required"
        }
        $requiredFile = Get-Item -LiteralPath $requiredPath -Force
        if ($requiredFile.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Required portable source is linked or unsafe: $required"
        }
        if (-not (Test-AllowedSourceFile -File $requiredFile -ManifestPaths ([System.Collections.Generic.HashSet[string]]::new([string[]]@($required -replace '/', '\'), [StringComparer]::OrdinalIgnoreCase)))) {
            throw "Required portable source is excluded or unsafe: $required"
        }
        $manifestPaths.Add(($required -replace '/', '\')) | Out-Null
    }

    # The package is intentionally an allowlist. In particular, research/,
    # ignored checkout content, runtime caches, and model directories are not
    # package roots at all. This is safer than adding new denylist entries for
    # every possible SODA_STT_HOME layout.
    $rootFiles = @(
        'LICENSE', 'NOTICE', 'NETWORK.md', 'README.md', 'TRAINING_WORKFLOW.md',
        'native_session.py', 'requirements.txt', 'requirements-desktop.txt',
        'requirements-model-research.txt', 'run-desktop-windows.bat',
        'run-server-linux.sh', 'run-server-windows.bat', 'server.py',
        'soda_languages.py'
    )
    foreach ($relative in $rootFiles) {
        $filePath = Join-Path $RepoRoot $relative
        if (Test-Path -LiteralPath $filePath -PathType Leaf) {
            $file = Get-Item -LiteralPath $filePath
            if (Test-AllowedSourceFile -File $file -ManifestPaths $manifestPaths) {
                Copy-PortableFile -File $file
            }
        }
    }

    foreach ($directoryName in @('desktop', 'docs', 'ops', 'training', 'web', 'vi_enhance', 'benchmarks')) {
        $directoryPath = Join-Path $RepoRoot $directoryName
        if (-not (Test-Path -LiteralPath $directoryPath -PathType Container)) {
            continue
        }
        Assert-SourceDirectory -Path $directoryPath | Out-Null
        # Preflight all descendants before copying. A junction/symlink is a
        # rejected source root, not a path to follow into an external tree.
        foreach ($childDirectory in Get-ChildItem -LiteralPath $directoryPath -Force -Recurse -Directory) {
            if ($childDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing to traverse linked source directory: $($childDirectory.FullName)"
            }
            Get-RelativeRepoPath -Path $childDirectory.FullName | Out-Null
        }
        foreach ($file in Get-ChildItem -LiteralPath $directoryPath -Force -Recurse -File) {
            Get-RelativeRepoPath -Path $file.FullName | Out-Null
            if (Test-AllowedSourceFile -File $file -ManifestPaths $manifestPaths) {
                Copy-PortableFile -File $file
            }
        }
    }

    # Keep the enhancement regression test explicitly allowlisted instead of
    # packaging every repository test (some tests require local tools/data).
    $enhancementTest = Join-Path $RepoRoot 'tests/test_vi_enhance.py'
    if (Test-Path -LiteralPath $enhancementTest -PathType Leaf) {
        $file = Get-Item -LiteralPath $enhancementTest
        if (Test-AllowedSourceFile -File $file -ManifestPaths $manifestPaths) {
            Copy-PortableFile -File $file
        }
    }

    # Only redistribution notices and integrity manifests are retained from
    # third_party; no model/component payload is copied from that tree.
    $thirdPartyPath = Join-Path $RepoRoot 'third_party'
    if (Test-Path -LiteralPath $thirdPartyPath -PathType Container) {
        foreach ($file in Get-ChildItem -LiteralPath $thirdPartyPath -Force -Recurse -File) {
            if ($file.Name -in @('THIRD_PARTY.md', 'NOTICE', 'NOTICE.txt', 'LICENSE', 'LICENSE.txt', 'SHA256SUMS') -and
                -not (Test-ExcludedPath -File $file) -and
                -not ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                Copy-PortableFile -File $file
            }
        }
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $stagingRoot,
        $temporaryArchive,
        [IO.Compression.CompressionLevel]::Optimal,
        $false
    )

    # The temporary archive is in the destination directory, so replacement is
    # same-volume and atomic. File.Replace is used only for an existing ZIP;
    # non-ZIP source files were rejected before any staging work began.
    if (Test-Path -LiteralPath $OutputPath -PathType Leaf) {
        $backupArchive = Join-Path $outputParent ".soda-portable-backup-$([Guid]::NewGuid().ToString('N')).zip"
        [IO.File]::Replace($temporaryArchive, $OutputPath, $backupArchive, $true)
        Remove-Item -LiteralPath $backupArchive -Force -ErrorAction SilentlyContinue
    }
    else {
        [IO.File]::Move($temporaryArchive, $OutputPath)
    }
    $temporaryArchive = $null

    $zip = Get-Item -LiteralPath $OutputPath
    $sizeMiB = [Math]::Round($zip.Length / 1MB, 2)
    Write-Output "Portable ZIP: $($zip.FullName)"
    Write-Output "Size: $($zip.Length) bytes ($sizeMiB MiB)"
    Write-Output "Included files: $copiedFiles"
}
finally {
    if ($temporaryArchive -and (Test-Path -LiteralPath $temporaryArchive)) {
        Remove-Item -LiteralPath $temporaryArchive -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}

# Pure launcher helpers. This file has no side effects so it can be probed on
# Windows without WSL or a running backend.
function ConvertTo-SystemdPath {
    param([Parameter(Mandatory)][string]$Value)
    # systemd parses ExecStart words and directive values with C-style unit
    # escapes. Use \x20 for spaces (not shell-style ``\ ``), \x5c for a
    # literal backslash, and double percent signs so literals are not
    # interpreted as specifiers (all supported by systemd 249).
    return $Value.Replace('\', '\x5c').Replace(' ', '\x20').Replace('%', '%%')
}

function ConvertTo-SystemdExecPath {
    param([Parameter(Mandatory)][string]$Value)
    # ExecStart permits a quoted word; unlike WorkingDirectory, quoting keeps
    # a path containing spaces as one executable token and avoids systemd's
    # special-character executable-name rejection.
    $escaped = $Value.Replace('\', '\x5c').Replace('%', '%%')
    return '"' + $escaped + '"'
}

function ConvertTo-SystemdWorkingDirectory {
    param([Parameter(Mandatory)][string]$Value)
    # WorkingDirectory is the remainder of a unit-file assignment, not an
    # ExecStart word. Keep literal spaces unquoted (systemd 249 preserves
    # them here) and escape percent so a checkout name cannot become a unit
    # specifier. ExecStart has its own serializer above because it is parsed
    # as a list of words.
    return $Value.Replace('%', '%%')
}

function ConvertFrom-SystemdExecWords {
    param([Parameter(Mandatory)][string]$Value)
    $words = [Collections.Generic.List[string]]::new()
    $current = [Text.StringBuilder]::new()
    $escaped = $false
    $quoted = $false
    for ($i = 0; $i -lt $Value.Length; $i++) {
        $char = $Value[$i]
        if ($escaped) {
            [void]$current.Append($char)
            $escaped = $false
        }
        elseif ($char -eq '\') {
            if ($i + 3 -lt $Value.Length -and $Value[$i + 1] -eq 'x' -and
                $Value.Substring($i + 2, 2) -match '^[0-9A-Fa-f]{2}$') {
                [void]$current.Append([char][Convert]::ToInt32($Value.Substring($i + 2, 2), 16))
                $i += 3
            }
            else {
                $escaped = $true
            }
        }
        elseif ($char -eq '"') {
            $quoted = -not $quoted
        }
        elseif ($char -eq '%' -and $i + 1 -lt $Value.Length -and $Value[$i + 1] -eq '%') {
            [void]$current.Append('%')
            $i++
        }
        elseif ([char]::IsWhiteSpace($char) -and -not $quoted) {
            if ($current.Length) {
                $words.Add($current.ToString())
                [void]$current.Clear()
            }
        }
        else {
            [void]$current.Append($char)
        }
    }
    if ($quoted) { throw 'Generated systemd command has an unterminated quote.' }
    if ($escaped) { throw 'Generated systemd command ends with an escape.' }
    if ($current.Length) { $words.Add($current.ToString()) }
    return @($words)
}

function Test-BackendHealthIdentity {
    param(
        [Parameter(Mandatory)]$Health,
        [Parameter(Mandatory)][string]$ExpectedCheckoutId
    )
    return [bool]($Health -and $Health.ok -and
        $Health.checkout_id -eq $ExpectedCheckoutId)
}

function Assert-SystemdExecStart {
    param(
        [Parameter(Mandatory)][string]$Line,
        [Parameter(Mandatory)][string]$ExpectedPython,
        [string]$ExpectedModule = 'uvicorn'
    )
    if (-not $Line.StartsWith('ExecStart=', [StringComparison]::Ordinal)) {
        throw 'Generated systemd unit has no ExecStart directive.'
    }
    $words = @(ConvertFrom-SystemdExecWords -Value $Line.Substring('ExecStart='.Length))
    if ($words.Count -lt 4 -or $words[0] -ne $ExpectedPython -or
        $words[1] -ne '-m' -or $words[2] -ne $ExpectedModule -or
        $words[3] -notmatch '^server:app$') {
        throw "Generated systemd ExecStart does not parse to the expected launcher: $Line"
    }
    return $true
}

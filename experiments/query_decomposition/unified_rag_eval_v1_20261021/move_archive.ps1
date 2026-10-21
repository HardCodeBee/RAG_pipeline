$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$archiveRoot = [System.IO.Path]::GetFullPath((Join-Path $taskRoot 'archives\rag_evaluation_history_20261021'))
$indexPath = Join-Path $archiveRoot 'INDEX.json'
$index = Get-Content -LiteralPath $indexPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($index.status -ne 'prepared_not_moved') { throw 'Archive is not in prepared state' }
foreach ($op in $index.operations) {
    $sourceAbs = [System.IO.Path]::GetFullPath($op.source)
    $targetAbs = [System.IO.Path]::GetFullPath($op.target)
    if (-not $sourceAbs.StartsWith($taskRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Source escapes workspace' }
    if (-not $targetAbs.StartsWith($archiveRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Destination escapes archive' }
    $actualSource = (Resolve-Path -LiteralPath $sourceAbs).ProviderPath
    if ($actualSource -ne $sourceAbs) { throw 'Unexpected source resolution' }
    $sourceItem = Get-Item -LiteralPath $sourceAbs -Force
    if ($sourceItem.LinkType) { throw 'Source already is a link' }
    if (Test-Path -LiteralPath $targetAbs) { throw 'Destination already exists' }
    New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($targetAbs)) -Force | Out-Null
    Move-Item -LiteralPath $sourceAbs -Destination $targetAbs
    try {
        New-Item -ItemType Junction -Path $sourceAbs -Target $targetAbs | Out-Null
    } catch {
        # The same checked absolute paths are used for rollback; no deletion.
        if (-not (Test-Path -LiteralPath $sourceAbs)) { Move-Item -LiteralPath $targetAbs -Destination $sourceAbs }
        throw
    }
}
Write-Output ('Moved directories and created compatibility junctions: ' + $index.operations.Count)

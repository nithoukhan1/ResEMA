param(
  [string]$KaggleOwner="hebut123",
  [string]$SourceRoot="C:\Users\ik_pa\Downloads\results (15)bl00\BL00_CONTINUOUS_B16_OUTPUT",
  [string]$Destination="C:\Users\ik_pa\Downloads\BL00_EPOCH095_DATASET"
)
$ErrorActionPreference="Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$expected="05a87470d32ec7506f9d4212ce7ba5881ec703299883c2c445505bf2bc6e0c5d"
$zipName="BL00_EPOCH095_PORTABLE_RECOVERY.zip"
$zipFile=Join-Path $Destination $zipName
function Need([bool]$ok,[string]$reason) { if (-not $ok) { throw "BL00_PORTABLE_$reason" } }
function SHA([string]$p) {
  Need (Test-Path -LiteralPath $p -PathType Leaf) ("MISSING_FILE_"+$p)
  return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
}
function JSON([string]$p) { return (Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json) }
Need (Test-Path -LiteralPath $SourceRoot -PathType Container) "SOURCE_ROOT_MISSING"
$g4name="g4/BL00_EPOCH095_two_rank_manifest.json"
$g4path=Join-Path $SourceRoot "g4\BL00_EPOCH095_two_rank_manifest.json"
Need ((SHA $g4path) -eq $expected) "G4_SHA_MISMATCH"
$g4=JSON $g4path
Need ($g4.arm -eq "BL00" -and $g4.complete -eq $true -and
      [int]$g4.completed_epochs -eq 95 -and [int]$g4.next_epoch_index -eq 95 -and
      [int]$g4.optimizer_updates -eq 21767 -and $g4.b_test_access -eq "NONE") "G4_IDENTITY"
$pointer=JSON (Join-Path $SourceRoot "g4\LATEST_MANIFEST.json")
Need ($pointer.manifest -eq "BL00_EPOCH095_two_rank_manifest.json" -and
      $pointer.manifest_sha256 -eq $expected -and [int]$pointer.completed_epochs -eq 95) "LATEST_POINTER"
$g4members=@($g4.files.PSObject.Properties)
Need ($g4members.Count -eq 3) "G4_EXPECT_THREE_MEMBERS"
$boot=(JSON (Join-Path $SourceRoot "bootstrap\SOURCE_SHA256.json"))
$bootmembers=@($boot.PSObject.Properties)
Need ($bootmembers.Count -eq 13) "BOOTSTRAP_EXPECT_THIRTEEN_MEMBERS"
$payload=New-Object 'System.Collections.Generic.List[object]'
function AddFile([string]$relative,[string]$expectedSha) {
  $relative=$relative.Replace("\","/")
  Need (-not $relative.Contains("..") -and -not $relative.Contains(":") -and
        -not $relative.StartsWith("/")) ("UNSAFE_PATH_"+$relative)
  $src=Join-Path $SourceRoot ($relative.Replace("/","\"))
  $sha=SHA $src
  if ($expectedSha) { Need ($sha -eq $expectedSha.ToLowerInvariant()) ("SHA_"+$relative) }
  [void]$payload.Add([pscustomobject]@{rel=$relative;path=$src;sha=$sha})
}
AddFile $g4name $expected
AddFile "g4/LATEST_MANIFEST.json" ""
foreach($m in $g4members) {
  Need ([string]$m.Name -match '^BL00_EPOCH095_(best|rank0|rank1)\.pt$') "G4_UNEXPECTED_NAME"
  AddFile ("g4/"+$m.Name) ([string]$m.Value)
}
AddFile "bootstrap/SOURCE_SHA256.json" ""
foreach($m in $bootmembers) { AddFile ("bootstrap/"+$m.Name) ([string]$m.Value) }
Need ($payload.Count -eq 19) "EXPECTED_NINETEEN_FILES"
Need (@($payload | Select-Object -ExpandProperty rel -Unique).Count -eq 19) "DUPLICATE_FILE"
if (-not (Test-Path -LiteralPath $Destination)) {
  New-Item -ItemType Directory -Path $Destination | Out-Null
}
Need (-not (Test-Path -LiteralPath $zipFile)) "OUTPUT_ZIP_ALREADY_EXISTS"
Need (-not (Test-Path -LiteralPath "$zipFile.sha256")) "OUTPUT_SHA_ALREADY_EXISTS"
Need (-not (Test-Path -LiteralPath (Join-Path $Destination "dataset-metadata.json"))) "METADATA_EXISTS"
$index=[ordered]@{
  schema="BL00_EPOCH095_PORTABLE_TRANSPORT_V1"
  arm="BL00";completed_epochs=95;optimizer_updates=21767
  pinned_manifest_sha256=$expected;b_test_access="NONE";files_sha256=[ordered]@{}
}
foreach($entry in $payload) {
  $index.files_sha256["BL00_CONTINUOUS_B16_OUTPUT/"+$entry.rel]=$entry.sha
}
Write-Host "SHA PASS: Epoch-95 manifest + rank0/rank1/best + 13 pinned bootstrap files"
Write-Host "Packaging 19 source files into one portable ZIP..."
$zip=[System.IO.Compression.ZipFile]::Open($zipFile,[System.IO.Compression.ZipArchiveMode]::Create)
try {
  foreach($entry in $payload) {
    [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
      $zip,$entry.path,("BL00_CONTINUOUS_B16_OUTPUT/"+$entry.rel),
      [System.IO.Compression.CompressionLevel]::Fastest
    )
  }
  $indexEntry=$zip.CreateEntry("BL00_EPOCH095_PORTABLE_TRANSPORT.json")
  $writer=New-Object System.IO.StreamWriter($indexEntry.Open())
  try { $writer.Write(($index | ConvertTo-Json -Depth 10)) } finally { $writer.Dispose() }
} finally { $zip.Dispose() }
$zipSHA=SHA $zipFile
Set-Content -LiteralPath "$zipFile.sha256" -Encoding ASCII -Value ($zipSHA+"  "+$zipName)
$meta=@{title="BL00 Epoch 95 Private Recovery";id=($KaggleOwner+"/bl00-e95-recovery-20261109");licenses=@(@{name="other"})}
$meta | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Destination "dataset-metadata.json") -Encoding UTF8
Write-Host "BL00 PORTABLE DATASET PACKAGE: PASS"
Write-Host "DATASET DIR:" $Destination
Write-Host "DATASET ID:" $meta.id
Write-Host "ZIP SHA256:" $zipSHA
Write-Host "ZIP SIZE MiB:" ([math]::Round((Get-Item $zipFile).Length/1MB,2))
Write-Host "B-TEST: SEALED"
Write-Host "UPLOAD: kaggle datasets create -p <DATASET DIR> -t -r skip"

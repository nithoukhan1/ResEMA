<#
.SYNOPSIS
  Chat26 C1 Epoch-1: verify both Kaggle-exported G4 archives independently
  on Windows, then stage verified artifacts for a future Kaggle resume dataset.
.DESCRIPTION
  Nonofficial ONLY. Does NOT train, authorize an official run, access B-TEST,
  alter archived bytes, or declare offsite persistence independently of user
  action. Run only after actually downloading both original archives, their
  sidecars, and C1_OUTPUT_INDEX.json from a completed Kaggle Save Version.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$InputFolder,
  [string]$OutputFolder = ""
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Require([bool]$Condition,[string]$Reason) {
  if (-not $Condition) { throw ("CHAT26_OFFSITE_FAIL_CLOSED_" + $Reason) }
}
function Sha256([string]$Path) {
  return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
$Pin = 'fa6a7675daae385a04c7b77e69a9de29f9e1b2f3'
$Expected = @{
  MV00 = '4fc1c22adda6265a48917de15efea4b1f0766db9a8ed6502aaedfe208e7d44e4'
  MV01 = '5b064fdb8a41a3d525fc6f45650e79bdeb53395fd7fc1a09fcc3d5150a83f4e7'
}
$In = (Resolve-Path -LiteralPath $InputFolder -ErrorAction Stop).Path
Require (Test-Path -LiteralPath $In -PathType Container) 'INPUT_DIRECTORY'
if ($OutputFolder -eq '') {
  $OutputFolder = Join-Path $In 'CHAT26_EPOCH1_VERIFIED_FOR_KAGGLE'
}
$Out = [System.IO.Path]::GetFullPath($OutputFolder)
Require ($Out -ne $In) 'SEPARATE_STAGING_ROOT_REQUIRED'
Require (-not (Test-Path -LiteralPath $Out)) 'NO_OVERWRITE_OUTPUT_DIRECTORY'
$indexFiles = @(Get-ChildItem -LiteralPath $In -File -Recurse -Filter 'C1_OUTPUT_INDEX.json')
Require ($indexFiles.Count -eq 1) 'EXACTLY_ONE_OUTPUT_INDEX_REQUIRED'
$Index = Get-Content -LiteralPath $indexFiles[0].FullName -Raw -Encoding UTF8 | ConvertFrom-Json
Require ($Index.schema -eq 'DETECTION_CHAT25_PRODUCTION_SESSION_KAGGLE_OUTPUT_INDEX_C1') 'OUTPUT_SCHEMA'
Require ($Index.status -eq 'BOTH_ARMS_LOCAL_OUTPUTS_VERIFIED_EXTERNAL_PENDING') 'OUTPUT_STATUS'
Require ($Index.published_release_commit -eq $Pin) 'PINNED_GIT_RELEASE'
Require ($Index.run_class -eq 'NONOFFICIAL_FINALPATH_ACCEPTANCE') 'NONOFFICIAL_ONLY'
Require ($Index.b_test_access -eq 'NONE') 'BTEST_FORBIDDEN'
Require ($Index.both_arms_completed -eq $true) 'TWO_COMPLETED_ARMS'
Require ($Index.offsite_persistence_verified -eq $false) 'HISTORICAL_OFFSITE_FALSE'
Require ((@($Index.failures.PSObject.Properties).Count) -eq 0) 'SOURCE_FAILURES'
Require ((@($Index.arms.PSObject.Properties).Count) -eq 2) 'EXACT_TWO_ARM_RECORDS'
$pythonExe = (Get-Command python -ErrorAction Stop).Source
# Validate download and on-disk bytes BEFORE creating attestation.
$Files = @{}
foreach ($arm in @('MV00','MV01')) {
  $name = $arm + '_epoch001_G4_LOCAL_EXPORT.zip.zipblob'
  $found = @(Get-ChildItem -LiteralPath $In -File -Recurse -Filter $name)
  Require ($found.Count -eq 1) ('EXACT_ONE_ZIP_' + $arm)
  $blob = $found[0]
  $side = $blob.FullName + '.sha256'
  Require (Test-Path -LiteralPath $side -PathType Leaf) ('MISSING_SIDECAR_' + $arm)
  $actual = Sha256 $blob.FullName
  Require ($actual -eq $Expected[$arm]) ('ARCHIVE_HASH_PIN_' + $arm)
  $tokens = @((Get-Content -LiteralPath $side -Raw -Encoding UTF8).Trim() -split '\s+')
  Require ($tokens.Count -eq 2) ('SIDECAR_FORMAT_' + $arm)
  Require ($tokens[0].ToLowerInvariant() -eq $actual -and $tokens[1] -ceq $name) ('SIDECAR_PIN_' + $arm)
  $entry = $Index.arms.PSObject.Properties[$arm].Value
  Require ($null -ne $entry) ('INDEX_ARM_MISSING_' + $arm)
  Require ($entry.arm -eq $arm) ('INDEX_ARM_' + $arm)
  Require ($entry.run_id -eq ('C1ACCEPT_' + $arm + '_20261021')) ('INDEX_RUN_ID_' + $arm)
  Require ([int]$entry.completed_epoch_zero_based -eq 0 -and $entry.terminal -eq $false) ('INDEX_EPOCH_' + $arm)
  Require ($entry.zip_sha256 -eq $actual) ('INDEX_ZIP_SHA_' + $arm)
  Require ([string]$entry.zip_relative -eq ($arm + '/' + $name)) ('INDEX_ZIP_RELATIVE_' + $arm)
  Require ([long]$entry.zip_bytes -eq $blob.Length) ('INDEX_ZIP_SIZE_' + $arm)
  $Files[$arm] = @{ blob=$blob.FullName;side=$side;sha=$actual;name=$name }
  Write-Host "$arm ORIGINAL_DOWNLOADED_BYTES_VERIFIED_SHA256=$actual"
}
New-Item -ItemType Directory -Path $Out -ErrorAction Stop | Out-Null
$toolFile = Join-Path $Out 'R9_OFFSESSION_TRANSPORT_R1_PINNED.py'
$url = 'https://raw.githubusercontent.com/nithoukhan1/ResEMA/' + $Pin + '/research/r9_training/code/R9_OFFSESSION_TRANSPORT_R1.py'
Invoke-WebRequest -Uri $url -OutFile $toolFile -UseBasicParsing -ErrorAction Stop
Require (Test-Path -LiteralPath $toolFile -PathType Leaf) 'TRANSPORT_SOURCE_NOT_DOWNLOADED'
foreach ($arm in @('MV00','MV01')) {
  $dir = Join-Path $Out $arm
  New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
  $blobNew = Join-Path $dir $Files[$arm].name
  $sideNew = $blobNew + '.sha256'
  Copy-Item -LiteralPath $Files[$arm].blob -Destination $blobNew -ErrorAction Stop
  Copy-Item -LiteralPath $Files[$arm].side -Destination $sideNew -ErrorAction Stop
  Require ((Sha256 $blobNew) -eq $Expected[$arm]) ('STAGED_BYTES_HASH_' + $arm)
  $attest = Join-Path $dir ($arm + '_WINDOWS_OFFSESSION_VERIFY_R1.json')
  & $pythonExe -u $toolFile verify --archive $blobNew --sidecar $sideNew --attestation $attest --location-label ('WINDOWS_DOWNLOADED_INDEPENDENT_G4_EPOCH1_' + $arm)
  Require ($LASTEXITCODE -eq 0) ('TRANSPORT_INSPECTION_FAILED_' + $arm)
  $obj = Get-Content -LiteralPath $attest -Raw -Encoding UTF8 | ConvertFrom-Json
  Require ($obj.schema -eq 'R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE') ('ATTEST_SCHEMA_' + $arm)
  Require ($obj.status -eq 'PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION') ('ATTEST_STATUS_' + $arm)
  Require ($obj.arm -eq $arm -and [int]$obj.completed_epoch -eq 0 -and $obj.terminal -eq $false) ('ATTEST_CURSOR_' + $arm)
  Require ($obj.zip_sha256 -eq $Expected[$arm]) ('ATTEST_SHA_' + $arm)
  Require ($obj.external_persistence_confirmed_by_software -eq $false) ('NO_SOFTWARE_PERSISTENCE_CLAIM_' + $arm)
  Write-Host "$arm WINDOWS_OFFSESSION_BYTE_INSPECTION_PASS"
}
Copy-Item -LiteralPath $indexFiles[0].FullName -Destination (Join-Path $Out 'C1_OUTPUT_INDEX.json') -ErrorAction Stop
Write-Host "BOTH_WINDOWS_OFFSESSION_VERIFICATIONS_PASS"
Write-Host "STAGING_DIRECTORY=$Out"
Write-Host "OFFICIAL_TRAINING_AUTHORIZED_FALSE; B_TEST_NONE"
Write-Host "NOW_CREATE_PRIVATE_KAGGLE_DATASET_FROM_STAGING_FOLDER_FOR_VERSION2"

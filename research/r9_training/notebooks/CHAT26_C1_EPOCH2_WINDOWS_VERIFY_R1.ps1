<#
Chat26 C1 epoch2 independent Windows checkpoint verification.
Uses the exact frozen GitHub R9 transport verifier.
Never trains; never accesses B-TEST; does not modify downloaded ZIP.
#>
[CmdletBinding()]
param(
 [string]$SourceZip="C:\Users\ik_pa\Downloads\CHAT26_C1_EPOCH2_OFFSITE_TRANSFER.zip",
 [string]$EvidenceRoot="C:\Users\ik_pa\Downloads\CHAT26_C1_EPOCH2_WINDOWS_VERIFIED_R1",
 [string]$ExpectedOuterSha256=""
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest
function Need([bool]$Okay,[string]$Code) {
 if(-not $Okay){throw ("CHAT26_EPOCH2_FAIL_CLOSED_"+$Code)}
}
function SHA([string]$P){return (Get-FileHash -LiteralPath $P -Algorithm SHA256).Hash.ToLowerInvariant()}
function J([string]$P){return Get-Content -LiteralPath $P -Raw -Encoding UTF8 | ConvertFrom-Json}
$Release="fa6a7675daae385a04c7b77e69a9de29f9e1b2f3"
$Pins=@{
 MV00="9c436620f9b10af927b8f6a964ea4ce666ba81b30174afa59bd5cfd038aa2b22"
 MV01="66bc10de6abb9728752550999c70491018c0070ee56b886c156657d839ccc55e"
}
Write-Host "CHAT26_C1_EPOCH2_WINDOWS_INDEPENDENT_VERIFICATION"
Need (Test-Path -LiteralPath $SourceZip -PathType Leaf) "SOURCE_ZIP_MISSING"
Need (-not (Test-Path -LiteralPath $EvidenceRoot)) "NO_OVERWRITE_EVIDENCE_DIRECTORY"
$outer=SHA $SourceZip
Write-Host ("ORIGINAL_TRANSFER_ZIP_SHA256="+$outer)
$outerSidecar=$SourceZip+".sha256"
$outerPinConfirmed=$false
if(Test-Path -LiteralPath $outerSidecar -PathType Leaf){
 $t=@((Get-Content -LiteralPath $outerSidecar -Raw -Encoding UTF8).Trim() -split '\s+')
 Need ($t.Count -eq 2 -and $t[0].ToLowerInvariant() -eq $outer -and
     $t[1] -ceq [System.IO.Path]::GetFileName($SourceZip)) "ORIGINAL_ZIP_SIDECAR"
 $outerPinConfirmed=$true
 Write-Host "OUTER_ORIGINAL_SHA256_SIDECAR_PASS"
}
if($ExpectedOuterSha256 -ne ""){
 Need ($outer -eq $ExpectedOuterSha256.ToLowerInvariant()) "OUTER_KAGGLE_SHA256"
 $outerPinConfirmed=$true
 Write-Host "OUTER_KAGGLE_SHA256_MATCH_PASS"
}
if(-not $outerPinConfirmed){Write-Host "NOTE_NO_ORIGINAL_OUTER_PIN: compare printed SHA with Kaggle"}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$names=@(
 "C1_OUTPUT_INDEX.json","TRANSFER_MANIFEST.json",
 "plans/MV00.json","plans/MV01.json","plans/PLAN_REVIEW.json",
 "plans/C1_USER_ENABLED_NONOFFICIAL_ACCEPTANCE.json",
 "logs/MV00_native_stdout_stderr.log","logs/MV01_native_stdout_stderr.log",
 "logs/CHAT26_C1_SUPERVISOR_STDOUT_STDERR.log",
 "MV00/MV00_epoch002_G4_LOCAL_EXPORT.zip.zipblob",
 "MV00/MV00_epoch002_G4_LOCAL_EXPORT.zip.zipblob.sha256",
 "MV01/MV01_epoch002_G4_LOCAL_EXPORT.zip.zipblob",
 "MV01/MV01_epoch002_G4_LOCAL_EXPORT.zip.zipblob.sha256"
)
$z=[System.IO.Compression.ZipFile]::OpenRead($SourceZip)
try {
 $actual=@($z.Entries | ForEach-Object {$_.FullName})
 Need ($actual.Count -eq 13 -and @($actual|Select-Object -Unique).Count -eq 13) "ZIP_MEMBER_COUNT"
 foreach($name in $actual){
  Need (-not $name.Contains('\') -and -not $name.StartsWith('/') -and
      -not (($name -split '/') -contains '..') -and ($names -ccontains $name)) "ZIP_MEMBER_PATH"
 }
 foreach($name in $names){Need ($actual -ccontains $name) "ZIP_MEMBER_MISSING"}
} finally {$z.Dispose()}
Write-Host "THIRTEEN_SAFE_MEMBERS_PASS"
New-Item -ItemType Directory -Path $EvidenceRoot -ErrorAction Stop | Out-Null
$Unpack=Join-Path $EvidenceRoot "ORIGINAL_DOWNLOADED_EPOCH2"
New-Item -ItemType Directory -Path $Unpack -ErrorAction Stop | Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($SourceZip,$Unpack)
$m=J (Join-Path $Unpack "TRANSFER_MANIFEST.json")
Need ($m.schema -eq "CHAT26_C1_EPOCH2_OFFSITE_TRANSFER_MANIFEST_R1" -and
 $m.release_commit -eq $Release -and
 $m.run_class -eq "NONOFFICIAL_FINALPATH_ACCEPTANCE" -and
 $m.completed_epoch_zero_based -eq 1 -and
 $m.offsite_persistence_verified -eq $false -and $m.b_test_access -eq "NONE") "TRANSFER_MANIFEST_BOUNDARY"
Need (@($m.files.PSObject.Properties).Count -eq 12) "MANIFEST_MEMBER_COUNT"
foreach($property in @($m.files.PSObject.Properties)){
 $name=[string]$property.Name
 Need ($names -ccontains $name -and $name -ne "TRANSFER_MANIFEST.json") "MANIFEST_PATH"
 $file=Join-Path $Unpack ($name.Replace('/','\'))
 Need (Test-Path -LiteralPath $file -PathType Leaf) "MANIFEST_FILE_MISSING"
 Need ((SHA $file) -eq $property.Value.sha256 -and
  (Get-Item -LiteralPath $file).Length -eq [long]$property.Value.bytes) "MANIFEST_HASH_OR_SIZE"
}
Write-Host "ALL_TWELVE_ORIGINAL_MEMBERS_SHA256_AND_SIZES_PASS"
$index=J (Join-Path $Unpack "C1_OUTPUT_INDEX.json")
Need ($index.schema -eq "DETECTION_CHAT25_PRODUCTION_SESSION_KAGGLE_OUTPUT_INDEX_C1" -and
 $index.published_release_commit -eq $Release -and
 $index.run_class -eq "NONOFFICIAL_FINALPATH_ACCEPTANCE" -and
 $index.status -eq "BOTH_ARMS_LOCAL_OUTPUTS_VERIFIED_EXTERNAL_PENDING" -and
 $index.both_arms_completed -eq $true -and
 $index.offsite_persistence_verified -eq $false -and
 $index.b_test_access -eq "NONE") "OUTPUT_INDEX_BOUNDARY"
Need (@($index.arms.PSObject.Properties).Count -eq 2 -and
 @($index.failures.PSObject.Properties).Count -eq 0) "TWO_ARMS_NO_FAILURES"
$VerifyDir=Join-Path $EvidenceRoot "WINDOWS_ATTESTATIONS"
New-Item -ItemType Directory -Path $VerifyDir -ErrorAction Stop | Out-Null
$transport=Join-Path $VerifyDir "R9_OFFSESSION_TRANSPORT_R1_PINNED.py"
$url="https://raw.githubusercontent.com/nithoukhan1/ResEMA/$Release/research/r9_training/code/R9_OFFSESSION_TRANSPORT_R1.py"
Invoke-WebRequest -Uri $url -OutFile $transport -UseBasicParsing -ErrorAction Stop
Write-Host ("PINNED_R9_VERIFIER_SHA256="+(SHA $transport))
$Python=(Get-Command python -ErrorAction Stop).Source
$result=@{}
foreach($arm in @("MV00","MV01")){
 $name=$arm+"_epoch002_G4_LOCAL_EXPORT.zip.zipblob"
 $zipblob=Join-Path $Unpack ($arm+"\"+$name)
 $side=$zipblob+".sha256"
 Need ((Test-Path -LiteralPath $zipblob -PathType Leaf) -and
   (Test-Path -LiteralPath $side -PathType Leaf)) "G4_FILE_OR_SIDECAR_MISSING"
 $hash=SHA $zipblob
 Need ($hash -eq $Pins[$arm]) "EXACT_G4_EPOCH2_SHA256_PIN"
 $tokens=@((Get-Content -LiteralPath $side -Raw -Encoding UTF8).Trim() -split '\s+')
 Need ($tokens.Count -eq 2 -and $tokens[0].ToLowerInvariant() -eq $hash -and
    $tokens[1] -ceq $name) "G4_SIDECAR"
 $rec=$index.arms.PSObject.Properties[$arm].Value
 Need ($rec.arm -eq $arm -and
   $rec.run_id -eq ("C1ACCEPT_"+$arm+"_20261021") -and
   $rec.completed_epoch_zero_based -eq 1 -and $rec.terminal -eq $false -and
   $rec.zip_sha256 -eq $hash -and $rec.zip_relative -ceq ($arm+"/"+$name) -and
   [long]$rec.zip_bytes -eq (Get-Item -LiteralPath $zipblob).Length) "OUTPUT_INDEX_G4_LINK"
 Need ($m.archive_pins.PSObject.Properties[$arm].Value -eq $hash) "TRANSFER_PIN_LINK"
 $attestation=Join-Path $VerifyDir ($arm+"_WINDOWS_EPOCH2_OFFSESSION_VERIFY_R1.json")
 & $Python -u $transport verify --archive $zipblob --sidecar $side --attestation $attestation --location-label ("WINDOWS_ORIGINAL_DOWNLOADED_EPOCH2_"+$arm)
 Need ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $attestation -PathType Leaf)) "R9_TRANSPORT_VERIFICATION"
 $r=J $attestation
 Need ($r.schema -eq "R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE" -and
  $r.status -eq "PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION" -and
  $r.arm -eq $arm -and $r.completed_epoch -eq 1 -and
  $r.terminal -eq $false -and $r.zip_sha256 -eq $hash -and
  $r.b_test_access -eq "NONE" -and
  $r.external_persistence_confirmed_by_software -eq $false) "ORIGINAL_INTERNAL_G4_INSPECTION"
 $result[$arm]=@{archive_sha256=$hash;g4_manifest_sha256=$r.g4_manifest_sha256;
                  selected_best_checkpoint_sha256=$r.checkpoint_sha256}
 Write-Host ($arm+"_WINDOWS_EPOCH2_ARCHIVE_AND_INTERNAL_G4_PASS")
}
$summary=[ordered]@{
 schema="CHAT26_C1_EPOCH2_WINDOWS_VERIFICATION_R1"
 status="BOTH_INDEPENDENT_WINDOWS_G4_BYTE_INSPECTIONS_PASS"
 source_zip=$SourceZip
 original_transfer_zip_sha256=$outer
 compared_outer_hash_to_original_pin=$outerPinConfirmed
 release_commit=$Release
 verifier_git_pinned_source=$url
 verifier_sha256=(SHA $transport)
 model_hashes=$result
 user_performed_windows_verification=$true
 offsite_persistence_confirmed_by_software=$false
 official_training_authorized=$false
 b_test_access="NONE"
}
$report=Join-Path $EvidenceRoot "CHAT26_C1_EPOCH2_WINDOWS_VERIFICATION_R1.json"
$summary|ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $report -Encoding UTF8
Write-Host "=============================================================="
Write-Host "BOTH_WINDOWS_EPOCH2_OFFSESSION_ARCHIVE_INSPECTIONS_PASS"
Write-Host ("ORIGINAL_TRANSFER_ZIP_SHA256="+$outer)
Write-Host ("EVIDENCE_DIRECTORY="+$EvidenceRoot)
Write-Host ("WINDOWS_VERIFICATION_REPORT="+$report)
Write-Host "OFFICIAL_TRAINING_AUTHORIZED_FALSE; B_TEST_NONE"
Write-Host "CHAT26_TO_CHAT27_MIGRATION_CHECKPOINT_REACHED"
Write-Host "=============================================================="

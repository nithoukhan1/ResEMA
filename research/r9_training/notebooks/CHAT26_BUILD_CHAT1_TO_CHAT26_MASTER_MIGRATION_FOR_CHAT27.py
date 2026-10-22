#!/usr/bin/env python3
"""Chat26 -> Chat27 complete one-file migration builder (Python standard library).

Preserves the exact prior Chat1-25 master archive as an unchanged nested byte object;
includes the Chat26 final verified Windows evidence and pinned GitHub notebook snapshots.
Does not copy raw multi-hundred-MB .zipblob checkpoints: hashes/provenance and offsite
locations are supplied instead. Fail closed; never train or touch B-TEST.
"""
import argparse, hashlib, json, os, sys, tempfile, urllib.request, zipfile
from pathlib import Path
from datetime import datetime, timezone

STAMP="2026-10-22"
NAME="DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT26_FOR_CHAT27_"+STAMP+"_FINAL.zip"
REPO="nithoukhan1/ResEMA"
SHA_RELEASE="fa6a7675daae385a04c7b77e69a9de29f9e1b2f3"
VALID_WINDOWS_OUTER="59b41703df7a94768751a34647fe3d2133035ca34b9024aa7d8dd21630b0a38d"
PIN={
 "MV00":"9c436620f9b10af927b8f6a964ea4ce666ba81b30174afa59bd5cfd038aa2b22",
 "MV01":"66bc10de6abb9728752550999c70491018c0070ee56b886c156657d839ccc55e",
}
# Pinned exact immutable snapshots: each path is read from its historical Git commit.
SNAPSHOTS=[
 ("ef7c3cd56abaebd3711a14c35d9ca344fae4e824",
  "research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_FRESH_EPOCH1_T4X2.ipynb"),
 ("4e367ffdac2bc7da0f796ce4550ad3263857a62f",
  "research/r9_training/notebooks/CHAT26_C1_ONE_CLICK_NONOFFICIAL_RESUME_EPOCH2_T4X2.ipynb"),
 ("96613d99841d994a5d35d6433dce65821200b446",
  "research/r9_training/notebooks/CHAT26_C1_EPOCH1_WINDOWS_OFFSITE_VERIFY_AND_STAGE.ps1"),
 ("b7838be70586550578025280763b9a55cb4defef",
  "research/r9_training/notebooks/CHAT26_C1_EPOCH2_WINDOWS_VERIFY_R1.ps1"),
 ("71d984cd6f57356337420a3aed9eed6eb8e886b7",
  "research/r9_training/notebooks/CHAT26_C1_NATIVE_GPU_ACCEPTANCE_MIGRATION_ADDENDUM_2026-10-22.md"),
 ("4b5f173930f459b69cb7f7e2481b0f2e629e0867",
  "research/r9_training/notebooks/DETECTION_PROJECT_CHAT26_FINAL_EPOCH2_WINDOWS_VERIFIED_2026-10-22.md"),
]
def required(test,message):
 if not test: raise RuntimeError("CHAT26_MIGRATION_FAIL_CLOSED: "+message)
def sha_file(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def asjson(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def find_prior(downloads):
 patterns=(
  "DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT25_FOR_CHAT26_2026-10-21_FINAL*.zip",
  "DETECTION_PROJECT_MASTER_MIGRATION_CHAT1_TO_CHAT25_FOR_CHAT26_2026-10-21_PORTABLE*.zip",
 )
 for pattern in patterns:
  choices=sorted(downloads.glob(pattern))
  if choices:return choices[0]
 raise RuntimeError("MISSING_PRIOR_MASTER: download prior Chat1-25 FINAL.zip (or PORTABLE.zip) into "+str(downloads)+" or pass --prior exact_file_path")
def fetch(commit,relpath):
 url="https://raw.githubusercontent.com/"+REPO+"/"+commit+"/"+relpath
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Chat26-migration-builder"}),timeout=40) as r:
  content=r.read()
 required(len(content)>100,"EMPTY_GITHUB_SNAPSHOT_"+relpath)
 return content,url
def include_bytes(buffer,metadata,name,data,origin):
 required(name not in buffer,"DUPLICATE_ENTRY_"+name)
 required(not name.startswith("/") and ".." not in Path(name).parts and "\\" not in name,"UNSAFE_ENTRY_"+name)
 buffer[name]=data
 metadata[name]={"sha256":sha_bytes(data),"size_bytes":len(data),"origin":origin}
def include_file(files,metadata,name,path,origin):
 path=Path(path)
 required(path.is_file() and not path.is_symlink(),"MISSING_OR_SYMLINK_"+str(path))
 required(name not in metadata,"DUPLICATE_ENTRY_"+name)
 metadata[name]={"sha256":sha_file(path),"size_bytes":path.stat().st_size,"origin":origin}
 files[name]=path

def main():
 parser=argparse.ArgumentParser(description="Build one complete Chat1-26 master migration archive, preserving prior bytes.")
 parser.add_argument("--prior",default=None,help="Exact local path to earlier Chat1-25 FINAL or PORTABLE ZIP.")
 parser.add_argument("--evidence",default=None,help="Epoch2 Windows verified evidence folder.")
 parser.add_argument("--outdir",default=None,help="Output folder; defaults to Downloads.")
 args=parser.parse_args()
 downloads=Path.home()/"Downloads"
 prior=Path(args.prior) if args.prior else find_prior(downloads)
 evidence=Path(args.evidence) if args.evidence else downloads/"CHAT26_C1_EPOCH2_WINDOWS_VERIFIED_R1"
 outdir=Path(args.outdir) if args.outdir else downloads
 out=outdir/NAME
 outer_sidecar=outdir/(NAME+".sha256")
 required(prior.is_file() and prior.suffix.lower()==".zip","BASE_PRIOR_MUST_EXIST_AS_ZIP")
 required(evidence.is_dir(),"VERIFIED_WINDOWS_EVIDENCE_ROOT_REQUIRED")
 required(outdir.is_dir(),"OUTPUT_DIRECTORY_MISSING")
 required(not out.exists() and not outer_sidecar.exists(),"DO_NOT_OVERWRITE_EXISTING_MIGRATION")
 required(not prior.is_symlink(),"PRIOR_ZIP_SYMLINK_NOT_ALLOWED")
 with zipfile.ZipFile(prior,"r") as z:
  required(z.testzip() is None,"PRIOR_ZIP_CORRUPT")
  prior_members=z.namelist()
  required(len(prior_members)>0,"PRIOR_MASTER_EMPTY")
 prior_hash=sha_file(prior)
 # An adjacent sidecar, when supplied by the earlier migration, must match exactly.
 priorSidecar=Path(str(prior)+".sha256")
 if priorSidecar.is_file():
  tokens=priorSidecar.read_text(encoding="utf-8-sig").strip().split()
  required(len(tokens)>=1 and tokens[0].lower()==prior_hash,"PRIOR_MASTER_SIDECAR_SHA256_MISMATCH")
  print("PRIOR_MASTER_SIDECAR_HASH_MATCH_PASS")
 else:
  print("PRIOR_MASTER_SIDECAR_NOT_PRESENT: original ZIP content itself will be integrity-tested and nested unchanged")
 print("PRIOR_MASTER_SOURCE:",prior)
 print("PRIOR_MASTER_MEMBER_COUNT:",len(prior_members))
 print("PRIOR_MASTER_SHA256:",prior_hash)
 # Verify separately obtained Windows epoch2 evidence, not only the chat transcript.
 report=evidence/"CHAT26_C1_EPOCH2_WINDOWS_VERIFICATION_R1.json"
 r=asjson(report)
 required(r.get("status")=="BOTH_INDEPENDENT_WINDOWS_G4_BYTE_INSPECTIONS_PASS","WINDOWS_SUMMARY_NOT_PASS")
 required(r.get("original_transfer_zip_sha256")==VALID_WINDOWS_OUTER,"WINDOWS_OUTER_SHA_INVALID")
 required(r.get("release_commit")==SHA_RELEASE,"WINDOWS_RELEASE_MISMATCH")
 required(r.get("official_training_authorized") is False and r.get("b_test_access")=="NONE","WINDOWS_BOUNDARY")
 for arm in ("MV00","MV01"):
  rec=r.get("model_hashes",{}).get(arm,{})
  required(rec.get("archive_sha256")==PIN[arm],"WINDOWS_MODEL_SHA_"+arm)
  att=asjson(evidence/"WINDOWS_ATTESTATIONS"/(arm+"_WINDOWS_EPOCH2_OFFSESSION_VERIFY_R1.json"))
  required(att.get("schema")=="R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE" and
     att.get("arm")==arm and att.get("zip_sha256")==PIN[arm] and
     att.get("completed_epoch")==1 and att.get("terminal") is False and
     att.get("status")=="PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION",
     "WINDOWS_ATTESTATION_"+arm)
 print("BOTH_WIN_EPOCH2_G4_ATTESTATIONS_PASS")
 meta={}
 files={}
 small={}
 base_name="PRIOR_CHAT1_TO_CHAT25_MASTER/"+prior.name
 include_file(files,meta,base_name,prior,"PREVIOUS_CUMULATIVE_MASTER_ARCHIVE_UNMODIFIED")
 # Recorded independent Windows proof, including output plans, approval and native logs.
 include_file(files,meta,"CHAT26_WINDOWS_EVIDENCE/CHAT26_C1_EPOCH2_WINDOWS_VERIFICATION_R1.json",report,"WIN_OFFSESSION_VERIFICATION_SUMMARY")
 for arm in ("MV00","MV01"):
  p=evidence/"WINDOWS_ATTESTATIONS"/(arm+"_WINDOWS_EPOCH2_OFFSESSION_VERIFY_R1.json")
  include_file(files,meta,"CHAT26_WINDOWS_EVIDENCE/"+p.name,p,"ORIGINAL_WINDOWS_CHECKPOINT_VERIFIER")
 ep2root=evidence/"ORIGINAL_DOWNLOADED_EPOCH2"
 for relative in (
  "C1_OUTPUT_INDEX.json","TRANSFER_MANIFEST.json",
  "plans/MV00.json","plans/MV01.json","plans/PLAN_REVIEW.json",
  "plans/C1_USER_ENABLED_NONOFFICIAL_ACCEPTANCE.json",
  "logs/MV00_native_stdout_stderr.log","logs/MV01_native_stdout_stderr.log",
  "logs/CHAT26_C1_SUPERVISOR_STDOUT_STDERR.log"
 ):
  p=ep2root/relative
  include_file(files,meta,"CHAT26_EPOCH2_NATIVE_TEXT_EVIDENCE/"+relative,p,"WINDOWS_PRESERVED_VERSION2_TEXT_EVIDENCE")
 # Epoch1 input proof may be present on Windows; preserve attestation JSON and index when available.
 ep1root=downloads/"CHAT26_C1_EPOCH1_RAW"/"CHAT26_EPOCH1_VERIFIED_FOR_KAGGLE"
 if ep1root.is_dir():
  for name in ("C1_OUTPUT_INDEX.json",):
   p=ep1root/name
   if p.is_file():include_file(files,meta,"CHAT26_EPOCH1_WINDOWS_TEXT_EVIDENCE/"+name,p,"WIN_VERIFIED_EPOCH1_DATASET_INDEX")
  for arm in ("MV00","MV01"):
   p=ep1root/arm/(arm+"_WINDOWS_OFFSESSION_VERIFY_R1.json")
   if p.is_file():include_file(files,meta,"CHAT26_EPOCH1_WINDOWS_TEXT_EVIDENCE/"+p.name,p,"WIN_VERIFIED_EPOCH1_G4_ATTESTATION")
 # Fetch exact published notebook, verifier and history snapshots; immutable commit paths.
 for commit,relpath in SNAPSHOTS:
  data,url=fetch(commit,relpath)
  dest="CHAT26_FROZEN_GITHUB_SNAPSHOTS/"+Path(relpath).name
  include_bytes(small,meta,dest,data,url)
  print("SNAPSHOT_PASS:",dest,sha_bytes(data))
 # The next-chat prompt is itself held in the single archive.
 prompt=(
  "DETECTION PROJECT — CHAT27 FIRST PROMPT (2026-10-22)\n\n"
  "You are continuing our entire Detection Project from Chats1-26.\n"
  "Unpack and fully read the nested PRIOR_CHAT1_TO_CHAT25_MASTER original ZIP, "
  "both Chat26 migration addenda, independent epoch2 Windows verification "
  "summary, two G4 attestation JSON files, output index/plans and Kaggle worker logs, "
  "and the exact Github notebook snapshots bundled here. Preserve the full "
  "history and scientific frozen source; do not drop earlier C3k2+DySample+ResEMA "
  "experiments, Split-B governance or VGRA C1 line.\n\n"
  "Current C1 status: MV00 SCConv-Early and MV01 SCConv-Early+VGRA-V1 "
  "completed independent native fresh epoch1, Windows epoch1 verification, "
  "native authentic epoch2 resume and independent Windows epoch2 verification. "
  "Both nonofficial original archive SHA values and release hashes are preserved "
  "in CHAT26_CUMULATIVE_MIGRATION_MANIFEST.json and the final handoff. "
  "DO NOT claim official 100-epoch training was started; it was NOT. "
  "B-TEST remains SEALED. Do not assume comparative scientific performance "
  "from two-epoch GPU acceptance. Inspector checkpoint_sha256 identifies "
  "selected_best.pt, not necessarily last.pt; consider release QA for last.pt.\n\n"
  "NEXT REQUIRED ACTION: review the official release approval/governance gate "
  "and prepare a separately authorized, fully pinned MV00/MV01 fresh "
  "100-epoch plan (including multi-session G4 preservation and authentic resume) "
  "without mutating the frozen science; do not execute official training until "
  "the user expressly approves the new release and plan. "
  "Maintain an overall project tracker and work step by step.\n"
 )
 include_bytes(small,meta,"CHAT27_FIRST_PROMPT_2026-10-22.txt",prompt.encode("utf-8"),"BUILDER_GENERATED_NEXT_CHAT_INSTRUCTION")
 readme=(
  "# Detection Project Master Migration — Chat1 to Chat26 for Chat27\n\n"
  "Migration container: exact original Chat1-25 cumulative ZIP nested and byte-preserved; "
  "the complete Chat26 GPU/Windows independent verification addenda and complete epoch2 "
  "textual evidence files; exact committed one-click GPU notebooks, Windows verification "
  "scripts; and a Chat27 prompt. See manifest for every file SHA and source identity.\n\n"
  "## Scientific and authorization status\n"
  "Nonofficial two-arm fresh epoch1 + genuine native resume epoch2 and independently "
  "verified checkpoint archive bytes PASS. No official 100-epoch training has started "
  "and it is not authorized. B-TEST never accessed in C1 acceptance. "
  "Selected_best.pt hash unchanged between epoch1 and 2 does not alone imply failed "
  "resume; inspector returns selected_best checkpoint SHA. Optional last.pt cross-epoch "
  "hash validation remains release QA.\n\n"
  "## Important limitations\n"
  "Original .zipblob/large checkpoints are NOT bundled to avoid >500MB migration; "
  "their exact SHA and local evidence paths are preserved in documents. "
  "The user must retain original separately Windows-preserved checkpoint files "
  "and their original Kaggle private dataset. "
  "The outer epoch2 transfer SHA was not independently compared with a Kaggle outer "
  "pin, but all original 12 content files passed manifest checking and both original "
  "G4 archives matched SHA pinned in the previous Kaggle log.\n"
 )
 include_bytes(small,meta,"README_START_HERE.md",readme.encode("utf-8"),"BUILDER_GENERATED_README")
 manifest={
  "schema":"DETECTION_PROJECT_CHAT1_TO_CHAT26_FULL_MASTER_MIGRATION_R1",
  "creation_utc":datetime.now(timezone.utc).isoformat(),
  "previous_chat_range":"CHAT1_TO_CHAT25",
  "new_chat_range":"CHAT1_TO_CHAT26",
  "prior_archive_basename":prior.name,
  "prior_archive_sha256":prior_hash,
  "prior_archive_size_bytes":prior.stat().st_size,
  "prior_archive_original_member_count":len(prior_members),
  "frozen_release_commit":SHA_RELEASE,
  "nonofficial_c1_acceptance_epoch1_and_epoch2_pass":True,
  "epoch2_independent_windows_archive_verification_pass":True,
  "epoch2_original_transfer_zip_windows_sha256":VALID_WINDOWS_OUTER,
  "epoch2_outer_sha_independently_compared_to_kaggle":False,
  "epoch2_original_g4_archive_sha256":PIN,
  "selected_best_checkpoint_may_be_unchanged_between_epochs":True,
  "official_training_authorized":False,
  "official_training_started":False,
  "b_test_access":"NONE",
  "original_g4_zipblob_checkpoints_embedded":False,
  "file_details":meta
 }
 include_bytes(small,meta,"CHAT26_CUMULATIVE_MIGRATION_MANIFEST.json",
               (json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode("utf-8"),
               "BUILDER_GENERATED_FILE_PROVENANCE")
 # Atomic output creation. ZIP_STORED preserves nested prior ZIP exact bytes.
 stage=outdir/(NAME+".incomplete")
 required(not stage.exists(),"PARTIAL_OUTPUT_ALREADY_EXISTS")
 try:
  with zipfile.ZipFile(stage,"x",compression=zipfile.ZIP_STORED,allowZip64=True) as zout:
   for name,p in files.items():zout.write(p,name,compress_type=zipfile.ZIP_STORED)
   for name,data in small.items():zout.writestr(name,data,compress_type=zipfile.ZIP_STORED)
  with zipfile.ZipFile(stage,"r") as verify:
   required(verify.testzip() is None,"FINAL_ZIP_CRC_FAILED")
   required(len(verify.namelist())==len(files)+len(small),"FINAL_FILE_COUNT")
   for name,entry in meta.items():
    required(name in verify.namelist(),"FINAL_MISSING_"+name)
    required(sha_bytes(verify.read(name))==entry["sha256"],"FINAL_ENTRY_HASH_"+name)
  stage.rename(out)
 except Exception:
  # Keep incomplete as evidence for diagnosis, never declare successful.
  raise
 sha=sha_file(out)
 outer_sidecar.write_text(sha+"  "+out.name+"\n",encoding="utf-8")
 print("="*70)
 print("CHAT1_TO_CHAT26_FOR_CHAT27_FULL_MASTER_MIGRATION_READY")
 print("FILE:",out)
 print("ZIP_SHA256:",sha)
 print("SIDECAR:",outer_sidecar)
 print("BASE_PRIOR_ARCHIVE_PRESERVED_UNMODIFIED:",prior_hash)
 print("FILE_COUNT:",len(files)+len(small))
 print("OFFICIAL_TRAINING_AUTHORIZED_FALSE; B_TEST_NONE")
 print("="*70)
if __name__=="__main__":
 try:main()
 except Exception as exc:
  print(type(exc).__name__+": "+str(exc),file=sys.stderr)
  sys.exit(1)

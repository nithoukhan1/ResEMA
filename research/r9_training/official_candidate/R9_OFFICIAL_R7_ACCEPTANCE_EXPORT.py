#!/usr/bin/env python3
"""R7 two-arm nonofficial new-path output staging, local only; no cleanup.

A manifest/ZIP hash match demonstrates LOCAL COPY equality, never persistence
in another Kaggle version or independent storage. Original ZIPs are unchanged.
"""
from __future__ import annotations
import hashlib,json,os,shutil
from pathlib import Path

ARMS=('MV00','MV01')
SCHEMA='DETECTION_CHAT25_R7_NEWPATH_NONOFFICIAL_NATIVE_GPU_RESULT'

def need(ok,code):
    if not ok:raise RuntimeError('R9_R7_EXPORT_FAIL_CLOSED_'+code)

def hash_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def inspect_one(arm,proof,inspector):
    need(type(proof) is dict and set(proof)=={'report','zip','sidecar'},'PROOF_FIELDS')
    report,archive,sidecar=(Path(proof[k]) for k in ('report','zip','sidecar'))
    need(arm in ARMS and all(p.is_file() and not p.is_symlink() for p in
         (report,archive,sidecar)), 'PROOF_SOURCE')
    info=json.loads(report.read_text('utf-8'))
    need(info.get('schema')==SCHEMA and info.get('arm')==arm and
         info.get('status')=='NONOFFICIAL_NEWPATH_GPU_LOCAL_G4_VERIFIED_EXTERNAL_NOT_VERIFIED' and
         info.get('terminal') is False and
         info.get('official_training_authorized') is False and
         info.get('independent_offsite_verified') is False and
         info.get('b_test_access')=='NONE','REPORT_SCHEMA_AND_SCOPE')
    need(type(info.get('completed_epoch_zero_based')) is int and
         0<=info['completed_epoch_zero_based']<=3,'BOUNDED_EPOCH')
    need(type(info.get('run_id')) is str and info['run_id'] and
         info.get('local_zip')==str(archive) and
         archive.name == f'{arm}_epoch{info["completed_epoch_zero_based"]+1:03d}_G4_LOCAL_EXPORT.zip',
         'ZIP_CURSOR')
    digest=hash_file(archive)
    need(info.get('local_zip_sha256')==digest,'REPORT_ZIP_HASH')
    need(sidecar==archive.with_name(archive.name+'.sha256'),'SIDECAR_NAME')
    parts=sidecar.read_text('utf-8').strip().split()
    need(parts==[digest,archive.name],'SIDECAR_CONTENT')
    checked=inspector(archive,sidecar=sidecar)
    need(checked.get('status')=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION' and
         checked.get('arm')==arm and checked.get('completed_epoch')==info['completed_epoch_zero_based'] and
         checked.get('terminal') is False and checked.get('zip_sha256')==digest,
         'G4_INSPECTED_ARCHIVE')
    return {'arm':arm,'run_id':info['run_id'],
        'completed_epoch_zero_based':info['completed_epoch_zero_based'],
        'zip_sha256':digest,'zip_bytes':archive.stat().st_size,
        'source_paths':proof}

def stage_two(*,proofs,root,inspector,max_bytes=18*(1<<30)):
    need(type(proofs) is dict and set(proofs)==set(ARMS),'BOTH_ARMS_REQUIRED')
    need(type(max_bytes) is int and 0<max_bytes<=20*(1<<30),'MAX_BYTES')
    dest=Path(root)
    need(dest.parent.is_dir() and not dest.exists() and not dest.is_symlink(),
         'DEST_NEW_DIRECTORY')
    entries={arm:inspect_one(arm,proofs[arm],inspector) for arm in ARMS}
    need(entries['MV00']['run_id']!=entries['MV01']['run_id'],'COLLIDING_RUN_IDS')
    total=sum(sum(Path(proofs[a][k]).stat().st_size for k in ('report','zip','sidecar')) for a in ARMS)
    need(total<=max_bytes,'OUTPUT_BUDGET')
    dest.mkdir(mode=0o700)
    try:
        for arm in ARMS:
            folder=dest/arm;folder.mkdir(mode=0o700)
            for typ in ('zip','sidecar','report'):
                p=Path(proofs[arm][typ]);target=folder/(p.name+'.zipblob' if typ=='zip' else p.name)
                with p.open('rb') as r,target.open('xb') as w:
                    shutil.copyfileobj(r,w,1<<20);w.flush();os.fsync(w.fileno())
                need(hash_file(p)==hash_file(target),'BYTE_COPY')
                entries[arm][typ+'_relative_path']=str(target.relative_to(dest))
            entries[arm].pop('source_paths')
        summary={'schema':'DETECTION_CHAT25_R7_NEWPATH_LOCAL_OUTPUT_INDEX',
            'status':'TWO_ARM_LOCAL_BYTES_COPIED_OFFSITE_UNVERIFIED',
            'arms':entries,'bytes':total,'offsite_verified':False,
            'official_training_authorized':False,'b_test_access':'NONE'}
        output=dest/'R7_LOCAL_OUTPUT_INDEX.json'
        with output.open('x',encoding='utf-8') as f:
            json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
            f.flush();os.fsync(f.fileno())
        return summary
    except BaseException:
        shutil.rmtree(dest)
        raise

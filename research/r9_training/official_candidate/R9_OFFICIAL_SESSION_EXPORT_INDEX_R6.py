#!/usr/bin/env python3
"""R6 session-output index: original ZIP bytes, no overwrite; NOT offsite proof.

Only locally inspected G4 checkpoint outputs may be included. Stage the latest
completed capsule for EACH arm independently. Kaggle version output files do not
by themselves establish external persistence. Never delete original sources.
"""
from __future__ import annotations
import hashlib, json, os, re, shutil
from pathlib import Path

ARMS=('MV00','MV01')
HEX=re.compile(r'[0-9a-f]{64}\Z')


def need(v,k):
    if not v:raise RuntimeError('R9_R6_EXPORT_FAIL_CLOSED_'+k)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def inspect_session_proof(arm,report_path,archive_path,sidecar_path,inspector):
    report_path,archive_path,sidecar_path=map(Path,(report_path,archive_path,sidecar_path))
    need(arm in ARMS,'ARM')
    need(all(p.is_file() and not p.is_symlink() for p in
             (report_path,archive_path,sidecar_path)),'SOURCES')
    report=json.loads(report_path.read_text('utf8'))
    need(report.get('schema')=='DETECTION_CHAT25_R6_LOCAL_NATIVE_SESSION_RESULT' and
         report.get('status')=='GPU_EXECUTION_LOCAL_BYTES_ONLY_OFFSITE_PENDING' and
         report.get('arm')==arm and report.get('b_test_access')=='NONE' and
         report.get('independent_offsite_verified') is False, 'LOCAL_REPORT')
    need(report.get('local_zip')==str(archive_path) and
         report.get('local_zip_sha256')==digest(archive_path),'LOCAL_ZIP_IDENTITY')
    check=inspector(archive_path,sidecar=sidecar_path)
    need(check.get('status')=='PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION'
         and check['arm']==arm and
         check['completed_epoch']==report['completed_epoch_zero_based'] and
         check['terminal']==report['terminal'] and
         check['zip_sha256']==report['local_zip_sha256'],'INSPECTED_G4_CONTENT')
    need(sidecar_path.name==archive_path.name+'.sha256','SIDECAR_NAME')
    return {'arm':arm,'run_id':report['run_id'],
            'completed_epoch_zero_based':report['completed_epoch_zero_based'],
            'terminal':report['terminal'],'zip_sha256':report['local_zip_sha256'],
            'zip_bytes':archive_path.stat().st_size,
            'b_test_access':'NONE'}


def create_output_index(*, proof_paths,output_root,inspector,max_bytes=18*(1<<30)):
    """Stage two locally verified original ZIPs into a NEW Save-Version output.

    The output must be copied/downloaded and SHA-verified independently before
    any claim of offsite persistence. This routine never garbage collects.
    """
    need(type(proof_paths) is dict and set(proof_paths)==set(ARMS),'TWO_ARMS')
    need(type(max_bytes) is int and 0<max_bytes<=20*(1<<30),'OUTPUT_BUDGET')
    root=Path(output_root)
    need(root.parent.is_dir() and not root.exists() and not root.is_symlink(),
         'NEW_OUTPUT_DIRECTORY')
    entries={};total=0
    for arm in ARMS:
        paths=proof_paths[arm]
        need(type(paths) is dict and set(paths)=={'report','zip','sidecar'},'PATH_KEYS')
        info=inspect_session_proof(arm,paths['report'],paths['zip'],paths['sidecar'],inspector)
        entries[arm]=info
        total+=info['zip_bytes']+sum(Path(paths[k]).stat().st_size for k in ('report','sidecar'))
    need(total<=max_bytes,'OUTPUT_BUDGET_EXCEEDED')
    root.mkdir(mode=0o700)
    try:
        for arm in ARMS:
            dest=root/arm
            dest.mkdir(mode=0o700)
            paths=proof_paths[arm]
            for typ,extension in (('zip','.zipblob'),('sidecar','.sha256'),('report','.json')):
                source=Path(paths[typ]); out=dest/(source.name+extension if typ=='zip' else source.name)
                need(not out.exists(),'DEST_COLLISION')
                with source.open('rb') as i,out.open('xb') as o:
                    shutil.copyfileobj(i,o,1<<20);o.flush();os.fsync(o.fileno())
                need(digest(out)==digest(source),'COPIED_BYTES')
                entries[arm][typ+'_staged_relative']=str(out.relative_to(root))
        index={'schema':'DETECTION_CHAT25_R6_KAGGLE_SESSION_OUTPUT_INDEX',
               'status':'LOCAL_COPY_VERIFIED_EXTERNAL_PERSISTENCE_PENDING',
               'latest_state_per_arm':entries,'total_selected_bytes':total,
               'official_training_authorized_here':False,
               'offsite_persistence_verified':False,'b_test_access':'NONE'}
        with (root/'R6_SESSION_OUTPUT_INDEX.json').open('x') as f:
            json.dump(index,f,sort_keys=True,indent=2);f.write('\n')
            f.flush();os.fsync(f.fileno())
        return index
    except BaseException:
        shutil.rmtree(root)
        raise

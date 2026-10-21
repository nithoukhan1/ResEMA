#!/usr/bin/env python3
"""CPU-only byte-preserving Kaggle local output staging, never offsite proof.

Use only after R9's existing transport exporter produced and locally verified the
original ZIP. Stages a `.zipblob` copy to prevent zip auto-unpack in datasets.
Never overwrites or alters an original checkpoint. Offsite persistence remains
FALSE until independently checked on a different storage host/session.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def need(ok, msg):
    if not ok: raise RuntimeError('R9_R5_PRESERVE_FAIL_CLOSED_' + msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def stage_original_zip(*, arm, original_zip, sidecar, destination):
    """Atomic O_EXCL copy + exact original SHA; no semantic checkpoint parsing."""
    need(arm in ('MV00','MV01'), 'ARM')
    src,sc,dest = map(Path,(original_zip,sidecar,destination))
    need(src.is_file() and not src.is_symlink() and src.suffix=='.zip', 'ORIGINAL_ZIP')
    need(sc.is_file() and not sc.is_symlink() and
         sc == src.with_name(src.name+'.sha256'),'SIDECAR')
    need(dest.name==src.name+'.zipblob' and dest.parent.is_dir() and
         not dest.exists() and not dest.is_symlink(), 'NEW_ZIPBLOB_DEST')
    need(src.resolve() != dest.resolve() and dest.parent.resolve()!=src.parent.resolve(),
         'SEPARATE_STAGE_DIRECTORY')
    tokens = sc.read_text('utf-8').strip().split()
    need(len(tokens)==2 and bool(HEX64.fullmatch(tokens[0])) and tokens[1]==src.name,
         'SIDECAR_FORMAT')
    original_sha = sha(src)
    need(original_sha==tokens[0], 'ORIGINAL_BYTES')
    temp = dest.parent / ('.'+dest.name+'.tmp-'+os.urandom(8).hex())
    try:
        with src.open('rb') as inp, temp.open('xb') as out:
            for part in iter(lambda:inp.read(1048576),b''):
                out.write(part)
            out.flush();os.fsync(out.fileno())
        need(sha(temp)==original_sha, 'COPIED_BYTES')
        # O_EXCL hard link refuses overwrite, atomic only on same filesystem.
        os.link(temp,dest)
        need(sha(dest)==original_sha, 'COMMITTED_BYTES')
    finally:
        temp.unlink(missing_ok=True)
    receipt = {'schema':'R9_R5_LOCAL_ZIPBLOB_STAGING_ONLY', 'arm':arm,
               'original_filename':src.name, 'zip_sha256':original_sha,
               'staged_filename':dest.name, 'bytes':dest.stat().st_size,
               'original_zip_preserved':True, 'offsite_verified':False,
               'gpu_training_performed_by_this_tool':False,'b_test_access':'NONE'}
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('MV00','MV01'),required=True)
    p.add_argument('--original-zip',required=True);p.add_argument('--sidecar',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    out=Path(a.receipt)
    need(out.parent.is_dir() and not out.exists() and not out.is_symlink(),'NEW_RECEIPT')
    record=stage_original_zip(arm=a.arm,original_zip=a.original_zip,
                              sidecar=a.sidecar,destination=a.destination)
    with out.open('x',encoding='utf8') as f:
        json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
        f.flush();os.fsync(f.fileno())
    print(json.dumps(record,indent=2,sort_keys=True))
    print('ONLY_KAGGLE_LOCAL_COPY__EXTERNAL_DOWNLOAD_AND_VERIFICATION_REQUIRED')

if __name__=='__main__': main()

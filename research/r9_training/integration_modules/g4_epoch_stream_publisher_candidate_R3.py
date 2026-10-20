"""Chat21: streamed, immutable epoch-bundle publisher for authenticated *candidate* trainers.

Not a remote-storage client or proof of native-loop integration. Must be called
only after all guards, selected-best authentication and capsule checks passed.
Never overwrites a committed epoch; no pickle deserialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import re
import shutil
import stat

SCHEMA = 'CHAT21_G4C_EPOCH_BUNDLE_CANDIDATE_R2_WITH_AUTHENTIC_RECEIPT'
RECOVERY_RECEIPT_SCHEMA = 'D4E_TRUSTED_EPOCH_BOUNDARY_RECOVERY_R2_AUGMENT'
ALLOWED_ARMS = frozenset(('MV00', 'MV01'))


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n').encode('ascii')


def _fsync_dir(path):
    fd=os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _private(root):
    root = Path(root)
    if (os.name != 'posix' or not root.is_absolute() or not root.is_dir() or
            root.is_symlink() or root.resolve(strict=True) != root or
            root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode)&0o077):
        raise RuntimeError('G4C_PRIVATE_EXISTING_ROOT_REQUIRED')
    return root


def _copy_exclusive(source, dest):
    src=Path(source)
    if not src.is_file() or src.is_symlink() or src.stat().st_size == 0:
        raise RuntimeError('G4C_SOURCE_FILE_INVALID')
    inp_fd=os.open(src, os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
    out_fd=None
    try:
        out_fd=os.open(dest, os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        h=hashlib.sha256(); n=0
        with os.fdopen(inp_fd,'rb',closefd=False) as f, os.fdopen(out_fd,'wb',closefd=False) as out:
            for part in iter(lambda:f.read(1<<20), b''):
                out.write(part); h.update(part); n+=len(part)
            out.flush();os.fsync(out.fileno())
        if src.stat().st_size!=n or _hash_file(src)!=h.hexdigest():
            raise RuntimeError('G4C_SOURCE_MUTATED_DURING_PUBLISH')
        return {'sha256':h.hexdigest(),'bytes':n}
    finally:
        os.close(inp_fd)
        if out_fd is not None:os.close(out_fd)


def publish_epoch(*, root, arm, epoch, last_path, best_path, capsule_path=None,
                  capsule_receipt_path=None, capsule_receipt_pin=None, terminal=False,
                  evidence=None, failpoint=None):
    """One immutable on-session bundle. External transfer/receipt remains MANDATORY.

    Epoch is 0-indexed. Evidence must be derived from actual native-loop guard hooks,
    not caller-declared simulation, for any future production invocation.
    """
    root=_private(root)
    if arm not in ALLOWED_ARMS or type(epoch) is not int or not 0<=epoch<100 or type(terminal) is not bool:
        raise RuntimeError('G4C_ARM_EPOCH_OR_TERMINAL_DRIFT')
    if not isinstance(evidence,dict) or set(evidence) != {'pre_save_epoch','boundary_epoch','validator_pass','selected_best_authenticated','recovery_identity_verified','b_test_access'}:
        raise RuntimeError('G4C_EVIDENCE_SCHEMA_INVALID')
    if (evidence['pre_save_epoch'] != epoch or evidence['boundary_epoch'] != epoch or
            evidence['validator_pass'] is not True or evidence['selected_best_authenticated'] is not True or
            evidence['recovery_identity_verified'] is not True or evidence['b_test_access'] != 'NONE'):
        raise RuntimeError('G4C_MANDATORY_GUARD_EVIDENCE_MISSING')
    if terminal:
        if capsule_path is not None or capsule_receipt_path is not None or capsule_receipt_pin is not None:
            raise RuntimeError('G4C_TERMINAL_CAPSULE_PROHIBITED')
    elif (capsule_path is None or capsule_receipt_path is None or
          not isinstance(capsule_receipt_pin,str) or
          len(capsule_receipt_pin)!=64 or any(c not in '0123456789abcdef' for c in capsule_receipt_pin)):
        raise RuntimeError('G4C_NONTERMINAL_VERIFIED_CAPSULE_REQUIRED')
    # Verify both the original .pt AND the separate authenticated receipt, without
    # deserializing any checkpoint bytes. The capsule reader explicitly REQUIRES
    # <checkpoint_stem>.receipt.json; omitting it makes cross-session recovery fail.
    if not terminal:
        cp=Path(capsule_path)
        rp=Path(capsule_receipt_path)
        if not cp.is_file() or cp.is_symlink() or not rp.is_file() or rp.is_symlink():
            raise RuntimeError('G4C_CAPSULE_OR_RECEIPT_NOT_REGULAR_FILE')
        if (not re.fullmatch(r'[A-Za-z0-9_-]{1,160}\.pt', cp.name) or
                cp.name in {'last.pt','selected_best.pt'}):
            raise RuntimeError('G4C_CAPSULE_BASENAME_NOT_PORTABLE')
        if rp != cp.with_suffix('.receipt.json'):
            raise RuntimeError('G4C_CAPSULE_RECEIPT_SIDECAR_NAME_MISMATCH')
        if _hash_file(rp) != capsule_receipt_pin:
            raise RuntimeError('G4C_CAPSULE_RECEIPT_EXTERNAL_PIN_MISMATCH')
        meta=json.loads(rp.read_bytes())
        if (meta.get('schema') != RECOVERY_RECEIPT_SCHEMA or
                meta.get('checkpoint_filename') != cp.name or
                meta.get('checkpoint_sha256') != _hash_file(cp) or
                not isinstance(meta.get('identity'),dict) or
                meta['identity'].get('arm') != arm or
                not isinstance(meta.get('cursor'),dict) or
                meta['cursor'].get('completed_epoch') != epoch or
                meta['cursor'].get('next_epoch') != epoch+1):
            raise RuntimeError('G4C_CAPSULE_RECEIPT_CONTENTS_MISMATCH')
    if failpoint not in (None,'after_last','after_all','before_commit'):
        raise RuntimeError('G4C_INVALID_TEST_FAULT')
    folder=root/(f'{arm}_epoch{epoch+1:03d}')
    if folder.exists() or folder.is_symlink():raise FileExistsError('G4C_REFUSE_REPLACE_COMMITTED_EPOCH')
    lease=root/f'.{arm}.writer.lock'
    fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,'O_NOFOLLOW',0),0o600)
    os.close(fd)
    staging=root/f'.g4c_tmp_{secrets.token_hex(12)}'
    committed=False
    try:
        staging.mkdir(mode=0o700)
        sources={'last.pt':last_path,'selected_best.pt':best_path}
        if not terminal:
            # Preserve the signed original basename: the recovery reader checks
            # receipt['checkpoint_filename'] == checkpoint_path.name. Renaming
            # run_id_epochNNN.pt would break authenticated cross-session resume.
            sources[cp.name]=capsule_path
            sources[rp.name]=capsule_receipt_path
        files={}
        for name,src in sources.items():
            files[name]=_copy_exclusive(src,staging/name)
            if name=='last.pt' and failpoint=='after_last':raise RuntimeError('G4C_TEST_ABORT_AFTER_LAST')
        if not terminal:
            if files[rp.name]['sha256'] != capsule_receipt_pin:
                raise RuntimeError('G4C_COPIED_RECEIPT_PIN_MISMATCH')
            if files[cp.name]['sha256'] != meta['checkpoint_sha256']:
                raise RuntimeError('G4C_COPIED_CAPSULE_PIN_MISMATCH')
        if failpoint=='after_all':raise RuntimeError('G4C_TEST_ABORT_AFTER_ALL')
        manifest={'schema':SCHEMA,'arm':arm,'completed_epoch':epoch,'terminal':terminal,
                  'files':files,'capsule_filename':None if terminal else cp.name,
                  'capsule_receipt_pin':capsule_receipt_pin,
                  'b_test_access':'NONE','exported_off_session':False,
                  'official_training_authorized':False}
        raw=_bytes(manifest)
        with (staging/'manifest.json').open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        _fsync_dir(staging)
        if failpoint=='before_commit':raise RuntimeError('G4C_TEST_ABORT_BEFORE_COMMIT')
        if folder.exists() or folder.is_symlink():raise FileExistsError('G4C_COMMIT_DESTINATION_RACE')
        os.rename(staging,folder)
        _fsync_dir(root)
        committed=True
        return {'status':'G4C_LOCAL_IMMUTABLE_BUNDLE_COMMITTED_EXPORT_NOT_VERIFIED',
                'arm':arm,'completed_epoch':epoch,'terminal':terminal,
                'manifest_sha256':hashlib.sha256(raw).hexdigest(),'directory':folder.name}
    finally:
        if not committed and staging.is_dir():shutil.rmtree(staging)
        lease.unlink(missing_ok=True)
        _fsync_dir(root)


def verify_epoch(*,root,arm,epoch,external_manifest_sha256):
    root=_private(root)
    if arm not in ALLOWED_ARMS or type(epoch) is not int or not 0<=epoch<100:
        raise RuntimeError('G4C_INVALID_ARM_EPOCH')
    if (not isinstance(external_manifest_sha256,str) or len(external_manifest_sha256)!=64 or
            any(c not in '0123456789abcdef' for c in external_manifest_sha256)):
        raise RuntimeError('G4C_PIN_FROM_EXTERNAL_EVIDENCE_REQUIRED')
    d=root/f'{arm}_epoch{epoch+1:03d}'
    if not d.is_dir() or d.is_symlink():raise RuntimeError('G4C_MISSING_BUNDLE')
    m=d/'manifest.json'
    if not m.is_file() or m.is_symlink() or _hash_file(m)!=external_manifest_sha256:
        raise RuntimeError('G4C_MANIFEST_EXTERNAL_PIN_FAILED')
    obj=json.loads(m.read_bytes())
    if (obj.get('schema')!=SCHEMA or obj.get('arm')!=arm or obj.get('completed_epoch')!=epoch or
            type(obj.get('terminal')) is not bool or
            obj.get('b_test_access')!='NONE' or obj.get('exported_off_session') is not False or
            obj.get('official_training_authorized') is not False):
        raise RuntimeError('G4C_BUNDLE_IDENTITY_INVALID')
    capsule_name=obj.get('capsule_filename')
    if obj.get('terminal') is True:
        if capsule_name is not None:raise RuntimeError('G4C_TERMINAL_CAPSULE_NAME_PROHIBITED')
        expected={'last.pt','selected_best.pt'}
    else:
        if (not isinstance(capsule_name,str) or
                not re.fullmatch(r'[A-Za-z0-9_-]{1,160}\.pt',capsule_name) or
                capsule_name in {'last.pt','selected_best.pt'}):
            raise RuntimeError('G4C_INVALID_ORIGINAL_CAPSULE_NAME')
        receipt_name=Path(capsule_name).with_suffix('.receipt.json').name
        expected={'last.pt','selected_best.pt',capsule_name,receipt_name}
    if set(obj.get('files',{}))!=expected or {x.name for x in d.iterdir()} != expected|{'manifest.json'}:
        raise RuntimeError('G4C_FILE_SET_INVALID')
    for name,meta in obj['files'].items():
        p=d/name
        if not p.is_file() or p.is_symlink() or set(meta)!={'sha256','bytes'} or _hash_file(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:
            raise RuntimeError('G4C_FILE_PIN_INVALID')
    if obj.get('terminal') is not True:
        pin=obj.get('capsule_receipt_pin')
        if (not isinstance(pin,str) or obj['files'][receipt_name]['sha256'] != pin):
            raise RuntimeError('G4C_BUNDLE_RECEIPT_PIN_DRIFT')
        meta=json.loads((d/receipt_name).read_bytes())
        if (meta.get('schema')!=RECOVERY_RECEIPT_SCHEMA or
                meta.get('checkpoint_filename')!=capsule_name or
                meta.get('checkpoint_sha256')!=obj['files'][capsule_name]['sha256'] or
                meta.get('cursor',{}).get('completed_epoch')!=epoch or
                meta.get('cursor',{}).get('next_epoch')!=epoch+1):
            raise RuntimeError('G4C_BUNDLE_CAPSULE_RECEIPT_RELATION_DRIFT')
    return {'status':'G4C_LOCAL_VERIFICATION_PASS_ONLY','arm':arm,'completed_epoch':epoch,'terminal':obj['terminal']}


def recovery_restore_contract(*, root, arm, epoch, external_manifest_sha256):
    """Derive safe relative names and hashes from externally pinned, verified bytes.

    Return metadata only. The actual R2 recovery loader must independently read the
    checkpoint+receipt with weights_only=True and the frozen complete identity.
    Do not treat this as an actual cross-session or remote-export verification.
    """
    checked=verify_epoch(root=root,arm=arm,epoch=epoch,
                         external_manifest_sha256=external_manifest_sha256)
    if checked['terminal']:
        raise RuntimeError('G4C_TERMINAL_BUNDLE_NOT_RESUMABLE')
    folder=Path(root)/(f'{arm}_epoch{epoch+1:03d}')
    meta=json.loads((folder/'manifest.json').read_bytes())
    filename=meta['capsule_filename']
    pin=meta['capsule_receipt_pin']
    if not isinstance(pin,str) or len(pin)!=64:
        raise RuntimeError('G4C_NO_AUTHENTICATED_RECEIPT_PIN')
    return {
        'schema':'CHAT21_G4C_R3_REHYDRATE_CONTRACT',
        'arm':arm,'completed_epoch':epoch,
        'checkpoint_filename':filename,
        'receipt_filename':Path(filename).with_suffix('.receipt.json').name,
        'trusted_receipt_sha256':pin,
        'selected_best_filename':'selected_best.pt',
        'selected_best_sha256':meta['files']['selected_best.pt']['sha256'],
        'last_sha256':meta['files']['last.pt']['sha256'],
        'manifest_sha256':external_manifest_sha256,
        'b_test_access':'NONE',
        'training_performed':False,
        'offsession_transfer_verified':False,
    }

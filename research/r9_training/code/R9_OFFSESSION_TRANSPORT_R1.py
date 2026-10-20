#!/usr/bin/env python3
"""R9A candidate: immutable G4 epoch-bundle transport/inspection (CPU only).

Requires a separately pinned exact G4 manifest. Never unpickles checkpoints,
opens B-TEST, executes trainers, or claims external persistence by itself.
This is candidate release engineering, NOT official training authorization.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, stat, sys, tempfile, zipfile
from pathlib import Path

G4_SCHEMA = 'CHAT21_G4C_EPOCH_BUNDLE_CANDIDATE_R2_WITH_AUTHENTIC_RECEIPT'
RECEIPT_SCHEMA = 'D4E_TRUSTED_EPOCH_BOUNDARY_RECOVERY_R2_AUGMENT'
TRANSPORT_SCHEMA = 'CHAT23_R9A_LOCAL_TRANSFER_R1_NOT_EXTERNALLY_ACCEPTED'
ARMS = ('MV00', 'MV01')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
SAFE_CAPSULE = re.compile(r'^[A-Za-z0-9_-]{1,160}\.pt$')
CHUNK = 1024 * 1024


def require(ok, msg):
    if not ok:
        raise ValueError('R9_TRANSPORT_' + msg)


def hfile(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''): h.update(b)
    return h.hexdigest()


def json_bytes(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=True)+'\n').encode('ascii')


def regular(p):
    p = Path(p)
    require(p.is_file() and not p.is_symlink(), 'REQUIRE_REGULAR_FILE')
    require(p.stat().st_size > 0, 'EMPTY_FILE')
    return p


def inspect_bundle(folder, arm, index, pin):
    """Inspection of a committed G4 directory; no trust in caller filenames."""
    require(arm in ARMS and type(index) is int and 0 <= index < 100, 'ARM_OR_EPOCH')
    require(bool(HEX64.fullmatch(pin)), 'MANIFEST_PIN')
    folder=Path(folder)
    require(folder.is_dir() and not folder.is_symlink() and folder.name == f'{arm}_epoch{index+1:03d}', 'FOLDER_ID')
    mf=regular(folder/'manifest.json')
    require(hfile(mf)==pin, 'ORIGINAL_MANIFEST_MISMATCH')
    data=json.loads(mf.read_bytes())
    require(data.get('schema')==G4_SCHEMA and data.get('arm')==arm and data.get('completed_epoch')==index, 'G4_SCHEMA_OR_CURSOR')
    require(data.get('b_test_access')=='NONE' and data.get('exported_off_session') is False, 'G4_GOVERNANCE')
    # G4's historical field is retained: it describes the origin bundle and is NOT rewritten.
    require(data.get('official_training_authorized') is False, 'G4_HISTORICAL_AUTH_STATUS')
    require(type(data.get('terminal')) is bool, 'TERMINAL_FIELD')
    cp=data.get('capsule_filename')
    expected={'last.pt','selected_best.pt'}
    if data['terminal']:
        require(cp is None and data.get('capsule_receipt_pin') is None, 'TERMINAL_CAPSULE')
    else:
        require(isinstance(cp,str) and SAFE_CAPSULE.fullmatch(cp) and cp not in expected, 'CAPSULE_BASENAME')
        expected |= {cp,Path(cp).with_suffix('.receipt.json').name}
    require(set(data.get('files',{}))==expected, 'FILE_MANIFEST_KEYS')
    require({p.name for p in folder.iterdir()}==expected|{'manifest.json'}, 'UNEXPECTED_BUNDLE_MEMBER')
    for name in expected:
        p=regular(folder/name)
        info=data['files'][name]
        require(set(info)=={'sha256','bytes'} and type(info['bytes']) is int, 'FILE_RECORD_SCHEMA')
        require(p.stat().st_size==info['bytes'] and hfile(p)==info['sha256'], 'FILE_BYTES_DRIFT_'+name)
    if cp is not None:
        rp=folder/Path(cp).with_suffix('.receipt.json').name
        rec=json.loads(rp.read_bytes())
        require(hfile(rp)==data['capsule_receipt_pin'] and rec.get('schema')==RECEIPT_SCHEMA, 'RECEIPT_PIN')
        require(rec.get('checkpoint_filename')==cp and rec.get('checkpoint_sha256')==data['files'][cp]['sha256'], 'RECEIPT_CAPSULE_LINK')
        cur=rec.get('cursor',{})
        require(cur.get('completed_epoch')==index and cur.get('next_epoch')==index+1, 'RECEIPT_CURSOR')
        require(rec.get('identity',{}).get('arm')==arm, 'RECEIPT_ARM')
    return data


def inspected_archive(path, *, sidecar=None):
    """Stream-check all uncompressed members and linkage; no extraction/deserialization."""
    path=regular(path)
    if sidecar is not None:
        required_pin=Path(sidecar).read_text('utf-8').split()[0].strip().lower()
        require(bool(HEX64.fullmatch(required_pin)) and hfile(path)==required_pin, 'ZIP_EXTERNAL_PIN_MISMATCH')
    with zipfile.ZipFile(path) as z:
        members=z.infolist(); names=[m.filename for m in members]
        require(len(names)==len(set(names)), 'ZIP_DUPLICATE_NAMES')
        require('R9_TRANSPORT_RECORD.json' in names, 'MISSING_TRANSPORT_RECEIPT')
        record=json.loads(z.read('R9_TRANSPORT_RECORD.json'))
        require(record.get('schema')==TRANSPORT_SCHEMA and record.get('b_test_access')=='NONE', 'TRANSPORT_SCHEMA')
        arm=record['arm']; index=record['completed_epoch']; prefix=f'{arm}_epoch{index+1:03d}/'
        require(arm in ARMS and type(index) is int and 0<=index<100,'ZIP_ARM_CURSOR')
        for m in members:
            require(not m.is_dir() and not m.filename.startswith('/') and '..' not in Path(m.filename).parts, 'ZIP_PATH_TRAVERSAL')
            require(m.compress_type == zipfile.ZIP_STORED, 'ZIP_NOT_STORED')
        require(set(names)==set(record['member_sha256'])|{'R9_TRANSPORT_RECORD.json'}, 'ZIP_MEMBER_SET')
        for name, meta in record['member_sha256'].items():
            require(name.startswith(prefix) and '/' not in name[len(prefix):], 'ZIP_MEMBER_SCOPE')
            require(set(meta)=={'sha256','bytes'} and type(meta['bytes']) is int, 'ZIP_RECORD_SCHEMA')
            digest=hashlib.sha256(); size=0
            with z.open(name) as f:
                for part in iter(lambda: f.read(CHUNK),b''):
                    digest.update(part); size+=len(part)
            require(size==meta['bytes'] and digest.hexdigest()==meta['sha256'], 'ZIP_FILE_BYTES_'+name)
        mf=prefix+'manifest.json'
        require(record.get('g4_manifest_sha256') == record['member_sha256'][mf]['sha256'], 'ZIP_G4_MANIFEST_PIN')
        g4=json.loads(z.read(mf))
        require(g4.get('schema')==G4_SCHEMA and g4.get('arm')==arm and g4.get('completed_epoch')==index, 'ZIP_G4_CURSOR')
        expected={'last.pt','selected_best.pt'}
        if g4.get('terminal') is True:
            require(g4.get('capsule_filename') is None, 'ZIP_TERMINAL_CAPSULE')
        else:
            cp=g4.get('capsule_filename')
            require(isinstance(cp,str) and SAFE_CAPSULE.fullmatch(cp) and cp not in expected, 'ZIP_CAPSULE')
            expected |= {cp, Path(cp).with_suffix('.receipt.json').name}
        require(set(g4.get('files',{})) == expected, 'ZIP_G4_FILES')
        require(set(record['member_sha256']) == {prefix+n for n in expected|{'manifest.json'}}, 'ZIP_SET_G4')
        for n in expected:
            require(record['member_sha256'][prefix+n] == g4['files'][n], 'ZIP_VS_G4_PIN_'+n)
        if not g4['terminal']:
            rec=json.loads(z.read(prefix+Path(g4['capsule_filename']).with_suffix('.receipt.json').name))
            require(rec.get('schema')==RECEIPT_SCHEMA and rec.get('checkpoint_filename')==g4['capsule_filename'], 'ZIP_REC_SCHEMA')
            require(rec.get('checkpoint_sha256')==g4['files'][g4['capsule_filename']]['sha256'], 'ZIP_REC_CHECKPOINT')
            require(rec.get('cursor',{}).get('next_epoch')==index+1 and rec.get('identity',{}).get('arm')==arm,'ZIP_REC_CURSOR')
            require(g4.get('capsule_receipt_pin')==g4['files'][Path(g4['capsule_filename']).with_suffix('.receipt.json').name]['sha256'],'ZIP_REC_PIN')
        return {'status':'PASS_BYTE_LEVEL_ARCHIVE_VERIFICATION_NOT_GPU_REPRODUCTION', 'arm':arm,
                'completed_epoch':index,'terminal':g4['terminal'], 'zip_sha256':hfile(path),
                'g4_manifest_sha256':record['g4_manifest_sha256'],'checkpoint_sha256':g4['files']['selected_best.pt']['sha256'],
                'offsession_persistence_confirmed':False, 'training_performed':False,'b_test_access':'NONE'}


def export(args):
    folder=Path(args.publication_root)/f'{args.arm}_epoch{args.completed_epoch+1:03d}'
    g4=inspect_bundle(folder,args.arm,args.completed_epoch,args.manifest_sha256)
    dest=Path(args.destination)
    require(dest.parent.is_dir() and not dest.parent.is_symlink(), 'DESTINATION_DIRECTORY')
    require(not dest.exists() and not Path(str(dest)+'.sha256').exists(), 'NO_OVERWRITE')
    require(dest.suffix=='.zip', 'DESTINATION_ZIP_EXTENSION')
    names=sorted({'manifest.json'}|set(g4['files']))
    prefix=folder.name+'/'
    files={prefix+n:{'sha256':hfile(folder/n),'bytes':(folder/n).stat().st_size} for n in names}
    record={'schema':TRANSPORT_SCHEMA,'arm':args.arm,'completed_epoch':args.completed_epoch,'terminal':g4['terminal'],
            'g4_manifest_sha256':args.manifest_sha256,'member_sha256':files,'b_test_access':'NONE',
            'training_performed':False,'offsession_persistence_confirmed':False,
            'original_g4_manifest_governance_preserved':True}
    tmp=dest.parent/('.'+dest.name+'.incomplete.'+os.urandom(6).hex())
    try:
        with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
            for n in names:
                z.write(folder/n,prefix+n,compress_type=zipfile.ZIP_STORED)
            z.writestr('R9_TRANSPORT_RECORD.json',json_bytes(record),compress_type=zipfile.ZIP_STORED)
        # Verify complete temporary archive before committing a named export.
        checked=inspected_archive(tmp)
        inspect_bundle(folder,args.arm,args.completed_epoch,args.manifest_sha256)
        os.link(tmp,dest)  # atomic no-overwrite hard-link in same filesystem
        side=Path(str(dest)+'.sha256')
        with side.open('x') as f:
            f.write(f'{checked["zip_sha256"]}  {dest.name}\n');f.flush();os.fsync(f.fileno())
        print(json.dumps({**checked,'export_location':str(dest),'external_download_still_required':True},indent=2))
    finally:
        tmp.unlink(missing_ok=True)


def verify(args):
    report=inspected_archive(args.archive,sidecar=args.sidecar)
    if args.attestation:
        target=Path(args.attestation)
        require(not target.exists() and target.parent.is_dir(), 'ATTESTATION_NO_OVERWRITE')
        record={**report,'schema':'R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE',
                'actual_verified_file':str(Path(args.archive).absolute()),
                'location_label_user_supplied':args.location_label or 'UNSPECIFIED',
                'external_persistence_confirmed_by_software':False}
        with target.open('x') as f:
            json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report,indent=2))


def stage(args):
    """Stage externally verified bytes for later native recovery; NEVER load .pt."""
    checked=inspected_archive(args.archive,sidecar=args.sidecar)
    require(checked['arm']==args.expected_arm and checked['completed_epoch']==args.expected_completed_epoch, 'STAGE_CURSOR_OR_ARM')
    require(checked['terminal'] is False, 'TERMINAL_NOT_RESUMABLE')
    root=Path(args.destination)
    require(root.parent.is_dir() and not root.exists() and not root.is_symlink(), 'STAGE_DESTINATION_FRESH')
    # This does NOT itself establish that a user actually saved the bytes outside Kaggle.
    if args.external_attestation:
        attest=json.loads(regular(args.external_attestation).read_bytes())
        require(attest.get('zip_sha256')==checked['zip_sha256'] and attest.get('arm')==checked['arm'] and
                attest.get('completed_epoch')==checked['completed_epoch'] and
                attest.get('schema')=='R9_OFFSITE_CHECK_NOT_NATIVE_RESUME_ACCEPTANCE', 'EXTERNAL_ATTESTATION_DRIFT')
    else:
        raise ValueError('R9_STAGE_REQUIRES_SEPARATE_EXTERNAL_VERIFICATION_ATTESTATION')
    root.mkdir(mode=0o700)
    try:
        prefix=f'{checked["arm"]}_epoch{checked["completed_epoch"]+1:03d}/'
        with zipfile.ZipFile(args.archive) as z:
            mf=json.loads(z.read(prefix+'manifest.json'))
            for name in sorted(mf['files']):
                require('/' not in name and '..' not in name and name!='manifest.json', 'UNSAFE_STAGE_NAME')
                dest=root/name
                with z.open(prefix+name) as i, dest.open('xb') as o:
                    for part in iter(lambda: i.read(CHUNK),b''): o.write(part)
                    o.flush();os.fsync(o.fileno())
                require(hfile(dest)==mf['files'][name]['sha256'], 'STAGED_BYTES_CORRUPT')
            (root/'manifest.json').write_bytes(z.read(prefix+'manifest.json'))
        report={'schema':'R9_NATIVE_RESTORE_BYTES_STAGE_NOT_GPU_RESUME','arm':checked['arm'],
                'completed_epoch':checked['completed_epoch'],'next_epoch':checked['completed_epoch']+1,
                'selected_best_path':str(root/'selected_best.pt'),
                'capsule_path':str(root/mf['capsule_filename']),
                'trusted_receipt_sha256':mf['capsule_receipt_pin'],
                'transport_archive_sha256':checked['zip_sha256'],
                'external_byte_verification_attested':True,
                'native_recovery_executed':False,'official_training_authorized':False,'b_test_access':'NONE'}
        (root/'R9_STAGE_RECEIPT.json').write_bytes(json_bytes(report))
        print(json.dumps(report,indent=2))
    except BaseException:
        # Stage is disposable: never leave partially verified files as authentic recovery.
        import shutil
        shutil.rmtree(root)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    x=sub.add_parser('export'); x.add_argument('--publication-root',required=True);x.add_argument('--arm',choices=ARMS,required=True)
    x.add_argument('--completed-epoch',type=int,required=True);x.add_argument('--manifest-sha256',required=True)
    x.add_argument('--destination',required=True)
    y=sub.add_parser('verify');y.add_argument('--archive',required=True);y.add_argument('--sidecar')
    y.add_argument('--attestation');y.add_argument('--location-label')
    z=sub.add_parser('stage');z.add_argument('--archive',required=True);z.add_argument('--sidecar',required=True)
    z.add_argument('--external-attestation',required=True);z.add_argument('--destination',required=True)
    z.add_argument('--expected-arm',choices=ARMS,required=True);z.add_argument('--expected-completed-epoch',type=int,required=True)
    args=ap.parse_args()
    try:
        {'export':export,'verify':verify,'stage':stage}[args.command](args)
    except Exception as e:
        print(f'R9_TRANSPORT_FAIL_CLOSED={type(e).__name__}:{e}',file=sys.stderr)
        raise SystemExit(2)

if __name__=='__main__':main()

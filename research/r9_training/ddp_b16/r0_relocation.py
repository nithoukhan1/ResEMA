"""Restore exact pinned R0 file paths from already mounted Kaggle Inputs.

No training, no B-TEST access, no config modification, no network.
Source choices are accepted only if bytes match original frozen R0 SHA pins
(or the original Git blob pin for native_original).
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

SEAL = {'test', 'b-test', 'b_test'}
CHUNK = 4 * 1024 * 1024


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def git_blob_file(path):
    p=Path(path)
    raw=p.read_bytes()
    canonical=raw.replace(b'\r\n', b'\n') if b'\r\n' in raw else raw
    return hashlib.sha1(b'blob '+str(len(canonical)).encode('ascii')+b'\0'+canonical).hexdigest()


def safe_parts(p):
    return not any(x.casefold() in SEAL for x in Path(p).parts)


def regular(path):
    p=Path(path)
    return (p.is_file() and not p.is_symlink() and
            safe_parts(p) and stat.S_ISREG(p.stat().st_mode))


def path_is_inside(child, parent):
    return Path(child).is_relative_to(Path(parent))


def validate_source(p, expected, mode):
    if not regular(p):
        return False
    if mode == 'gitblob':
        return git_blob_file(p)==expected
    return sha256_file(p)==expected


def basename_index(roots):
    """Only list the six already attached source/provenance datasets.

    NEVER walk split-B image dataset or any TEST tree.
    """
    index={}
    for root in roots:
        root=Path(root)
        if not root.is_dir():
            raise RuntimeError('MISSING_SOURCE_DATASET:'+str(root))
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:]=sorted(d for d in dirs if d.casefold() not in SEAL
                           and not Path(base,d).is_symlink())
            if not safe_parts(base):
                dirs[:]=[]
                continue
            for name in names:
                p=Path(base,name)
                if name.casefold() in SEAL or not regular(p):
                    continue
                index.setdefault(name,[]).append(p)
    return index


def stage_copy(src, dest, expected, mode):
    """Copy to an exclusive temporary file, hash, then rename; no overwrite."""
    dest=Path(dest)
    if dest.exists() or dest.is_symlink():
        raise RuntimeError('REFUSE_OVERWRITE:'+str(dest))
    if not path_is_inside(dest, '/kaggle/working'):
        raise RuntimeError('REFUSE_WRITE_OUTSIDE_KAGGLE_WORKING:'+str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in [dest.parent,*dest.parent.parents]):
        raise RuntimeError('SYMLINK_IN_TARGET_PATH')
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=dest.parent,
                                         prefix='.r0_pinned_', delete=False) as out:
            tmp=Path(out.name)
            with Path(src).open('rb') as f:
                shutil.copyfileobj(f,out,length=CHUNK)
            out.flush()
            os.fsync(out.fileno())
        if not validate_source(tmp,expected,mode):
            raise RuntimeError('TEMP_COPY_SHA_MISMATCH:'+dest.name)
        tmp.rename(dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    if not validate_source(dest,expected,mode):
        raise RuntimeError('POST_COPY_HASH_MISMATCH:'+str(dest))


def attempt_archive_member(name, expected, mode, archives, destination):
    """Optional exact-name extraction from already attached archives only.

    No full extraction, no unverified data acceptance. Does not recurse zip.
    """
    if not archives:
        return None
    for archive in archives:
        archive=Path(archive)
        if not regular(archive):
            continue
        try:
            if not zipfile.is_zipfile(archive):
                continue
            with zipfile.ZipFile(archive) as z:
                hits=[x for x in z.infolist() if not x.is_dir() and
                      Path(x.filename).name==name and
                      safe_parts(x.filename) and x.file_size <= (1<<31)]
                for info in hits:
                    dst=Path(destination)
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    tmp=None
                    try:
                        with tempfile.NamedTemporaryFile(mode='wb',dir=dst.parent,
                            prefix='.r0_zip_',delete=False) as out:
                            tmp=Path(out.name)
                            with z.open(info) as f:
                                shutil.copyfileobj(f,out,length=CHUNK)
                            out.flush();os.fsync(out.fileno())
                        if validate_source(tmp,expected,mode):
                            if dst.exists() or dst.is_symlink():
                                raise RuntimeError('REFUSE_OVERWRITE:'+str(dst))
                            tmp.rename(dst)
                            return f'{archive.name}::{info.filename}'
                    finally:
                        if tmp is not None:
                            tmp.unlink(missing_ok=True)
        except (zipfile.BadZipFile,OSError,RuntimeError):
            # If an archive is present but unreadable, it cannot be treated
            # as trusted source. Other archives may still contain pinned bytes.
            continue
    return None


def restore_r0_config(config_path, roots, asset_pins, overlay_pins,
                      code_pins, native_blob_pin, runtime_yaml_pin,
                      mirrored_handoff=None, archives_roots=()):
    config_path=Path(config_path)
    config=json.loads(config_path.read_text('utf-8'))
    if set(config)!={'assets','overlays','r21_code','native_original','runtime_yaml'}:
        raise RuntimeError('R0_SCHEMA_DRIFT')
    mapping={}
    for category,pins in [('assets',asset_pins),('overlays',overlay_pins),('r21_code',code_pins)]:
        if set(config[category])!=set(pins):
            raise RuntimeError('R0_'+category+'_KEY_SET_DRIFT')
        for key,expected in pins.items():
            mapping[f'{category}/{key}']=(Path(config[category][key]),expected,'sha256')
    mapping['native_original']=(Path(config['native_original']),native_blob_pin,'gitblob')
    mapping['runtime_yaml']=(Path(config['runtime_yaml']),runtime_yaml_pin,'sha256')
    missing=[]
    for label,(p,pin,mode) in mapping.items():
        if not p.is_absolute() or '..' in p.parts or not safe_parts(p):
            raise RuntimeError('R0_PATH_NOT_ALLOWED:'+label)
        if p.is_file() and not validate_source(p,pin,mode):
            raise RuntimeError('R0_EXISTING_FILE_HASH_DRIFT:'+label+':'+str(p))
        if not p.is_file():
            missing.append((label,p,pin,mode))
    print('R0 CONFIG FILE COUNT:',len(mapping),'EXISTING PINNED:',len(mapping)-len(missing),
          'MISSING:',len(missing),flush=True)
    if not missing:
        return {'existing':len(mapping),'restored':0,'records':[]}
    index=basename_index(roots)
    zipped=[]
    for root in archives_roots:
        root=Path(root)
        for base,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d.casefold() not in SEAL and not Path(base,d).is_symlink()]
            if not safe_parts(base): dirs[:]=[];continue
            zipped += [Path(base,n) for n in files if n.lower().endswith(('.zip','.zipblob'))]
    records=[]; unresolved=[]
    for label,dest,pin,mode in missing:
        if not path_is_inside(dest,'/kaggle/working'):
            unresolved.append((label,dest,pin,'MISSING_READ_ONLY_DESTINATION'))
            continue
        candidates=[]
        if mirrored_handoff is not None:
            mirror=Path(mirrored_handoff)/dest.relative_to('/kaggle/working')
            candidates.append(mirror)
        candidates+=index.get(dest.name,[])
        found=None
        for p in dict.fromkeys(candidates):
            if validate_source(p,pin,mode):
                found=p
                break
        if found is not None:
            stage_copy(found,dest,pin,mode)
            records.append({'label':label,'target':str(dest),'source':str(found),
                            'pin':pin,'source_kind':'DIRECT_PINNED_FILE'})
            print('RESTORED:',label,'FROM',found,flush=True)
        else:
            src=attempt_archive_member(dest.name,pin,mode,zipped,dest)
            if src is not None:
                records.append({'label':label,'target':str(dest),'source':src,
                                'pin':pin,'source_kind':'PINNED_ZIP_MEMBER'})
                print('RESTORED:',label,'FROM ARCHIVE',src,flush=True)
            else:
                unresolved.append((label,dest,pin,'NOT_IN_ATTACHED_INPUTS'))
                print('MISSING:',label,'EXPECTED NAME:',dest.name,'PIN:',pin,flush=True)
    if unresolved:
        raise RuntimeError('R0_RELOCATION_INCOMPLETE: '+json.dumps([
            {'label':a,'target':str(b),'pin':c,'reason':d}
            for a,b,c,d in unresolved],sort_keys=True))
    for label,(p,pin,mode) in mapping.items():
        if not validate_source(p,pin,mode):
            raise RuntimeError('R0_FINAL_PIN_FAILURE:'+label)
    return {'existing':len(mapping)-len(missing),'restored':len(records),'records':records}

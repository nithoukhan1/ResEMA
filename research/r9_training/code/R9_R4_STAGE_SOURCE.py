#!/usr/bin/env python3
"""Stage the EXACT archived R9A source, integration modules and candidate recipe.

CPU-only, no training; uses the authentic embedded ZIP hierarchy from Chat 23.
"""
from __future__ import annotations
import argparse, hashlib, io, json, shutil, sys, zipfile
from pathlib import Path

R3_NAME='DETECTION_PROJECT_CHAT23_R9B_NATIVE_WORKER_INTEGRATION_R3_CPU_CANDIDATE.zip'
R2_NAME='DETECTION_PROJECT_CHAT23_R9B_NATIVE_BOUNDARY_INTEGRATION_R2.zip'
R1_NAME='DETECTION_PROJECT_CHAT23_R9B_CONTROL_PLANE_CANDIDATE_R1.zip'
R9A_NAME='DETECTION_PROJECT_CHAT23_R9A_RELEASE_ENGINEERING_CANDIDATE_R1.zip'
PINS={R2_NAME:'12aa1e806d430c62f52363957527c7468bab3500f09fd936d344bc8c434c431c',
      R1_NAME:'c354dd49ad76c9317ae7e6f39864a526a44648a53f92988485c98bb2caed7b81',
      R9A_NAME:'c9aa976bff0710a2581b619008b7056ab398f498201cb857e86ed96e0e9ae734'}

def need(ok,msg):
    if not ok: raise RuntimeError('R9_R4_STAGE_FAIL_CLOSED_'+msg)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--r3-archive',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args(); root=Path(a.output)
    need(root.parent.is_dir() and not root.exists(),'OUTPUT_MUST_BE_NEW')
    path=Path(a.r3_archive);need(path.is_file(),'R3_ARCHIVE_MISSING')
    with zipfile.ZipFile(path) as r3:
        r2=r3.read(R2_NAME)
    need(hashlib.sha256(r2).hexdigest()==PINS[R2_NAME],'R2_PIN')
    with zipfile.ZipFile(io.BytesIO(r2)) as z:r1=z.read(R1_NAME)
    need(hashlib.sha256(r1).hexdigest()==PINS[R1_NAME],'R1_PIN')
    with zipfile.ZipFile(io.BytesIO(r1)) as z:r9a=z.read(R9A_NAME)
    need(hashlib.sha256(r9a).hexdigest()==PINS[R9A_NAME],'R9A_PIN')
    import R9B_R3_NATIVE_WORKER_INTEGRATION as native
    root.mkdir(mode=0o700)
    try:
        source=root/'source';mods=root/'modules';source.mkdir();mods.mkdir()
        with zipfile.ZipFile(io.BytesIO(r9a)) as z:
            for item in z.infolist():
                name=item.filename
                if item.is_dir():continue
                if name.startswith('R9_ISOLATED_SOURCE_CANDIDATE/'):
                    relative=name.removeprefix('R9_ISOLATED_SOURCE_CANDIDATE/');base=source
                elif name.startswith('R9_INTEGRATION_MODULES_UNCHANGED/'):
                    relative=name.removeprefix('R9_INTEGRATION_MODULES_UNCHANGED/');base=mods
                else:continue
                pieces=Path(relative)
                need(relative and not pieces.is_absolute() and '..' not in pieces.parts,'UNSAFE_ZIP_ENTRY')
                dst=base/pieces;dst.parent.mkdir(parents=True,exist_ok=True)
                need(not dst.exists(),'DUPLICATE_ENTRY')
                with dst.open('xb') as f:f.write(z.read(item))
            recipes={'R9_RELEASE_RECIPE_CANDIDATE_R1.json':'dd0ef5903dc3229abfe0c5e9298d0c288e3c682d2d14a8314530caf4ad3ad074'}
            for fn,pin in recipes.items():
                b=z.read(fn);need(hashlib.sha256(b).hexdigest()==pin,'RECIPE_BYTES')
                (root/fn).write_bytes(b)
        manifest=Path(__file__).with_name('R9_CANDIDATE_MANIFEST_R1.json')
        checked=native.verify_science_source(source,mods,manifest)
        report={'schema':'R9_R4_VERIFIED_SCIENCE_STAGE_CPU_ONLY',
                'status':'PASS_SOURCE_BYTE_VERIFICATION_NO_GPU',
                **checked,'recipe_sha256':recipes['R9_RELEASE_RECIPE_CANDIDATE_R1.json'],
                'recipe_path':str(root/'R9_RELEASE_RECIPE_CANDIDATE_R1.json'),
                'source_root':str(source),'modules_root':str(mods),
                'source_manifest_path':str(manifest),'gpu_training_performed':False,
                'official_training_authorized':False,'b_test_access':'NONE'}
        with (root/'STAGE_REPORT.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True)
        print(json.dumps(report,indent=2,sort_keys=True))
    except BaseException:
        shutil.rmtree(root)
        raise
if __name__=='__main__':main()

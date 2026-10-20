#!/usr/bin/env python3
"""Source manifest + file-change rejection using actual R9A archived bytes, CPU-only."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import importlib.util, json
root=Path(__file__).parent
spec=importlib.util.spec_from_file_location('w',root/'R9B_R3_NATIVE_WORKER_INTEGRATION.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
a=root/'DETECTION_PROJECT_CHAT23_R9B_NATIVE_BOUNDARY_INTEGRATION_R2.zip'
# R2 nests R9B R1, which nests R9A; no need to re-download history.
with ZipFile(a) as z: parent=z.read('DETECTION_PROJECT_CHAT23_R9B_CONTROL_PLANE_CANDIDATE_R1.zip')
from io import BytesIO
with ZipFile(BytesIO(parent)) as z:r9a=z.read('DETECTION_PROJECT_CHAT23_R9A_RELEASE_ENGINEERING_CANDIDATE_R1.zip')
with TemporaryDirectory() as td:
    temp=Path(td);s=temp/'source';m=temp/'modules';s.mkdir();m.mkdir()
    with ZipFile(BytesIO(r9a)) as z:
        for name in z.namelist():
            if name.startswith('R9_ISOLATED_SOURCE_CANDIDATE/') and not name.endswith('/'):
                n=name.removeprefix('R9_ISOLATED_SOURCE_CANDIDATE/');d=s/n
            elif name.startswith('R9_INTEGRATION_MODULES_UNCHANGED/') and not name.endswith('/'):
                n=name.removeprefix('R9_INTEGRATION_MODULES_UNCHANGED/');d=m/n
            else:continue
            d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(z.read(name))
    result=w.verify_science_source(s,m,root/'R9_CANDIDATE_MANIFEST_R1.json')
    assert result=={'source_files_verified':336,'integration_modules_verified':7},result
    candidate=s/'ultralytics/models/yolo/detect/vgra_trainer.py'
    old=candidate.read_bytes();candidate.write_bytes(old+b'\n# changed\n')
    try:w.verify_science_source(s,m,root/'R9_CANDIDATE_MANIFEST_R1.json')
    except RuntimeError:pass
    else:raise RuntimeError('ALTERED_SCIENCE_SOURCE_ACCEPTED')
    candidate.write_bytes(old)
    assert w.verify_science_source(s,m,root/'R9_CANDIDATE_MANIFEST_R1.json')==result
    print(json.dumps({'schema':'CHAT23_R9B_R3_REAL_SOURCE_STAGE_CPU','status':'PASS',
         'authentic_source_files':336,'authentic_integration_modules':7,
         'tampering_rejected':True,'restored_source_reaccepted':True,
         'training_performed':False,'gpu_used':False,'b_test_access':'NONE'},indent=2))

#!/usr/bin/env python3
"""D4-E Step14.4B / Chat20 read-only pinned-input identity preflight.

This script does NOT modify original sources, stage code, deserialize model weights,
access dataset images beyond a single already-pinned TRAIN PNG, access B-TEST, or train.
Passing proves file-byte identity only, not GPU replay or approved production source.

Usage:
    python d4e_r3_read_only_identity_preflight.py --config /path/to/paths.json \
      --approved-runtime-yaml-sha256 HASH --arm MV00 --run-id r3_mv00_control \
      --out /some/new/path/receipt.json
The runtime YAML hash must be taken from independently reviewed preparation evidence.
Do NOT approve a hash merely by computing it from an unreviewed input file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

SOURCE_COMMIT = '48cc059cfa133ae5fbfcec607f15daeda843d3ba'
NATIVE_BLOB_SHA1 = 'a3ad643da2141a425b92cee6c51bcd3417ec3506'
ASSET_PINS = {
    'pinned_git_snapshot_zip': 'ed176294b219159072e47f0baa34e7d6110530345afe8ffc66c91df2502f544e',
    'bridge_zip': 'd6e9d412e33e57a59f41d4faa75d5b376ba2046c1a3d065a1f06e5c2e9e7c510',
    'early_archive_zip': '4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396',
    'early_best_pt': '620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1',
    'visibility_counts_json': '9de034fa7df668d17effa843674c097341af9dd6461963f9e2c454c5f358b289',
    'recovered_train_png': 'fe119a7e3deb1792aa1b70a79dddcd9e6891d4c57fe7adf273a1218a87a9a2fe',
    'assignment_csv': '87aa5e9089c6282df14ae32917946d70d9f56eaa9a5d63443985877ab9dd1ef4',
}
OVERLAY_PINS = {
    'd4e_fixed_conf_metric.py': '9a6f27b746084d483f229d735d750673c86364d7a2b91e592f52afe0cd9c7617',
    'd4e_visibility_binding_guard.py': 'bcc8ce97cb5260f28ca05f67c1fa34cecd5d30d698596081a26ec89865452e50',
    'early_paired_trainer.py': '5dbbb09a12a934d7d7821d70ae3c4ca8f0ff44864465187b30171bbfa5d88b4f',
    'early_paired_validator.py': 'cacd9ed61d55fd702a5d20aefdc442a04a2c8a5567464195d1e8d74537804fc6',
    'vgra_fail_closed.py': 'e1947b645202f3d73f6db035ffa673350837bd162ddd2822cc5ee6cd5bde082f',
    'vgra_pretrained_boundary.py': 'a2210aa18c57f1e4c347c44859acd1d74450c7fec80fd7e4ed4e7014a14e0575',
    'vgra_trainer.py': 'eba9392f8b9a40e1daa081b46c12c07f7beaae0f785e8cd89faa63971df32486',
    'vgra_validator.py': '97636dc58ff4dac1e945df64eee589c2111c35bc7fb0cde60ebf1a02224501ad',
}
# Code bytes checked against R2.1 archived ZIP members. No changes to those files.
R21_PINS = {
    'd4e_recovery_integration.py': '275378143733d38937aa0813e615fc4a6bd8c5d17b1748ecb3e383a92bd76c42',
    'd4e_trusted_recovery_capsule.py': '8c0335b6cafe1af6b101af904153eb855013110f1e1e4d4d9d22ed52429557a3',
    'd4e_apply_private_native_recovery_patch.py': '226d10e8712ad7df55018551ce7a179d1f2f2813fc130e91fe86e8cf8639a9e1',
}
CANDIDATE_RECIPE_SHA256 = 'dd0ef5903dc3229abfe0c5e9298d0c288e3c682d2d14a8314530caf4ad3ad074'
SCHEMA = 'D4E_CHAT20_R0_READ_ONLY_IDENTITY_PREFLIGHT_NOT_GPU_AUTHORITY'
SHA_RE = re.compile(r'^[0-9a-f]{64}$')


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(path, *, exclude_test=True):
    """Reject links, special nodes, path traversal; do not search or infer locations."""
    if not isinstance(path, str) or not path or not os.path.isabs(path):
        raise ValueError('R0_REQUIRE_EXPLICIT_ABSOLUTE_PATH')
    given = Path(path)
    if '..' in given.parts:
        raise ValueError('R0_PARENT_TRAVERSAL_FORBIDDEN')
    if exclude_test and any(part.lower() in {'test', 'b-test', 'b_test'} for part in given.parts):
        raise ValueError('R0_SEALED_TEST_PATH_FORBIDDEN')
    for part in [given, *given.parents]:
        if part.is_symlink():
            raise ValueError('R0_SYMLINK_ANCESTOR_FORBIDDEN')
    metadata = given.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError('R0_REGULAR_UNLINKED_FILE_REQUIRED')
    return given


def require_pinned_file(path, expected, label):
    if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
        raise ValueError(f'R0_{label}_EXPECTED_SHA_INVALID')
    p = safe_file(path)
    actual = file_sha(p)
    if actual != expected:
        raise ValueError(f'R0_{label}_SHA256_MISMATCH')
    return {'sha256': actual, 'size_bytes': p.stat().st_size}


def git_blob_sha(b):
    return hashlib.sha1(b'blob ' + str(len(b)).encode('ascii') + b'\x00' + b).hexdigest()


def verify_native_blob(path):
    """Read-only CRLF-to-LF proof; never normalize/mutate originals on disk."""
    p = safe_file(path)
    raw = p.read_bytes()
    if git_blob_sha(raw) == NATIVE_BLOB_SHA1:
        mode, approved = 'PINNED_LF', raw
    else:
        if b'\r\n' not in raw:
            raise ValueError('R0_NATIVE_GIT_BLOB_UNRECOGNIZED')
        approved = raw.replace(b'\r\n', b'\n')
        if git_blob_sha(approved) != NATIVE_BLOB_SHA1:
            raise ValueError('R0_NATIVE_EOL_NORMALIZED_GIT_BLOB_DRIFT')
        mode = 'NORMALIZED_IN_MEMORY_ONLY'
    return {'original_sha256': sha(raw), 'canonical_sha256': sha(approved),
            'canonical_git_blob_sha1': NATIVE_BLOB_SHA1, 'mode': mode}


def verify_recipe(candidate):
    """Only narrow candidate assertions; not a user-authorized final recipe freeze."""
    if not isinstance(candidate, dict) or candidate.get('schema') != 'CHAT23_R9_100EPOCH_AMP_BATCH8_VAL16_RECIPE_CANDIDATE_R1':
        raise ValueError('R0_RECIPE_SCHEMA_UNRECOGNIZED')
    if candidate.get('status') != 'RELEASE_CANDIDATE_NOT_AUTHORIZED':
        raise ValueError('R0_RECIPE_STATUS_UNSAFE')
    controls = candidate.get('controls')
    exact = {'train_batch': 8, 'val_batch': 16, 'imgsz': 1024, 'amp': True,
             'optimizer': 'SGD', 'seed': 42, 'nbs': 64, 'workers': 0,
             'epochs_configuration_only': 100, 'save': True, 'val': True,
             'mosaic': 0.0, 'mixup': 0.0, 'cutmix': 0.0, 'copy_paste': 0.0,
             'deterministic': True}
    if candidate.get('source_commit') != SOURCE_COMMIT or set(candidate) != {'schema','status','source_commit','b_test_access','official_training_authorized','controls'}:
        raise ValueError('R0_RECIPE_TOP_LEVEL_DRIFT')
    if not isinstance(controls, dict) or any(type(controls.get(k)) != type(v) or controls[k] != v for k,v in exact.items()):
        raise ValueError('R0_RECIPE_COMMON_CONTROL_DRIFT')
    if set(controls) != set(exact):
        raise ValueError('R0_RECIPE_UNREVIEWED_CONTROL_KEYS')
    if candidate.get('b_test_access') != 'NONE' or candidate.get('official_training_authorized') is not False:
        raise ValueError('R0_RECIPE_GOVERNANCE_VIOLATION')


def verify_runtime_yaml(path, external_approved_sha):
    if not SHA_RE.fullmatch(external_approved_sha or ''):
        raise ValueError('R0_EXTERNAL_APPROVED_RUNTIME_YAML_PIN_REQUIRED')
    p = safe_file(path)
    raw = p.read_bytes()
    if sha(raw) != external_approved_sha:
        raise ValueError('R0_RUNTIME_YAML_NOT_EXTERNALLY_PINNED')
    # Defense in depth: do not use/parse/access B-TEST. Check YAML first-level key syntax.
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError('R0_PYYAML_REQUIRED_FOR_TRAIN_VAL_ONLY_GUARD') from e
    content = yaml.safe_load(raw)
    if not isinstance(content, dict) or not {'train', 'val'} <= set(content):
        raise ValueError('R0_YAML_MISSING_TRAIN_VAL')
    if 'test' in content or any('test' in str(k).lower() for k in content):
        raise ValueError('R0_SEALED_TEST_YAML_KEY_FORBIDDEN')
    for k in ('path', 'train', 'val'):
        if k == 'path' and k not in content:
            continue
        if not isinstance(content[k], str) or not content[k].strip():
            raise ValueError('R0_DATASET_PATH_MUST_BE_EXPLICIT_STRING')
        components = re.split(r'[/\\]', content[k].lower())
        if any(comp in {'test', 'b-test', 'b_test'} for comp in components):
            raise ValueError('R0_TEST_DATASET_REFERENCE_FORBIDDEN')
    return {'sha256': sha(raw), 'size_bytes': len(raw), 'train_val_keys_only': True}


def verify_inputs(config, arm, run_id, approved_yaml_sha, recipe_path, *, pins=ASSET_PINS,
                  overlay_pins=OVERLAY_PINS, code_pins=R21_PINS, native_check=True):
    if arm not in ('MV00', 'MV01') or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', run_id):
        raise ValueError('R0_ARM_OR_RUN_ID_INVALID')
    # No user-supplied hashes accepted as an alternative to pinned constants.
    if not isinstance(config, dict) or set(config) != {'assets', 'overlays', 'r21_code', 'native_original', 'runtime_yaml'}:
        raise ValueError('R0_CONFIG_SCHEMA_INVALID')
    for category, expected in [('assets', pins), ('overlays', overlay_pins), ('r21_code', code_pins)]:
        if not isinstance(config[category], dict) or set(config[category]) != set(expected):
            raise ValueError(f'R0_{category.upper()}_KEYS_MISMATCH')
    assets = {k:require_pinned_file(config['assets'][k], pin, k) for k,pin in pins.items()}
    overlays = {k:require_pinned_file(config['overlays'][k], pin, 'OVERLAY_'+k) for k,pin in overlay_pins.items()}
    code = {k:require_pinned_file(config['r21_code'][k], pin, 'CODE_'+k) for k,pin in code_pins.items()}
    native = verify_native_blob(config['native_original']) if native_check else {'fixture_only': True}
    recipe = json.loads(safe_file(recipe_path).read_text('utf-8'))
    verify_recipe(recipe)
    recipe_sha = file_sha(Path(recipe_path))
    if recipe_sha != CANDIDATE_RECIPE_SHA256:
        raise ValueError('R0_RECIPE_FILE_BYTES_DRIFT')
    runtime = verify_runtime_yaml(config['runtime_yaml'], approved_yaml_sha)
    overlay_composite = sha(canonical({k: overlays[k]['sha256'] for k in sorted(overlays)}))
    identity = {'arm': arm, 'source_commit': SOURCE_COMMIT,
                'pretrained_sha256': assets['early_best_pt']['sha256'],
                'recipe_sha256': recipe_sha,
                'data_yaml_sha256': runtime['sha256'],
                'assignment_sha256': assets['assignment_csv']['sha256'],
                'overlays_sha256': overlay_composite, 'run_id': run_id}
    synthetic = pins is not ASSET_PINS or overlay_pins is not OVERLAY_PINS or code_pins is not R21_PINS or not native_check
    record = {'schema': SCHEMA, 'status': ('PASS_SYNTHETIC_FIXTURE_ONLY' if synthetic else 'PASS_LOCAL_FILES_MATCH_PINNED_BYTES_NOT_GPU_ACCEPTANCE'),
              'identity': identity, 'assets': assets, 'overlays': overlays,
              'r21_code': code, 'native_trainer': native,
              'runtime_yaml': runtime, 'recipe_sha256': recipe_sha,
              'b_test_access': 'NONE', 'patient_png_hashed_as_bytes_only': True, 'patient_images_decoded': False,
              'deserialization_performed': False, 'gpu_used': False,
              'official_training_authorized': False, 'R3_GPU_PROVEN': False,
              'unresolved': ['R2.1 mixin still accepts caller-made identity without mandatory verifier',
                             'External immutable receipt binding R2.1 code/persistence must be integrated',
                             'Paired interrupted-vs-uninterrupted GPU replay not yet run']}
    record['all_input_hashes_sha256'] = sha(canonical({k:v for k,v in record.items() if k != 'all_input_hashes_sha256'}))
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True,help='JSON exact path mapping, no hash overrides')
    ap.add_argument('--recipe',required=True)
    ap.add_argument('--approved-runtime-yaml-sha256',required=True)
    ap.add_argument('--arm',choices=('MV00','MV01'),required=True)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    config=json.loads(safe_file(args.config).read_text('utf-8'))
    rec=verify_inputs(config,args.arm,args.run_id,args.approved_runtime_yaml_sha256,args.recipe)
    out=Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError('R0_RECEIPT_MUST_BE_NEW')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        json.dump(rec,f,indent=2,sort_keys=True)
        f.write('\n')
    print('PASS_LOCAL_PINNED_INPUT_PREFLIGHT_ONLY',rec['identity']['arm'], 'RECEIPT_SHA256',file_sha(out))


if __name__=='__main__':
    main()

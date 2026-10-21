#!/usr/bin/env python3
"""Post-R8 official-run approval gate. No approval is shipped with the code.

The frozen R6 candidate/G4 authority fields stay unchanged. A later separately
approved authorization record attaches official-run identity to a particular
published Git commit and both distinct run IDs. A JSON flag is an approval
record, not a cryptographic signature or an offsite preservation receipt.
"""
from __future__ import annotations
import re
from pathlib import Path

BASE = 'b783eb3a470ab010785a8a24a90a666e9a3f2ed6'
ARMS = {'MV00': '0', 'MV01': '1'}
SCHEMA = 'DETECTION_CHAT25_OFFICIAL_FRESH_100EPOCH_RUN_AUTHORIZATION_C1'
PURPOSE = 'USER_APPROVED_FINAL_PRODUCTION_RELEASE_AND_EXACT_FRESH_100EPOCH_MV00_MV01'
KEYS = {'schema', 'purpose', 'approval_record', 'release_commit', 'frozen_science_commit',
        'run_ids', 'official_training_authorized', 'b_test_access', 'max_epochs',
        'checkpoint_retention', 'fresh_pretrained_only'}
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
RUN = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')

def need(ok, message):
    if not ok:
        raise RuntimeError('CHAT25_PROD_APPROVAL_FAIL_CLOSED_' + message)

def validate(specs, approval, pinned_commit, *, execute=False):
    need(type(pinned_commit) is str and HEX40.fullmatch(pinned_commit) is not None,
         'EXTERNAL_RELEASE_COMMIT_PIN')
    need(pinned_commit != BASE, 'APPROVED_RELEASE_MUST_NOT_BE_BASE')
    need(type(specs) is dict and set(specs) == set(ARMS), 'EXACT_TWO_ARMS')
    for arm in ARMS:
        s = specs[arm]
        need(type(s) is dict and s.get('arm') == arm and s.get('physical_gpu') == ARMS[arm],
             'ARM_GPU_'+arm)
        need(s.get('mode') in ('FRESH_FROM_EARLY', 'RESUME_FROM_OUR_OWN_RUN'), 'MODE_'+arm)
        start, target = s.get('start_epoch_one_based'), s.get('target_epoch_one_based')
        need(type(start) is int and type(target) is int and 1 <= start <= target <= 100,
             'CURSOR_'+arm)
        need(target - start + 1 <= 12, 'SESSION_AT_MOST_TWELVE_EPOCHS_'+arm)
        if s['mode'] == 'FRESH_FROM_EARLY':
            need(start == 1, 'FRESH_START_'+arm)
        else:
            need(start >= 2, 'RESUME_START_'+arm)
        # Reserve at least 30 minutes for final checkpoint packaging, and do not
        # request >10 hours of training inside a nominal 12-hour Kaggle version.
        need(type(s.get('monotonic_session_seconds')) in (int,float) and
             0 < s['monotonic_session_seconds'] <= 36000,
             'SESSION_TIME_'+arm)
        need(type(s.get('reserve_export_seconds')) in (int,float) and
             s['reserve_export_seconds'] >= 1800, 'PUBLISH_RESERVE_'+arm)
    need(specs['MV00']['run_id'] != specs['MV01']['run_id'], 'RUN_ID_COLLISION')
    if not execute:
        return {'status':'STATIC_RELEASE_SPEC_REVIEW_ONLY', 'release_commit':pinned_commit,
                'official_training_authorized':False, 'gpu_executed':False}
    need(type(approval) is dict and set(approval) == KEYS, 'APPROVAL_SCHEMA_KEYS')
    need(approval['schema'] == SCHEMA and approval['purpose'] == PURPOSE,
         'EXPLICIT_USER_AUTHORITY')
    need(approval['approval_record'] == 'EXPLICIT_FINAL_USER_APPROVAL_RECORDED',
         'FINAL_APPROVAL_ABSENT')
    need(approval['official_training_authorized'] is True and
         approval['b_test_access'] == 'NONE' and approval['max_epochs'] == 100 and
         approval['fresh_pretrained_only'] is True,
         'OFFICIAL_APPROVAL_AND_BTEST_BOUNDARY')
    need(approval['checkpoint_retention'] == 'KEEP_ORIGINAL_UNTIL_EXTERNAL_SHA_VERIFIED',
         'NO_UNVERIFIED_CLEANUP')
    need(approval['release_commit'] == pinned_commit and
         approval['frozen_science_commit'] == BASE, 'APPROVAL_RELEASE_IDENTITY')
    need(type(approval['run_ids']) is dict and set(approval['run_ids']) == set(ARMS),
         'APPROVED_TWO_RUN_IDS')
    for arm in ARMS:
        run = approval['run_ids'][arm]
        need(type(run) is str and RUN.fullmatch(run) is not None and
             specs[arm]['run_id'] == run, 'APPROVED_RUN_ID_'+arm)
    need(approval['run_ids']['MV00'] != approval['run_ids']['MV01'], 'APPROVED_COLLISION')
    return {'status':'OFFICIAL_RUN_APPROVAL_RECORD_MATCHED_NOT_EXECUTION_PROOF',
            'release_commit':pinned_commit, 'run_ids':approval['run_ids'],
            'b_test_access':'NONE', 'offsite_checkpoint_verified':False}

ACCEPTANCE_SCHEMA = 'DETECTION_CHAT25_FINALPATH_NONOFFICIAL_GPU_ACCEPTANCE_C1'
ACCEPTANCE_PURPOSE = 'I_APPROVE_BOUNDED_NONOFFICIAL_C1_FINALPATH_GPU_TEST_NOT_OFFICIAL_TRAINING'
ACCEPTANCE_KEYS = {'schema','purpose','release_commit','frozen_science_commit','run_ids',
                   'official_training_authorized','b_test_access','max_completed_epoch_one_based'}

def validate_acceptance(specs, consent, release_sha):
    """Independent NONOFFICIAL approval; same native path, max epoch two."""
    need(type(release_sha) is str and HEX40.fullmatch(release_sha) is not None
         and release_sha != BASE,'ACCEPTANCE_RELEASE_PIN')
    need(type(specs) is dict and set(specs)==set(ARMS), 'ACCEPTANCE_TWO_ARMS')
    need(type(consent) is dict and set(consent)==ACCEPTANCE_KEYS, 'ACCEPTANCE_KEYS')
    need(consent['schema']==ACCEPTANCE_SCHEMA and
         consent['purpose']==ACCEPTANCE_PURPOSE,'ACCEPTANCE_EXPLICIT_CONSENT')
    need(consent['official_training_authorized'] is False and
         consent['b_test_access']=='NONE','ACCEPTANCE_NOT_OFFICIAL')
    need(consent['release_commit']==release_sha and
         consent['frozen_science_commit']==BASE,'ACCEPTANCE_SHA')
    need(type(consent['max_completed_epoch_one_based']) is int and
         consent['max_completed_epoch_one_based']==2,'ACCEPTANCE_MAX_TWO')
    need(type(consent['run_ids']) is dict and set(consent['run_ids'])==set(ARMS),
         'ACCEPTANCE_RUN_IDS')
    for arm,gpu in ARMS.items():
        ss=specs[arm]
        need(type(ss) is dict and ss.get('arm')==arm and
             ss.get('physical_gpu')==gpu and
             ss.get('run_id')==consent['run_ids'][arm], 'ACCEPTANCE_ARM_'+arm)
        st,tar=ss.get('start_epoch_one_based'),ss.get('target_epoch_one_based')
        need(type(st) is int and type(tar) is int and 1<=st<=tar<=2,
             'ACCEPTANCE_BOUND_'+arm)
        need((ss.get('mode')=='FRESH_FROM_EARLY' and st==1) or
             (ss.get('mode')=='RESUME_FROM_OUR_OWN_RUN' and st==2),
             'ACCEPTANCE_MODE_'+arm)
    need(consent['run_ids']['MV00']!=consent['run_ids']['MV01'],
         'ACCEPTANCE_RUN_IDS_COLLISION')
    return {'status':'NONOFFICIAL_FINALPATH_ACCEPTANCE_APPROVED_NOT_GPU_PROOF',
            'release_commit':release_sha,'official_training_authorized':False,
            'b_test_access':'NONE'}

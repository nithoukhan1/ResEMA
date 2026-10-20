#!/usr/bin/env python3
"""Exact source and recipe verification; no GPU, no training, no B-TEST."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import R9B_R3_NATIVE_WORKER_INTEGRATION as r3
summary=r3.verify_science_source(HERE/'frozen_source',HERE/'integration_modules',HERE/'code/R9_CANDIDATE_MANIFEST_R1.json')
assert r3.file_sha(HERE/'R9_RELEASE_RECIPE_CANDIDATE_R1.json')==r3.SOURCE_RECIPE_SHA
print(json.dumps({'status':'PASS_PINNED_R9_SCIENCE_336_PLUS_7_AND_RECIPE',**summary,'official_training_authorized':False,'b_test_access':'NONE'},indent=2,sort_keys=True))

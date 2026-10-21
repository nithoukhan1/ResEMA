#!/usr/bin/env python3
"""Post-publication read-only release identity report; no GPU/authorization."""
import argparse,json,sys
from R9_OFFICIAL_RELEASE_IDENTITY_R8 import validate_checkout

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--checkout',required=True)
 p.add_argument('--pinned-release-commit',required=True)
 a=p.parse_args()
 print(json.dumps(validate_checkout(a.checkout,a.pinned_release_commit),indent=2,sort_keys=True))
 print('PIN_IDENTITY_ONLY_FROZEN_SOURCE_BYTES_MUST_BE_VERIFIED_SEPARATELY')

if __name__=='__main__':
 try:main()
 except Exception as e:
  print('R8 RELEASE PIN VERIFICATION FAILED:',repr(e),file=sys.stderr);sys.exit(2)

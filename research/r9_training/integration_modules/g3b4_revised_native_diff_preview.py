#!/usr/bin/env python3
"""Gate 3-B4: generate read-only source diff at scientific Git pin. NEVER applies it.

Output is NOT executable without a reviewed class/MRO binding, official
per-epoch export/terminal artifact contract, and explicit D4-F authorization.
"""
from __future__ import annotations
import argparse, ast, difflib, hashlib, importlib.util, json, subprocess
from pathlib import Path

PIN='48cc059cfa133ae5fbfcec607f15daeda843d3ba'
BLOB='a3ad643da2141a425b92cee6c51bcd3417ec3506'
REFERENCE_SHA256='226d10e8712ad7df55018551ce7a179d1f2f2813fc130e91fe86e8cf8639a9e1'
SOURCE_SHA256='c635a6d5aafeb9aee754e711a00c23355507f9058f600151a8283a8c1e8f886f'
YAML_PINS={
'ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml':'01caddd418e280684380c4629cb2b594a8ce628c37666216b37bef4226b11882',
'ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml':'cee2646cca95d337e36f3ad1a92914da2610e184e136e23596a5013b4566faed'}

def sha(b):return hashlib.sha256(b).hexdigest()
def gitblob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def git(repo,*args):
    p=subprocess.run(['git','-C',str(repo),*args],capture_output=True,timeout=90)
    if p.returncode:raise RuntimeError('G3B4_READ_ONLY_GIT_FAILED '+p.stderr.decode(errors='replace')[:250])
    return p.stdout

def replace_once(body,old,new,label):
    if body.count(old)!=1 or body.count(new)>0:raise RuntimeError('G3B4_MISSING_DUPLICATE_OR_ALREADY_MODIFIED_ANCHOR:'+label)
    return body.replace(old,new,1)

def proposed_r2_native(r2:bytes):
    src=r2.decode('utf-8')
    start='    def _do_train(self):\n';end='    def auto_batch(self, max_num_obj=0):\n'
    if src.count(start)!=1 or src.count(end)!=1:raise RuntimeError('G3B4_NATIVE_BOUNDARY_DRIFT')
    pre,rest=src.split(start,1);body,post=rest.split(end,1)
    for expect in ['self.d4e_recovery_start(nb)','self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)']:
        if body.count(expect)!=1:raise RuntimeError('G3B4_R2_HOOK_DRIFT:'+expect)
    edits=[
      ('mandatory_guard_class_binding',
       '        if hasattr(self, "d4e_recovery_start") and not getattr(self, "_d4e_recovery_configured", False):\n            raise RuntimeError("D4E_R2_GOVERNED_RECOVERY_CONFIGURATION_REQUIRED")\n',
       '        if hasattr(self, "d4e_recovery_start") and not getattr(self, "_d4e_recovery_configured", False):\n            raise RuntimeError("D4E_R2_GOVERNED_RECOVERY_CONFIGURATION_REQUIRED")\n'
       '        if hasattr(self, "d4e_recovery_start") and not hasattr(self, "d4e_g3b_guard_preflight"):\n            raise RuntimeError("G3B4_MANDATORY_GUARD_MIXIN_MISSING")\n'
       '        if getattr(self, "_d4e_recovery_configured", False) and not getattr(self, "_g3b4_ready", False):\n            raise RuntimeError("G3B4_EXPLICIT_GUARD_BINDING_MISSING")\n'),
      ('preflight_after_verified_resume',
       '            last_opt_step = self.d4e_recovery_start(nb)\n',
       '            last_opt_step = self.d4e_recovery_start(nb)\n'
       '            self.d4e_g3b_guard_preflight()\n'),
      ('all_loss_components_before_backward',
       '                    self.loss = loss.sum()\n',
       '                    self.loss = loss.sum()\n'
       '                    if getattr(self, "_d4e_recovery_configured", False):\n'
       '                        self.d4e_g3b_guard_loss(epoch=epoch, batch_index=i)\n'),
      ('every_accumulation_microbatch_gradient_check',
       '                self.scaler.scale(self.loss).backward()\n',
       '                self.scaler.scale(self.loss).backward()\n'
       '                if getattr(self, "_d4e_recovery_configured", False):\n'
       '                    self.d4e_g3b_guard_after_backward(epoch=epoch, batch_index=i)\n'),
      ('pre_native_optimizer_update',
       '                if ni - last_opt_step >= self.accumulate:\n',
       '                if ni - last_opt_step >= self.accumulate:\n'
       '                    if getattr(self, "_d4e_recovery_configured", False):\n'
       '                        self.d4e_g3b_guard_pre_step(epoch=epoch, batch_index=i)\n'),
      ('post_native_optimizer_and_ema_update',
       '                    self.optimizer_step()\n',
       '                    self.optimizer_step()\n'
       '                    if getattr(self, "_d4e_recovery_configured", False):\n'
       '                        self.d4e_g3b_guard_post_step(epoch=epoch, batch_index=i)\n'),
      ('validation_before_native_retry',
       '                self.metrics, self.fitness = self.validate()\n',
       '                self.metrics, self.fitness = self.validate()\n'
       '                if getattr(self, "_d4e_recovery_configured", False):\n'
       '                    self.d4e_g3b_guard_validation(epoch=epoch)\n'),
      ('no_native_nan_recovery_in_governed_runs',
       '            if self._handle_nan_recovery(epoch):\n                continue\n',
       '            if getattr(self, "_d4e_recovery_configured", False):\n'
       '                self.d4e_g3b_guard_no_native_retry(epoch=epoch)\n'
       '            elif self._handle_nan_recovery(epoch):\n                continue\n'),
      ('guard_before_metrics_and_first_checkpoint_write',
       '            if RANK in {-1, 0}:\n                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})\n',
       '            if RANK in {-1, 0}:\n'
       '                if getattr(self, "_d4e_recovery_configured", False):\n'
       '                    self.d4e_g3b_guard_pre_save(epoch=epoch)\n'
       '                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})\n'),
      ('epoch_boundary_before_resumable_capsule',
       '                self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)\n',
       '                self.d4e_g3b_guard_epoch_boundary(epoch=epoch)\n'
       '                self.d4e_recovery_epoch_end(epoch, nb, last_opt_step)\n'),
      ('terminal_artifact_protocol_fail_closed',
       '        self.final_eval()\n',
       '        self.final_eval()\n'
       '        if getattr(self, "_d4e_recovery_configured", False):\n'
       '            self.d4e_g3b_guard_terminal()  # BLOCKED until terminal export independently accepted\n')
    ]
    for label,old,new in edits:body=replace_once(body,old,new,label)
    result=(pre+start+body+end+post).encode('utf-8')
    ast.parse(result.decode())
    return result,[x[0] for x in edits]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('repo',type=Path);ap.add_argument('--output-dir',required=True,type=Path)
    a=ap.parse_args();repo=a.repo.resolve(strict=True)
    if Path(git(repo,'rev-parse','--show-toplevel').decode().strip()).resolve()!=repo:raise RuntimeError('G3B4_EXPECT_EXACT_REPO_ROOT')
    if git(repo,'cat-file','-t',PIN).strip()!=b'commit':raise RuntimeError('G3B4_SCIENTIFIC_COMMIT_NOT_LOCALLY_AVAILABLE')
    for path,pin in YAML_PINS.items():
        if sha(git(repo,'show',PIN+':'+path))!=pin:raise RuntimeError('G3B4_MODEL_YAML_DRIFT '+path)
    native=git(repo,'show',PIN+':ultralytics/engine/trainer.py')
    if sha(native)!=SOURCE_SHA256 or gitblob(native)!=BLOB:raise RuntimeError('G3B4_NATIVE_SOURCE_PIN_FAILURE')
    ref=Path(__file__).with_name('g3b_authenticated_r2_native_patch_reference.py')
    if sha(ref.read_bytes())!=REFERENCE_SHA256:raise RuntimeError('G3B4_R2_PATCH_REFERENCE_DRIFT')
    s=importlib.util.spec_from_file_location('g3b4_auth_r2',ref);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    r2=m.patch_native(native);candidate,labels=proposed_r2_native(r2)
    diff=''.join(difflib.unified_diff(native.decode().splitlines(keepends=True),candidate.decode().splitlines(keepends=True),fromfile='a/ultralytics/engine/trainer.py (pinned)',tofile='b/ultralytics/engine/trainer.py (REVIEW ONLY, NOT APPLIED)',n=5)).encode()
    if not diff:raise RuntimeError('G3B4_EMPTY_DIFF')
    out=a.output_dir.resolve()
    if out.exists() or out.is_symlink():raise RuntimeError('G3B4_REFUSE_EXISTING_OUTPUT_DIR')
    out.mkdir(parents=False)
    fname='G3B4_REVISED_NATIVE_REVIEW_ONLY.patch'
    (out/fname).write_bytes(diff)
    result={'schema':'CHAT21_GATE3B4_READ_ONLY_REVISED_HOOKS_R1',
        'status':'PASS_READ_ONLY_REVISED_HOOK_PREVIEW_WITH_OPEN_PRODUCTION_BLOCKERS',
        'scientific_commit':PIN,'pinned_native_blob':BLOB,'original_native_sha256':sha(native),
        'authenticated_r2_native_sha256':sha(r2),'proposed_native_sha256':sha(candidate),
        'review_patch_sha256':sha(diff),'review_patch_bytes':len(diff),
        'patch_filename':fname,'hook_labels':labels,'hook_count':len(labels),
        'git_branch_at_audit':git(repo,'branch','--show-current').decode().strip(),
        'git_head_at_audit':git(repo,'rev-parse','HEAD').decode().strip(),
        'git_status_at_audit':git(repo,'status','--porcelain=v1').decode(),
        'class_mro_binding_not_included':True,'durable_all_epoch_export_not_included':True,
        'terminal_export_intentionally_blocked':True,'actual_trainer_cpu_integration_not_yet_verified':True,
        'production_training_authorized':False,'b_test_access':'NONE','repo_mutated':False}
    (out/'G3B4_NATIVE_REVIEW_ONLY_REPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('G3B4_REVIEW_HOOK_COUNT='+str(len(labels)))
    print('G3B4_PATCH_SHA256='+sha(diff))
    print('G3B4_PREVIEW=PASS_READ_ONLY_NOT_EXECUTABLE_NO_D4F_AUTHORIZATION')
    print('G3B4_REPORT='+str(out/'G3B4_NATIVE_REVIEW_ONLY_REPORT.json'))
    print('B_TEST=SEALED; D4F_TRAINING=LOCKED')

if __name__=='__main__':main()

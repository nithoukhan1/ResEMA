# ======================================================================
# SMS-01 SELF-CONTAINED KAGGLE SAVE-VERSION RESUME
# CANDIDATE 1 — SCConv-Early
#
# Fresh Save Version workflow:
#   1. verify GPU + saved parent run
#   2. clone exact frozen scientific commit
#   3. install exact checkout
#   4. run governed resume preflight
#   5. run governed resume execution
#
# NO scientific overrides.
# NO test access.
# ======================================================================

from pathlib import Path
import hashlib
import subprocess
import sys
import torch

REPO_URL = "https://github.com/nithoukhan1/ResEMA.git"
BRANCH = "research/single-module-screen-01"
EXECUTION_COMMIT = "9d65b7adae3d488f1cb70856476fef0488e9749a"
TRAINING_SOURCE_COMMIT = "660475b3aa3614f58f21c98f760561c72e3f1436"

EXPERIMENT_ID = "BORG-PT-S42-SCCONV-EARLY-E100"
REPO = Path("/kaggle/working/ResEMA_SMS01_SAVE_RESUME_C1")

SOURCE_RUN = Path(
    "/kaggle/input/datasets/nithoukhan/borg-pt-s42-scconv-early-e100-1/ResEMA_single_module_runs/BORG-PT-S42-SCCONV-EARLY-E100"
)

DESTINATION_RUN = (
    Path("/kaggle/working/ResEMA_single_module_runs")
    / EXPERIMENT_ID
)

EXPECTED_RESULTS_SHA256 = "71692809e6c93eb8d04b999184f728936b3ca2248570faded6d1bc204d680d6f"
EXPECTED_LAST_SHA256 = "c9d1f06f123d1b7ab8083d7dbbcd536fc41d6aeb31693f81269783ea0df3d2d8"
EXPECTED_COMPLETED_EPOCHS = 99
EXPECTED_NEXT_EPOCH = 100
EXPECTED_TOTAL_EPOCHS = 100

def run(cmd, *, cwd=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        check=True,
    )

def capture(cmd, *, cwd=None):
    return subprocess.check_output(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
    ).strip()

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

print("=" * 110)
print("CANDIDATE 1 — SCConv-Early — SELF-CONTAINED SAVE-VERSION RESUME")
print("=" * 110)

# 1. GPU gate
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

gpu_names = [
    torch.cuda.get_device_name(i)
    for i in range(torch.cuda.device_count())
]

print("GPU_NAMES =", gpu_names)

if gpu_names != ["Tesla T4", "Tesla T4"]:
    raise RuntimeError(
        f"Frozen SMS-01 runtime requires Tesla T4 x2; observed {gpu_names}"
    )

print("GPU_GATE = PASS")

# 2. Source run binding
required = [
    SOURCE_RUN / "results.csv",
    SOURCE_RUN / "args.yaml",
    SOURCE_RUN / "weights/last.pt",
    SOURCE_RUN / "weights/best.pt",
    SOURCE_RUN / "governance/PRETRAIN_PREFLIGHT.json",
    SOURCE_RUN / "governance/RUNTIME_MANIFEST.json",
    SOURCE_RUN / "governance/SMS01_MODEL_RUNTIME.json",
]

for path in required:
    if not path.is_file():
        raise RuntimeError(f"Required saved-run file missing: {path}")

observed_results_sha = sha256_file(SOURCE_RUN / "results.csv")
observed_last_sha = sha256_file(SOURCE_RUN / "weights/last.pt")

print("EXPECTED_RESULTS_SHA256 =", EXPECTED_RESULTS_SHA256)
print("OBSERVED_RESULTS_SHA256 =", observed_results_sha)
print("EXPECTED_LAST_SHA256    =", EXPECTED_LAST_SHA256)
print("OBSERVED_LAST_SHA256    =", observed_last_sha)

if observed_results_sha != EXPECTED_RESULTS_SHA256:
    raise RuntimeError("Saved results.csv hash mismatch.")

if observed_last_sha != EXPECTED_LAST_SHA256:
    raise RuntimeError("Saved last.pt hash mismatch.")

print("SAVED_RUN_BINDING = PASS")

# 3. Fresh working state
if REPO.exists():
    raise RuntimeError(
        f"Fresh Save Version expected no existing checkout: {REPO}"
    )

if DESTINATION_RUN.exists():
    raise RuntimeError(
        f"Fresh Save Version expected no destination run: {DESTINATION_RUN}"
    )

print("FRESH_WORKING_STATE = PASS")

# 4. Exact repository checkout
run(["git", "clone", "--no-checkout", REPO_URL, str(REPO)])
run(["git", "fetch", "origin", BRANCH], cwd=REPO)
run(["git", "checkout", "-B", BRANCH, EXECUTION_COMMIT], cwd=REPO)

observed_branch = capture(["git", "branch", "--show-current"], cwd=REPO)
observed_head = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
observed_parent = capture(["git", "rev-parse", "HEAD^"], cwd=REPO)
dirty = capture(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=REPO,
)

print("OBSERVED_BRANCH =", observed_branch)
print("OBSERVED_HEAD   =", observed_head)
print("OBSERVED_PARENT =", observed_parent)
print("WORKTREE_DIRTY  =", bool(dirty))

if observed_branch != BRANCH:
    raise RuntimeError("Branch mismatch.")

if observed_head != EXECUTION_COMMIT:
    raise RuntimeError("Execution commit mismatch.")

if observed_parent != TRAINING_SOURCE_COMMIT:
    raise RuntimeError("Training-source commit mismatch.")

if dirty:
    raise RuntimeError("Repository is not clean.")

print("GIT_PROVENANCE_GATE = PASS")

# 5. Install exact checkout
run(
    [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
    cwd=REPO,
)

dirty_after_install = capture(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=REPO,
)

if dirty_after_install:
    print(dirty_after_install)
    raise RuntimeError("Editable install dirtied repository.")

print("POST_INSTALL_GIT_GATE = PASS")

# 6. Governed preflight in this same fresh Save Version
print("\n" + "=" * 110)
print("GOVERNED RESUME PREFLIGHT")
print("=" * 110)

preflight_cmd = [
    sys.executable,
    "-u",
    "-m",
    "research.runtime.single_module_resume_runner",
    "--experiment-id",
    EXPERIMENT_ID,
    "--mode",
    "preflight-only",
    "--input-root",
    "/kaggle/input",
    "--work-root",
    "/kaggle/working",
]

run(preflight_cmd, cwd=REPO)

if DESTINATION_RUN.exists():
    raise RuntimeError(
        "Preflight unexpectedly materialized destination run."
    )

print("POST_PREFLIGHT_DESTINATION_ABSENT = PASS")
print("EXPECTED_COMPLETED_EPOCHS =", EXPECTED_COMPLETED_EPOCHS)
print("EXPECTED_NEXT_EPOCH       =", EXPECTED_NEXT_EPOCH)
print("EXPECTED_TOTAL_EPOCHS      =", EXPECTED_TOTAL_EPOCHS)

# 7. Governed execution
print("\n" + "=" * 110)
print("GOVERNED RESUME EXECUTION")
print("=" * 110)

execute_cmd = [
    sys.executable,
    "-u",
    "-m",
    "research.runtime.single_module_resume_runner",
    "--experiment-id",
    EXPERIMENT_ID,
    "--mode",
    "execute",
    "--input-root",
    "/kaggle/input",
    "--work-root",
    "/kaggle/working",
]

run(execute_cmd, cwd=REPO)

print("\n" + "=" * 110)
print("CANDIDATE 1 — SCConv-Early — SAVE-VERSION RESUME RETURNED NORMALLY")
print("TEST_ACCESS = NONE")
print("=" * 110)

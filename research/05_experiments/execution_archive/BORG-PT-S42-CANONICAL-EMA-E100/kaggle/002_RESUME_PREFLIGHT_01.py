# ======================================================================
# SMS-01 GOVERNED RESUME PREFLIGHT — CANDIDATE 4 Canonical EMA
#
# Experiment:
#   BORG-PT-S42-CANONICAL-EMA-E100
#
# Expected saved state:
#   completed epochs = 89
#   next epoch       = 90
#   target           = 100
#
# IMPORTANT:
#   PREFLIGHT ONLY — MUST NOT START TRAINING.
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
EXPERIMENT_ID = "BORG-PT-S42-CANONICAL-EMA-E100"

REPO = Path("/kaggle/working/ResEMA_SMS01_RESUME_C4")
EXPECTED_SOURCE_RUN = Path(
    "/kaggle/input/datasets/abasitive/borg-pt-s42-canonical-ema-e100-1/ResEMA_single_module_runs/BORG-PT-S42-CANONICAL-EMA-E100"
)

EXPECTED_RESULTS_SHA256 = "238fe45143f52d2107833f4cd399e8976df6f64a3300351ff1a4cdab2964c812"
EXPECTED_LAST_SHA256 = "f2fcd6ba122e2fd520ae9f0049293dbcacfee760e5c9291e50b4d158977b2fb7"

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

print("=" * 100)
print("SMS-01 CANDIDATE 4 Canonical EMA — GOVERNED RESUME PREFLIGHT")
print("=" * 100)

print("\n--- GPU GATE ---")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

gpu_names = [
    torch.cuda.get_device_name(i)
    for i in range(torch.cuda.device_count())
]

print("GPU_COUNT =", len(gpu_names))
print("GPU_NAMES =", gpu_names)

if gpu_names != ["Tesla T4", "Tesla T4"]:
    raise RuntimeError("Frozen SMS-01 runtime requires Tesla T4 x2.")

print("GPU_GATE = PASS")

print("\n--- SAVED RUN BINDING ---")

if not EXPECTED_SOURCE_RUN.is_dir():
    raise RuntimeError(
        "Expected saved run was not found:\n"
        f"{EXPECTED_SOURCE_RUN}"
    )

results_csv = EXPECTED_SOURCE_RUN / "results.csv"
last_pt = EXPECTED_SOURCE_RUN / "weights/last.pt"
best_pt = EXPECTED_SOURCE_RUN / "weights/best.pt"

for path in (
    results_csv,
    last_pt,
    best_pt,
    EXPECTED_SOURCE_RUN / "args.yaml",
    EXPECTED_SOURCE_RUN / "governance/PRETRAIN_PREFLIGHT.json",
    EXPECTED_SOURCE_RUN / "governance/RUNTIME_MANIFEST.json",
    EXPECTED_SOURCE_RUN / "governance/SMS01_MODEL_RUNTIME.json",
):
    if not path.is_file():
        raise RuntimeError(f"Required prior-run file missing: {path}")

observed_results_sha = sha256_file(results_csv)
observed_last_sha = sha256_file(last_pt)

print("EXPECTED_RESULTS_SHA256 =", EXPECTED_RESULTS_SHA256)
print("OBSERVED_RESULTS_SHA256 =", observed_results_sha)
print("EXPECTED_LAST_SHA256    =", EXPECTED_LAST_SHA256)
print("OBSERVED_LAST_SHA256    =", observed_last_sha)

if observed_results_sha != EXPECTED_RESULTS_SHA256:
    raise RuntimeError(
        "results.csv does not match SAVED-RUN AUDIT 01."
    )

if observed_last_sha != EXPECTED_LAST_SHA256:
    raise RuntimeError(
        "last.pt does not match SAVED-RUN AUDIT 01."
    )

print("SAVED_RUN_BINDING = PASS")

destination = Path(
    "/kaggle/working/ResEMA_single_module_runs"
) / EXPERIMENT_ID

print("\n--- DESTINATION GATE ---")
print("DESTINATION_RUN    =", destination)
print("DESTINATION_EXISTS =", destination.exists())

if destination.exists():
    raise RuntimeError(
        "Resume destination already exists. "
        "Do not delete/overwrite it automatically."
    )

print("DESTINATION_GATE = PASS")

print("\n--- REPOSITORY CHECKOUT ---")

if REPO.exists():
    raise RuntimeError(f"Resume repo already exists: {REPO}")

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
    raise RuntimeError("Training-source parent mismatch.")
if dirty:
    raise RuntimeError("Repository is not clean.")

print("GIT_PROVENANCE_GATE = PASS")

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

print("\n" + "=" * 100)
print("RUNNING GOVERNED RESUME PREFLIGHT ONLY")
print("=" * 100)

run(
    [
        sys.executable,
        "-u",
        "research/runtime/single_module_resume_runner.py",
        "--experiment-id",
        EXPERIMENT_ID,
        "--mode",
        "preflight-only",
        "--input-root",
        "/kaggle/input",
        "--work-root",
        "/kaggle/working",
    ],
    cwd=REPO,
)

print("\n" + "=" * 100)
print("PREFLIGHT PROCESS RETURNED NORMALLY")
print("EXPECTED_COMPLETED_EPOCHS = 89")
print("EXPECTED_NEXT_EPOCH       = 90")
print("EXPECTED_PARAMETERS       = 9435423")
print("EXPECTED_STATE_ITEMS      = 527")
print("TRAINING_STARTED          = FALSE")
print("=" * 100)

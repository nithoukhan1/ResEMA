# ======================================================================
# SMS-01-T1
# GOVERNED FRESH EXECUTION — CANDIDATE 3
#
# Candidate:
#   BORG-PT-S42-DYSAMPLE-E100
#
# Scientific condition:
#   Split-B Original
#   Pretrained
#   Seed 42
#   E100
#   imgsz 1024
#   global batch 16
#   SGD
#
# IMPORTANT:
#   - NO Split-B test access
#   - NO manual hyperparameter overrides
#   - NO other candidate in this run
#   - NO combinations
# ======================================================================

from pathlib import Path
import os
import subprocess
import sys
import torch


# ----------------------------------------------------------------------
# 0. FROZEN IDENTIFIERS
# ----------------------------------------------------------------------

REPO_URL = "https://github.com/nithoukhan1/ResEMA.git"

BRANCH = "research/single-module-screen-01"

EXECUTION_COMMIT = (
    "9d65b7adae3d488f1cb70856476fef0488e9749a"
)

TRAINING_SOURCE_COMMIT = (
    "660475b3aa3614f58f21c98f760561c72e3f1436"
)

EXPERIMENT_ID = (
    "BORG-PT-S42-DYSAMPLE-E100"
)

# Use a dedicated clean execution checkout.
REPO = Path(
    "/kaggle/working/ResEMA_SMS01_EXEC"
)

RUN_DIR = (
    Path("/kaggle/working/ResEMA_single_module_runs")
    / EXPERIMENT_ID
)


# ----------------------------------------------------------------------
# 1. HELPERS
# ----------------------------------------------------------------------

def run(cmd, *, cwd=None, env=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)

    subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
    )


def capture(cmd, *, cwd=None):
    return subprocess.check_output(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        text=True,
    ).strip()


print("=" * 100)
print("SMS-01 — CANDIDATE 3 GOVERNED EXECUTION")
print("=" * 100)


# ----------------------------------------------------------------------
# 2. GPU GATE
# ----------------------------------------------------------------------

print("\n--- GPU GATE ---")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Enable Kaggle GPU accelerator before continuing."
    )

gpu_count = torch.cuda.device_count()

print("CUDA_AVAILABLE =", torch.cuda.is_available())
print("GPU_COUNT      =", gpu_count)

for i in range(gpu_count):
    print(
        f"GPU_{i}          =",
        torch.cuda.get_device_name(i),
    )

if gpu_count < 2:
    raise RuntimeError(
        "Frozen SMS-01 runtime requires two GPUs "
        "(intended Kaggle T4x2). "
        f"Detected only {gpu_count}."
    )


# ----------------------------------------------------------------------
# 3. REQUIRE FRESH EXECUTION STATE
# ----------------------------------------------------------------------

print("\n--- FRESH-RUN GATE ---")

print("EXPECTED_RUN_DIR =", RUN_DIR)
print("RUN_DIR_EXISTS   =", RUN_DIR.exists())

if RUN_DIR.exists():
    raise RuntimeError(
        "\nCandidate run directory already exists:\n"
        f"{RUN_DIR}\n\n"
        "DO NOT delete it and DO NOT launch fresh training over it.\n"
        "If it contains an interrupted training run, use the "
        "governed SMS resume workflow instead."
    )

print("FRESH_RUN_DIR_GATE = PASS")


# ----------------------------------------------------------------------
# 4. CREATE EXACT CLEAN REPOSITORY CHECKOUT
# ----------------------------------------------------------------------

print("\n--- EXACT REPOSITORY CHECKOUT ---")

if REPO.exists():
    raise RuntimeError(
        "\nDedicated execution repository already exists:\n"
        f"{REPO}\n\n"
        "Do not overwrite it automatically. "
        "Share this output before proceeding."
    )

run(
    [
        "git",
        "clone",
        "--no-checkout",
        REPO_URL,
        str(REPO),
    ]
)

run(
    [
        "git",
        "fetch",
        "origin",
        BRANCH,
    ],
    cwd=REPO,
)

run(
    [
        "git",
        "checkout",
        "-B",
        BRANCH,
        EXECUTION_COMMIT,
    ],
    cwd=REPO,
)


# ----------------------------------------------------------------------
# 5. VERIFY EXACT GIT PROVENANCE
# ----------------------------------------------------------------------

print("\n--- GIT PROVENANCE GATE ---")

observed_branch = capture(
    ["git", "branch", "--show-current"],
    cwd=REPO,
)

observed_head = capture(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO,
)

observed_parent = capture(
    ["git", "rev-parse", "HEAD^"],
    cwd=REPO,
)

dirty = capture(
    [
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ],
    cwd=REPO,
)

print("EXPECTED_BRANCH          =", BRANCH)
print("OBSERVED_BRANCH          =", observed_branch)

print("EXPECTED_EXECUTION_COMMIT=", EXECUTION_COMMIT)
print("OBSERVED_HEAD            =", observed_head)

print("EXPECTED_SOURCE_COMMIT   =", TRAINING_SOURCE_COMMIT)
print("OBSERVED_PARENT          =", observed_parent)

print("WORKTREE_DIRTY           =", bool(dirty))

if observed_branch != BRANCH:
    raise RuntimeError(
        "Branch mismatch."
    )

if observed_head != EXECUTION_COMMIT:
    raise RuntimeError(
        "Execution commit mismatch."
    )

if observed_parent != TRAINING_SOURCE_COMMIT:
    raise RuntimeError(
        "Authorization parent/source mismatch."
    )

if dirty:
    print("\nDIRTY CONTENT:")
    print(dirty)

    raise RuntimeError(
        "Repository is not clean."
    )

print("GIT_PROVENANCE_GATE = PASS")


# ----------------------------------------------------------------------
# 6. INSTALL EXACT CHECKOUT WITHOUT DEPENDENCY MUTATION
# ----------------------------------------------------------------------

print("\n--- EDITABLE INSTALL ---")

run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--no-deps",
    ],
    cwd=REPO,
)


# ----------------------------------------------------------------------
# 7. RE-CHECK CLEAN WORKTREE AFTER INSTALL
# ----------------------------------------------------------------------

dirty_after_install = capture(
    [
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ],
    cwd=REPO,
)

print(
    "\nWORKTREE_DIRTY_AFTER_INSTALL =",
    bool(dirty_after_install),
)

if dirty_after_install:
    print("\nDIRTY CONTENT:")
    print(dirty_after_install)

    raise RuntimeError(
        "Editable install changed tracked/untracked repository state. "
        "Stopping before governed execution."
    )

print("POST_INSTALL_GIT_GATE = PASS")


# ----------------------------------------------------------------------
# 8. GOVERNED EXECUTION
# ----------------------------------------------------------------------

print("\n" + "=" * 100)
print("STARTING GOVERNED SMS-01 CANDIDATE 3")
print("=" * 100)

print("EXPERIMENT_ID =", EXPERIMENT_ID)
print("MODE          = execute")
print("TEST_ACCESS   = NONE")
print("COMBINATIONS  = FALSE")
print()

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

run(
    [
        sys.executable,
        "-u",
        "research/runtime/single_module_runner.py",
        "--experiment-id",
        EXPERIMENT_ID,
        "--mode",
        "execute",
    ],
    cwd=REPO,
    env=env,
)

print("\n" + "=" * 100)
print("SMS-01 CANDIDATE 3 EXECUTION RETURNED NORMALLY")
print("=" * 100)

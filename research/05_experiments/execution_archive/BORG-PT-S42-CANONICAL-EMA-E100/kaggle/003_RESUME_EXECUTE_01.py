# ======================================================================
# SMS-01 GOVERNED RESUME EXECUTE
# CANDIDATE 4 — Canonical EMA
#
# Uses exact frozen resume runner and scientific configuration.
# ======================================================================

from pathlib import Path
import subprocess
import sys

REPO = Path("/kaggle/working/ResEMA_SMS01_RESUME_C4")
EXPERIMENT_ID = "BORG-PT-S42-CANONICAL-EMA-E100"
DESTINATION_RUN = (
    Path("/kaggle/working/ResEMA_single_module_runs")
    / EXPERIMENT_ID
)

print("=" * 100)
print("CANDIDATE 4 — Canonical EMA — GOVERNED RESUME EXECUTION")
print("=" * 100)

if not REPO.is_dir():
    raise RuntimeError(
        f"Expected governed resume checkout missing: {REPO}"
    )

if DESTINATION_RUN.exists():
    raise RuntimeError(
        "Resume destination already exists. "
        "Do not delete or overwrite it automatically."
    )

branch_name = subprocess.check_output(
    ["git", "branch", "--show-current"],
    cwd=str(REPO),
    text=True,
).strip()

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=str(REPO),
    text=True,
).strip()

dirty = subprocess.check_output(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    cwd=str(REPO),
    text=True,
).strip()

print("BRANCH =", branch_name)
print("HEAD   =", head)
print("DIRTY  =", bool(dirty))

if branch_name != "research/single-module-screen-01":
    raise RuntimeError("Branch mismatch.")

if head != "9d65b7adae3d488f1cb70856476fef0488e9749a":
    raise RuntimeError("Execution commit mismatch.")

if dirty:
    print(dirty)
    raise RuntimeError("Repository is not clean.")

cmd = [
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

print("\n>>>", " ".join(cmd), flush=True)

subprocess.run(
    cmd,
    cwd=str(REPO),
    check=True,
)

print("\n" + "=" * 100)
print("CANDIDATE 4 — Canonical EMA GOVERNED RESUME PROCESS RETURNED NORMALLY")
print("=" * 100)

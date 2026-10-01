# SMS-01 Resume Preflight Invocation Failure — 2026-10-01

## Classification

Non-scientific wrapper/invocation failure.

## Affected experiments

- BORG-PT-S42-SCCONV-EARLY-E100
- BORG-PT-S42-SCCONV-4STAGE-E100
- BORG-PT-S42-CANONICAL-EMA-E100

## Observed failure

Direct invocation:

```
python -u research/runtime/single_module_resume_runner.py ...
```

failed before governed resume analysis with:

```
ModuleNotFoundError: No module named 'research'
```

## Root cause

At frozen execution commit
`9d65b7adae3d488f1cb70856476fef0488e9749a`,
`research/runtime/single_module_runner.py` bootstraps the repository root
into `sys.path` before importing the project-local `research` namespace,
while `research/runtime/single_module_resume_runner.py` does not.

When the resume runner is executed by file path, Python places
`research/runtime` rather than the repository root at the front of the
module search path.

## Corrective action

Do not modify the frozen scientific source during SMS-01.

Invoke the same exact frozen module from the repository root with:

```
python -u -m research.runtime.single_module_resume_runner ...
```

This changes package resolution only. Scientific source, configuration,
checkpoint, data binding, authorization, and test firewall remain unchanged.

## Scientific impact

None. The failure occurred before resume analysis/materialization/training.
Saved-run binding and Git provenance gates had already passed.

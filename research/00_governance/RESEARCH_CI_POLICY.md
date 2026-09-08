# Research CI Policy

## Why two workflows

### `research-integrity.yml`
Lightweight and always used for research-documentation/code-integrity changes.
It does not install PyTorch.

Checks:
- repository descends from frozen base commit;
- origin repository identity;
- clean Git state in CI;
- required governance/provenance/history files;
- experiment/result/artifact registry integrity;
- valid SHA256 strings;
- no checkpoint binaries inside `research/`.

### `research-model-smoke.yml`
Runs only when model-source/model-YAML files or its smoke test change.

Checks:
- baseline YOLO model builds;
- historical SC-e50 YAML builds;
- historical DySample YAML builds;
- baseline CPU forward pass.

It installs the CPU training package and is intentionally not run on documentation-only commits.

## Important inherited-workflow note

The upstream `CI` workflow in this fork is configured for:
- pushes to `main`;
- all pull requests.

Therefore pushing `research-v7-publication` itself should not trigger the full upstream CI, but opening a pull request can.

Until inherited workflow triggers are separately audited/filtered, use:
- pushed feature branches;
- local `git diff` review;
- the research-specific push workflows;
- local merge into `research-v7-publication`.

Do not open a PR solely for provenance if it would trigger the full inherited Ultralytics matrix.

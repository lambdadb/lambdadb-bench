# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Test and lint commands are in README.md "Development". Only `ruff check` is clean; many files are not `ruff format`-clean, so do not reformat unrelated files.
- Unit tests need no cloud access. Cloud integration tests are opt-in (`LAMBDADB_BENCH_RUN_INTEGRATION=1`, `scripts/run-integration-tests.sh`).
- `DevelopServer` in `tests/test_lambdadb_adapter.py` drives the real `lambdadb` SDK with lambdadb develop response shapes (201 create, `numDocs`, `took`, signed bulk-upload headers). Update it when the server or SDK contract changes.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.

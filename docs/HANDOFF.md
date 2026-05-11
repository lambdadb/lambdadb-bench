# LambdaDB Bench Handoff

Last updated: 2026-05-11

This repository is ready to continue from Phase 2-4 in a new Codex session.

## Current State

Remote:

- `https://github.com/lambdadb/lambdadb-bench`

Current branch:

- `main`

Latest committed work:

```text
f0d36ee Add exact ground truth generation
8979cab Add normalized dataset record outputs
f011802 Add dataset prepare skeleton
6501a9a Add dry-run adapter planning
7886ca4 Add config and manifest core
6155d6b Add Python project skeleton
f2dcb0a Refine benchmark consistency and setup design
23908ba Add initial benchmark design
```

Current working tree was clean before this handoff document was added.

## Completed Work

### Phase 1

Phase 1 is complete.

- Python package skeleton with `uv`, `hatchling`, and CLI entrypoint `ldbbench`.
- `config validate` command.
- scenario and target YAML loading.
- environment variable interpolation.
- redacted target config output.
- run manifest initialization.
- adapter capability model.
- dry-run adapters for LambdaDB, Qdrant, and Pinecone.
- dry-run run planning with `supported`, `partial`, and `unsupported` states.
- `strong` consistency unsupported targets are reported as `N/A`.
- `recreate` requires `--allow-destructive`.

### Phase 2-1

Dataset prepare skeleton is complete.

- `ldbbench dataset prepare`
- Hugging Face `datasets` dependency.
- safe `--dry-run` path.
- `--limit` for smoke tests.
- dataset cache manifest.

### Phase 2-2

Normalized dataset record format is complete.

Prepared dataset artifacts:

- `raw_records.jsonl`: provider source rows.
- `queries.jsonl`: held-out query rows.
- `records.jsonl`: records intended for database loading.
- `dataset_manifest.json`: source, split, row counts, artifact paths, and checksums.

Current normalized record shape:

```json
{
  "id": "string",
  "vector": [0.1, 0.2],
  "metadata": {}
}
```

The current split strategy reads `query_count + rows` source rows:

- first `query_count` rows -> `queries.jsonl`
- next `rows` rows -> `records.jsonl`

The Cohere Wikipedia scenario now maps:

- `id_field: _id`
- `vector_field: emb`
- `text_field: text`
- `metric: cosine`

### Phase 2-3

Exact ground truth generation is complete for small/local datasets.

- `ldbbench dataset ground-truth`
- `--top-k`
- `--metric cosine|dot`
- `--backend exact`
- `--limit-queries`
- `--dry-run`
- reads `records.jsonl`
- reads `queries.jsonl`
- writes `ground_truth.jsonl`
- writes `ground_truth_manifest.json`
- deterministic ranking by `score desc`, then `id asc`
- excludes same-id matches

This is brute-force exact search. It is good for fixtures and small smoke datasets. A scalable FAISS/cloud runner strategy is still needed before serious 1M/10M ground truth runs.

## Validation Commands

Use these from repo root:

```bash
uv sync --extra dev
uv run ruff check .
uv run python -m pytest
git diff --check
```

Current test count after Phase 2-3:

```text
31 passed
```

Useful smoke commands:

```bash
uv run ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --dry-run \
  --limit 100 \
  --query-count 10 \
  --out /tmp/ldbbench-dataset-dryrun
```

```bash
QDRANT_URL=https://example.qdrant.io \
  uv run ldbbench run --dry-run \
    --scenario scenarios/cohere-wikipedia-1m.yaml \
    --target configs/qdrant-cloud.example.yaml \
    --out /tmp/ldbbench-dry-run
```

## Next Work

Continue with Phase 2-4.

### Phase 2-4: LambdaDB Adapter

Goal:

- Add real LambdaDB adapter using the official Python SDK.
- Keep tests deterministic with fakes/mocks.
- Do not require a real LambdaDB endpoint for unit tests.
- Add gated integration tests only if credentials are explicitly present.

Recommended subtasks:

1. Add LambdaDB SDK dependency.
2. Extend adapter protocol beyond `check` to include:
   - `prepare`
   - `upsert_batch`
   - `query`
   - optionally `fetch`
3. Implement `src/ldbbench/adapters/lambdadb.py`.
4. Register real LambdaDB adapter while preserving dry-run behavior.
5. Add config fields needed by LambdaDB:
   - endpoint/base URL
   - project name or project host
   - API key env var
   - collection name
   - vector field name/index config field
6. Add unit tests with fake LambdaDB SDK objects.
7. Add optional integration test gate, for example:
   - `LAMBDADB_BENCH_RUN_INTEGRATION=1`
   - `LAMBDADB_API_KEY`
   - `LAMBDADB_ENDPOINT` or `LAMBDADB_BASE_URL`
   - `LAMBDADB_PROJECT_NAME`
8. Update README with LambdaDB adapter setup.

Important consistency mapping:

- benchmark `query.consistency: eventual` -> LambdaDB `consistent_read=False`
- benchmark `query.consistency: strong` -> LambdaDB `consistent_read=True`

Strong consistency should remain a LambdaDB-supported capability. Qdrant and Pinecone should keep strong consistency as `N/A`.

## LambdaDB SDK Notes

Package check:

- PyPI package: `lambdadb`
- Observed version from local install: `0.7.5`

Local introspection command used:

```bash
uv pip install lambdadb
uv run python - <<'PY'
import inspect
from lambdadb import LambdaDB
client = LambdaDB(base_url='https://api.lambdadb.ai', project_name='demo', project_api_key='key')
print(inspect.signature(LambdaDB))
print([a for a in dir(client.collections) if not a.startswith('_')])
print([a for a in dir(client.collections.docs) if not a.startswith('_')])
print(inspect.signature(client.collections.create))
print(inspect.signature(client.collections.get))
print(inspect.signature(client.collections.query))
print(inspect.signature(client.collections.docs.upsert))
print(inspect.signature(client.collections.docs.fetch))
PY
```

Observed constructor shape:

```text
LambdaDB(
  project_api_key=None,
  base_url=None,
  project_name=None,
  project_host=None,
  server_idx=None,
  server_url=None,
  ...
)
```

`server_url` emitted a deprecation warning during earlier introspection. Prefer `base_url` + `project_name`, or `project_host` if that better matches current product usage.

Observed collection methods:

- `client.collections.create(...)`
- `client.collections.get(...)`
- `client.collections.delete(...)`
- `client.collections.query(...)`

Observed docs methods:

- `client.collections.docs.upsert(...)`
- `client.collections.docs.fetch(...)`
- `client.collections.docs.delete(...)`
- `client.collections.docs.bulk_upsert(...)`
- `client.collections.docs.get_bulk_upsert(...)`

Relevant observed signatures:

```text
collections.create(
  *,
  collection_name: str,
  index_configs = None,
  partition_config = None,
  ...
)
```

```text
collections.get(
  *,
  collection_name: str,
  ...
)
```

```text
collections.query(
  *,
  collection_name: str,
  query: dict,
  size: int | None = None,
  consistent_read: bool | None = False,
  include_vectors: bool | None = False,
  ...
)
```

```text
collections.docs.upsert(
  *,
  collection_name: str,
  docs: list[dict],
  ...
)
```

```text
collections.docs.fetch(
  *,
  collection_name: str,
  ids: list[str],
  consistent_read: bool | None = False,
  include_vectors: bool | None = False,
  ...
)
```

Open detail to verify before final adapter implementation:

- exact LambdaDB query DSL for vector search using normalized `vector` field.
- exact `index_configs` shape for creating a vector collection/index.
- whether benchmark normalized field should be named `vector`, `emb`, or configurable per target.
- whether `collection` in target YAML should be renamed to `collection_name` for clarity.
- whether `base_url` + `project_name` is enough for all LambdaDB environments.

## Phase 2 Remaining After LambdaDB Adapter

### Phase 2-5: Qdrant Adapter

- Add `qdrant-client`.
- Use gRPC by default.
- Implement create/existing target handling.
- Implement upsert.
- Implement query.
- Keep `strong` consistency result as `N/A`.
- Add unit tests with fake client.
- Add gated integration tests for Qdrant Cloud.

### Phase 2-6: 1M Dry-to-Real Runner

- Add real load stage.
- Add real query stage.
- Write `ingest_events.jsonl`.
- Write `query_events.jsonl`.
- Write `summary.json`.
- Compute latency percentiles and QPS.
- Compute recall using `ground_truth.jsonl`.
- Keep 1M real run opt-in because it can incur cost/time.

## Later Work

- Pinecone Serverless adapter.
- FAISS-backed or cloud-runner ground truth for 1M/10M.
- 10M scenario.
- filtered search scenario.
- search-under-ingest scenario.
- idle-to-burst serverless scenario.
- report generator.

## Notes For Next Session

- Start by reading this file and `docs/DESIGN.md`.
- The next implementation should not require real LambdaDB credentials for normal tests.
- Prefer fakes/mocks for SDK unit tests.
- Add real integration tests behind explicit env gates only.
- Keep public claims careful: the repo is a reproducible harness, not a leaderboard.

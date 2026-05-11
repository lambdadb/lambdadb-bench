# LambdaDB Bench Handoff

Last updated: 2026-05-12

This repository is ready to continue from Phase 2-6 stage 2 in a new Codex session.

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

Current working tree contains the Phase 2-6 stage 1 runner changes.

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

### Phase 2-4

LambdaDB adapter surface is complete.

- Added `lambdadb>=0.7.5`.
- Extended the adapter protocol beyond `check` to include:
  - `prepare`
  - `upsert_batch`
  - `query`
  - `fetch`
- Implemented `src/ldbbench/adapters/lambdadb.py`.
- Registered the real LambdaDB adapter while preserving dry-run planning behavior for `ldbbench run --dry-run`.
- Added LambdaDB target config fields:
  - `endpoint`
  - `project_name`
  - `api_key_env`
  - `collection_name` with legacy `collection` compatibility
  - `vector_field`
  - `index_configs`
- Added fake-SDK unit tests for check, prepare, upsert, query, fetch, and missing credential behavior.
- Added gated LambdaDB integration coverage behind `LAMBDADB_BENCH_RUN_INTEGRATION=1`.
- Updated `README.md` and `configs/lambdadb.example.yaml`.

Implementation choices:

- benchmark `query.consistency: eventual` maps to LambdaDB `consistent_read=False`.
- benchmark `query.consistency: strong` maps to LambdaDB `consistent_read=True`.
- LambdaDB query uses `{"knn": {"field": <vector_field>, "queryVector": [...], "k": top_k}}`.
- LambdaDB create mode uses explicit `target.index_configs` when provided; otherwise it builds a minimal unmanaged vector field from dataset dimensions and metric.
- `endpoint` is passed to the LambdaDB SDK as `base_url`.
- `collection_name` is preferred in new YAML, but existing `collection` is still accepted.

### Phase 2-5

Qdrant adapter surface is complete.

- Added `qdrant-client>=1.15.0`; local lock resolved `qdrant-client==1.18.0`.
- Implemented `src/ldbbench/adapters/qdrant.py`.
- Registered the real Qdrant adapter while preserving dry-run planning behavior for `ldbbench run --dry-run`.
- Qdrant client construction uses `QdrantClient(url=..., api_key=..., prefer_grpc=True)` by default.
- Added Qdrant target config support:
  - `endpoint`
  - `api_key_env`
  - `collection_name` with legacy `collection` compatibility
  - optional `vector_field` for named vectors
  - optional `prefer_grpc`, defaulting to `true`
- Implemented `existing`, `create`, and `recreate` preparation modes.
- Implemented `upsert_batch`, `query`, and `fetch`.
- Added fake-client unit tests for check, prepare, upsert, query, fetch, gRPC default, missing credential behavior, and strong-consistency rejection.
- Added gated Qdrant integration coverage behind `QDRANT_BENCH_RUN_INTEGRATION=1`.
- Updated `README.md` and `configs/qdrant-cloud.example.yaml`.

Implementation choices:

- Qdrant remains eventual-only in the portable benchmark consistency model.
- benchmark `query.consistency: strong` should continue to plan as `N/A` for Qdrant.
- unnamed Qdrant vectors are the default; setting `target.vector_field` switches create/upsert/query to named-vector mode.
- Qdrant create mode maps benchmark metric `cosine|dot|dot_product|euclidean` to Qdrant distances `Cosine|Dot|Euclid`.

### Phase 2-6 Stage 1

Sequential dry-to-real runner surface is complete without requiring real endpoints
for normal tests.

- Added `src/ldbbench/runner/execute.py`.
- `ldbbench run` now supports non-dry-run execution when `--dataset-dir` is provided.
- Added run limits:
  - `--max-records`
  - `--max-queries`
  - `--allow-large-run`
- Real runs over 1M scenario rows require `--allow-large-run` unless `--max-records` keeps the run below the threshold.
- Added optional `--ground-truth`; by default the runner uses `<dataset-dir>/ground_truth.jsonl` when present.
- Implemented sequential `prepare -> load -> query` execution using the common adapter interface.
- Wrote runner artifacts:
  - `ingest_events.jsonl`
  - `query_events.jsonl`
  - `summary.json`
- Summary includes load/query counts, duration, QPS, latency percentiles, and mean recall when ground truth is available.
- Added fake-adapter unit/smoke tests for real runner behavior, output files, recall, limits, and large-run opt-in.

Remaining Phase 2-6 work:

- Add true concurrent query stages based on `scenario.query.stages`.
- Add duration-based query loops rather than one pass over `queries.jsonl`.
- Add richer error event handling and partial-run summaries.
- Run real LambdaDB/Qdrant endpoint smoke tests once credentials are available.

## Validation Commands

Use these from repo root:

```bash
uv sync --extra dev
uv run ruff check .
uv run python -m pytest
git diff --check
```

Current test count after Phase 2-6 stage 1:

```text
59 passed, 2 skipped
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
QDRANT_ENDPOINT=https://example.qdrant.io \
  uv run ldbbench run --dry-run \
    --scenario scenarios/cohere-wikipedia-1m.yaml \
    --target configs/qdrant-cloud.example.yaml \
    --out /tmp/ldbbench-dry-run
```

## Next Work

Continue with Phase 2-6 stage 2.

## Qdrant SDK Notes

Package check:

- PyPI package: `qdrant-client`
- Observed version from local install: `1.18.0`

Local introspection command used:

```bash
uv run python - <<'PY'
import inspect
from qdrant_client import QdrantClient, models
print(inspect.signature(QdrantClient))
print(inspect.signature(QdrantClient.collection_exists))
print(inspect.signature(QdrantClient.get_collection))
print(inspect.signature(QdrantClient.create_collection))
print(inspect.signature(QdrantClient.recreate_collection))
print(inspect.signature(QdrantClient.upsert))
print(inspect.signature(QdrantClient.query_points))
print(inspect.signature(QdrantClient.retrieve))
print(list(models.Distance))
PY
```

Relevant observed calls:

```text
QdrantClient(url=..., api_key=..., prefer_grpc=True)
client.collection_exists(collection_name=...)
client.get_collection(collection_name=...)
client.create_collection(collection_name=..., vectors_config=...)
client.recreate_collection(collection_name=..., vectors_config=...)
client.upsert(collection_name=..., points=..., wait=True)
client.query_points(collection_name=..., query=..., using=..., limit=...)
client.retrieve(collection_name=..., ids=...)
```

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

`server_url` and `project_host` are legacy SDK parameters. The adapter config uses `endpoint` as SDK `base_url` plus `project_name`.

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

## Phase 2 Remaining

### Phase 2-6 Stage 2: Concurrent/Duration Runner

- Execute `scenario.query.stages` with configured concurrency and duration.
- Keep sequential one-pass mode for smoke tests.
- Preserve `ingest_events.jsonl`, `query_events.jsonl`, and `summary.json`.
- Add robust error events and non-zero error-rate summaries.
- Validate against real LambdaDB and Qdrant endpoints when credentials are available.

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
- The next implementation should not require real database credentials for normal tests.
- Prefer fakes/mocks for SDK unit tests.
- Add real integration tests behind explicit env gates only.
- Keep public claims careful: the repo is a reproducible harness, not a leaderboard.

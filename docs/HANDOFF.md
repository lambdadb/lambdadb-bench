# LambdaDB Bench Handoff

Last updated: 2026-05-13

This repository includes Phase 2-6 runner follow-up work and the Phase 3
Pinecone Serverless adapter. Real LambdaDB/Qdrant Phase 2 endpoint validation
has been completed by the user.

## Current State

Remote:

- `https://github.com/lambdadb/lambdadb-bench`

Current branch:

- `main`

Recent committed milestones:

```text
aff49a9 Add concurrent duration query runner
9e019a6 Polish smoke benchmark workflow
24660f9 Add sequential benchmark runner
2adb193 Add Qdrant adapter
06e5ec9 Add LambdaDB adapter
95ac266 Add handoff notes for benchmark phases
f0d36ee Add exact ground truth generation
8979cab Add normalized dataset record outputs
f011802 Add dataset prepare skeleton
```

Current code includes Phase 2-6 stage 2 plus follow-up hardening:

- LambdaDB recreate waits for asynchronous collection deletion, and
  create/existing prepare waits for collection `ACTIVE` status before load or
  query starts.
- Load failures write partial `summary.json` and skip query execution cleanly.
- Load batches can be capped by approximate payload size.
- Load can run with concurrent upsert workers via `load.concurrency`.
- Load can split that total concurrency across multiple worker processes with
  `load.processes` to use more CPU cores while preserving checkpoint/event
  semantics.
- Load writes `load_checkpoint.json` and can resume interrupted large ingests
  with `--resume-load` from the highest contiguous successful batch watermark.
- The runner can wait for loaded records to become query-visible before query
  stages start.
- `--load-only` and `--query-only` support separate load/query validation.
- FAISS-backed ground truth is available through `--backend faiss` with the
  optional `groundtruth` extra.
- Long-running CLI commands emit `progress:` logs for dataset prepare,
  ground-truth generation, load, visibility wait, and query stages.
- Pinecone Serverless is registered as a real adapter using the official
  `pinecone>=9.0.0` SDK.
- `.env.example` and `scripts/run-integration-tests.sh` provide an explicit
  local path for gated LambdaDB/Qdrant/Pinecone integration tests without
  auto-loading secrets during normal pytest runs.
- Dataset prepare computes artifact SHA256 digests while writing JSONL files,
  so full 1M prepares no longer pause silently after `wrote artifacts` to
  re-read large files for checksums.
- Qdrant and Pinecone adapters cache SDK clients/index handles per target
  settings instead of recreating them for every batch/query.
- LambdaDB caches SDK clients per target settings and worker thread to avoid
  sharing the underlying HTTP transport at high load/query concurrency.
- Load summaries now separate runner-side batching time from adapter upsert
  attempt time with `records_read`, `batching_duration_seconds`,
  `batching_records_per_second`, and `upsert_attempt_duration_seconds`.
- Load record reading uses `orjson` and reuses prepared JSONL line byte sizes for
  `load.max_batch_bytes`, avoiding per-record float normalization/norm
  calculation and per-record `json.dumps` during batching.
- Dataset prepare now writes compact `records.msgpack` / `queries.msgpack`
  float32 caches, and `ldbbench dataset optimize --dataset-dir ...` can add
  those caches to older prepared datasets without re-downloading source data.
  Runs automatically prefer the msgpack cache when present while preserving the
  original JSONL byte estimate for `load.max_batch_bytes`.
- Staged query runs now decouple request workers from JSONL event writes with a
  writer thread and batched flushes, avoiding a per-query write lock/flush
  ceiling at high configured concurrency. The Qdrant query adapter also reuses
  list vectors directly instead of copying every query vector.
- Staged query runs can split each stage's total concurrency across multiple
  worker processes with `query.processes`.
- `ldbbench report` combines one or more result directories into a Markdown
  report plus sibling load/query-stage CSV files for spreadsheet-friendly
  comparison.

Any remaining local files should be benchmark artifacts or local configs ignored
by `.gitignore`.

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

Ground truth generation is complete for small/local datasets and has a FAISS
backend for larger datasets.

- `ldbbench dataset ground-truth`
- `--top-k`
- `--metric cosine|dot`
- `--backend exact`
- `--backend faiss`
- `--batch-size` for FAISS query search batches
- `--limit-queries`
- `--dry-run`
- reads `records.jsonl`
- reads `queries.jsonl`
- writes `ground_truth.jsonl`
- writes `ground_truth_manifest.json`
- exact backend ranking is deterministic by `score desc`, then `id asc`
- excludes same-id matches

The `exact` backend is brute-force Python search. It is good for fixtures and
small smoke datasets. The `faiss` backend uses optional `faiss-cpu` dependencies
from `uv sync --extra groundtruth`, builds an in-memory `IndexFlatIP`, normalizes
vectors for cosine search, and records index/batch/normalization settings in
`ground_truth_manifest.json`.

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
- LambdaDB recreate mode waits for asynchronous collection deletion to become
  visible before creating the replacement collection. Configure with
  `delete_wait_timeout_seconds` and `delete_wait_poll_seconds` in the target if
  needed.
- LambdaDB create/existing prepare modes wait for collection status `ACTIVE`
  before returning. Configure with `create_wait_timeout_seconds` and
  `create_wait_poll_seconds` in the target if needed.

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

Phase 2-6 implementation is complete. Real LambdaDB/Qdrant endpoint validation
has been completed by the user. There are no remaining Phase 2 implementation
items tracked in this handoff.

## Validation Commands

Use these from repo root:

```bash
uv sync --extra dev
uv run ruff check .
uv run python -m pytest
git diff --check
```

Current test count after LambdaDB ACTIVE wait work:

```text
102 passed, 3 skipped
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

Run gated Pinecone endpoint validation when credentials are available, then move
to another Later Work item unless priorities change.

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

## Pinecone SDK Notes

Package check:

- PyPI package: `pinecone`
- Observed version from local lock: `9.0.0`

Local introspection confirmed:

```text
Pinecone(api_key=..., timeout=..., connection_pool_maxsize=...)
Pinecone.create_index(name=..., dimension=..., metric=..., spec=..., timeout=...)
Pinecone.delete_index(name=..., timeout=...)
Pinecone.has_index(name=...)
Pinecone.describe_index(name=...)
Pinecone.Index(name=..., host=..., pool_threads=..., connection_pool_maxsize=...)
Index.upsert(vectors=..., namespace=...)
Index.query(vector=..., top_k=..., namespace=..., include_values=..., include_metadata=..., filter=...)
Index.fetch(ids=..., namespace=...)
ServerlessSpec(cloud=..., region=..., read_capacity=..., schema=...)
```

## Phase 3 Status

Pinecone Serverless adapter surface is complete for normal tests without
requiring real endpoints.

- Added `pinecone>=9.0.0`; local lock resolved `pinecone==9.0.0`.
- Implemented `src/ldbbench/adapters/pinecone.py`.
- Registered the real Pinecone adapter while preserving dry-run planning
  behavior for `ldbbench run --dry-run`.
- Added Pinecone target config support:
  - `endpoint` as optional index host for data-plane calls.
  - `api_key_env`
  - `collection_name` with legacy `collection` compatibility as index name.
  - `region`
  - optional `cloud`, defaulting to `aws`.
  - optional `namespace`, defaulting to the default namespace.
  - optional `pool_threads` and `connection_pool_maxsize`.
  - optional `spec`, `tags`, `create_timeout_seconds`, and
    `delete_timeout_seconds`.
- Implemented `existing`, `create`, and `recreate` preparation modes for
  serverless dense-vector indexes.
- Implemented `upsert_batch`, `query`, and `fetch`.
- Added fake-client unit tests for check, prepare, upsert, query, fetch,
  missing credential behavior, and strong-consistency rejection.
- Added gated Pinecone integration coverage behind
  `PINECONE_BENCH_RUN_INTEGRATION=1`.
- Added `configs/pinecone-serverless.example.yaml`.
- Added `.env.example` and `scripts/run-integration-tests.sh` for explicit
  local integration-test env loading.
- Updated `README.md` and `docs/DESIGN.md`.

Implementation choices:

- Pinecone remains eventual-only in the portable benchmark consistency model.
- benchmark `query.consistency: strong` should continue to plan as `N/A` for
  Pinecone.
- `target.endpoint` maps to the Pinecone index host. If omitted, the adapter
  targets the index by `collection_name`, which causes the SDK to resolve the
  host through the control plane.
- Pinecone create mode maps benchmark metric `cosine|dot|dot_product|euclidean`
  to Pinecone metrics `cosine|dotproduct|euclidean`.

## Phase 2 Status

### Phase 2-6 Stage 2: Concurrent/Duration Runner

Concurrent/duration query execution is implemented for normal tests without
requiring real endpoints.

- `ldbbench run` keeps bounded one-pass query mode when `--max-queries` is set.
- Without `--max-queries`, the runner executes `scenario.query.stages` by
  cycling through prepared query rows for each configured concurrency/duration
  stage.
- Added threaded query execution with per-event `query_stage_index` and
  `worker_index`.
- `query_events.jsonl` now records query errors as structured events instead of
  aborting the whole run.
- `summary.json` now includes query mode, attempts, errors, error rate, and
  per-stage summaries.
- CLI output reports `completed_with_errors`, error count, and error rate when
  query attempts fail.
- Added fake-adapter tests for staged duration mode and query error summaries.
- Load failures now write partial `ingest_events.jsonl` and `summary.json`,
  skip query execution, and report run status `failed`.
- `load.max_batch_bytes` is now enforced by the runner. This was added after a
  real LambdaDB 1M run failed on the first 500-record batch with HTTP 413
  `Request Too Long`.
- `load.concurrency` now controls concurrent upsert workers so load throughput
  can be measured under parallel writes instead of single-threaded batch
  submission.
- `load.wait_until_query_visible: true` now waits for a loaded-record sample to
  be visible via vector query before query execution starts.
- `ldbbench run --load-only` loads records and writes a skipped query summary
  without executing query attempts. The scenario still needs a `query` section
  because query rows and compatibility validation are scenario-level concerns.
- `ldbbench run --query-only` skips loading and runs query attempts against an
  existing target. It requires `prepare.mode: existing` to avoid accidental
  collection creation or recreation before querying.
- `ldbbench run --resume-load` resumes from the existing `load_checkpoint.json`
  in the result directory. It requires `prepare.mode: existing`, validates that
  dataset/target/load settings match the checkpoint context, appends new ingest
  events, and skips only the highest contiguous successful batch watermark so
  out-of-order concurrent successes cannot hide a failed earlier batch.

Completed validation:

- Real LambdaDB and Qdrant endpoint validation was completed directly by the
  user.
- Pinecone adapter unit coverage was added. Real Pinecone endpoint validation is
  gated behind `PINECONE_BENCH_RUN_INTEGRATION=1`.

## Optional Scale Validation Checklist

The user has already completed real LambdaDB/Qdrant endpoint validation. Keep
these commands only as a reference if scale validation needs to be rerun or
extended later.

Set credentials:

```bash
export QDRANT_ENDPOINT=...
export QDRANT_API_KEY=...
export LAMBDADB_ENDPOINT=https://api.lambdadb.ai
export LAMBDADB_PROJECT_NAME=...
export LAMBDADB_API_KEY=...
```

Run target checks:

```bash
uv run ldbbench target check --target configs/qdrant-cloud.local.yaml
uv run ldbbench target check --target configs/lambdadb.local.yaml
```

Run one-pass smoke tests:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.local.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-records 100 \
  --max-queries 10 \
  --out results/qdrant-smoke
```

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/lambdadb.local.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-records 100 \
  --max-queries 10 \
  --out results/lambdadb-smoke
```

For optional future scale checks, remove `--max-queries` after a smoke run is
stable. Prepare 1k/10k datasets before any full 1M workload, and use
`--allow-large-run` only after cost/resource approval.

## Later Work

- Cloud-runner or persisted-index ground truth for 10M+ workloads.
- 10M scenario.
- filtered search scenario.
- search-under-ingest scenario.
- idle-to-burst serverless scenario.

## Notes For Next Session

- Start by reading this file and `docs/DESIGN.md`.
- The next implementation should not require real database credentials for normal tests.
- Prefer fakes/mocks for SDK unit tests.
- Add real integration tests behind explicit env gates only.
- Keep public claims careful: the repo is a reproducible harness, not a leaderboard.

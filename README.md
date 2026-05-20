# lambdadb-bench

Reproducible benchmark harness for LambdaDB and comparable managed vector
databases.

The initial benchmark design focuses on the Cohere Wikipedia embedding workload,
with LambdaDB, Qdrant Cloud, and Pinecone Serverless as the first target
adapters.

See [docs/DESIGN.md](docs/DESIGN.md) for the current design decisions, workload
model, adapter contract, result format, and implementation phases.

## Quickstart

Install the package in editable mode with development dependencies:

```bash
uv sync --extra dev
```

For FAISS-backed ground truth generation on larger datasets, install the
optional ground truth dependencies too:

```bash
uv sync --extra dev --extra groundtruth
```

Check the CLI:

```bash
uv run ldbbench doctor
```

Long-running commands print `progress:` lines for major phases and periodic
counts. This includes dataset preparation, ground-truth generation, load,
visibility wait, and query stages.
The load path uses `orjson` and prepared-record byte estimates so large JSONL
loads do not reserialize every record just to form size-capped batches.

### 1. Prepare a smoke dataset

Start with a tiny row-limited dataset. This avoids a costly 1M-row run while
verifying the end-to-end flow.

This step does not use a LambdaDB, Qdrant, or Pinecone target config. It reads
the scenario dataset source and writes local files under `--out`. Set `HF_TOKEN`
in your environment if you want authenticated Hugging Face downloads with higher
rate limits.

```bash
uv run ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --limit 100 \
  --query-count 10 \
  --out data/datasets/cohere-wikipedia-1m-smoke
```

Dataset preparation writes:

- `raw_records.jsonl`: source rows as received from the dataset provider.
- `queries.jsonl`: held-out query vectors.
- `records.jsonl`: records intended for database loading.
- `queries.msgpack` and `records.msgpack`: compact float32 caches used
  automatically by `ldbbench run` when present.
- `dataset_manifest.json`: dataset source, row counts, artifact paths, and
  checksums.

Compute exact ground truth for the smoke dataset:

```bash
uv run ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --top-k 10 \
  --backend exact
```

For larger datasets, use the FAISS backend. It builds an in-memory
`IndexFlatIP`; cosine ground truth normalizes vectors before indexing and
querying.

```bash
uv run --extra groundtruth ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --top-k 10 \
  --backend faiss \
  --batch-size 100
```

### 2. Configure a target

Use one target config per database. The checked-in files are examples:

- `configs/lambdadb.example.yaml`
- `configs/qdrant-cloud.example.yaml`
- `configs/pinecone-serverless.example.yaml`

For LambdaDB, set:

```bash
export LAMBDADB_ENDPOINT=https://api.lambdadb.ai
export LAMBDADB_PROJECT_NAME=your-project
export LAMBDADB_COLLECTION_NAME=your-collection
export LAMBDADB_API_KEY=...
```

For Qdrant Cloud, set:

```bash
export QDRANT_ENDPOINT=https://example.qdrant.io
export QDRANT_COLLECTION_NAME=your-collection
export QDRANT_API_KEY=...
```

For Pinecone Serverless, set:

```bash
export PINECONE_INDEX_NAME=your-index
export PINECONE_API_KEY=...
```

For local integration-test credentials, copy `.env.example` to `.env`, fill in
the target credentials, and set only the gates you want to run to `1`. `.env` is
ignored by git. The helper script loads `.env` explicitly:

```bash
cp .env.example .env
$EDITOR .env
scripts/run-integration-tests.sh
```

Before a real run, make sure the target config points at the collection you want
to use. For smoke testing, `prepare.mode: create` can create the collection from
the scenario dimensions. For existing collections, keep `prepare.mode: existing`.

For a first Qdrant smoke run, copy the example target and switch it to create a
fresh smoke collection:

```bash
cp configs/qdrant-cloud.example.yaml configs/qdrant-cloud.smoke.yaml
```

Then edit `configs/qdrant-cloud.smoke.yaml`:

```yaml
collection_name: cohere_wikipedia_1m_smoke

prepare:
  mode: create
```

### 3. Validate the target

LambdaDB:

```bash
uv run ldbbench target check --target configs/lambdadb.example.yaml
```

Qdrant:

```bash
uv run ldbbench target check --target configs/qdrant-cloud.smoke.yaml
```

Pinecone:

```bash
uv run ldbbench target check --target configs/pinecone-serverless.example.yaml
```

Validate the scenario plus target plan:

```bash
uv run ldbbench config validate \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml
```

### 4. Dry-run the benchmark plan

Dry-run writes run metadata without contacting the database:

```bash
uv run ldbbench run --dry-run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --out results/example-qdrant-dry-run
```

### 5. Run a real smoke benchmark

Use the smoke dataset first. This contacts the configured database.
Supplying `--max-queries` keeps the query step in bounded one-pass smoke mode.
Without `--max-queries`, `run` uses `scenario.query.stages` and repeats the
prepared query set for each configured concurrency/duration stage.
If `scenario.load.wait_until_query_visible` is true, the runner waits for a
small sample of loaded records to be returned by vector query before starting
the query stage.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-records 100 \
  --max-queries 10 \
  --out results/example-qdrant-smoke
```

For LambdaDB, use the LambdaDB target instead:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-records 100 \
  --max-queries 10 \
  --out results/example-lambdadb-smoke
```

To load records without running queries, add `--load-only`:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-records 100 \
  --load-only \
  --out results/example-qdrant-load-only
```

Load runs write `load_checkpoint.json` in the result directory. If a large load
is interrupted or fails after some batches succeed, rerun with the same
`--out`, same dataset/load settings, and `--resume-load` to skip the highest
contiguous successful batch watermark. The target config must use
`prepare.mode: existing` for the resume command so the already-loaded collection
is not recreated.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --load-only \
  --resume-load \
  --out results/example-qdrant-load-only
```

To query an already-loaded collection without loading records again, use
`--query-only`. The target must use `prepare.mode: existing` so the command does
not create or recreate collections before querying.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --max-queries 10 \
  --query-only \
  --out results/example-qdrant-query-only
```

Real runs write:

- `ingest_events.jsonl`: one event per upsert batch, including load errors.
- `load_checkpoint.json`: resumable load watermark and matching load context.
- `query_events.jsonl`: one event per query attempt, including query errors.
- `search_under_ingest_events.jsonl`: one event per upload-and-ask probe when
  `workload: search_under_ingest` is used.
- `summary.json`: load/query counts, latency percentiles, QPS, per-stage query
  summaries, load batching/upsert timing, error rates, recall when
  `ground_truth.jsonl` is present, and search-under-ingest metrics when
  applicable.

### Search-under-ingest read-after-write runs

Search-under-ingest workloads measure whether newly written document sets are
search-visible immediately after write acknowledgement. The included Cohere
Wikipedia scenario uses held-out `queries.msgpack` records as upload-and-ask
probes, groups chunks by `metadata.url`, upserts one URL group, then immediately
queries with one chunk vector from that group.

This workload reports read-after-write document visibility metrics separately
from normal FAISS recall:

- `read_after_write_exact_chunk_hit_rate_at_k`
- `read_after_write_same_document_hit_rate_at_k`
- `read_after_write_same_document_recall_at_k`
- `write_latency_ms`
- `immediate_query_latency_ms`
- `time_to_visible_ms`

The first implementation supports `probe_source: queries`,
`probe_concurrency: 1`, and `probe_queries_per_document: 1`. Use `upsert`, not
`bulk_upsert`, because this workload models interactive read-after-write
behavior.

Follow this sequence for a first run with a 100k preloaded background corpus.
The scenario still declares the full 1M Cohere Wikipedia source; `--limit
100000` creates a smaller local dataset for this first run.

1. Prepare local records and held-out probes.

```bash
uv run ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m-search-under-ingest.yaml \
  --limit 100000 \
  --query-count 1000 \
  --out data/datasets/cohere-wikipedia-search-under-ingest-100k
```

This writes 100k load records plus 1,000 held-out query/probe records. The
search-under-ingest stage uses the held-out `queries.msgpack` rows as new
document groups to upload and immediately query.

2. Configure the target collection.

Copy the target example for the database you want to test, then set a dedicated
collection name. For a fresh preload, the target must create or recreate the
collection:

```bash
cp configs/lambdadb.example.yaml configs/lambdadb-search-under-ingest.yaml
```

```yaml
collection_name: cohere_wikipedia_search_under_ingest_100k

prepare:
  mode: create
```

Use `mode: recreate` only when you intentionally want to delete and rebuild an
existing benchmark collection.

3. Preload the 100k background corpus.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-search-under-ingest.yaml \
  --target configs/lambdadb-search-under-ingest.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-search-under-ingest-100k \
  --max-records 100000 \
  --load-only \
  --out results/example-lambdadb-search-under-ingest-preload-100k
```

After this succeeds, change the same target config to `prepare.mode: existing`.
`--query-only` requires `existing` so the runner does not create, recreate, or
delete the collection before probing it.

4. Run the search-under-ingest probes.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-search-under-ingest.yaml \
  --target configs/lambdadb-search-under-ingest.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-search-under-ingest-100k \
  --query-only \
  --allow-large-run \
  --out results/example-lambdadb-search-under-ingest
```

For this workload, `--query-only` skips only the background load stage. Each
probe still upserts one held-out document group, immediately queries for it, and
writes `search_under_ingest_events.jsonl`. `--allow-large-run` is still needed
because the scenario declares a 1M-row dataset, even though the example preload
uses only 100k background records.

5. Check the result summary.

```bash
jq '.search_under_ingest' results/example-lambdadb-search-under-ingest/summary.json
```

The CLI prints the probe count and same-document hit rate. The full
`summary.json` also includes exact-chunk hit rate, same-document recall, write
latency, immediate query latency, and time-to-visible metrics.

For LambdaDB, `search_under_ingest.consistency: strong` maps to
`consistent_read=True`. Targets that do not declare a comparable portable
strong read-after-write query guarantee plan strong variants as `N/A`.

Combine one or more run directories into Markdown and CSV report artifacts:

```bash
uv run ldbbench report \
  results/example-qdrant-smoke results/example-lambdadb-smoke \
  --out reports/cohere-wikipedia-smoke.md
```

The report command writes the Markdown file plus sibling `*-load.csv` and
`*-query-stages.csv` files for spreadsheet-friendly comparisons. The Markdown
report also includes a Search-Under-Ingest Results section when runs contain
that workload.

Runs at 1M rows or larger require `--allow-large-run` unless `--max-records`
keeps the run below that threshold.

### 6. Scale validation

After both targets pass the 100-row smoke run, scale up gradually before the
full 1M run:

```bash
uv run ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --limit 1000 \
  --query-count 100 \
  --out data/datasets/cohere-wikipedia-1m-1k
```

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.smoke.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m-1k \
  --max-records 1000 \
  --max-queries 100 \
  --out results/example-qdrant-1k
```

For the full scenario, prepare the full dataset and opt into the large run:

```bash
uv run ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --out data/datasets/cohere-wikipedia-1m
```

If the dataset was prepared before binary caches were available, build them
without re-downloading Hugging Face data:

```bash
uv run ldbbench dataset optimize \
  --dataset-dir data/datasets/cohere-wikipedia-1m
```

To enable sharded load, split records into msgpack shards:

```bash
uv run ldbbench dataset optimize \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --shards 16
```

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --allow-large-run \
  --out results/example-qdrant-1m
```

## Target Config Reference

### LambdaDB

The LambdaDB adapter uses the official `lambdadb` Python SDK. Configure
LambdaDB targets with:

- `endpoint`: API base URL, for example `https://api.lambdadb.ai`.
- `project_name`: LambdaDB project name.
- `api_key_env`: environment variable name containing the project API key.
- `collection_name`: target collection.
- `vector_field`: field that stores normalized benchmark vectors. Defaults to
  `vector`.
- `index_configs`: LambdaDB collection index config used by create/recreate
  preparation modes.
- `partition_config`: optional LambdaDB hash partition config for create/recreate
  preparation modes. See `configs/lambdadb-partitioned.example.yaml`.
- `delete_wait_timeout_seconds`: recreate-mode deletion wait timeout. Defaults
  to `60`.
- `delete_wait_poll_seconds`: recreate-mode deletion polling interval. Defaults
  to `1`.
- `create_wait_timeout_seconds`: create/existing-mode ACTIVE status wait
  timeout before load or query starts. Defaults to `300`.
- `create_wait_poll_seconds`: create/existing-mode ACTIVE status polling
  interval. Defaults to `1`.

The LambdaDB adapter keeps SDK clients thread-local during load/query execution
so concurrent workers do not share the same underlying HTTP transport. High
concurrency can therefore open more sockets; raise the process file descriptor
limit on benchmark hosts when testing large `load.concurrency` values.

Useful load settings:

- `write_mode`: `upsert` for regular batched writes, or `bulk_upsert` for
  LambdaDB's S3-backed `bulk_upsert_docs()` import path.
- `batch_size`: maximum records per write batch. For LambdaDB `upsert` this is
  a direct docs upsert request; for LambdaDB `bulk_upsert` this is one bulk
  object upload and import trigger.
- `concurrency`: number of concurrent upsert workers. Defaults to `1`.
- `processes`: optional process count for CPU parallelism. Defaults to `1`.
  `concurrency` remains the total in-flight upsert worker count; when
  `processes > 1`, the runner splits that total across worker processes.
- `sharded_records`: when true, load workers read prepared record shards
  directly instead of receiving parsed batches from the parent process. Prepare
  shards first with `ldbbench dataset optimize --shards N`.
- `shard_count`: optional assertion for the expected number of record shards.
- `max_batch_bytes`: optional approximate request payload cap. The runner
  splits batches by both `batch_size` and this byte limit to avoid oversized
  requests. The first sharded load path does not support `max_batch_bytes`;
  remove this setting when `sharded_records: true`.
- `wait_until_query_visible`: when true, wait for a loaded-record sample to be
  visible through vector query before the query stage starts.
- `query_visibility_timeout`: optional duration string, defaults to `60s`.
- `query_visibility_poll_interval`: optional duration string, defaults to `1s`.

For staged queries, `query.processes` works the same way: each
`query.stages[].concurrency` value remains the total in-flight query count, and
the runner splits that total across worker processes when `query.processes > 1`.

Partition-pruned query workloads can set `query.partition_filter` with a target
field and query metadata source field. These runs intentionally skip global
recall reporting because the query searches a restricted partition subset. See
`docs/PARTITIONING_WORKLOAD.md` and
`scenarios/cohere-wikipedia-1m-partitioned.yaml`.

Optional integration coverage is gated behind:

```bash
LAMBDADB_BENCH_RUN_INTEGRATION=1
LAMBDADB_API_KEY=...
LAMBDADB_ENDPOINT=https://api.lambdadb.ai
LAMBDADB_PROJECT_NAME=...
LAMBDADB_COLLECTION_NAME=...
```

### Qdrant

The Qdrant adapter uses the official `qdrant-client` package with gRPC
preferred by default. Configure Qdrant targets with:

- `endpoint`: Qdrant Cloud or self-managed Qdrant URL.
- `api_key_env`: environment variable name containing the Qdrant API key.
- `collection_name`: target collection. Existing `collection` configs are still
  accepted.
- `vector_field`: optional named vector to use. Omit this for Qdrant's default
  unnamed vector.
- `prefer_grpc`: boolean, defaults to `true`.

Optional integration coverage is gated behind:

```bash
QDRANT_BENCH_RUN_INTEGRATION=1
QDRANT_ENDPOINT=https://example.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=...
```

## Development

Run tests and linting:

```bash
uv run python -m pytest
uv run ruff check .
```

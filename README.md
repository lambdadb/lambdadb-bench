# lambdadb-bench

Reproducible benchmark harness for LambdaDB and comparable managed vector
databases.

The initial benchmark design focuses on the Cohere Wikipedia embedding workload,
with LambdaDB, Qdrant Cloud, and Pinecone Serverless as the first target
adapters.

See [docs/DESIGN.md](docs/DESIGN.md) for the current design decisions, workload
model, adapter contract, result format, and implementation phases.

Future evaluation of Meta/Faiss DINO10B as a 1024-dimensional,
billion-scale byte-vector workload is recorded in
[docs/DINO10B_FUTURE_WORK.md](docs/DINO10B_FUTURE_WORK.md).

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
`IndexFlatIP` for cosine/dot metrics and `IndexFlatL2` for `euclidean`. Cosine
ground truth normalizes vectors before indexing and querying. For Euclidean
ground truth, pass `--metric euclidean`; its scores are squared L2 distances,
matching FAISS ranking semantics.

```bash
uv run --extra groundtruth ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --top-k 10 \
  --backend faiss \
  --batch-size 100
```

Prepared records include deterministic synthetic bucket metadata for approximate
50%, 10%, 5%, 1%, and 0.1% equality filters. Generate the 5% filtered ground
truth with:

```bash
uv run --extra groundtruth ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --top-k 10 \
  --backend faiss \
  --batch-size 100 \
  --filter-name synthetic_bucket_5pct \
  --filter-field filter_bucket_20 \
  --filter-value-source eligible-record-buckets \
  --filter-seed 20260511 \
  --filter-min-candidates 10
```

The ground-truth and load paths deterministically backfill `filter_bucket_20`
when reusing dataset artifacts prepared by an older version. A database
collection loaded before this field was added still needs the
`filter_bucket_20` metadata index and a full record upsert before running
`scenarios/cohere-wikipedia-1m-filtered-5pct.yaml`.

Ground-truth filenames include the metric so multiple variants can coexist,
for example `ground_truth.cosine.jsonl`, `ground_truth.euclidean.jsonl`, and
`ground_truth.euclidean.filtered.synthetic_bucket_1pct.jsonl`. Manifest files
use the same stem with `.manifest.json`. `ldbbench run` selects the unfiltered
artifact for the scenario metric by default; pass `--ground-truth` explicitly
for filtered or legacy artifacts.

When a ground-truth JSONL file is selected, its matching manifest is required.
Before querying, the runner verifies the scenario metric and `top_k`, the JSONL
filename and SHA-256, and any filtered name/field/operator configuration.

Ground-truth recall is tie-aware at the kth boundary. Candidates with a score
strictly better than the kth score must be returned, while any candidates whose
ground-truth score is exactly equal to the kth score may fill the remaining
slots. Ground-truth generation stores optional `recall_groups` only when a tie
crosses that boundary; artifacts without this metadata keep the legacy ID-set
recall behavior. Ties use exact equality from the selected ground-truth backend
(`float32` FAISS scores for `faiss`, Python float scores for `exact`) rather than
an added tolerance.

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
After a successful load, the runner waits until the collection is ready and
reports as many documents as were loaded before any query runs (LambdaDB:
ACTIVE and `numDocs` equal to the loaded count). If that wait times out, the
run is marked `failed` and queries are skipped with `doc_count_timeout`.
Adapters that cannot report a document count record the wait as skipped.
The wait only checks that the indexed count reaches the expected total, so it
does not confirm that a load which overwrites existing documents without
changing the count has been applied; comparing against a pre-load baseline is
follow-up work.
`ldbbench run` exits with status 1 whenever the run status is `failed`, for
example after load errors or a wait timeout, so scripts can stop before a
later `--query-only` run measures an unfinished index.
If `scenario.load.wait_until_query_visible` is true, the runner then waits for a
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

- `run_manifest.json`: scenario and target fingerprints plus tool provenance:
  `tool.git_commit`, `tool.git_dirty` (uncommitted tracked changes), and the
  vendor SDK in `tool.sdk_package` and `tool.sdk_version`. When available,
  `tool.sdk_direct_url` records the SDK's PEP 610 install source, including its
  VCS commit or editable checkout path. The tool commit comes from the checkout
  for editable installs and from the recorded commit for git installs; it is
  `null` when neither is available, and reports warn when it is `null`.
- `ingest_events.jsonl`: one event per upsert batch, including load errors.
- `load_checkpoint.json`: resumable load watermark and matching load context.
- `delete_events.jsonl`: one event per document-ID delete batch in
  `--delete-only` runs.
- `deletion_state.json`: written by `--delete-only` with the cumulative delete
  checkpoint, target identity fingerprint, dataset/deletion-plan checksums,
  and sampled delete-visibility result. By default, only a state whose sampled
  IDs are no longer fetch-visible is marked `completed`; when
  `delete.wait_until_deletion_visible` is false, successful delete requests
  produce a completed state with visibility marked `skipped`.
- `query_events.jsonl`: one event per query attempt, including query errors.
  Successful events include `server_took_ms` when the database reports its own
  query time (LambdaDB `took`). LambdaDB `took` is the coordinator's whole query
  time for routing, fan-out, retrieve, and merge; it excludes the gateway and
  transport.
- `search_under_ingest_events.jsonl`: one event per upload-and-ask probe when
  `search_under_ingest.pattern: upload_and_ask` is used. Parallel
  upsert/query runs write their operation events to `ingest_events.jsonl` and
  `query_events.jsonl`.
- `summary.json`: load/query counts, latency percentiles, QPS, per-stage query
  summaries, load batching/upsert timing, error rates, recall when
  a ground-truth artifact is present, and search-under-ingest metrics when
  applicable. `load.doc_count` records the post-load document-count wait
  (`status`, `expected`, `observed`, `attempts`, `duration_seconds`), and
  `query.server_took_ms` summarizes server-reported query time with the same
  percentiles as the client-side `query.latency_ms`. `server_took_ms` is not
  end-to-end client latency; for LambdaDB it excludes gateway and transport
  time.

### Delete-only and search-after-delete runs

Deletion scenarios use a deterministic plan and split mutation from search:

```text
load-only -> delete-only -> query-only with survivor ground truth
```

Use `scenarios/cohere-wikipedia-1m-delete-sequential.yaml` to delete in prepared
record order, or `scenarios/cohere-wikipedia-1m-delete-random.yaml` for a seeded
random order. `delete.checkpoints_pct` lists the allowed cumulative checkpoints;
each `--delete-only` invocation advances to exactly one checkpoint. The delete
block also controls post-delete verification through
`wait_until_deletion_visible`, `visibility_timeout`,
`visibility_poll_interval`, and `visibility_sample_size`. Visibility waiting is
enabled by default. When disabled, the delete-only run finishes after all delete
requests succeed and records `visibility.status: skipped`; a subsequent query
may still observe stale deleted documents in an eventually consistent target.

First create the checksummed deletion plan:

```bash
uv run ldbbench dataset delete-plan \
  --scenario scenarios/cohere-wikipedia-1m-delete-random.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m
```

Then compute survivor ground truth outside the timed database run. Repeat this
command for each checkpoint that will be queried:

```bash
uv run ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --metric cosine \
  --backend faiss \
  --top-k 10 \
  --deletion-plan data/datasets/cohere-wikipedia-1m/deletion_plan.random.seed-20260723.jsonl \
  --delete-checkpoint-pct 25
```

After the collection has been loaded and the target config has been changed to
`prepare.mode: existing`, advance the database to the 25% checkpoint:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-delete-random.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --delete-only \
  --deletion-plan data/datasets/cohere-wikipedia-1m/deletion_plan.random.seed-20260723.jsonl \
  --delete-checkpoint-pct 25 \
  --allow-destructive \
  --allow-large-run \
  --out results/lambdadb-delete-random-25
```

Search the remaining corpus using the matching state and survivor GT. A
query-only run using a deletion scenario requires both arguments and rejects a
state whose order, seed, or checkpoint is not configured by the scenario:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-delete-random.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --query-only \
  --deletion-state results/lambdadb-delete-random-25/deletion_state.json \
  --ground-truth data/datasets/cohere-wikipedia-1m/ground_truth.cosine.deleted.random.seed-20260723.pct-025.jsonl \
  --allow-large-run \
  --out results/lambdadb-search-after-delete-random-25
```

To advance from 25% to 50%, pass the 25% state into the next delete-only run.
Only the additional plan slice is sent to the database:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-delete-random.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --delete-only \
  --deletion-plan data/datasets/cohere-wikipedia-1m/deletion_plan.random.seed-20260723.jsonl \
  --delete-checkpoint-pct 50 \
  --deletion-state results/lambdadb-delete-random-25/deletion_state.json \
  --allow-destructive \
  --allow-large-run \
  --out results/lambdadb-delete-random-50
```

The runner rejects mismatched target fingerprints (including endpoint and
configured project/region/namespace identity), records and queries checksums,
deletion-plan checksum, scenario order/seed/checkpoint metadata, metric, top-k,
or survivor GT checksum. Do not recreate the same remote collection between
delete checkpoints or switch the credentials behind the configured API-key
environment variable: a local fingerprint cannot distinguish those remote
identity changes when the target config itself is unchanged.

Delete-only, checkpoint-resume, and query-only validation compare the
`records_sha256` recorded in their manifests instead of re-reading the complete
`records.jsonl` on every run. The deletion plan itself is still checksummed and
fully parsed. Regenerate the dataset artifacts and deletion plan together after
manually changing `records.jsonl`.

Delete completion verifies a deterministic sample from the full cumulative
deleted prefix after all delete requests return. It is not an exhaustive scan
of every deleted ID; any stale deleted hits that remain outside the sample still
occupy result slots and reduce survivor recall. Query reports and
`*-query-stages.csv` include delete order, seed, checkpoint percentage, and
remaining-record count so checkpoint results can be compared directly.
Filtered survivor GT is not supported in the initial implementation.

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

There are two search-under-ingest patterns:

- `upload_and_ask`: upserts one held-out document group, then immediately
  queries for that same group. This is the read-after-write visibility check.
- `parallel_upsert_query`: runs upsert workers and query workers at the same
  time until the configured duration or record stream is exhausted. This is the
  concurrent ingest/query load workload. It uses `load.processes` for upsert
  workers and `query.processes` for query workers; each concurrency value remains
  the total in-flight worker count and is split across the configured processes.

The `upload_and_ask` implementation supports `probe_source: queries`,
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

To run concurrent upserts and queries instead, use the parallel scenario:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-parallel-search-under-ingest.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-search-under-ingest-100k \
  --max-records 100000 \
  --allow-large-run \
  --out results/example-lambdadb-parallel-search-under-ingest
```

The parallel pattern writes upsert events to `ingest_events.jsonl`, query events
to `query_events.jsonl`, and a `search_under_ingest` summary with concurrent
records/s, queries/s, write latency, query latency, and recall when ground truth
is available.

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
- `timeout_ms`: per-request SDK timeout in milliseconds. The example targets
  use `30000` to keep long-tail concurrent ingest/query calls bounded.
- `index_configs`: LambdaDB collection index config used by create/recreate
  preparation modes. Metadata fields that need indexing should be declared
  under a `metadata` object field with `objectIndexConfigs`; the example targets
  index `metadata.text` for text search and `metadata.filter_bucket_*` for
  filtered-vector workloads.
- `partition_config`: optional LambdaDB hash partition config for create/recreate
  preparation modes. Nested metadata partition fields use dotted paths such as
  `metadata.url`. See `configs/lambdadb-partitioned.example.yaml`.
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
- `wait_until_doc_count`: when true (the default), wait after the load until the
  collection's full `numDocs` count matches this run's expected document count,
  including records loaded by an earlier run that `--resume-load` skipped. The
  runner captures `numDocs` and LambdaDB `dataUpdatedAt` before loading; when
  this run writes records, the count must match and LambdaDB's data head must
  advance past that baseline. This checks collection-level publication, not
  query visibility or a sample of documents. An observed count above expected
  fails immediately. Adapters without collection statistics skip this wait.
- `doc_count_timeout`: optional duration string, defaults to `1h`. The run fails
  when the expected count and required data-head advance do not appear within
  this time.
- `doc_count_poll_interval`: optional duration string, defaults to `5s`.
- `wait_until_query_visible`: when true, wait for a loaded-record sample to be
  visible through vector query before the query stage starts.
- `query_visibility_timeout`: optional duration string, defaults to `60s`.
- `query_visibility_poll_interval`: optional duration string, defaults to `1s`.

For staged queries, `query.processes` works the same way: each
`query.stages[].concurrency` value remains the total in-flight query count, and
the runner splits that total across worker processes when `query.processes > 1`.
Each query stage can stop by elapsed `duration`, by `max_requests`, or by
whichever limit is reached first when both are set. `max_requests` counts query
attempts for that stage, including failed requests, so it is useful when you
want a fixed-size concurrent query run instead of a duration-based load test.
The `parallel_upsert_query` search-under-ingest workload also uses
`query.processes`, with `search_under_ingest.query_concurrency` as the total
in-flight query count.

`query.warmup` can set `enabled: true` and a positive `query_count`. The runner
sends that many queries after the post-load waits and before measured query
stages. Warmup requests are not written to `query_events.jsonl` and do not
contribute to measured query summaries; the runner cycles through the available
query vectors when the warmup count is larger than the query set. Unknown keys
under `scenario.query` are rejected instead of ignored.

Partition-pruned query workloads can set `query.partition_filter` with a target
field such as `metadata.url` and query metadata source field such as `url`.
These runs intentionally skip global recall reporting because the query searches
a restricted partition subset. See `docs/PARTITIONING_WORKLOAD.md` and
`scenarios/cohere-wikipedia-1m-partitioned.yaml`.

Full-text-only workloads set `workload: full_text_search` and
`query.full_text`. The runner builds deterministic query strings from the held
out query rows' metadata text and sends them to adapters that declare
`supports_full_text_search`. These runs do not use vector ground truth; summary
and report output focus on latency, QPS, error rate, returned-count
distribution, and empty-result rate. LambdaDB uses `queryString` with
`defaultField: metadata.text`, matching the example target's nested metadata
text index.

The full-text scenario can reuse the same prepared dataset directory as the
dense-vector scenario because `dataset prepare` already stores each source
row's text under record/query `metadata.text`. It does not require running
`dataset ground-truth`; `recall_at_k` is reported as `null` with
`recall_skip_reason: full_text_no_ground_truth`.

`query.full_text` fields have separate source and target meanings:

- `field`: database query field, such as `metadata.text` for LambdaDB's nested
  metadata text index.
- `metadata_field`: metadata key to read from each held-out query row, such as
  `text`.
- `max_terms`: maximum number of leading tokens used to build the deterministic
  query string from the query row's metadata text.

Full-text-only workloads may also set `query.partition_filter`, using the same
`field` and `metadata_field` semantics as partition-pruned vector workloads.
For LambdaDB this sends the text `queryString` and the `partition_filter` in the
same query request, so the collection can be hash-partitioned on a field such as
`metadata.url` while searching `metadata.text`.

At the moment, LambdaDB declares both `supports_full_text_search` and
`supports_query_partition_filter`. Qdrant and Pinecone full-text support is
intentionally reported as N/A until their exact equivalent behavior is verified.

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-text-only.yaml \
  --target configs/lambdadb.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --allow-large-run \
  --out results/example-lambdadb-text-only
```

For query-time partition pruning on `metadata.url`, use the partitioned target
and scenario:

```bash
uv run ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m-text-only-partitioned.yaml \
  --target configs/lambdadb-partitioned.example.yaml \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --allow-large-run \
  --out results/example-lambdadb-text-only-partitioned
```

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

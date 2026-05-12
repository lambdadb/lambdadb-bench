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
export LAMBDADB_API_KEY=...
```

For Qdrant Cloud, set:

```bash
export QDRANT_ENDPOINT=https://example.qdrant.io
export QDRANT_API_KEY=...
```

For Pinecone Serverless, set:

```bash
export PINECONE_INDEX_HOST=https://example-index.svc.us-east-1-aws.pinecone.io
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
- `summary.json`: load/query counts, latency percentiles, QPS, per-stage query
  summaries, error rates, and recall when `ground_truth.jsonl` is present.

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
- `delete_wait_timeout_seconds`: recreate-mode deletion wait timeout. Defaults
  to `60`.
- `delete_wait_poll_seconds`: recreate-mode deletion polling interval. Defaults
  to `1`.

Useful load settings:

- `batch_size`: maximum records per upsert request.
- `concurrency`: number of concurrent upsert workers. Defaults to `1`.
- `max_batch_bytes`: optional approximate request payload cap. The runner
  splits batches by both `batch_size` and this byte limit to avoid oversized
  requests.
- `wait_until_query_visible`: when true, wait for a loaded-record sample to be
  visible through vector query before the query stage starts.
- `query_visibility_timeout`: optional duration string, defaults to `60s`.
- `query_visibility_poll_interval`: optional duration string, defaults to `1s`.

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

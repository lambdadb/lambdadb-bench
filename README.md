# lambdadb-bench

Reproducible benchmark harness for LambdaDB and comparable managed vector
databases.

The initial benchmark design focuses on the Cohere Wikipedia embedding workload
used by `tpuf-benchmark`, with LambdaDB, Qdrant Cloud, and Pinecone Serverless as
the first target adapters.

See [docs/DESIGN.md](docs/DESIGN.md) for the current design decisions, workload
model, adapter contract, result format, and implementation phases.

## Quickstart

Install the package in editable mode with development dependencies:

```bash
uv sync --extra dev
```

Check the CLI:

```bash
uv run ldbbench doctor
```

### 1. Prepare a smoke dataset

Start with a tiny row-limited dataset. This avoids a costly 1M-row run while
verifying the end-to-end flow.

This step does not use a LambdaDB or Qdrant target config. It reads the scenario
dataset source and writes local files under `--out`. Set `HF_TOKEN` in your
environment if you want authenticated Hugging Face downloads with higher rate
limits.

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

### 2. Configure a target

Use one target config per database. The checked-in files are examples:

- `configs/lambdadb.example.yaml`
- `configs/qdrant-cloud.example.yaml`

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

Real runs write:

- `ingest_events.jsonl`: one event per upsert batch.
- `query_events.jsonl`: one event per query.
- `summary.json`: load/query counts, latency percentiles, QPS, and recall when
  `ground_truth.jsonl` is present.

Runs at 1M rows or larger require `--allow-large-run` unless `--max-records`
keeps the run below that threshold.

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

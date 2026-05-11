# lambdadb-bench

Reproducible benchmark harness for LambdaDB and comparable managed vector
databases.

The initial benchmark design focuses on the Cohere Wikipedia embedding workload
used by `tpuf-benchmark`, with LambdaDB, Qdrant Cloud, and Pinecone Serverless as
the first target adapters.

See [docs/DESIGN.md](docs/DESIGN.md) for the current design decisions, workload
model, adapter contract, result format, and implementation phases.

## Development

Install the package in editable mode with development dependencies:

```bash
uv sync --extra dev
```

Or create a virtual environment manually:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Check the CLI:

```bash
ldbbench doctor
```

Plan the dataset cache layout without downloading rows:

```bash
ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --dry-run
```

Prepare a tiny row-limited dataset cache for a smoke test:

```bash
ldbbench dataset prepare \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --limit 100 \
  --query-count 10 \
  --out data/datasets/cohere-wikipedia-1m-smoke
```

Dataset preparation writes a raw source sample plus normalized benchmark
artifacts:

- `raw_records.jsonl`: source rows as received from the dataset provider.
- `queries.jsonl`: held-out query vectors.
- `records.jsonl`: records intended for database loading.
- `dataset_manifest.json`: dataset source, row counts, artifact paths, and
  checksums.

Compute exact ground truth for a prepared smoke dataset:

```bash
ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m-smoke \
  --top-k 10 \
  --backend exact
```

Validate the example scenario and target config:

```bash
QDRANT_URL=https://example.qdrant.io \
  ldbbench config validate \
    --scenario scenarios/cohere-wikipedia-1m.yaml \
    --target configs/qdrant-cloud.example.yaml
```

Initialize a result directory with reproducibility artifacts:

```bash
QDRANT_URL=https://example.qdrant.io \
  ldbbench manifest init \
    --scenario scenarios/cohere-wikipedia-1m.yaml \
    --target configs/qdrant-cloud.example.yaml \
    --out results/example-qdrant-1m
```

Check target adapter capabilities:

```bash
QDRANT_URL=https://example.qdrant.io \
  ldbbench target check --target configs/qdrant-cloud.example.yaml
```

The real Qdrant adapter uses the official `qdrant-client` package with gRPC
preferred by default. Configure Qdrant targets with:

- `endpoint`: Qdrant Cloud or self-managed Qdrant URL.
- `api_key_env`: environment variable name containing the Qdrant API key.
- `collection_name`: target collection. Existing `collection` configs are still
  accepted.
- `vector_field`: optional named vector to use. Omit this for Qdrant's default
  unnamed vector.
- `prefer_grpc`: boolean, defaults to `true`.

Normal unit tests do not contact Qdrant. Optional integration coverage is gated
behind:

```bash
QDRANT_BENCH_RUN_INTEGRATION=1
QDRANT_URL=https://example.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=...
```

Check a LambdaDB target config:

```bash
LAMBDADB_ENDPOINT=https://api.lambdadb.ai \
LAMBDADB_PROJECT_NAME=my-project \
  ldbbench target check --target configs/lambdadb.example.yaml
```

The real LambdaDB adapter uses the official `lambdadb` Python SDK. Configure
LambdaDB targets with:

- `endpoint`: API base URL, for example `https://api.lambdadb.ai`.
- `project_name`: LambdaDB project name.
- `api_key_env`: environment variable name containing the project API key.
- `collection_name`: target collection.
- `vector_field`: field that stores normalized benchmark vectors. Defaults to
  `vector`.
- `index_configs`: LambdaDB collection index config used by create/recreate
  preparation modes.

Normal unit tests do not contact LambdaDB. Optional integration coverage is
gated behind:

```bash
LAMBDADB_BENCH_RUN_INTEGRATION=1
LAMBDADB_API_KEY=...
LAMBDADB_ENDPOINT=https://api.lambdadb.ai
LAMBDADB_PROJECT_NAME=...
LAMBDADB_COLLECTION_NAME=...
```

Dry-run a benchmark plan without contacting a database:

```bash
QDRANT_URL=https://example.qdrant.io \
  ldbbench run --dry-run \
    --scenario scenarios/cohere-wikipedia-1m.yaml \
    --target configs/qdrant-cloud.example.yaml \
    --out results/example-qdrant-1m
```

Run tests and linting:

```bash
uv run python -m pytest
uv run ruff check .
```

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

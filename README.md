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

Run tests and linting:

```bash
uv run python -m pytest
uv run ruff check .
```

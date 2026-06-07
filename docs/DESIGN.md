# LambdaDB Bench Design

`lambdadb-bench` is a reproducible benchmark harness for LambdaDB and comparable
managed vector databases. The first benchmark target is the Cohere Wikipedia
embedding workload, with an initial focus on LambdaDB, Qdrant Cloud, and
Pinecone Serverless.

The goal is not to force every database into the same infrastructure shape. The
goal is to run the same workload against user-provisioned targets, record the
target configuration transparently, and compare the observable outcomes:
latency, throughput, recall, write visibility, errors, and cost assumptions.

## Current Decisions

- Repository name: `lambdadb-bench`.
- Primary implementation language: Python.
- Database access path: official Python SDKs, not hand-written raw HTTP calls.
- First comparison target: LambdaDB and Qdrant Cloud.
- First expansion target: Pinecone Serverless.
- Pinecone pod-based indexes are out of scope.
- First meaningful dataset size: `1M` rows.
- First public-scale dataset size: `10M` rows.
- Cluster, index, and region setup are user-managed.
- Target preparation supports `existing` and `create` from the first release.
- `recreate` must require an explicit destructive flag.
- Qdrant uses gRPC by default.
- Public reports use a user-provided target label plus a redacted hostname.
- `10M` exact ground truth generation is handled as a separate prepare step.
- Public results must include redacted target metadata and user-declared
  deployment settings.

## Design Goals

- Reproduce a realistic vector search workload based on Cohere Wikipedia
  embeddings.
- Make LambdaDB, Qdrant Cloud, and Pinecone Serverless comparable under the same
  dataset, query set, concurrency schedule, and measurement rules.
- Keep database provisioning user-managed so users can test the cluster, region,
  and pricing model they actually care about.
- Record enough run metadata to make public results auditable.
- Support additional vector databases through a stable adapter interface.
- Separate serverless-relevant measurements from generic vector search
  measurements instead of mixing them into one opaque score.

## Non-Goals

- The benchmark does not prescribe a single Qdrant Cloud cluster size.
- The benchmark does not prescribe a single Pinecone project region.
- The benchmark does not claim one global winner across all workloads.
- The benchmark does not hide vendor-specific configuration behind a synthetic
  normalized tier.
- The benchmark does not test Pinecone pod-based indexes.
- The first version does not attempt to benchmark every feature of each database.

## Initial Scope

### Targets

The first implementation should support:

- `lambdadb` using the official LambdaDB Python SDK.
- `qdrant` using the official Qdrant Python client.
- `pinecone` using the official Pinecone Python SDK against serverless indexes.

The adapter layer should be written so that adding Turbopuffer, Weaviate,
Milvus, or other managed vector databases later does not require changes to the
runner.

### Dataset

Primary dataset:

- `CohereLabs/wikipedia-2023-11-embed-multilingual-v3`
- English subset for the initial workload.
- Vector field: `emb`
- Dimension: `1024`
- Metric: cosine similarity

Planned sizes:

- `1M`: first implementation and first useful comparison.
- `10M`: first public-scale report candidate.
- Small smoke-test subsets can exist for CI, but should be clearly marked as
  non-representative.

### Workloads

Initial workload:

- Dense vector search.
- `top_k = 10`
- Concurrency sweep, for example `[1, 8, 32, 64, 128]`.
- Query vectors sampled deterministically from held-out dataset rows.
- Optional warmup stage before measured stages.

Follow-up workloads:

- Filtered vector search.
- Search while ingesting.
- Idle-to-burst serverless workload.
- Write visibility workload.

### Write Modes

Write workloads should make the write path explicit. This matters because
different write APIs can have different acknowledgement and consistency
semantics.

Initial write modes:

- `upsert`: regular per-record or batched SDK upsert path.
- `bulk_upsert`: bulk loading path optimized for larger imports.

For LambdaDB, the first write benchmark should prefer `upsert` because it
matches the read-after-write path most users exercise in interactive agent and
RAG applications. `bulk_upsert` should remain available as a separate scenario
for large offline imports.

Read consistency should also be explicit:

- `eventual`: query using the default eventual-consistency path.
- `strong`: query using the database's strong-consistency or consistent-read
  option, when supported.

For LambdaDB `upsert` with strong consistency enabled, query-time
read-after-write consistency is expected, so the benchmark should not report
query visibility lag as if it were an indexing delay. It should instead report
the cost and latency of the strong-consistency read path separately from the
default eventual-consistency query path.

Not every adapter will support every consistency level. If a scenario requests
`strong` and the target cannot provide a comparable read-after-write guarantee,
the runner should mark that result as `N/A` rather than silently falling back to
eventual reads or approximating strong consistency with a sleep.

Vendor-specific consistency controls should be recorded separately from the
portable `eventual`/`strong` scenario enum. For example, Qdrant exposes replica
read-consistency settings such as `all`, `majority`, and `quorum` in distributed
deployments; those settings should be captured in target metadata, but should
not be treated as equivalent to LambdaDB's strong read-after-write query path
unless the adapter can validate the same guarantee for this benchmark.

## Fairness Model

The benchmark should compare user-visible outcomes under the same workload,
rather than pretending that provisioned and serverless products have identical
deployment knobs.

The runner should not enforce a single cluster size, pod size, or region.
Instead, target configuration is supplied by the benchmark user and copied into
the run manifest.

Required run metadata:

- Vendor and adapter version.
- SDK name and SDK version.
- Protocol, when relevant, such as REST or gRPC.
- Adapter capabilities, including supported consistency levels.
- User-provided target label.
- Endpoint hostname with secrets redacted.
- User-declared deployment mode.
- User-declared region.
- Client machine region or location, when available.
- Dataset name, split, sample seed, and row count.
- Scenario file hash.
- Target config file hash with secrets redacted.
- Batch size, timeout, retry policy, and concurrency schedule.
- User-declared pricing assumptions.
- User-declared database configuration notes.

If the client region and database region differ, the benchmark should continue
but the report should show a warning.

## User-Managed Provisioning

Targets should default to existing user-managed endpoints. This keeps the
benchmark useful for real-world comparisons and avoids hard-coding debatable
cluster choices into the harness.

Target preparation modes:

- `existing`: use an existing collection/index/table.
- `create`: create a target with minimal required settings.
- `recreate`: delete and create a clean target. This should require an explicit
  destructive flag.

The first release should support both `existing` and `create`. `existing` keeps
the benchmark useful for real user-managed deployments. `create` is important
for convenient setup and repeatable smoke testing. `recreate` is useful, but it
must require an explicit destructive flag.

Example target config:

```yaml
vendor: qdrant
name: qdrant-cloud-user-cluster
endpoint: ${QDRANT_ENDPOINT}
api_key_env: QDRANT_API_KEY
collection: cohere_wikipedia_1m
region: us-east-1

prepare:
  mode: existing

metadata:
  deployment_mode: cloud
  user_declared_config: "1 node, user-selected size, no quantization"
  pricing_notes: "User-provided monthly cluster cost"
  report_label: "qdrant-cloud-grpc"
```

## Scenario Format

Scenarios define the workload. Targets define the database connection and
deployment metadata.

Example scenario:

```yaml
name: cohere-wikipedia-1m-vector
description: Cohere Wikipedia 1M dense vector search workload.

dataset:
  provider: huggingface
  source: CohereLabs/wikipedia-2023-11-embed-multilingual-v3
  subset: en
  rows: 1000000
  vector_field: emb
  dimensions: 1024
  metric: cosine
  seed: 20260511

load:
  write_mode: upsert
  batch_size: 500
  max_batch_bytes: 200MB
  wait_until_query_visible: true

query:
  top_k: 10
  query_count: 1000
  query_source: heldout_dataset_vectors
  consistency: eventual
  include_vectors: false
  warmup:
    enabled: true
    query_count: 100
  stages:
    - concurrency: 1
      duration: 5m
    - concurrency: 8
      duration: 5m
    - concurrency: 32
      duration: 5m
    - concurrency: 64
      duration: 5m
    - concurrency: 128
      duration: 5m

quality:
  ground_truth: exact_top_k
  recall_at: [10]
  min_recall_at_10: 0.95

metrics:
  latency_percentiles: [50, 95, 99]
  include_qps: true
  include_error_rate: true
  include_recall: true
  include_cost_estimates: true
```

## CLI Shape

Suggested command shape:

```bash
ldbbench dataset prepare scenarios/cohere-wikipedia-1m.yaml

ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/qdrant-cloud.yaml \
  --out results/qdrant-1m

ldbbench run \
  --scenario scenarios/cohere-wikipedia-1m.yaml \
  --target configs/lambdadb.yaml \
  --out results/lambdadb-1m

ldbbench report results/* --out reports/cohere-wikipedia-1m.md
```

Recommended subcommands:

- `dataset prepare`: download, sample, normalize, and cache dataset files.
- `dataset ground-truth`: compute exact nearest neighbors for query vectors as
  a separate prepare step.
- `target check`: validate credentials, endpoint reachability, and target
  metadata.
- `load`: load records without running queries.
- `run`: run load, visibility wait, query stages, and reporting for one target.
- `report`: combine one or more result directories into Markdown and CSV.

## Adapter Contract

Each database adapter should implement the same logical interface:

```python
class VectorDBAdapter:
    def describe(self) -> TargetDescription: ...
    def check(self) -> CheckResult: ...
    def prepare(self, mode: PrepareMode) -> None: ...
    def upsert_batch(self, records: list[VectorRecord]) -> UpsertResult: ...
    def wait_until_query_visible(self, sample_ids: list[str], deadline: float) -> VisibilityResult: ...
    def query(self, vector: list[float], top_k: int, filter: dict | None = None) -> QueryResult: ...
    def delete_all(self) -> None: ...
```

Adapter implementations may use vendor-specific SDK features, but the runner
should only depend on the common interface.

Adapters should declare capabilities before a run starts:

```python
class AdapterCapabilities:
    supported_write_modes: set[str]
    supported_query_consistency: set[str]
    supports_read_after_write_strong: bool
    vendor_consistency_options: dict[str, object]
```

The runner should use these capabilities to decide whether a scenario is
supported, unsupported, or partially reportable. Unsupported measurements should
be reported as `N/A` with a short reason.

## Measurements

### Ingest

Measure:

- Total records loaded.
- Accepted records per second.
- Batch latency percentiles.
- Failed batch count.
- Retry count.
- Time to fetch-visible, where supported.
- Time to query-visible.

Write visibility should be reported separately from API acknowledgement time.
This matters for systems with asynchronous indexing. If a scenario uses a write
mode and query consistency option that guarantees read-after-write visibility,
the report should mark visibility as guaranteed by configuration rather than
measuring it as a lagging background-indexing process.

### Query

Measure for each concurrency stage:

- Successful queries.
- Failed queries.
- Timeout count.
- Rate-limit count.
- QPS.
- p50, p95, and p99 latency.
- Recall@10, when ground truth is available.
- Consistency mode used for the query path.
- Unsupported consistency results as `N/A`, with a reason.

Latency should be measured client-side around the SDK call. The report should
state that this includes network and SDK overhead.

### Cost

The benchmark should not guess hidden vendor costs by default. It should support
user-provided cost assumptions and compute normalized derived metrics:

- Cost per 1M successful queries.
- Cost per 1M loaded vectors.
- Cost per stored GB per month, if supplied.
- Monthly cost under a declared workload shape.

Reports should clearly mark cost data as user-supplied unless the adapter uses a
documented pricing API or a checked-in pricing model.

## Output Files

Each run directory should contain:

```text
results/<run-id>/
  run_manifest.json
  scenario.resolved.yaml
  target.redacted.yaml
  ingest_events.jsonl
  query_events.jsonl
  summary.json
  summary.csv
```

`run_manifest.json` is the source of truth for reproducibility. Public reports
should link back to the manifest and include enough target metadata for readers
to understand what was actually compared.

## Report Shape

Reports should lead with the workload and configuration, not a winner.

Suggested sections:

- Workload summary.
- Target configurations.
- Data loading results.
- Query performance by concurrency.
- Recall and quality gates.
- Error and retry behavior.
- Cost assumptions and normalized cost.
- Notes, warnings, and limitations.

If a result fails the recall gate, the report should still show latency and QPS
but mark the result as not meeting the quality target.

## Implementation Phases

### Phase 1: Harness Skeleton

- Python package and CLI.
- Scenario parser.
- Target config parser.
- Result directory and manifest writer.
- Adapter base interface.
- Dataset cache layout.
- Target `existing` and `create` preparation modes.

### Phase 2: 1M Cohere Wikipedia

- Deterministic 1M dataset preparation.
- Held-out query sampling.
- Exact ground truth generation for recall@10.
- LambdaDB adapter.
- Qdrant adapter using gRPC by default.
- 1M ingest and query run.

### Phase 3: Pinecone Serverless

- Pinecone adapter. (Implemented)
- Same 1M workload.
- Report comparing LambdaDB, Qdrant Cloud, and Pinecone Serverless.

### Phase 4: 10M Scale Run

- 10M dataset preparation.
- Separate exact ground truth prepare step that can run on a suitable local
  machine or cloud runner.
- Long-running ingest and query reporting.
- Public report template.

### Phase 5: Serverless-Specific Workloads

- Idle-to-burst workload.
- Search-under-ingest workload.
- Write visibility workload.
- Optional filtered search workload. See
  [FILTERED_SEARCH_WORKLOAD.md](FILTERED_SEARCH_WORKLOAD.md).

## Resolved Decisions

- LambdaDB Python SDK: use the latest stable SDK as the first supported minimum.
- Qdrant protocol: use gRPC by default.
- `10M` exact ground truth: generate it as a separate prepare step.
- Public endpoint display: use a user-provided target label plus a redacted
  hostname.
- Target setup: support `existing` and `create` in the first release.

## References

- `VectorDBBench`: https://github.com/zilliztech/vectordbbench
- Qdrant vector database benchmark: https://github.com/qdrant/vector-db-benchmark
- Cohere Wikipedia embedding dataset:
  https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3
- Pinecone serverless documentation:
  https://docs.pinecone.io/guides/indexes/understanding-indexes

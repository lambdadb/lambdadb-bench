# Partitioned Search Workload Design

This document defines how `lambdadb-bench` should test hash-based partitioning
without mixing it into the existing global vector-recall benchmark.

## Goal

Measure whether LambdaDB partitioning improves query latency, throughput, and
cost shape when an application can route queries to one or more known
partitions.

The first implementation focuses on LambdaDB because the benchmark already has
a real LambdaDB adapter and the LambdaDB SDK exposes both collection partition
configuration and query-time partition filtering.

## Non-Goals

- Do not report partition-filtered runs as directly comparable global recall
  results.
- Do not use the existing global FAISS ground truth for partition-filtered
  queries.
- Do not emulate partition pruning on vendors that do not expose an equivalent
  physical partition filter.
- Do not combine partitioned and unpartitioned results into one opaque winner
  score.

## Why Recall Is Different

The current benchmark computes ground truth over the full loaded corpus. A
partition-filtered query intentionally searches only a subset of that corpus.
That means the top-k result set is allowed to differ from the global top-k
result set, even when the database behaves correctly.

For the first partitioning workload, report recall as `null` / `N/A` and mark
the skip reason as `partition_filtered`. This keeps latency, throughput, error
rate, and cost signals clean without implying that lower global recall is a
quality regression.

Partition-aware recall can be added later by building ground truth over the same
partition-filtered candidate set used by each query.

## Dataset Requirements

The Cohere Wikipedia dataset already contains fields such as `url`. Dataset
prepare preserves non-vector source fields in normalized record metadata, so
records loaded into LambdaDB include `metadata.url`.

The first workload should use:

- Partition field: `url`
- Partition field type: `keyword`
- Query partition value source: held-out query metadata field `url`

Using `url` is intentionally high-cardinality. It exercises hash-based physical
distribution while keeping query routing deterministic from the query row.

## Target Configuration

Add optional LambdaDB target configuration:

```yaml
partition_config:
  field_name: url
  data_type: keyword
  num_partitions: 16
```

The LambdaDB adapter should pass this through during collection creation:

```python
client.collections.create(
    collection_name=...,
    index_configs=...,
    partition_config={
        "field_name": "url",
        "data_type": "keyword",
        "num_partitions": 16,
    },
)
```

For `prepare.mode: existing`, the adapter should not assume that the existing
collection matches the target file. It should record the configured
`partition_config` in the run manifest and, if the SDK response exposes the
actual collection partition config, include that in prepare details or target
check output.

## Query Workload Configuration

Add optional query configuration:

```yaml
query:
  partition_filter:
    field: url
    metadata_field: url
```

At query time, the runner should read `metadata[metadata_field]` from each query
record. For LambdaDB, it should pass:

```python
partition_filter={
    "field": "url",
    "in_": [query.metadata["url"]],
}
```

This is separate from the existing vector query `filter_query`. The partition
filter is a routing/pruning hint, while `knn.filter` remains a logical query
filter inside the search request.

If a query row does not contain the configured metadata field, the runner should
fail the scenario before starting the measured stage. Silently falling back to
unpartitioned search would make the benchmark misleading.

## Result Matrix

Run three LambdaDB variants for a clean first comparison:

| Variant | Collection partitioned | Query partition filter | Recall |
| --- | --- | --- | --- |
| Baseline | No | No | Global recall |
| Partitioned collection only | Yes | No | Global recall |
| Partition-pruned query | Yes | Yes | N/A |

The first two variants show whether merely partitioning the collection changes
global query behavior. The third variant measures the intended product path:
the application knows a partition key and routes the query to the relevant
partition subset.

## Metrics

Keep the standard metrics:

- query latency percentiles
- queries per second
- attempts per second
- error rate
- load records per second
- load latency percentiles

Add partition-specific run metadata:

- `target.partition_config`
- `query.partition_filter`
- `query.partition_filter_applied: true | false`
- `query.recall_skip_reason: partition_filtered` when applicable

For cost analysis, the report should allow partition-pruned runs to stand alone
instead of averaging them into global vector-search recall tables.

## Adapter Support Rules

LambdaDB:

- Supports `partition_config` on collection create.
- Supports `partition_filter` on collection query and fetch APIs.
- First implementation should support query partition filters only. Fetch can be
  extended later if visibility checks need partition-aware fetches.

Qdrant and Pinecone:

- Do not receive LambdaDB partition filters.
- If a scenario requires `query.partition_filter`, the planner should mark these
  targets as `N/A` unless an adapter explicitly declares equivalent physical
  partition pruning support.

Milvus:

- Can be considered later as a separate adapter because it has native partition
  concepts, but the first benchmark should not block on Milvus support.

## Implementation Status

Initial LambdaDB support is implemented:

- `AdapterCapabilities` declares partition-filter support.
- `TargetConfig` preserves optional `partition_config`.
- The LambdaDB adapter validates and passes `partition_config` during
  create/recreate.
- When `partition_config.field_name` is present, LambdaDB load copies the
  matching metadata value to a top-level document field so the partition key is
  present in loaded documents.
- The runner parses `query.partition_filter`, validates query metadata before a
  measured query stage starts, and sends per-query partition filters to the
  adapter.
- LambdaDB forwards query partition filters to `collections.query(...)`.
- Non-supporting adapters are planned as `N/A` for partition-filter scenarios.
- Query summaries and reports mark partition-filtered recall as skipped.
- A partitioned Cohere Wikipedia scenario and LambdaDB target example are
  available.

## Open Questions

- What default `num_partitions` should public reports use for 1M and 10M runs?
- Should partition-pruned query stages use the same concurrency schedule as
  global query stages, or a separate lower-latency schedule?
- Should the first partition-aware ground truth implementation sample candidate
  records by partition value from the normalized dataset, or should it be built
  from loaded database fetch/list output?

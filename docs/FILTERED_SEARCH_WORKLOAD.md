# Filtered Vector Search Workload Design

This document defines a workload for measuring vector-search latency,
throughput, and recall when every query also applies a logical metadata filter.

## Goal

Measure whether a target can preserve nearest-neighbor quality and serving
performance when search is constrained to a filtered candidate set.

The workload should answer:

- How does filter selectivity affect query latency and throughput?
- Does approximate vector search preserve recall inside the filtered subset?
- Do highly selective filters return enough results, and when do they fail to
  return `top_k` matches?
- Are metadata filters pushed into the search path in a way that behaves like a
  first-class vector-search constraint rather than a post-processing step?

This is separate from partition pruning. Partition pruning measures application
routing to a physical subset. Filtered vector search measures logical predicate
semantics inside a normal vector query.

## Non-Goals

- Do not reuse global unfiltered ground truth for filtered recall.
- Do not report partition-pruned recall as filtered vector-search recall.
- Do not use high-cardinality point filters as the default public scenario.
- Do not compare vendor-specific filter languages directly in scenario files
  without a portable filter spec.
- Do not hide low candidate counts. If a filter has fewer than `top_k` eligible
  records, report that explicitly instead of treating it as a recall failure.

## Why Recall Is Different

The existing global workload computes exact top-k over the full loaded corpus.
A filtered vector query intentionally searches only records matching a metadata
predicate. Correct filtered search can return a result set that is completely
different from the global top-k result set.

Filtered recall must therefore be computed against exact top-k over the same
filtered candidate set:

```text
candidate_set = records where filter(record.metadata) is true
expected = exact_top_k(query.vector, candidate_set, k)
actual = database.query(query.vector, filter, k)
recall_at_k = overlap(actual[:k], expected[:k]) / len(expected[:k])
```

When `candidate_set` has fewer than `top_k` records, the denominator should be
the expected result count, not `top_k`. The event should also record
`candidate_count` and `expected_count` so reports can distinguish quality
regressions from intentionally small filtered subsets.

## Workload Shape

The first implementation should use the same prepared Cohere Wikipedia corpus
as the global workload, add deterministic synthetic filter buckets to record
metadata, and assign per-query filter values from eligible loaded buckets.

Use selectivity buckets rather than natural fields as the default benchmark
axis. Natural fields such as `url` are useful for partition pruning and
upload-and-ask probes, but they can be too high-cardinality or distributionally
uneven for a clean filtered-recall benchmark.

Recommended bucket fields:

| Field | Approximate selectivity | Purpose |
| --- | ---: | --- |
| `filter_bucket_2` | 50% | broad filter sanity check |
| `filter_bucket_10` | 10% | common application segment |
| `filter_bucket_100` | 1% | selective production filter |
| `filter_bucket_1000` | 0.1% | stress case for narrow candidate sets |

Bucket assignment should be deterministic from record id or source ordinal so
the same dataset artifacts produce the same filters across runs:

```text
bucket_value = stable_hash(record.id) % bucket_count
```

Each query should receive a bucket value selected from the loaded record bucket
distribution, not directly from query id. The assignment should be deterministic
for reproducibility, but it should only use bucket values whose loaded
candidate count is at least the query `top_k`.

Recommended assignment:

1. Build `bucket_value -> candidate_count` from loaded records for the selected
   filter field.
2. Keep only values where `candidate_count >= top_k`.
3. Assign query filter values by seeded round-robin or seeded sampling across
   the eligible values.
4. Persist the assigned per-query filter value in the filtered ground-truth
   artifact or its companion manifest.

This keeps public benchmark runs from accidentally measuring empty or
underfilled filters caused by small smoke datasets, `--max-records`, or uneven
natural distribution. Query-id hashing is acceptable only as an internal smoke
fallback when the run also records candidate counts and does not claim public
comparability.

## Scenario Configuration

Add an optional logical filter configuration under `query`. This should be
separate from `partition_filter`.

```yaml
query:
  top_k: 10
  query_count: 1000
  query_source: heldout_dataset_vectors
  consistency: eventual
  include_vectors: false
  filter:
    name: synthetic_bucket_1pct
    field: filter_bucket_100
    operator: eq
    value_source:
      type: eligible_record_buckets
      seed: 20260511
      min_candidates: top_k
    expected_selectivity: 0.01
  warmup:
    enabled: true
    query_count: 100
  stages:
    - concurrency: 1
      duration: 5m
    - concurrency: 8
      duration: 5m
```

For a selectivity matrix, prefer separate scenario files at first:

```text
scenarios/cohere-wikipedia-1m-filtered-50pct.yaml
scenarios/cohere-wikipedia-1m-filtered-10pct.yaml
scenarios/cohere-wikipedia-1m-filtered-1pct.yaml
scenarios/cohere-wikipedia-1m-filtered-0_1pct.yaml
```

This keeps run manifests and report rows easy to audit. A later implementation
can support multiple filter variants in one scenario if the runner grows
multi-stage quality summaries.

For public reports, use the `1%` scenario as the primary filtered-search
headline. Keep `50%`, `10%`, and `0.1%` as supporting rows that show how latency,
throughput, recall, and result fill rate change as selectivity moves from broad
to very narrow.

## Portable Filter Spec

The scenario should describe logical filter semantics, not raw vendor request
payloads.

Initial portable operators:

| Operator | Meaning | Example |
| --- | --- | --- |
| `eq` | field equals one value | `filter_bucket_100 == "42"` |
| `in` | field is in a list of values | `country in ["US", "CA"]` |

Initial MVP can support only `eq`. Add `in` once multi-value selectivity tests
are useful.

Runner-generated portable filter object:

```json
{
  "field": "filter_bucket_100",
  "operator": "eq",
  "value": "42"
}
```

Each adapter should translate the portable filter into its native query filter:

- LambdaDB: `knn.filter`
- Qdrant: `query_filter`
- Pinecone: `filter`

Adapter translation belongs in adapters, not scenario YAML.

## Ground Truth

Filtered ground truth should be generated by the existing `dataset ground-truth`
command with additional filter options. It should still write a distinct
artifact because it is not interchangeable with global ground truth.

This keeps the CLI model simple: filtered recall is still ground truth over a
prepared dataset with the same `top_k`, metric, and backend choices. The
filter-specific behavior should live in options or a referenced filter spec,
not in a separate command family.

Suggested command shape:

```bash
uv run --extra groundtruth ldbbench dataset ground-truth \
  --dataset-dir data/datasets/cohere-wikipedia-1m \
  --top-k 10 \
  --backend faiss \
  --filter-name synthetic_bucket_1pct \
  --filter-field filter_bucket_100 \
  --filter-operator eq \
  --filter-value-source eligible-record-buckets
```

Suggested artifact names:

```text
ground_truth.filtered.synthetic_bucket_1pct.jsonl
ground_truth.filtered.synthetic_bucket_1pct.manifest.json
```

Each line should include filter metadata in addition to matches:

```json
{
  "query_id": "20231101.en_123",
  "filter": {
    "field": "filter_bucket_100",
    "operator": "eq",
    "value": "42"
  },
  "candidate_count": 10012,
  "expected_count": 10,
  "matches": ["doc-a", "doc-b"]
}
```

The ground-truth manifest should include:

- source dataset manifest checksum
- filter spec
- metric
- top_k
- backend (`exact` or `faiss`)
- record count
- query count
- candidate count summary

For the first implementation, exact filtered ground truth is acceptable for
smoke and small runs. At 1M rows, FAISS-backed filtered ground truth can be
implemented by grouping record vectors by bucket value and building one exact
index per bucket value, or by scanning candidate ids for small internal runs.

Filtered recall is only comparable when the loaded target corpus matches the
record set used to generate the filtered ground truth. If a run loads only a
subset, such as `--max-records 10000` against a 1M ground-truth artifact, the
latency, error, returned-count, and underfilled-result metrics are still useful
as a smoke test, but `recall_at_k` will be biased low because many expected
neighbors were never loaded into the target. Generate ground truth over the same
loaded subset before treating recall as a quality result.

## Query Events

Add filter-specific fields to successful query events:

```json
{
  "stage": "query",
  "query_id": "20231101.en_123",
  "filter": {
    "field": "filter_bucket_100",
    "operator": "eq",
    "value": "42"
  },
  "filter_name": "synthetic_bucket_1pct",
  "filter_selectivity": 0.01,
  "candidate_count": 10012,
  "expected_count": 10,
  "returned_count": 10,
  "matches": ["doc-a", "doc-b"],
  "latency_ms": 12.4,
  "recall_at_k": 0.97,
  "status": "ok"
}
```

For errors, include `filter` and `filter_name` when they were known before the
adapter call.

## Summary Metrics

Keep the standard query summary:

- latency percentiles
- queries per second
- attempts per second
- error rate
- recall_at_k
- recall_samples

Add filtered-search summary fields:

- `filter`
- `filter_name`
- `filter_selectivity`
- `candidate_count.min`
- `candidate_count.p50`
- `candidate_count.p95`
- `candidate_count.max`
- `expected_count.min`
- `expected_count.p50`
- `returned_count.min`
- `returned_count.p50`
- `returned_count.p95`
- `returned_count.max`
- `underfilled_result_rate`

`underfilled_result_rate` is the fraction of successful queries where
`returned_count < expected_count`. This catches post-filtering or planner issues
that latency and recall alone may not make obvious.

## Result Matrix

Run the filtered scenario alongside the global baseline:

| Variant | Query filter | Ground truth | Recall |
| --- | --- | --- | --- |
| Global baseline | none | global exact top-k | global recall |
| Filtered 50% | `filter_bucket_2 = assigned_query_value` | filtered exact top-k | filtered recall |
| Filtered 10% | `filter_bucket_10 = assigned_query_value` | filtered exact top-k | filtered recall |
| Filtered 1% | `filter_bucket_100 = assigned_query_value` | filtered exact top-k | filtered recall |
| Filtered 0.1% | `filter_bucket_1000 = assigned_query_value` | filtered exact top-k | filtered recall |

Use the same load settings, query `top_k`, consistency, warmup, and concurrency
schedule across variants unless the report explicitly calls out a difference.

## Implementation Plan

1. Add synthetic filter bucket metadata during dataset preparation.
2. Add scenario validation for `query.filter`.
3. Add a small `LogicalFilterSpec` model in the runner.
4. Generate per-query portable filters from query metadata.
5. Pass translated filters through existing `adapter.query(..., filter_query=...)`.
6. Add filtered ground-truth generation and loading.
7. Record filter fields, candidate counts, expected counts, returned counts, and
   recall in query events.
8. Extend summaries and reports with filtered-search metrics.
9. Add one MVP scenario for `filter_bucket_100` before expanding to the full
   selectivity matrix.

For the MVP, implement only `eq` filters. Record `in` filter support as a later
multi-tenant or access-control-style workload rather than including it in the
first public matrix.

## Implementation Status

Initial `eq` filtered-search support is implemented:

- Dataset preparation adds deterministic `filter_bucket_2`, `filter_bucket_10`,
  `filter_bucket_100`, and `filter_bucket_1000` metadata.
- The load path backfills those bucket metadata fields for existing prepared
  dataset caches that were created before filtered-search support existed.
- Scenario validation accepts `query.filter` with
  `value_source.type: eligible_record_buckets`.
- `dataset ground-truth` accepts filter options and writes distinct
  `ground_truth.filtered.<name>.jsonl` artifacts.
- Filtered ground-truth rows store per-query filter values, candidate counts,
  expected counts, and exact expected matches.
- Filtered ground-truth manifests include eligible bucket candidate-count
  summaries.
- The runner reads filtered ground-truth rows, forwards portable logical filters
  through `adapter.query(..., filter_query=...)`, and records filter/count
  fields in query events and summaries.
- LambdaDB, Qdrant, and Pinecone adapters translate the portable `eq` filter
  into their native filter shape.
- LambdaDB create/recreate preparation includes `filter_bucket_*` keyword index
  configs, and the LambdaDB adapter copies those bucket metadata fields to
  top-level document fields so `queryString` filters can match them.
- The report includes filter fields, candidate/expected/returned count
  summaries, and `underfilled_result_rate`.
- `scenarios/cohere-wikipedia-1m-filtered-1pct.yaml` is available as the first
  public-headline scenario.

## Adapter Support Rules

LambdaDB:

- Supports logical vector filter through `knn.filter`.
- Filter fields must be present in collection `index_configs`; otherwise
  LambdaDB will not evaluate the filter field correctly.
- Should be included in the first filtered-search MVP.

Qdrant:

- Supports payload filters through `query_filter`.
- Can participate once the portable `eq` filter is translated to Qdrant's
  filter shape.

Pinecone:

- Supports metadata filters through `filter`.
- Can participate once the portable `eq` filter is translated to Pinecone's
  filter shape.

Targets that cannot translate the portable filter spec should be marked `N/A`
by the planner for filtered scenarios.

## Resolved Decisions

- Query filter values should be assigned from eligible loaded record bucket
  values so public runs guarantee enough candidates for `top_k`.
- Public reports should use `1%` selectivity as the primary filtered-search
  headline and keep other selectivities as supporting rows.
- Filtered ground truth should be an option on the existing
  `dataset ground-truth` command, with distinct filtered output artifacts.
- `in` filters should be deferred to a later multi-tenant or
  access-control-style workload. The first public matrix should use `eq`.

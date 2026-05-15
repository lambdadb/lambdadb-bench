# Search-Under-Ingest Read-After-Write Design

This document defines a workload for measuring search behavior immediately
after document-set writes while ingest is still active.

## Goal

Show whether a target can make newly written document sets visible to vector
search immediately after write acknowledgement.

This workload is intended to highlight LambdaDB's strong read-after-write query
path separately from the default eventual-consistency query path. It should also
make eventual-consistency visibility lag visible for vendors that do not expose
a comparable query-time strong read-after-write guarantee.

The primary realistic pattern is upload-and-ask:

1. A base dataset is already loaded.
2. The scenario selects one held-out document set, such as all Wikipedia chunks
   that share the same `url`.
3. The runner upserts that document set.
4. As soon as the write API acknowledges the upsert, the runner searches using
   a vector related to that just-written document set.
5. The runner records whether the response contains the exact probe chunk and
   whether it contains any chunk from the same document set.
6. The runner repeats the loop until the configured duration or probe count is
   reached.

## Non-Goals

- Do not mix this workload into the existing global FAISS recall benchmark.
- Do not call vendor-specific replica read settings "strong" unless the adapter
  can declare a comparable read-after-write query guarantee.
- Do not add sleeps to approximate strong consistency for vendors that do not
  support it.
- Do not require a large background ingest stream for the first version. The
  first version should focus on precise foreground upload-and-ask probes rather
  than maximum write throughput.

## Why This Is Not Normal Recall

The existing query workload measures approximate nearest-neighbor quality over a
loaded corpus using ground truth generated before the run.

Search-under-ingest asks a different question:

> After a document-set write is acknowledged, can a query related to that
> just-written document set find that document immediately?

The expected result is known without FAISS because the probe query uses a vector
from the newly written document set. The metric should therefore be a
read-after-write document visibility metric, not global vector recall.

Use names such as:

- `read_after_write_exact_chunk_hit_rate_at_k`
- `read_after_write_same_document_hit_rate_at_k`
- `read_after_write_same_document_recall_at_k`
- `time_to_visible_ms`
- `immediate_query_latency_ms`

If the report uses the word recall, qualify it clearly as
`read_after_write_same_document_recall_at_k`, not normal `recall_at_k`.

`same_document_recall_at_k` should be treated as a secondary metric. If a
document set contains 100 chunks and `top_k=10`, the maximum possible recall is
0.10. The primary user-facing metric should therefore be
`read_after_write_same_document_hit_rate_at_k`: did the immediate search return
at least one chunk from the just-written document set?

## Workload Shape

The first implementation should run in two required phases, with optional
background ingest after the foreground metric is stable.

### 1. Base Corpus Load

Load an existing base corpus into the target before measured probes begin.

Options:

- Use a previously loaded collection with `prepare.mode: existing`.
- Or run the existing load stage with `max_records` for a base corpus.

The base corpus gives the target a realistic collection size and prevents the
probe from measuring an empty-index special case. Base corpus records and probe
document sets must not overlap. If they overlap, a hit can come from a
previously loaded record and the read-after-write result becomes ambiguous.

### 2. Upload-And-Ask Probe Loop

Read held-out records that were not part of the base corpus and group them into
document sets.

For Cohere Wikipedia, group records by `metadata.url`:

```text
document_set_id = metadata.url
document_set = all held-out chunks with that url
```

For each probe document set:

1. Upsert all chunks in the document set.
2. Wait only for the write API acknowledgement.
3. Immediately query using one selected chunk vector from that document set.
4. Record exact chunk hit and same-document hit/recall from the returned ids.
5. Optionally poll until same-document visibility to measure
   `time_to_visible_ms`.

The first implementation can run document-set probes sequentially or with low
controlled concurrency. Higher-concurrency search-under-ingest can be a follow-up
once the metric contract is stable.

### 3. Optional Background Ingest

After the immediate probe loop is working, add optional background ingest:

- A background writer continuously upserts small batches from another held-out
  stream.
- Foreground probes continue to write one held-out document set and immediately
  query it.

This separates "visibility after my document upload" from generic load
throughput.

## Scenario Configuration

Add a workload mode instead of overloading the existing staged query mode:

```yaml
workload: search_under_ingest

load:
  write_mode: upsert
  base_records: 1000000

search_under_ingest:
  pattern: upload_and_ask
  probe_source: queries
  document_group_field: url
  max_probe_documents: 1000
  duration: 10m
  min_chunks_per_document: 1
  max_chunks_per_document: 20
  probe_queries_per_document: 1
  probe_concurrency: 1
  top_k: 10
  consistency: strong
  poll_until_visible: true
  visibility_timeout: 5s
  visibility_poll_interval: 25ms
  background_ingest:
    enabled: false
```

For LambdaDB, `consistency: strong` maps to `consistent_read=True`.

For targets that do not support the portable `strong` consistency enum, the
planner should mark the strong variant as `N/A`, consistent with the existing
run-plan behavior.

If both `max_probe_documents` and `duration` are set, the stage should stop when
the first limit is reached.

## Dataset Split

The dataset needs at least two non-overlapping record ranges:

- Base corpus records.
- Probe records.

The current prepare flow already separates query rows from load records. For
this workload, prefer an explicit probe artifact so probe records are not
confused with normal query records:

```text
records.msgpack
queries.msgpack
probes.msgpack
```

Initial implementation options:

- Reuse `queries.msgpack` as probes for a first internal smoke test.
- Add `probe_count` to dataset prepare and write `probes.jsonl` /
  `probes.msgpack` as a separate artifact.

The production-quality path should add explicit probe artifacts.

For the current Cohere Wikipedia dataset, each normalized record preserves
metadata fields such as `url`, `title`, and `text`. This is enough to implement
the first upload-and-ask workload by grouping held-out probe records on
`metadata.url`.

`queries.msgpack` is acceptable for the first version because the current
prepare flow writes query rows before record rows. That means query records are
held out from the base `records.msgpack` corpus. However, query rows were
created for global recall, not document-set probing, so `probes.msgpack` should
be added once the metric behavior is validated.

## Metrics

Per-probe event fields:

```json
{
  "stage": "search_under_ingest",
  "probe_document_index": 1,
  "document_group_field": "url",
  "document_group_value": "https://en.wikipedia.org/wiki/Dialect%20levelling",
  "probe_chunk_ids": ["20231101.en_34146519_43"],
  "query_chunk_id": "20231101.en_34146519_43",
  "write_latency_ms": 12.3,
  "immediate_query_latency_ms": 8.7,
  "top_k": 10,
  "exact_chunk_hit_at_1": true,
  "exact_chunk_hit_at_k": true,
  "same_document_hit_at_1": true,
  "same_document_hit_at_k": true,
  "same_document_recall_at_k": 1.0,
  "visible": true,
  "time_to_visible_ms": 21.5,
  "matches": ["20231101.en_34146519_43"],
  "consistency": "strong",
  "status": "ok"
}
```

Summary fields:

```json
{
  "mode": "search_under_ingest",
  "pattern": "upload_and_ask",
  "probe_documents": 1000,
  "probe_chunks": 8432,
  "errors": 0,
  "consistency": "strong",
  "read_after_write_exact_chunk_hit_rate_at_1": 0.998,
  "read_after_write_exact_chunk_hit_rate_at_k": 1.0,
  "read_after_write_same_document_hit_rate_at_1": 0.999,
  "read_after_write_same_document_hit_rate_at_k": 1.0,
  "read_after_write_same_document_recall_at_k": 0.82,
  "time_to_visible_ms": {
    "p50": 14.2,
    "p95": 33.8,
    "p99": 71.4
  },
  "write_latency_ms": {
    "p50": 11.1,
    "p95": 19.8,
    "p99": 41.0
  },
  "immediate_query_latency_ms": {
    "p50": 9.0,
    "p95": 16.2,
    "p99": 30.5
  }
}
```

If `poll_until_visible` is false, omit `time_to_visible_ms` or set it to null
with a skip reason.

For same-document metrics, the expected set is the set of just-written chunk ids
in the current document set. A result id is a same-document hit if it is in that
expected set. The first version should avoid metadata-based match inference in
the response because not every adapter returns identical metadata shapes.

## Expected Comparisons

Run at least these variants:

1. LambdaDB eventual:
   - `consistency: eventual`
   - Expected: possible immediate misses under visibility lag.

2. LambdaDB strong:
   - `consistency: strong`
   - Expected: higher immediate hit rate, with query latency/cost recorded.

3. Qdrant Cloud eventual:
   - `consistency: eventual`
   - Expected: measured as-is. Do not label as strong even if writes use
     `wait=True`.

4. Qdrant Cloud strong:
   - `consistency: strong`
   - Expected: `N/A` unless a future adapter declares a comparable
     read-after-write query guarantee.

5. Pinecone Serverless eventual:
   - `consistency: eventual`
   - Expected: measured as-is.

6. Pinecone Serverless strong:
   - `consistency: strong`
   - Expected: `N/A`.

## Fairness Notes

- Keep base corpus size the same across vendors.
- Use the same probe document sets and probe ordering across vendors.
- Keep top-k identical.
- Report both visibility hit rate and latency. A strong read path should not be
  treated as free; its value is correctness under an explicit consistency
  contract.
- Do not retry immediate queries before recording the immediate hit/miss.
  Polling for `time_to_visible_ms` should happen after the immediate query
  event is recorded.
- Ensure probe document sets are not already present in the base corpus.
- Report document-set sizing because larger document sets can make
  `same_document_recall_at_k` look lower even when same-document hit rate is
  high.

## Runner Changes

Required implementation pieces:

1. Extend scenario config to identify `workload: search_under_ingest`.
2. Add explicit probe artifact support in dataset prepare, or initially reuse
   query artifacts behind a clear internal flag.
3. Add probe grouping by a metadata field, initially `metadata.url` for Cohere
   Wikipedia.
4. Add a `run_search_under_ingest_stage` runner path.
5. Add event file output, likely `search_under_ingest_events.jsonl`.
6. Add summary section, likely `search_under_ingest`, separate from `query`.
7. Add report columns for exact-chunk hit rate, same-document hit rate,
   same-document recall, and time-to-visible percentiles.
8. Ensure run manifests record the workload mode, probe source, consistency
   setting, and adapter capabilities.

## Adapter Requirements

Adapters already expose enough surface for the first version:

- `upsert_batch(...)`
- `query(...)`

The stage should use the same `write_mode` path as normal load. For LambdaDB,
the first workload should use `write_mode: upsert`, not `bulk_upsert`, because
the benchmark is about interactive read-after-write behavior.

## Failure Behavior

- If write fails, record the probe as an error and do not query it.
- If immediate query fails, record the probe as an error.
- If immediate query succeeds but misses the exact query chunk, record
  `exact_chunk_hit_at_1=false` and `exact_chunk_hit_at_k=false`.
- If immediate query succeeds but misses all just-written chunks from the same
  document set, record `same_document_hit_at_k=false`.
- If polling times out, record `visible=false` and
  `time_to_visible_ms=null`.

## Initial Implementation Plan

1. Add this workload only for `--load-only`-style controlled runs or as a new
   `workload` mode that skips normal staged query execution.
2. Reuse prepared query records as the first probe source.
3. Group probes by `metadata.url` for Cohere Wikipedia.
4. Upsert one document set, immediately query one selected chunk vector, and
   record exact-chunk and same-document visibility metrics.
5. Implement LambdaDB eventual vs strong locally.
6. Add Qdrant/Pinecone eventual runs and ensure strong variants plan as `N/A`.
7. Add explicit probe artifacts after the metric behavior is validated.
8. Add report support once event/summary shape stabilizes.

## Open Questions

- Should probe artifacts be generated during `dataset prepare` by default, or
  only for scenarios with `workload: search_under_ingest`?
- Should background ingest be part of the first public benchmark, or saved for a
  second workload after the immediate visibility probe is validated?
- Should the stage support `probe_concurrency > 1` in the first version, or keep
  it sequential to make time-to-visible easier to reason about?
- Should strong LambdaDB visibility be verified with query only, or also with a
  strongly consistent fetch as a diagnostic field?
- Should probe query chunk selection be first chunk, random seeded chunk, or one
  query per chunk within each document set?

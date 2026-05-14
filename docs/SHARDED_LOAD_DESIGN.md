# Sharded Load Path Design

This document defines a sharded dataset/load path for high-throughput load
benchmarks.

## Goal

Let load worker processes read, parse, batch, and write their own assigned
dataset shards directly, instead of having one parent process read all records
and send Python objects through a multiprocessing queue.

The intended result is to reduce parent-process CPU pressure and inter-process
serialization overhead during large load-only runs, especially when the target
can absorb more write traffic than the current single-reader pipeline can feed.

## Current Bottleneck

The current load path is centralized:

1. The parent process opens `records.msgpack`.
2. The parent process sequentially decodes records.
3. The parent process groups records into batches.
4. Worker threads or processes receive fully parsed `VectorRecord` batches.
5. Workers call the adapter write method.

This keeps ordering and checkpoint semantics simple, but it means decode,
record conversion, batch sizing, and multiprocessing queue serialization all
flow through one coordinator.

For high concurrency, this can make the benchmark client the bottleneck even
when the database target still has spare capacity.

## Non-Goals

- Do not change query-stage execution in this design.
- Do not remove the current single-reader load path. It should remain the
  default until the sharded path is validated.
- Do not require sharding for small smoke tests.
- Do not make JSONL the optimized load path. JSONL may remain a fallback, but
  the sharded path should prefer msgpack artifacts.
- Do not silently change existing checkpoint semantics for non-sharded runs.

## Proposed Dataset Artifacts

Dataset prepare or optimize should be able to write sharded record artifacts:

```text
data/datasets/<name>/
  records.msgpack
  records-00000.msgpack
  records-00001.msgpack
  records-00002.msgpack
  ...
  dataset_manifest.json
```

`records.msgpack` can remain as the canonical single-file artifact for backward
compatibility. Sharded artifacts should be optional and recorded in the dataset
manifest.

Manifest shape:

```json
{
  "artifacts": {
    "records_msgpack": "data/datasets/.../records.msgpack",
    "records_msgpack_sha256": "...",
    "records_shards": [
      {
        "path": "data/datasets/.../records-00000.msgpack",
        "sha256": "...",
        "records": 62500,
        "first_record_index": 1,
        "last_record_index": 62500
      }
    ]
  }
}
```

`first_record_index` and `last_record_index` are one-based indexes in the
global prepared record stream. They make shard provenance auditable and let the
runner compute stable global batch indexes.

## Shard Creation

Two entry points are acceptable:

- `dataset prepare --shards N`: write sharded artifacts while streaming the
  source dataset.
- `dataset optimize --shards N`: split an existing prepared artifact into
  shards without re-downloading the source.

The first implementation should prefer `dataset optimize --shards N` because it
is less disruptive and works with already prepared 1M/10M datasets.

Shard assignment should be contiguous by record order:

```text
shard 0: records [1, 62500]
shard 1: records [62501, 125000]
...
```

Contiguous shards preserve deterministic ordering and make checkpoint reasoning
simpler than round-robin sharding.

## Runner Configuration

Add optional load settings:

```yaml
load:
  sharded_records: true
  shard_count: 16
```

Proposed semantics:

- `sharded_records: false` or missing: use the existing parent-reader path.
- `sharded_records: true`: require `records_shards` in the dataset manifest.
- `shard_count`: optional assertion. If set, the manifest must contain exactly
  this many record shards.

`load.processes` remains the number of worker processes. It does not have to
equal shard count:

- If `processes < shards`, each process owns multiple shards.
- If `processes == shards`, each process owns one shard.
- If `processes > shards`, extra processes should not be started.

For clarity, the CLI progress line should include both:

```text
progress: load: starting write_mode=upsert batch_size=500 concurrency=16 processes=8 sharded_records=true shards=16
```

## Worker-Owned Read Path

In the sharded path, each worker process should:

1. Receive shard descriptors, target config, write mode, and batching settings.
2. Open its assigned shard file locally.
3. Decode msgpack records in that process.
4. Build batches using the same `batch_size` and `max_batch_bytes` rules.
5. Call the adapter write method directly.
6. Emit compact event summaries back to the parent.

The parent process should no longer send record batches through the task queue.
It should only assign shard work and aggregate result events.

Result events should remain one JSON line per write batch, not one line per
shard. This keeps existing metrics and report logic mostly intact.

## Batch Indexing

Checkpoint/resume depends on stable batch indexes. The sharded path should
avoid using "arrival order" as the batch index because concurrent workers
finish out of order.

Use deterministic global batch indexes derived from shard metadata:

```text
global_batch_index = shard_batch_base + local_batch_index
```

`shard_batch_base` can be precomputed in the parent by estimating the number of
batches in each shard from its record count and the configured `batch_size`.

When `max_batch_bytes` is not set, this is exact:

```text
shard_batch_count = ceil(shard.records / batch_size)
```

When `max_batch_bytes` is set, batch count depends on record sizes and cannot be
known from record count alone. For the first implementation, use one of these
safe options:

- Disable sharded load with `max_batch_bytes` and return a clear config error.
- Add per-shard batch indexes during shard creation.

The first implementation should choose the config error. It is simpler and
avoids unstable resume behavior.

## Checkpoint Semantics

The existing load checkpoint records the highest contiguous successful batch
index. The sharded path can preserve that contract if global batch indexes are
stable.

Checkpoint context should include:

```json
{
  "record_source": "sharded_msgpack",
  "records_shards": [
    {"path": ".../records-00000.msgpack", "sha256": "...", "records": 62500}
  ],
  "write_mode": "upsert",
  "batch_size": 500,
  "max_batch_bytes": null
}
```

Resume behavior:

- Workers may skip batches whose global batch index is less than or equal to
  the checkpoint watermark.
- The parent continues to aggregate successful batch indexes and advances the
  highest contiguous watermark exactly as today.
- A failed run may still have successful batches beyond the contiguous
  watermark. Those are not considered resume-safe unless the checkpoint format
  is extended to track sparse successes per shard.

This preserves the current conservative resume behavior.

## Visibility Samples

The current runner captures a small sample of loaded records for post-load query
visibility checks. With worker-owned readers, the parent does not naturally see
record objects.

Workers should include a small bounded sample from their first successful
batches in result messages. The parent aggregates until `visibility_sample_size`
is reached.

This keeps visibility checks working without sending every loaded record through
the parent.

## Metrics

Existing load metrics should keep the same meaning:

- `records`: successfully written records.
- `records_read`: records decoded by workers.
- `batches`: successful write batches.
- `attempts`: attempted write batches.
- `duration_seconds`: end-to-end load wall time.
- `records_per_second`: successful records over wall time.
- `upsert_attempt_duration_seconds`: sum of write attempt latencies.

Add sharded diagnostics:

```json
{
  "record_source": {
    "format": "msgpack",
    "path": null,
    "sharded": true,
    "shards": 16
  },
  "sharded_records": true,
  "shard_count": 16,
  "worker_shards": [2, 2, 2, 2, 2, 2, 2, 2]
}
```

The report generator can use these fields to distinguish single-reader and
sharded-reader runs.

## Failure Behavior

If one worker reports an error:

- The parent should stop assigning new shard work.
- Already running write attempts may finish and emit events.
- The load stage should finish with `status: failed`.
- The checkpoint should preserve the highest contiguous successful batch index.

This matches the existing concurrent load behavior.

## Implementation Plan

1. Add dataset shard metadata models and manifest validation.
2. Add `ldbbench dataset optimize --shards N` to split existing msgpack records.
3. Add load config parsing for `sharded_records` and optional `shard_count`.
4. Implement a new sharded load execution path gated by `sharded_records`.
5. Keep the current single-reader path unchanged for default runs.
6. Add unit tests for shard manifest parsing, sharded batching, global batch
   indexes, checkpoint context, resume skipping, and visibility samples.
7. Run local smoke tests with a tiny sharded dataset.
8. Run a real 100K/1M load comparison:
   - single-reader path
   - sharded path with `processes == CPU cores`
   - sharded path with more shards than processes

## Initial Limitations

- `max_batch_bytes` should be unsupported in the first sharded load
  implementation unless shard-time batch indexes are added.
- Sharded load should require msgpack record shards.
- Query records remain single-file for now.
- Exact record ordering in `ingest_events.jsonl` will follow event completion
  order, not global batch order. Each event must include its deterministic
  `batch_index`.

## Open Questions

- Should shard files replace `records.msgpack` for very large datasets, or
  remain additional artifacts only?
- Should `dataset prepare` write shards directly once `dataset optimize
  --shards` is validated?
- Should the runner support per-shard resume state later to avoid re-writing
  successful non-contiguous batches after a partial failure?
- Should bulk-upsert workloads prefer larger shard-local batch sizes than
  regular upsert workloads?

## Initial Implementation Status

Implemented:

- `ldbbench dataset optimize --shards N` writes `records-00000.msgpack` style
  shard artifacts and records them under `artifacts.records_shards`.
- `load.sharded_records: true` enables the sharded load path.
- `load.shard_count` can assert the expected manifest shard count.
- Worker processes read shard files directly, decode records, batch locally,
  and send only write event summaries plus bounded visibility samples to the
  parent process.
- Global batch indexes are deterministic for the first implementation because
  sharded load rejects `max_batch_bytes`.
- The load summary records `sharded_records`, `shard_count`, `worker_shards`,
  `effective_shard_count`, `manifest_shard_count`, and a sharded
  `record_source`.

Still intentionally deferred:

- `max_batch_bytes` support for sharded load.
- Query sharding.
- Per-shard sparse resume state.
- Direct shard writing during `dataset prepare`.

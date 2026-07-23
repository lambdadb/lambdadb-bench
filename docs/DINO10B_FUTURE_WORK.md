# DINO10B Dataset Future Work

This document records DINO10B as a future high-dimensional, billion-scale
dataset candidate for `lambdadb-bench`. It is a design note, not an implemented
dataset source or a committed benchmark result.

## Status

- Candidate only.
- No DINO10B download, conversion, scenario, or adapter integration is
  implemented yet.
- LambdaDB byte-vector support is the preferred prerequisite for a practical
  billion-scale run.
- The dataset is useful for high-dimensional ANN, ingest, delete, compaction,
  and recall testing. It must not be presented as a text-RAG embedding workload.

## Published Dataset

Meta's Faiss project added DINO10B in Faiss 1.13.1. The published dataset
contains:

- 10 billion 1024-dimensional image-patch vectors.
- Features extracted from YFCC100M images with
  `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Unsigned byte vectors stored in `.bvecs` files.
- L2 distance.
- 100,000 test queries.
- 99 million training vectors.
- Top-10 ground truth for these supported prefix sizes:
  `100K`, `200K`, `500K`, `1M`, `2M`, `5M`, `10M`, `20M`, `50M`, `100M`,
  `200M`, `500M`, `1B`, `2B`, `5B`, and `10B`.

The base corpus is split into 50 chunks. Each chunk contains 200 million
vectors and is approximately 200 GB. The full download is approximately 10 TB.

Primary references:

- Faiss integration:
  https://github.com/facebookresearch/faiss/pull/4686
- Download and usage instructions:
  https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/README.md
- Published file inventory:
  https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B/file_list.txt

Before an implementation or public run, recheck that the download URLs,
release terms, and all required artifacts are still available.

## Why It Is Interesting

The current Cohere Wikipedia workload is semantically closer to text retrieval,
but it does not provide a ready-made 1B-vector test at approximately 1024
dimensions.

DINO10B provides a useful complementary workload:

- Its 1024 dimensions are much closer to current production embedding sizes
  than classic 96-to-200-dimensional billion-scale ANN datasets.
- It is large enough to test 1B and multi-billion-vector storage and lifecycle
  behavior.
- It supplies queries and scale-specific ground truth instead of requiring the
  benchmark to compute 1B-scale exact neighbors from scratch.
- Its prefix-sized ground truths can support some deterministic shrink
  experiments.

The limitation is equally important: these are image-patch descriptors, not
modern text embeddings. Results may support claims about vector-system scale
and high-dimensional ANN behavior, but not claims about text-RAG semantic
quality.

## Storage Estimate

The following values cover vector payload only and exclude document IDs,
metadata, manifests, indexes, replication, and temporary build space.

| Base vectors | Published `uint8` payload | Expanded `float32` payload |
| ---: | ---: | ---: |
| 100M | approximately 100 GB | approximately 410 GB |
| 1B | approximately 1 TB | approximately 4.1 TB |
| 10B | approximately 10 TB | approximately 41 TB |

Byte-vector support therefore changes whether a 1B run is operationally
reasonable. A streaming float conversion avoids a 4.1 TB intermediate file,
but it does not avoid the four-times-larger LambdaDB ingest and stored vector
payload.

## Byte-Vector Compatibility

DINO10B `.bvecs` records contain a four-byte dimension header followed by 1024
unsigned bytes. Even with byte-vector support, ingestion must parse this framing,
assign document IDs, and serialize LambdaDB records. The expected improvement is
to avoid widening every component to `float32`, not to adopt `.bvecs` objects
without parsing.

Two LambdaDB representations are possible:

### Native `uint8`

If LambdaDB defines byte vectors as values in `[0, 255]` and scores L2 with
unsigned semantics, the DINO vector and query payloads can be preserved as-is
after removing `.bvecs` framing.

### Signed `int8`

If LambdaDB defines byte vectors as values in `[-128, 127]`, transform every
base and query component as:

```text
signed_value = unsigned_value - 128
```

For packed bytes this can be implemented as XOR with `0x80`. Applying the same
translation to both sides preserves squared L2 distance exactly:

```text
((u - 128) - (v - 128))^2 = (u - v)^2
```

The published L2 ground truth therefore remains valid after this translation.
The same statement does not hold for cosine similarity or dot product. The
initial DINO10B workload should be Euclidean-only.

Practical byte-vector support must be end-to-end:

- Collection field schema and element type.
- Ingest and bulk-ingest request representation.
- WAL, delta, and durable posting formats.
- Query-vector transport.
- Exact and approximate scorers.
- Snapshot manifest encoding contract.
- Benchmark dataset, record-cache, and adapter paths.

Adding a byte representation only inside the final index would still leave
float conversion and expanded transport on the ingest path.

## Metadata and Partitioning Limitation

The published DINO10B package does not include record metadata. In particular,
it does not expose:

- Original YFCC image ID or a patch-to-image mapping.
- User, timestamp, geography, URL, tag, or category fields.
- A tenant-like or document-like grouping key.

The only directly available identifiers are the implicit global vector ordinal
and the physical chunk number. Neither is a realistic application partition
key.

For a pure physical partition-scaling experiment, the benchmark can generate a
deterministic synthetic key:

```text
seed = UTF8(record_id + ":" + global_vector_ordinal)
digest = SHA256(seed)
partition_key = UINT64_BE(digest[0:8]) % partition_count
```

This matches the stable hashing approach already used by the benchmark's
synthetic filter buckets and avoids implementation-defined process-local hash
behavior. It is preferable to using `chunk_id`, because chunk-based assignment
is correlated with source order, prefix-sized ground truth, and sequential
deletion. The report must label the key as synthetic and must not interpret the
result as a realistic tenant, category, or time-partition workload.

A metadata-filtered or semantically partitioned DINO10B workload requires an
authoritative vector-to-source mapping that is not present in the published
package. Do not infer such a mapping from ordinal arithmetic without a
documented generation contract.

## Delete and Recall Workloads

The supplied ground truth is top-10 and is computed for specific prefix sizes.
This creates two different post-delete cases.

### Prefix-Preserving Shrink

If a run loads a supported prefix and deletes only its tail so that the
survivors are another supported prefix, the corresponding supplied ground truth
can be reused.

Example:

```text
load first 2B vectors
delete ordinal range [1B, 2B)
search the remaining first 1B vectors
use gts_dino_patch_1000000000_k10.npy
```

This is the preferred first DINO10B delete/recall experiment because it avoids
1B-scale ground-truth regeneration.

The workload must explicitly implement tail deletion. A generic "sequential"
delete mode that removes the first records would leave a suffix, not the prefix
represented by the supplied ground truth.

### Random Or Arbitrary Deletion

The supplied prefix ground truth does not describe an arbitrary survivor set.
Filtering deleted IDs from top-10 ground truth is also insufficient: deleted
neighbors may need to be replaced by candidates outside the original top 10.

Random-delete recall therefore requires one of:

- Exact ground-truth regeneration over the survivor set.
- A separately published or generated deeper candidate set with a proven
  sufficiency rule.
- A smaller dataset size at which exact regeneration is operationally
  acceptable.

Do not report random-delete recall from a merely filtered top-10 list.

## Proposed Delivery Sequence

1. Define LambdaDB byte-vector semantics:
   `uint8` versus signed `int8`, Euclidean scoring, query transport, and
   manifest encoding.
2. Add a streaming `.bvecs` dataset reader and byte-capable prepared artifact
   format without materializing float-expanded vectors.
3. Validate `100K`, `1M`, and `10M` subsets against the supplied top-10 ground
   truth.
4. Add a DINO10B Euclidean scenario and make its non-text semantics explicit in
   run manifests and reports.
5. Run a storage and ingest validation at 100M before committing resources to
   1B.
6. Run a 1B baseline using the first five chunks and the supplied 1B ground
   truth.
7. Add a prefix-preserving tail-delete experiment, initially at a smaller
   supported pair such as `10M -> 5M`, before attempting `2B -> 1B`.
8. Treat synthetic partitioning and arbitrary-delete ground-truth generation as
   separate follow-up work.

## Open Questions

- Should LambdaDB expose native `uint8`, signed `int8`, or an explicit
  byte-vector encoding with a zero point?
- Can the bulk API carry packed vector bytes, or would JSON/base64 overhead
  dominate ingest?
- Should byte-vector collections be Euclidean-only initially?
- How should prepared-artifact manifests distinguish source dtype, stored dtype,
  signedness, and any zero-point transform?
- What checksum and resumable-download strategy should be required for
  200-GB chunks?
- What temporary space and object-store replication factor are required for a
  1B build?
- Is top-10 ground truth sufficient for the intended recall report, or should a
  deeper exact artifact be generated at smaller scales?
- Can Meta provide an authoritative patch-to-YFCC metadata mapping for future
  filtered or document-grouped workloads?

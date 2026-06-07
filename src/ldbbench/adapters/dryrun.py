"""Dry-run adapters used before real database SDKs are wired."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ldbbench.adapters.base import (
    AdapterCapabilities,
    CheckResult,
    PrepareResult,
    QueryResult,
    UpsertResult,
    VectorRecord,
)
from ldbbench.config import TargetConfig


@dataclass(frozen=True)
class StaticAdapter:
    vendor: str
    capabilities: AdapterCapabilities

    def check(self, target: TargetConfig) -> CheckResult:
        if target.vendor != self.vendor:
            return CheckResult(
                ok=False,
                message=f"adapter {self.vendor} cannot check target {target.vendor}",
            )
        return CheckResult(
            ok=True,
            message="target metadata is valid for dry-run checks",
            details={"vendor": target.vendor, "target": target.name},
        )

    def prepare(
        self,
        target: TargetConfig,
        *,
        dimensions: int | None = None,
        metric: str | None = None,
    ) -> PrepareResult:
        raise NotImplementedError("dry-run adapters do not prepare real targets")

    def upsert_batch(
        self,
        target: TargetConfig,
        records: list[dict[str, Any] | VectorRecord],
        *,
        write_mode: str = "upsert",
    ) -> UpsertResult:
        raise NotImplementedError("dry-run adapters do not upsert records")

    def query(
        self,
        target: TargetConfig,
        *,
        vector: list[float],
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        filter_query: dict[str, Any] | None = None,
        partition_filter: dict[str, Any] | None = None,
    ) -> QueryResult:
        raise NotImplementedError("dry-run adapters do not query real targets")

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: list[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("dry-run adapters do not fetch real documents")


LAMBDADB_DRYRUN = StaticAdapter(
    vendor="lambdadb",
    capabilities=AdapterCapabilities(
        supported_write_modes=frozenset({"upsert", "bulk_upsert"}),
        supported_query_consistency=frozenset({"eventual", "strong"}),
        supports_read_after_write_strong=True,
        supports_query_filter=True,
        supports_query_partition_filter=True,
        vendor_consistency_options={"consistent_read": True},
    ),
)

QDRANT_DRYRUN = StaticAdapter(
    vendor="qdrant",
    capabilities=AdapterCapabilities(
        supported_write_modes=frozenset({"upsert"}),
        supported_query_consistency=frozenset({"eventual"}),
        supports_read_after_write_strong=False,
        supports_query_filter=True,
        supports_query_partition_filter=False,
        vendor_consistency_options={
            "read_consistency": ["all", "majority", "quorum"],
            "write_ordering": ["weak", "medium", "strong"],
            "default_protocol": "grpc",
        },
    ),
)

PINECONE_DRYRUN = StaticAdapter(
    vendor="pinecone",
    capabilities=AdapterCapabilities(
        supported_write_modes=frozenset({"upsert"}),
        supported_query_consistency=frozenset({"eventual"}),
        supports_read_after_write_strong=False,
        supports_query_filter=True,
        supports_query_partition_filter=False,
        vendor_consistency_options={"data_freshness_model": "eventual"},
    ),
)

DRYRUN_ADAPTERS = {
    adapter.vendor: adapter
    for adapter in [LAMBDADB_DRYRUN, QDRANT_DRYRUN, PINECONE_DRYRUN]
}

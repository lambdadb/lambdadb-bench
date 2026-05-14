"""Adapter contracts shared by database implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ldbbench.config import TargetConfig


@dataclass(frozen=True)
class AdapterCapabilities:
    supported_write_modes: frozenset[str]
    supported_query_consistency: frozenset[str]
    supports_read_after_write_strong: bool = False
    supports_query_partition_filter: bool = False
    supported_prepare_modes: frozenset[str] = frozenset(
        {"existing", "create", "recreate"}
    )
    vendor_consistency_options: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "supported_write_modes": sorted(self.supported_write_modes),
            "supported_query_consistency": sorted(self.supported_query_consistency),
            "supports_read_after_write_strong": self.supports_read_after_write_strong,
            "supports_query_partition_filter": self.supports_query_partition_filter,
            "supported_prepare_modes": sorted(self.supported_prepare_modes),
            "vendor_consistency_options": self.vendor_consistency_options,
        }


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareResult:
    ok: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    estimated_size_bytes: int | None = None


@dataclass(frozen=True)
class UpsertResult:
    count: int
    raw_response: object | None = None


@dataclass(frozen=True)
class QueryMatch:
    id: str
    score: float | None
    document: Mapping[str, Any]


@dataclass(frozen=True)
class QueryResult:
    matches: list[QueryMatch]
    raw_response: object | None = None


class VectorDBAdapter(Protocol):
    vendor: str
    capabilities: AdapterCapabilities

    def check(self, target: TargetConfig) -> CheckResult:
        """Validate target metadata without running a benchmark."""

    def prepare(
        self,
        target: TargetConfig,
        *,
        dimensions: int | None = None,
        metric: str | None = None,
    ) -> PrepareResult:
        """Prepare the target collection/index for loading."""

    def upsert_batch(
        self,
        target: TargetConfig,
        records: Sequence[Mapping[str, Any] | VectorRecord],
    ) -> UpsertResult:
        """Upsert a batch of normalized benchmark records."""

    def query(
        self,
        target: TargetConfig,
        *,
        vector: Sequence[float],
        top_k: int,
        consistency: str,
        include_vectors: bool = False,
        filter_query: Mapping[str, Any] | None = None,
        partition_filter: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        """Run one vector query against the target."""

    def fetch(
        self,
        target: TargetConfig,
        *,
        ids: Sequence[str],
        consistency: str,
        include_vectors: bool = False,
    ) -> Sequence[Mapping[str, Any]]:
        """Fetch documents by ID from the target."""

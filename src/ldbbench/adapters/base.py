"""Adapter contracts shared by database implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ldbbench.config import TargetConfig


@dataclass(frozen=True)
class AdapterCapabilities:
    supported_write_modes: frozenset[str]
    supported_query_consistency: frozenset[str]
    supports_read_after_write_strong: bool = False
    supported_prepare_modes: frozenset[str] = frozenset(
        {"existing", "create", "recreate"}
    )
    vendor_consistency_options: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "supported_write_modes": sorted(self.supported_write_modes),
            "supported_query_consistency": sorted(self.supported_query_consistency),
            "supports_read_after_write_strong": self.supports_read_after_write_strong,
            "supported_prepare_modes": sorted(self.supported_prepare_modes),
            "vendor_consistency_options": self.vendor_consistency_options,
        }


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)


class VectorDBAdapter(Protocol):
    vendor: str
    capabilities: AdapterCapabilities

    def check(self, target: TargetConfig) -> CheckResult:
        """Validate target metadata without running a benchmark."""

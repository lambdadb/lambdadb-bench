"""Dry-run execution planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ldbbench.adapters.base import AdapterCapabilities
from ldbbench.config import ScenarioConfig, TargetConfig

PlanStatus = Literal["supported", "partial", "unsupported"]


@dataclass(frozen=True)
class RunPlan:
    status: PlanStatus
    scenario_name: str
    target_name: str
    vendor: str
    write_mode: str
    query_consistency: str
    prepare_mode: str
    capabilities: AdapterCapabilities
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)

    @property
    def can_run(self) -> bool:
        return self.status != "unsupported"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "scenario_name": self.scenario_name,
            "target_name": self.target_name,
            "vendor": self.vendor,
            "write_mode": self.write_mode,
            "query_consistency": self.query_consistency,
            "prepare_mode": self.prepare_mode,
            "capabilities": self.capabilities.as_dict(),
            "warnings": self.warnings,
            "unsupported": self.unsupported,
            "not_applicable": self.not_applicable,
        }


def build_run_plan(
    *,
    scenario: ScenarioConfig,
    target: TargetConfig,
    capabilities: AdapterCapabilities,
    allow_destructive: bool = False,
) -> RunPlan:
    write_mode = str(scenario.load.get("write_mode"))
    query_consistency = str(scenario.query.get("consistency", "eventual"))
    partition_filter_requested = scenario.query.get("partition_filter") is not None
    unsupported: list[str] = []
    not_applicable: list[str] = []
    warnings: list[str] = []

    if write_mode not in capabilities.supported_write_modes:
        unsupported.append(
            f"write_mode {write_mode!r} is not supported by {target.vendor}"
        )

    if query_consistency not in capabilities.supported_query_consistency:
        if query_consistency == "strong":
            not_applicable.append(
                f"query consistency 'strong' is N/A for {target.vendor}: "
                "no comparable read-after-write strong guarantee is declared"
            )
        else:
            unsupported.append(
                f"query consistency {query_consistency!r} is not supported by "
                f"{target.vendor}"
            )

    if target.prepare_mode not in capabilities.supported_prepare_modes:
        unsupported.append(
            f"prepare mode {target.prepare_mode!r} is not supported by {target.vendor}"
        )

    if target.prepare_mode == "recreate" and not allow_destructive:
        unsupported.append("prepare mode 'recreate' requires --allow-destructive")

    if partition_filter_requested and not capabilities.supports_query_partition_filter:
        not_applicable.append(
            f"query partition_filter is N/A for {target.vendor}: "
            "no equivalent physical partition pruning support is declared"
        )

    if not target.endpoint:
        warnings.append(
            "target endpoint is not set; real adapter checks may fail later"
        )

    if unsupported:
        status: PlanStatus = "unsupported"
    elif not_applicable:
        status = "partial"
    else:
        status = "supported"

    return RunPlan(
        status=status,
        scenario_name=scenario.name,
        target_name=target.name,
        vendor=target.vendor,
        write_mode=write_mode,
        query_consistency=query_consistency,
        prepare_mode=target.prepare_mode,
        capabilities=capabilities,
        warnings=warnings,
        unsupported=unsupported,
        not_applicable=not_applicable,
    )

"""Run manifest creation."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ldbbench.__about__ import __version__
from ldbbench.config import (
    ScenarioConfig,
    TargetConfig,
    dump_yaml,
    redact_target_config,
)


@dataclass(frozen=True)
class ManifestPaths:
    run_manifest: Path
    scenario_resolved: Path
    target_redacted: Path


def initialize_run_artifacts(
    *,
    scenario: ScenarioConfig,
    target: TargetConfig,
    scenario_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
) -> ManifestPaths:
    """Write the initial reproducibility artifacts for a benchmark run."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenario_resolved = out / "scenario.resolved.yaml"
    target_redacted = out / "target.redacted.yaml"
    run_manifest = out / "run_manifest.json"

    dump_yaml(scenario.raw, scenario_resolved)
    redacted_target = redact_target_config(target)
    dump_yaml(redacted_target, target_redacted)

    manifest = build_run_manifest(
        scenario=scenario,
        target=target,
        scenario_path=scenario_path,
        target_path=target_path,
        redacted_target=redacted_target,
    )
    run_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ManifestPaths(
        run_manifest=run_manifest,
        scenario_resolved=scenario_resolved,
        target_redacted=target_redacted,
    )


def build_run_manifest(
    *,
    scenario: ScenarioConfig,
    target: TargetConfig,
    scenario_path: str | Path,
    target_path: str | Path,
    redacted_target: dict[str, Any],
) -> dict[str, Any]:
    scenario_file = Path(scenario_path)
    target_file = Path(target_path)
    endpoint_redacted = redacted_target.get("endpoint")
    metadata = target.metadata

    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {
            "name": "lambdadb-bench",
            "version": __version__,
        },
        "scenario": {
            "name": scenario.name,
            "path": str(scenario_file),
            "sha256": sha256_file(scenario_file),
            "dataset": {
                "source": scenario.dataset.get("source"),
                "subset": scenario.dataset.get("subset"),
                "rows": scenario.dataset.get("rows"),
                "dimensions": scenario.dataset.get("dimensions"),
                "seed": scenario.dataset.get("seed"),
            },
            "query": {
                "consistency": scenario.query.get("consistency", "eventual"),
                "top_k": scenario.query.get("top_k"),
            },
            "load": {
                "write_mode": scenario.load.get("write_mode"),
            },
        },
        "target": {
            "vendor": target.vendor,
            "name": target.name,
            "report_label": metadata.get("report_label", target.name),
            "path": str(target_file),
            "redacted_sha256": sha256_mapping(redacted_target),
            "endpoint": endpoint_redacted,
            "region": target.region,
            "prepare_mode": target.prepare_mode,
            "deployment_mode": metadata.get("deployment_mode"),
            "user_declared_config": metadata.get("user_declared_config"),
            "pricing_notes": metadata.get("pricing_notes"),
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(data: dict[str, Any]) -> str:
    encoded = yaml.safe_dump(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


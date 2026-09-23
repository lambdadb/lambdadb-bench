"""Run manifest creation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import yaml

from ldbbench.__about__ import __version__
from ldbbench.config import (
    ScenarioConfig,
    TargetConfig,
    dump_yaml,
    redact_target_config,
)

TOOL_DISTRIBUTION = "lambdadb-bench"


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
    sdk_package: str | None = None,
    adapter_capabilities: dict[str, object] | None = None,
    dry_run_plan: dict[str, object] | None = None,
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
        sdk_package=sdk_package,
        adapter_capabilities=adapter_capabilities,
        dry_run_plan=dry_run_plan,
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
    sdk_package: str | None = None,
    adapter_capabilities: dict[str, object] | None = None,
    dry_run_plan: dict[str, object] | None = None,
) -> dict[str, Any]:
    scenario_file = Path(scenario_path)
    target_file = Path(target_path)
    endpoint_redacted = redacted_target.get("endpoint")
    metadata = target.metadata
    git_commit, git_dirty = tool_git_state(_tool_direct_url())

    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "tool": {
            "name": TOOL_DISTRIBUTION,
            "version": __version__,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "sdk_package": sdk_package,
            "sdk_version": _installed_version(sdk_package) if sdk_package else None,
        },
        "scenario": {
            "name": scenario.name,
            "workload": scenario.workload,
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
                "partition_filter": scenario.query.get("partition_filter"),
            },
            "load": {
                "write_mode": scenario.load.get("write_mode"),
            },
            "search_under_ingest": scenario.search_under_ingest,
            "delete": scenario.delete,
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
            "partition_config": target.partition_config,
            "deployment_mode": metadata.get("deployment_mode"),
            "user_declared_config": metadata.get("user_declared_config"),
            "pricing_notes": metadata.get("pricing_notes"),
            "adapter_capabilities": adapter_capabilities or {},
        },
        "dry_run_plan": dry_run_plan,
    }


def tool_git_state(
    direct_url: Mapping[str, Any] | None,
) -> tuple[str | None, bool | None]:
    """Return the tool's source commit and dirty flag from PEP 610 install data.

    A git install is built from a clean checkout of the recorded commit. An
    editable install runs the checkout in place, so its git state is read live.
    """

    if direct_url is None:
        return None, None
    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, Mapping):
        commit = vcs_info.get("commit_id")
        if vcs_info.get("vcs") == "git" and isinstance(commit, str):
            return commit, False
        return None, None
    dir_info = direct_url.get("dir_info")
    url = direct_url.get("url")
    if isinstance(dir_info, Mapping) and dir_info.get("editable") and url:
        return _checkout_git_state(Path(url2pathname(urlparse(str(url)).path)))
    return None, None


def _checkout_git_state(checkout: Path) -> tuple[str | None, bool | None]:
    if not (checkout / ".git").exists():
        return None, None
    try:
        commit = _git(checkout, "rev-parse", "HEAD")
        changes = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(changes)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tool_direct_url() -> dict[str, Any] | None:
    try:
        distribution = importlib_metadata.distribution(TOOL_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    return json.loads(text) if text else None


def _installed_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_mapping(data: dict[str, Any]) -> str:
    encoded = yaml.safe_dump(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

"""Scenario and target configuration loading."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SECRET_KEY_PARTS = ("api_key", "apikey", "secret", "token", "password", "credential")
VALID_WRITE_MODES = {"upsert", "bulk_upsert"}
VALID_QUERY_CONSISTENCY = {"eventual", "strong"}
VALID_PREPARE_MODES = {"existing", "create", "recreate"}
VALID_WORKLOADS = {"standard", "search_under_ingest"}
VALID_SEARCH_UNDER_INGEST_PATTERNS = {"upload_and_ask", "parallel_upsert_query"}
VALID_SEARCH_UNDER_INGEST_PROBE_SOURCES = {"queries"}


class ConfigError(ValueError):
    """Raised when a benchmark configuration file is invalid."""


@dataclass(frozen=True)
class ScenarioConfig:
    """Validated benchmark scenario configuration."""

    name: str
    dataset: dict[str, Any]
    load: dict[str, Any]
    query: dict[str, Any]
    description: str | None = None
    workload: str = "standard"
    search_under_ingest: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ScenarioConfig:
        raw = _as_dict(data, "scenario")
        name = _required_str(raw, "name")
        dataset = _required_mapping(raw, "dataset")
        load = _required_mapping(raw, "load")
        query = _required_mapping(raw, "query")
        workload = raw.get("workload", "standard")
        if workload not in VALID_WORKLOADS:
            raise ConfigError(
                "scenario.workload must be one of "
                f"{sorted(VALID_WORKLOADS)}"
            )
        search_under_ingest = _optional_mapping(raw, "search_under_ingest")

        _validate_positive_int(dataset, "rows")
        _validate_positive_int(dataset, "dimensions")

        write_mode = load.get("write_mode")
        if write_mode not in VALID_WRITE_MODES:
            raise ConfigError(
                "scenario.load.write_mode must be one of "
                f"{sorted(VALID_WRITE_MODES)}"
            )
        if workload == "search_under_ingest" and write_mode != "upsert":
            raise ConfigError(
                "scenario.load.write_mode must be 'upsert' for "
                "workload 'search_under_ingest'"
            )
        _validate_optional_positive_int(load, "concurrency")
        _validate_optional_positive_int(load, "processes")
        _validate_optional_bool(load, "sharded_records")
        _validate_optional_positive_int(load, "shard_count")

        consistency = query.get("consistency", "eventual")
        if consistency not in VALID_QUERY_CONSISTENCY:
            raise ConfigError(
                "scenario.query.consistency must be one of "
                f"{sorted(VALID_QUERY_CONSISTENCY)}"
            )
        _validate_optional_positive_int(query, "processes")
        _validate_partition_filter(query)
        _validate_search_under_ingest(
            search_under_ingest,
            workload=str(workload),
            default_consistency=str(consistency),
        )

        stages = query.get("stages", [])
        if stages is not None:
            if not isinstance(stages, list):
                raise ConfigError("scenario.query.stages must be a list")
            for index, stage in enumerate(stages):
                stage_dict = _as_dict(stage, f"scenario.query.stages[{index}]")
                _validate_positive_int(stage_dict, "concurrency")
                if not isinstance(stage_dict.get("duration"), str):
                    raise ConfigError(
                        f"scenario.query.stages[{index}].duration must be a string"
                    )

        return cls(
            name=name,
            description=_optional_str(raw, "description"),
            dataset=dataset,
            load=load,
            query=query,
            workload=str(workload),
            search_under_ingest=search_under_ingest,
            quality=_optional_mapping(raw, "quality"),
            metrics=_optional_mapping(raw, "metrics"),
            raw=raw,
        )


@dataclass(frozen=True)
class TargetConfig:
    """Validated target database configuration."""

    vendor: str
    name: str
    endpoint: str | None
    api_key_env: str | None
    collection_name: str | None
    project_name: str | None
    vector_field: str | None
    index_configs: dict[str, Any]
    partition_config: dict[str, Any] | None
    region: str | None
    prepare_mode: str
    metadata: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TargetConfig:
        raw = _as_dict(data, "target")
        vendor = _required_str(raw, "vendor")
        name = _required_str(raw, "name")
        endpoint = _optional_str(raw, "endpoint")
        api_key_env = _optional_str(raw, "api_key_env")
        collection_name = _collection_name(raw)
        project_name = _optional_str(raw, "project_name")
        vector_field = _optional_str(raw, "vector_field")
        index_configs = _optional_mapping(raw, "index_configs")
        partition_config = _optional_mapping_or_none(raw, "partition_config")
        region = _optional_str(raw, "region")
        prepare = _optional_mapping(raw, "prepare")
        prepare_mode = prepare.get("mode", "existing")
        if prepare_mode not in VALID_PREPARE_MODES:
            raise ConfigError(
                "target.prepare.mode must be one of "
                f"{sorted(VALID_PREPARE_MODES)}"
            )

        return cls(
            vendor=vendor,
            name=name,
            endpoint=endpoint,
            api_key_env=api_key_env,
            collection_name=collection_name,
            project_name=project_name,
            vector_field=vector_field,
            index_configs=index_configs,
            partition_config=partition_config,
            region=region,
            prepare_mode=prepare_mode,
            metadata=_optional_mapping(raw, "metadata"),
            raw=raw,
        )


def load_scenario(path: str | Path) -> ScenarioConfig:
    return ScenarioConfig.from_mapping(load_yaml(path))


def load_target(path: str | Path) -> TargetConfig:
    return TargetConfig.from_mapping(load_yaml(path))


def load_yaml(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load YAML and expand ${VAR} references in string values."""

    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML file {config_path}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")

    return expand_env(data, environ=environ)


def expand_env(value: Any, *, environ: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ${VAR} references in strings."""

    env = os.environ if environ is None else environ
    if isinstance(value, str):
        return _expand_env_string(value, env)
    if isinstance(value, list):
        return [expand_env(item, environ=env) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item, environ=env) for key, item in value.items()}
    return value


def redact_target_config(target: TargetConfig) -> dict[str, Any]:
    """Return a target config safe to write into public run artifacts."""

    return _redact_value(target.raw)


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False),
        encoding="utf-8",
    )


def _expand_env_string(value: str, environ: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return environ[name]
        except KeyError as exc:
            raise ConfigError(f"environment variable {name} is not set") from exc

    return ENV_PATTERN.sub(replace, value)


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_secret_key(key):
        return "<redacted>"
    if key == "endpoint" and isinstance(value, str):
        return _redact_endpoint(value)
    if isinstance(value, dict):
        return {
            item_key: _redact_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _redact_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, "<redacted-host>", parsed.path, "", ""))
    return "<redacted-host>"


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_env"):
        return False
    return any(part in normalized for part in SECRET_KEY_PARTS)


def _as_dict(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be a mapping")
    return dict(data)


def _required_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _collection_name(data: Mapping[str, Any]) -> str | None:
    collection = _optional_str(data, "collection")
    collection_name = _optional_str(data, "collection_name")
    if collection and collection_name and collection != collection_name:
        raise ConfigError("target collection and collection_name must match")
    return collection_name or collection


def _required_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _optional_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _optional_mapping_or_none(
    data: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _validate_partition_filter(query: Mapping[str, Any]) -> None:
    value = query.get("partition_filter")
    if value is None:
        return
    if not isinstance(value, dict):
        raise ConfigError("scenario.query.partition_filter must be a mapping")
    field = value.get("field")
    metadata_field = value.get("metadata_field")
    if not isinstance(field, str) or not field:
        raise ConfigError("scenario.query.partition_filter.field must be a string")
    if not isinstance(metadata_field, str) or not metadata_field:
        raise ConfigError(
            "scenario.query.partition_filter.metadata_field must be a string"
        )


def _validate_search_under_ingest(
    config: Mapping[str, Any],
    *,
    workload: str,
    default_consistency: str,
) -> None:
    if not config:
        if workload == "search_under_ingest":
            raise ConfigError(
                "scenario.search_under_ingest must be set for "
                "workload 'search_under_ingest'"
            )
        return

    pattern = config.get("pattern", "upload_and_ask")
    if pattern not in VALID_SEARCH_UNDER_INGEST_PATTERNS:
        raise ConfigError(
            "scenario.search_under_ingest.pattern must be one of "
            f"{sorted(VALID_SEARCH_UNDER_INGEST_PATTERNS)}"
        )

    probe_source = config.get("probe_source", "queries")
    if probe_source not in VALID_SEARCH_UNDER_INGEST_PROBE_SOURCES:
        raise ConfigError(
            "scenario.search_under_ingest.probe_source must be one of "
            f"{sorted(VALID_SEARCH_UNDER_INGEST_PROBE_SOURCES)}"
        )

    document_group_field = config.get("document_group_field", "url")
    if not isinstance(document_group_field, str) or not document_group_field:
        raise ConfigError(
            "scenario.search_under_ingest.document_group_field must be a string"
        )

    consistency = config.get("consistency", default_consistency)
    if consistency not in VALID_QUERY_CONSISTENCY:
        raise ConfigError(
            "scenario.search_under_ingest.consistency must be one of "
            f"{sorted(VALID_QUERY_CONSISTENCY)}"
        )

    duration = config.get("duration")
    if duration is not None and not isinstance(duration, str):
        raise ConfigError("scenario.search_under_ingest.duration must be a string")

    visibility_timeout = config.get("visibility_timeout")
    if visibility_timeout is not None and not isinstance(visibility_timeout, str):
        raise ConfigError(
            "scenario.search_under_ingest.visibility_timeout must be a string"
        )

    visibility_poll_interval = config.get("visibility_poll_interval")
    if visibility_poll_interval is not None and not isinstance(
        visibility_poll_interval,
        str,
    ):
        raise ConfigError(
            "scenario.search_under_ingest.visibility_poll_interval must be a string"
        )

    _validate_optional_positive_int(config, "max_probe_documents")
    _validate_optional_positive_int(config, "min_chunks_per_document")
    _validate_optional_positive_int(config, "max_chunks_per_document")
    _validate_optional_positive_int(config, "probe_queries_per_document")
    _validate_optional_positive_int(config, "probe_concurrency")
    _validate_optional_positive_int(config, "ingest_concurrency")
    _validate_optional_positive_int(config, "query_concurrency")
    _validate_optional_positive_int(config, "top_k")
    _validate_optional_bool(config, "poll_until_visible")

    min_chunks = config.get("min_chunks_per_document")
    max_chunks = config.get("max_chunks_per_document")
    if (
        isinstance(min_chunks, int)
        and isinstance(max_chunks, int)
        and max_chunks < min_chunks
    ):
        raise ConfigError(
            "scenario.search_under_ingest.max_chunks_per_document must be "
            "greater than or equal to min_chunks_per_document"
        )


def _validate_positive_int(data: Mapping[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{key} must be a positive integer")


def _validate_optional_positive_int(data: Mapping[str, Any], key: str) -> None:
    value = data.get(key)
    if value is not None and (not isinstance(value, int) or value <= 0):
        raise ConfigError(f"{key} must be a positive integer")


def _validate_optional_bool(data: Mapping[str, Any], key: str) -> None:
    value = data.get(key)
    if value is not None and not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")

"""Command line interface for lambdadb-bench."""

from __future__ import annotations

import argparse

from ldbbench.__about__ import __version__
from ldbbench.config import ConfigError, load_scenario, load_target
from ldbbench.manifest import initialize_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ldbbench",
        description=(
            "Reproducible benchmark harness for LambdaDB and comparable managed "
            "vector databases."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subcommands = parser.add_subparsers(dest="command")

    doctor = subcommands.add_parser(
        "doctor",
        help="Check the local CLI installation.",
    )
    doctor.set_defaults(func=run_doctor)

    config = subcommands.add_parser(
        "config",
        help="Validate benchmark configuration files.",
    )
    config_subcommands = config.add_subparsers(dest="config_command")
    validate = config_subcommands.add_parser(
        "validate",
        help="Validate a scenario and target config.",
    )
    validate.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    validate.add_argument("--target", required=True, help="Path to target YAML.")
    validate.set_defaults(func=run_config_validate)

    manifest = subcommands.add_parser(
        "manifest",
        help="Create run manifest artifacts.",
    )
    manifest_subcommands = manifest.add_subparsers(dest="manifest_command")
    init = manifest_subcommands.add_parser(
        "init",
        help="Initialize a result directory with reproducibility artifacts.",
    )
    init.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    init.add_argument("--target", required=True, help="Path to target YAML.")
    init.add_argument("--out", required=True, help="Output result directory.")
    init.set_defaults(func=run_manifest_init)

    return parser


def run_doctor(_args: argparse.Namespace) -> int:
    print(f"ldbbench {__version__}")
    print("status: ok")
    return 0


def run_config_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    print(f"scenario: {scenario.name}")
    print(f"target: {target.name} ({target.vendor})")
    print("status: ok")
    return 0


def run_manifest_init(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    target = load_target(args.target)
    paths = initialize_run_artifacts(
        scenario=scenario,
        target=target,
        scenario_path=args.scenario,
        target_path=args.target,
        output_dir=args.out,
    )
    print(f"wrote {paths.run_manifest}")
    print(f"wrote {paths.scenario_resolved}")
    print(f"wrote {paths.target_redacted}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except ConfigError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

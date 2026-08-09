"""Command-line interface for data, training, validation, and paper workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from sura import __version__
from sura.config import default_config_path, load_yaml
from sura.data.analysis import analyze_dataset, write_dataset_report
from sura.data.fingerprint import load_fingerprint_database
from sura.data.fingerprint_builder import build_fingerprint_database
from sura.data.layout import initialize_data_layout
from sura.data.migration import migrate_legacy_data
from sura.data.paths import (
    configured_data_path,
    data_root,
    fingerprint_database,
    paper_root,
    repository_root,
)
from sura.training.magnetic import train_magnetic_sequence
from sura.training.wifi import train_wifi_heatmap

COMMAND_SUMMARY = """Core commands
  sura doctor                         Check environment, data, and LaTeX tooling
  sura data init                      Create ignored raw/interim/processed/local folders
  sura data check                     Validate the processed fingerprint database
  sura data analyze                   Write compact dataset JSON and Markdown reports
  sura data migrate-legacy            Preview or apply migration from old Datasets/
  sura data build-fingerprint         Convert raw MagWi files to the processed database
  sura train wifi                     Train the Wi-Fi heatmap model
  sura train magnetic                 Train the standalone magnetic sequence CNN
  sura train all                      Train both canonical standalone models
  sura paper build                    Compile paper/main.tex
  sura commands                       Print this command list

Every command is also available as ``python -m sura ...``.
See COMMANDS.md for examples and all options.
"""


def command_commands(_args: argparse.Namespace) -> int:
    print(COMMAND_SUMMARY)
    return 0


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _dataset_configuration(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def _fingerprint_path(
    dataset_config_path: str | Path,
    explicit_data_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    configuration = _dataset_configuration(dataset_config_path)
    configured = configuration.get("fingerprint_db")
    if configured:
        path = configured_data_path(str(configured), explicit_data_root)
    else:
        building = str(configuration.get("building", "it_engineering"))
        path = fingerprint_database(building, explicit_data_root)
    return configuration, path


def command_doctor(args: argparse.Namespace) -> int:
    root = data_root(args.data_root)
    dataset_config, database_path = _fingerprint_path(
        args.dataset_config,
        args.data_root,
    )
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "pandas", "scipy", "torch", "matplotlib", "yaml", "xlrd")
    }
    checks: dict[str, Any] = {
        "version": __version__,
        "repository_root": str(repository_root()),
        "data_root": str(root),
        "data_directories": {
            name: (root / name).is_dir()
            for name in ("raw", "interim", "processed", "local")
        },
        "packages": packages,
        "latexmk": shutil.which("latexmk"),
        "fingerprint_database": str(database_path),
        "fingerprint_database_valid": False,
    }
    try:
        database = load_fingerprint_database(
            database_path,
            included_modes=dataset_config.get("included_modes"),
            require_wifi=True,
        )
        checks["fingerprint_database_valid"] = True
        checks["fingerprint_summary"] = database.summary()
    except (FileNotFoundError, ValueError) as error:
        checks["fingerprint_error"] = str(error)
    _json_print(checks)

    required_packages = all(packages.values())
    healthy = required_packages and checks["fingerprint_database_valid"]
    if args.strict and not healthy:
        return 1
    return 0


def command_data_init(args: argparse.Namespace) -> int:
    created = initialize_data_layout(args.data_root)
    _json_print({name: str(path) for name, path in created.items()})
    return 0


def command_data_migrate_legacy(args: argparse.Namespace) -> int:
    plan = migrate_legacy_data(
        legacy_directory=args.legacy_root,
        target_data_root=args.data_root,
        mode=args.mode,
        apply=args.apply,
    )
    _json_print(
        {
            "applied": bool(args.apply),
            "mode": args.mode,
            "items": plan,
        }
    )
    return 0


def command_data_check(args: argparse.Namespace) -> int:
    configuration, database_path = _fingerprint_path(
        args.dataset_config,
        args.data_root,
    )
    database = load_fingerprint_database(
        database_path,
        included_modes=configuration.get("included_modes"),
        require_wifi=not args.allow_missing_wifi,
    )
    _json_print(database.summary())
    return 0


def command_data_analyze(args: argparse.Namespace) -> int:
    configuration, database_path = _fingerprint_path(
        args.dataset_config,
        args.data_root,
    )
    report = analyze_dataset(
        data_directory=data_root(args.data_root),
        fingerprint_directory=database_path,
        included_modes=configuration.get("included_modes"),
    )
    if args.no_write:
        _json_print(report)
    else:
        output = args.output_dir or (
            repository_root() / "experiments" / "runs" / "dataset_analysis"
        )
        json_path, markdown_path = write_dataset_report(report, output)
        _json_print(
            {
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
                "fingerprint_database": report["fingerprint_database"],
            }
        )
    return 0


def command_build_fingerprint(args: argparse.Namespace) -> int:
    configuration = _dataset_configuration(args.dataset_config)
    building_slug = str(configuration.get("building", "it_engineering"))
    building_directory = args.building or str(
        configuration.get("building_directory", "IT Engineering")
    )
    wifi_building = args.wifi_building or str(
        configuration.get("wifi_building_directory", building_directory)
    )
    raw_relative = str(configuration.get("raw_dataset", "raw/magwi"))
    raw_root = configured_data_path(raw_relative, args.data_root)
    configured_output = configuration.get("fingerprint_db")
    output = (
        configured_data_path(str(configured_output), args.data_root)
        if configured_output
        else fingerprint_database(building_slug, args.data_root)
    )
    summary = build_fingerprint_database(
        raw_dataset_directory=raw_root,
        output_directory=output,
        building=building_directory,
        wifi_building=wifi_building,
        magnetic_static_subdirectory=str(
            configuration.get(
                "magnetic_static_subdirectory",
                "Magnetic field dataset/Static Data",
            )
        ),
        wifi_subdirectory=str(configuration.get("wifi_subdirectory", "WiFi dataset")),
        make_coverage_plot=not args.no_plot,
        dry_run=args.dry_run,
    )
    _json_print(summary.as_dict())
    return 0


def command_train_wifi(args: argparse.Namespace) -> int:
    result = train_wifi_heatmap(
        model_config_path=args.config,
        dataset_config_path=args.dataset_config,
        data_directory=args.data_root,
        output_directory=args.output_dir,
        split=args.split,
        device=args.device,
        run_name=args.run_name,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )
    _json_print(result)
    return 0


def command_train_magnetic(args: argparse.Namespace) -> int:
    result = train_magnetic_sequence(
        model_config_path=args.config,
        dataset_config_path=args.dataset_config,
        data_directory=args.data_root,
        output_directory=args.output_dir,
        device=args.device,
        run_name=args.run_name,
        epochs=args.epochs,
        sweep=args.sweep,
        dry_run=args.dry_run,
    )
    _json_print(result)
    return 0


def command_train_all(args: argparse.Namespace) -> int:
    base_name = args.run_name
    wifi = train_wifi_heatmap(
        model_config_path=args.wifi_config,
        dataset_config_path=args.dataset_config,
        data_directory=args.data_root,
        output_directory=args.output_dir,
        split="both",
        device=args.device,
        run_name=f"{base_name}-wifi" if base_name else None,
        epochs=args.wifi_epochs,
        dry_run=args.dry_run,
    )
    magnetic = train_magnetic_sequence(
        model_config_path=args.magnetic_config,
        dataset_config_path=args.dataset_config,
        data_directory=args.data_root,
        output_directory=args.output_dir,
        device=args.device,
        run_name=f"{base_name}-magnetic" if base_name else None,
        epochs=args.magnetic_epochs,
        sweep=args.magnetic_sweep,
        dry_run=args.dry_run,
    )
    _json_print({"wifi": wifi, "magnetic": magnetic})
    return 0


def command_paper_build(args: argparse.Namespace) -> int:
    executable = shutil.which("latexmk")
    if executable is None:
        print("latexmk is not installed or not on PATH", file=sys.stderr)
        return 2
    workspace = paper_root()
    build = workspace / "build"
    build.mkdir(parents=True, exist_ok=True)
    if args.clean:
        subprocess.run(
            [executable, "-C", f"-outdir={build}", "main.tex"],
            cwd=workspace,
            check=True,
        )
    subprocess.run(
        [
            executable,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build}",
            "main.tex",
        ],
        cwd=workspace,
        check=True,
    )
    print(build / "main.pdf")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sura",
        description="Reproducible workflows for the SURA indoor-localization project.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    commands = subcommands.add_parser("commands", help="print the command summary")
    commands.set_defaults(handler=command_commands)

    doctor = subcommands.add_parser("doctor", help="check environment and dataset readiness")
    doctor.add_argument("--data-root")
    doctor.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    data = subcommands.add_parser("data", help="initialize, validate, or analyze data")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    data_init = data_commands.add_parser("init", help="create the local data layout")
    data_init.add_argument("--data-root")
    data_init.set_defaults(handler=command_data_init)

    migrate = data_commands.add_parser(
        "migrate-legacy",
        help="preview or apply migration from the former Datasets/ directory",
    )
    migrate.add_argument("--legacy-root")
    migrate.add_argument("--data-root")
    migrate.add_argument("--mode", choices=("move", "copy"), default="move")
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="perform the migration; without this flag the command is a dry-run",
    )
    migrate.set_defaults(handler=command_data_migrate_legacy)

    data_check = data_commands.add_parser("check", help="validate processed fingerprints")
    data_check.add_argument("--data-root")
    data_check.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    data_check.add_argument("--allow-missing-wifi", action="store_true")
    data_check.set_defaults(handler=command_data_check)

    analyze = data_commands.add_parser("analyze", help="write a compact dataset report")
    analyze.add_argument("--data-root")
    analyze.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    analyze.add_argument("--output-dir")
    analyze.add_argument("--no-write", action="store_true")
    analyze.set_defaults(handler=command_data_analyze)

    build = data_commands.add_parser(
        "build-fingerprint",
        help="build nodes.csv and the AP vocabulary from raw MagWi files",
    )
    build.add_argument("--data-root")
    build.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    build.add_argument("--building")
    build.add_argument("--wifi-building")
    build.add_argument("--no-plot", action="store_true")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(handler=command_build_fingerprint)

    train = subcommands.add_parser("train", help="train canonical model stages")
    train_commands = train.add_subparsers(dest="train_command", required=True)

    wifi = train_commands.add_parser("wifi", help="train the Wi-Fi heatmap model")
    wifi.add_argument(
        "--config",
        default=str(default_config_path("wifi_heatmap.yaml")),
    )
    wifi.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    wifi.add_argument("--data-root")
    wifi.add_argument("--output-dir")
    wifi.add_argument("--split", choices=("random", "phone", "both"), default="both")
    wifi.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    wifi.add_argument("--run-name")
    wifi.add_argument("--epochs", type=int)
    wifi.add_argument("--dry-run", action="store_true")
    wifi.set_defaults(handler=command_train_wifi)

    magnetic = train_commands.add_parser(
        "magnetic",
        help="train the standalone magnetic sequence CNN",
    )
    magnetic.add_argument(
        "--config",
        default=str(default_config_path("magnetic_sequence.yaml")),
    )
    magnetic.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    magnetic.add_argument("--data-root")
    magnetic.add_argument("--output-dir")
    magnetic.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    magnetic.add_argument("--run-name")
    magnetic.add_argument("--epochs", type=int)
    magnetic.add_argument("--sweep", action="store_true")
    magnetic.add_argument("--dry-run", action="store_true")
    magnetic.set_defaults(handler=command_train_magnetic)

    train_all = train_commands.add_parser(
        "all",
        help="train the Wi-Fi and magnetic standalone models sequentially",
    )
    train_all.add_argument(
        "--wifi-config",
        default=str(default_config_path("wifi_heatmap.yaml")),
    )
    train_all.add_argument(
        "--magnetic-config",
        default=str(default_config_path("magnetic_sequence.yaml")),
    )
    train_all.add_argument(
        "--dataset-config",
        default=str(default_config_path("datasets/magwi_it_engineering.yaml")),
    )
    train_all.add_argument("--data-root")
    train_all.add_argument("--output-dir")
    train_all.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    train_all.add_argument("--run-name")
    train_all.add_argument("--wifi-epochs", type=int)
    train_all.add_argument("--magnetic-epochs", type=int)
    train_all.add_argument("--magnetic-sweep", action="store_true")
    train_all.add_argument("--dry-run", action="store_true")
    train_all.set_defaults(handler=command_train_all)

    paper = subcommands.add_parser("paper", help="compile the IEEE manuscript")
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)
    paper_build = paper_commands.add_parser("build", help="compile paper/main.tex")
    paper_build.add_argument("--clean", action="store_true")
    paper_build.set_defaults(handler=command_paper_build)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

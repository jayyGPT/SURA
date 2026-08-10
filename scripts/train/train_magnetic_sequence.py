#!/usr/bin/env python3
"""Train the standalone magnetic sequence CNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from sura.training.magnetic import train_magnetic_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "magnetic_sequence.yaml",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs"
            / "datasets"
            / "magwi_it_engineering.yaml"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--run-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = train_magnetic_sequence(
        model_config_path=args.config,
        dataset_config_path=args.dataset_config,
        data_directory=args.data_root,
        output_directory=args.output,
        device=args.device,
        run_name=args.run_name,
        epochs=args.epochs,
        sweep=args.sweep,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

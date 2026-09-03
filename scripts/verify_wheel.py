#!/usr/bin/env python3
"""Fail fast when an ArchCon wheel is missing runtime package files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

REQUIRED_MEMBERS = {
    "archcon/__init__.py",
    "archcon/__main__.py",
    "archcon/cli.py",
    "archcon/evaluate_egfr.py",
    "archcon/rebuild_global_normalization.py",
    "archcon/app.py",
    "archcon/batch.py",
    "archcon/data/__init__.py",
    "archcon/data/defaults.py",
    "archcon/data/geo_rma.py",
    "archcon/data/pretraining.py",
    "archcon/data/training.py",
    "archcon/data/training_sources.py",
    "archcon/data/downstream.py",
    "archcon/assets/__init__.py",
    "archcon/assets/rma_preprocess.R",
    "archcon/assets/molecular_mixed_models.R",
}


def verify_wheel(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Wheel does not exist: {path}")
    with ZipFile(path) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED_MEMBERS - names)
    if missing:
        formatted = "\n".join(f"  - {name}" for name in missing)
        raise SystemExit(f"Incomplete ArchCon wheel {path}:\n{formatted}")
    data_files = sorted(name for name in names if name.startswith("archcon/data/"))
    print(f"Wheel OK: {path}")
    print(f"archcon/data files: {len(data_files)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the already-complete IKEM local RMA into an isolated training overlay.

This does not modify the running rebuild or the production ``data/`` directory.
The overlay links the existing GEO store and eGFR table, while materializing the
new donor-safe IKEM per-study matrix from the rebuild HDF5 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROLE = "pretrain_train_no_egfr"
VALIDATION_ROLE = "pretrain_validation_no_egfr"
RELATED_ROLE = "held_out_related_to_measured_egfr"
MEASURED_ROLE = "held_out_measured_egfr"


def membership_sha256(namespace: str, sample_ids: list[str]) -> str:
    values = sorted(
        f"{namespace.upper()}:{str(sample_id).strip().upper()}"
        for sample_id in sample_ids
    )
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def link_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="GEO_DWNLD_TRAIN_REFERENCE_REBUILD directory",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        help="Default: PROJECT_ROOT/data-per-gse-ready",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    production = project / "data"
    work = args.work_root.expanduser().resolve()
    destination = (
        args.output_data_dir.expanduser().resolve()
        if args.output_data_dir
        else project / "data-per-gse-ready"
    )
    source_store = work / "IKEM_MATRIX_STORE"
    source_h5 = source_store / "ikem_expression_store.h5"
    complete_marker = source_store / "IKEM_LOCAL_RMA_COMPLETE.txt"
    if not source_h5.is_file() or not complete_marker.is_file():
        raise SystemExit(
            "IKEM local RMA is not certified complete; expected "
            f"{source_h5} and {complete_marker}."
        )

    try:
        import h5py
    except ImportError as error:
        raise SystemExit(
            "This one-time exporter requires h5py: python -m pip install 'h5py>=3.10,<4'"
        ) from error

    temporary = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    if destination.exists():
        if not args.force:
            raise SystemExit(f"Destination exists; use --force to replace it: {destination}")
        backup = destination.with_name(f"{destination.name}.previous")
        if backup.exists():
            shutil.rmtree(backup)
        destination.rename(backup)

    temporary.mkdir(parents=True)
    try:
        link_required(production / "GEO_NUMPY_STORE", temporary / "GEO_NUMPY_STORE")
        link_required(production / "egfr_data.xlsx", temporary / "egfr_data.xlsx")
        for optional in (
            "common_probes.pkl",
            "gene_annotations.csv",
            "sample_metadata.csv",
            "Klasifikator_20_3_24_v2.xlsx",
        ):
            source = production / optional
            if source.exists():
                (temporary / optional).symlink_to(source.resolve())

        output_store = temporary / "IKEM_CEL_NUMPY_STORE"
        output_store.mkdir()
        for name in (
            "sample_index.csv",
            "probe_index.csv",
            "ikem_cel_correspondence.csv",
            "ikem_rma_reference_samples.csv",
            "IKEM_LOCAL_RMA_COMPLETE.txt",
        ):
            source = source_store / name
            if source.is_file():
                shutil.copy2(source, output_store / name)

        samples = pd.read_csv(output_store / "sample_index.csv")
        probes = pd.read_csv(output_store / "probe_index.csv")
        expected_shape = (len(samples), len(probes))
        exported: dict[str, Path] = {}
        with h5py.File(source_h5, "r") as handle:
            for method in ("raw_original", "rma_per_gse"):
                dataset = handle[f"expression/{method}"]
                if tuple(dataset.shape) != expected_shape:
                    raise RuntimeError(
                        f"IKEM {method} HDF5 shape {dataset.shape} differs from "
                        f"{expected_shape}."
                    )
                output_path = output_store / f"{method}.npy"
                output = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape,
                )
                for start in range(0, expected_shape[0], 16):
                    stop = min(start + 16, expected_shape[0])
                    block = np.asarray(dataset[start:stop, :], dtype=np.float32)
                    if not np.isfinite(block).all():
                        raise RuntimeError(
                            f"Non-finite IKEM {method} values near row {start}."
                        )
                    output[start:stop] = block
                output.flush()
                del output
                exported[method] = output_path

        roles = samples["training_role"].astype(str)
        counts = roles.value_counts().to_dict()
        expected_counts = {
            TRAIN_ROLE: 24,
            VALIDATION_ROLE: 6,
            RELATED_ROLE: 4,
            MEASURED_ROLE: 254,
        }
        if counts != expected_counts:
            raise RuntimeError(f"Unexpected IKEM role counts: {counts}")
        train = samples.loc[roles.eq(TRAIN_ROLE)]
        validation = samples.loc[roles.eq(VALIDATION_ROLE)]
        provenance = {
            "format": 4,
            "cohort_samples": len(samples),
            "outcome_gate_unit": "donor",
            "split_unit": "donor",
            "split_seed": 20260915,
            "validation_fraction": 0.20,
            "no_measured_egfr_biopsies": 34,
            "donor_clean_pretraining_eligible_samples": 30,
            "pretraining_train_samples": 24,
            "pretraining_validation_samples": 6,
            "related_no_egfr_held_out_samples": 4,
            "held_out_measured_egfr_samples": 254,
            "pretraining_validation_donors": sorted(
                validation["donor_id"].astype(str).str.upper().unique().tolist()
            ),
            "ikem_train_sample_ids_sha256": membership_sha256(
                "SUPERVISED", train["sample_id"].astype(str).tolist()
            ),
            "ikem_validation_sample_ids_sha256": membership_sha256(
                "SUPERVISED", validation["sample_id"].astype(str).tolist()
            ),
            "interim_per_gse_overlay": True,
            "source_h5": str(source_h5),
            "methods": {
                "raw_original": {
                    "file": "raw_original.npy",
                    "sha256": sha256(exported["raw_original"]),
                    "shape": list(expected_shape),
                    "algorithm": (
                        "median of original PM CEL intensities within each common "
                        "probe set; standardization parameters are fitted later from "
                        "the frozen molecular-training partition"
                    ),
                    "uses_outcome_values_in_fit": False,
                    "uses_egfr_cv_fold": False,
                    "transductive_across_egfr_folds": False,
                },
                "rma_per_gse": {
                    "file": "rma_per_gse.npy",
                    "sha256": sha256(exported["rma_per_gse"]),
                    "shape": list(expected_shape),
                    "algorithm": (
                        "IKEM-specific RMA target and probe effects fitted only on "
                        "24 donor-clean no-eGFR pretraining-training biopsies"
                    ),
                    "uses_outcome_values_in_fit": False,
                    "uses_egfr_cv_fold": False,
                    "transductive_across_egfr_folds": False,
                }
            },
        }
        (output_store / "preprocessing_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Ready per-study training data: {destination}")
    print("This overlay did not modify the running rebuild or production data directory.")


if __name__ == "__main__":
    main()

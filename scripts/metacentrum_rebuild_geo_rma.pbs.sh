#!/bin/bash
#PBS -N archcon-cel-rma
#PBS -q default
#PBS -l select=1:ncpus=1:mem=64gb:scratch_local=1gb
#PBS -l walltime=96:00:00
#PBS -j oe

set -Eeuo pipefail

PROJECT_ROOT="${ARCHCON_PROJECT_ROOT:-/storage/praha1/home/anuarali/DP/ARCHCON}"
SWEEP_ROOT="${ARCHCON_SWEEP_ROOT:-$PROJECT_ROOT/sweeps/archcon-pretrain-0515}"
PERSIST_ROOT="${ARCHCON_RMA_STATE_ROOT:-$PROJECT_ROOT/data/GEO_DWNLD_TRAIN_REFERENCE_REBUILD}"
SOURCE_ROOT="${ARCHCON_RMA_INPUT_ROOT:-$PROJECT_ROOT/data/GEO_DWNLD}"
SCRIPT_ROOT="${ARCHCON_RMA_SCRIPT_ROOT:-$PROJECT_ROOT/scripts}"
R_LIBRARY="${ARCHCON_R_LIBS:-/storage/praha1/home/anuarali/Rpackages}"
PYTHON_BIN="${ARCHCON_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
GEO_NUMPY_STORE="$PROJECT_ROOT/data/GEO_NUMPY_STORE"
IKEM_STORE="$PROJECT_ROOT/data/IKEM_NUMPY_STORE"
IKEM_RAW_DIR="$PROJECT_ROOT/data/IKEM_CEL"
IKEM_CANDIDATE="$PERSIST_ROOT/IKEM_NUMPY_CANDIDATE"

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR is unset or unavailable. Run inside a PBS job." >&2
    exit 2
fi

set +u
source /cvmfs/software.metacentrum.cz/modulefiles/5.3.1/loadmodules
module load r/4.1.3-gcc-10.2.1-6xt26dl
set -u

export R_LIBS="$R_LIBRARY"
export R_LIBS_USER="$R_LIBRARY"
export R_MAKEVARS_USER=/dev/null
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RSCRIPT_BIN="$(command -v Rscript || true)"
if [[ -z "$RSCRIPT_BIN" ]]; then
    echo "ERROR: Rscript was not placed on PATH by the R module." >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable is unavailable: $PYTHON_BIN" >&2
    exit 2
fi

# Only executable source is staged to node-local scratch. CEL archives, RDS,
# HDF5, NumPy matrices and R temporary files all remain on praha1.
RUN_SCRIPTS="$SCRATCHDIR/archcon_rma_scripts"
RUN_TMP="$PROJECT_ROOT/data/R_TMP_RUN/${PBS_JOBID:-interactive}"
rm -rf -- "$RUN_SCRIPTS"
mkdir -p \
    "$RUN_SCRIPTS" \
    "$RUN_TMP" \
    "$PERSIST_ROOT" \
    "$SOURCE_ROOT/GEO_RAW" \
    "$IKEM_RAW_DIR" \
    "$IKEM_CANDIDATE" \
    "$GEO_NUMPY_STORE" \
    "$IKEM_STORE"
export TMPDIR="$RUN_TMP"
export TMP="$RUN_TMP"
export TEMP="$RUN_TMP"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    rm -rf -- "$RUN_TMP" "$RUN_SCRIPTS"
    echo "Pipeline exit status: $status"
    exit "$status"
}
trap cleanup EXIT INT TERM

cp -p \
    "$SCRIPT_ROOT/metacentrum_rebuild_geo_rma.R" \
    "$SCRIPT_ROOT/metacentrum_geo_download.R" \
    "$SCRIPT_ROOT/metacentrum_geo_per_gse_rma.R" \
    "$SCRIPT_ROOT/metacentrum_geo_train_reference_rma.R" \
    "$SCRIPT_ROOT/metacentrum_ikem_cel.R" \
    "$RUN_SCRIPTS/"

copy_if_missing() {
    local source_path=$1
    local destination_path=$2
    if [[ ! -f "$destination_path" ]]; then
        if [[ ! -f "$source_path" ]]; then
            echo "ERROR: required input not found: $source_path" >&2
            exit 2
        fi
        cp -p -- "$source_path" "$destination_path"
    fi
}

copy_if_missing "$SOURCE_ROOT/download_summary.csv" "$PERSIST_ROOT/download_summary.csv"
copy_if_missing "$SOURCE_ROOT/gsm_to_gse_mapping.csv" "$PERSIST_ROOT/gsm_to_gse_mapping.csv"

if [[ ! -f "$PROJECT_ROOT/data/common_probes.pkl" ]]; then
    echo "ERROR: required input not found: $PROJECT_ROOT/data/common_probes.pkl" >&2
    exit 2
fi
if [[ ! -f "$SWEEP_ROOT/prepared/sample_index.csv" ]]; then
    echo "ERROR: frozen split not found: $SWEEP_ROOT/prepared/sample_index.csv" >&2
    exit 2
fi
for required_ikem in expression.npy sample_index.csv probe_index.csv; do
    if [[ ! -f "$IKEM_STORE/$required_ikem" ]]; then
        echo "ERROR: existing IKEM correspondence input is missing: $IKEM_STORE/$required_ikem" >&2
        exit 2
    fi
done

# Reuse historical per-GSE products if the persistent rebuild directory does
# not already contain them. This copies only compact RDS checkpoints, never RAW.
if [[ ! -d "$PERSIST_ROOT/GEO_RMA" && -d "$SOURCE_ROOT/GEO_RMA" ]]; then
    cp -a "$SOURCE_ROOT/GEO_RMA" "$PERSIST_ROOT/GEO_RMA"
fi

export ARCHCON_GEO_RAW_DIR="$SOURCE_ROOT/GEO_RAW"
export ARCHCON_IKEM_RAW_DIR="$IKEM_RAW_DIR"
export ARCHCON_IKEM_LEGACY_STORE="$IKEM_STORE"

echo "Script execution root: $RUN_SCRIPTS"
echo "Persistent work root:  $PERSIST_ROOT"
echo "Persistent GEO CELs:   $ARCHCON_GEO_RAW_DIR"
echo "Persistent IKEM CELs:  $ARCHCON_IKEM_RAW_DIR"
echo "Frozen split:          $SWEEP_ROOT/prepared/sample_index.csv"
echo "R library:             $R_LIBS"
echo "R temporary directory: $TMPDIR"

"$RSCRIPT_BIN" --vanilla "$RUN_SCRIPTS/metacentrum_rebuild_geo_rma.R" \
    --work-root "$PERSIST_ROOT" \
    --frozen-split "$SWEEP_ROOT/prepared/sample_index.csv" \
    --numpy-store "$GEO_NUMPY_STORE" \
    --ikem-store "$IKEM_CANDIDATE" \
    --keep-raw true

# Reconstruct an all-cohort RMA only as an audit copy and compare it with the
# pre-existing Stadniuk/IKEM matrix. The deployed per-dataset-RMA matrix is a
# different, leakage-free frozen-train-reference transform.
"$PYTHON_BIN" - \
    "$IKEM_STORE" \
    "$IKEM_CANDIDATE" \
    "$GEO_NUMPY_STORE" \
    "$SWEEP_ROOT/prepared/sample_index.csv" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

legacy = Path(sys.argv[1]).resolve()
candidate = Path(sys.argv[2]).resolve()
geo_store = Path(sys.argv[3]).resolve()
split_path = Path(sys.argv[4]).resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def matrix_stats(path: Path) -> dict[str, object]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "file": path.name,
        "sha256": sha256(path),
        "shape": [int(v) for v in matrix.shape],
        "dtype": str(matrix.dtype),
    }

comparison = {
    "available": False,
    "max_abs": None,
    "mean_abs": None,
    "rmse": None,
    "n_values": 0,
}
old_path = legacy / "expression.npy"
new_path = candidate / "rma_cohort_legacy.npy"
if not old_path.is_file():
    raise RuntimeError(f"Required legacy IKEM expression matrix is missing: {old_path}")
old = np.load(old_path, mmap_mode="r", allow_pickle=False)
new = np.load(new_path, mmap_mode="r", allow_pickle=False)
if old.shape != new.shape:
    raise RuntimeError(
        f"Legacy IKEM expression shape {old.shape} differs from CEL RMA {new.shape}."
    )
sum_abs = 0.0
sum_sq = 0.0
max_abs = 0.0
count = 0
for start in range(0, old.shape[0], 8):
    delta = np.asarray(new[start : start + 8], dtype=np.float64)
    delta -= np.asarray(old[start : start + 8], dtype=np.float64)
    absolute = np.abs(delta)
    max_abs = max(max_abs, float(absolute.max(initial=0.0)))
    sum_abs += float(absolute.sum(dtype=np.float64))
    sum_sq += float(np.square(delta).sum(dtype=np.float64))
    count += int(delta.size)
comparison = {
    "available": True,
    "max_abs": max_abs,
    "mean_abs": sum_abs / count,
    "rmse": (sum_sq / count) ** 0.5,
    "n_values": count,
}
if comparison["rmse"] > 5e-4 or comparison["max_abs"] > 5e-3:
    raise RuntimeError(
        "CEL-derived cohort RMA does not reproduce the existing IKEM expression "
        f"matrix (RMSE={comparison['rmse']:.6g}, "
        f"max_abs={comparison['max_abs']:.6g}). Candidate files were preserved in "
        f"{candidate}; the existing IKEM store was not replaced."
    )

with (candidate / "legacy_rma_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparison))
    writer.writeheader()
    writer.writerow(comparison)

geo_provenance = {
    "format": 1,
    "method": "cel_level_train_reference_rma",
    "exact_cel_level_rma": True,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_split": str(split_path),
    "frozen_split_sha256": sha256(split_path),
    "output": matrix_stats(geo_store / "rma_global.npy"),
    "fit_scope": "GEO molecular-pretraining train arrays only",
}
(geo_store / "rma_global_provenance.json").write_text(
    json.dumps(geo_provenance, indent=2), encoding="utf-8"
)

ikem_provenance = {
    "format": 2,
    "source": "GSE290167",
    "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290167",
    "platform": "GPL15207 PrimeView Human Gene Expression Array",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_pretraining_split": str(split_path),
    "frozen_pretraining_split_sha256": sha256(split_path),
    "legacy_rma_comparison": comparison,
    "methods": {
        "raw_original": {
            **matrix_stats(candidate / "raw_original.npy"),
            "algorithm": "median of original PM CEL intensities within each common probe set",
            "uses_egfr": False,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        "rma_per_gse": {
            **matrix_stats(candidate / "rma_per_gse.npy"),
            "algorithm": (
                "IKEM-specific quantile target and median-polish probe effects fitted "
                "only on frozen outcome-free molecular-pretraining TRAIN samples; "
                "every other GSE290167 CEL transformed independently"
            ),
            "fit_scope": "frozen outcome-free IKEM molecular-pretraining TRAIN samples only",
            "reference_samples": "ikem_rma_reference_samples.csv",
            "uses_egfr": False,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        "rma_global": {
            **matrix_stats(candidate / "rma_global.npy"),
            "algorithm": (
                "per-array RMA background correction followed by the frozen GEO-train "
                "quantile target and frozen GEO-train median-polish probe effects"
            ),
            "fit_scope": "frozen GEO molecular-pretraining TRAIN samples only",
            "uses_egfr": False,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
    },
}
(candidate / "preprocessing_provenance.json").write_text(
    json.dumps(ikem_provenance, indent=2), encoding="utf-8"
)

install_names = [
    "raw_original.npy",
    "rma_per_gse.npy",
    "rma_global.npy",
    "sample_index.csv",
    "probe_index.csv",
    "ikem_gse290167_correspondence.csv",
    "ikem_rma_reference_samples.csv",
    "legacy_rma_comparison.csv",
    "preprocessing_provenance.json",
]
legacy.mkdir(parents=True, exist_ok=True)
for name in install_names:
    source = candidate / name
    destination = legacy / name
    temporary = legacy / f".{name}.new"
    with source.open("rb") as src, temporary.open("wb") as dst:
        while block := src.read(1024 * 1024):
            dst.write(block)
    os.replace(temporary, destination)

print("Verified and installed method-specific IKEM CEL matrices:", legacy)
if comparison["available"]:
    print(
        "Legacy RMA correspondence: "
        f"RMSE={comparison['rmse']:.6g}, max_abs={comparison['max_abs']:.6g}"
    )
PY

echo "GEO and IKEM CEL/RMA pipeline completed successfully."

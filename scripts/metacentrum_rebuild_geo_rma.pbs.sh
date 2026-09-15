#!/bin/bash
#PBS -N archcon-cel-rma
#PBS -q default
#PBS -l select=1:ncpus=1:mem=64gb:scratch_local=1gb
#PBS -l walltime=96:00:00
#PBS -j oe

set -Eeuo pipefail

# PASS 3 controls (the long 336-block calculation):
#   ARCHCON_RMA_MODE=stream  -> block-by-block HDF5 input (default)
#   ARCHCON_RMA_MODE=ram     -> cache the full probe-level matrix in RAM
#   ARCHCON_RMA_CPUS=N       -> start N independent probe-set workers
#
# The qsub resource request must provide at least the same number of CPUs.
# All HDF5 reads/writes and checkpoint commits remain in the parent process.

SCRIPT_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_HOME/.." && pwd -P)"
PROJECT_ROOT="${ARCHCON_PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
SWEEP_ROOT="${ARCHCON_SWEEP_ROOT:-$PROJECT_ROOT/sweeps/archcon-pretrain-0515}"
PERSIST_ROOT="${ARCHCON_RMA_STATE_ROOT:-$PROJECT_ROOT/data/GEO_DWNLD_TRAIN_REFERENCE_REBUILD}"
SOURCE_ROOT="${ARCHCON_RMA_INPUT_ROOT:-$PROJECT_ROOT/data/GEO_DWNLD}"
SCRIPT_ROOT="${ARCHCON_RMA_SCRIPT_ROOT:-$PROJECT_ROOT/scripts}"
R_LIBRARY="${ARCHCON_R_LIBS:-/storage/brno2/home/anuarali/Rpackages-r413-bookworm:/storage/praha1/home/anuarali/Rpackages}"
PYTHON_BIN="${ARCHCON_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
RMA_MODE="${ARCHCON_RMA_MODE:-stream}"
GEO_NUMPY_STORE="$PROJECT_ROOT/data/GEO_NUMPY_STORE"
IKEM_LEGACY_STORE="$PROJECT_ROOT/data/IKEM_NUMPY_STORE"
IKEM_OUTPUT_STORE="${ARCHCON_IKEM_OUTPUT_STORE:-$PROJECT_ROOT/data/IKEM_CEL_NUMPY_STORE}"
IKEM_RAW_DIR="$PROJECT_ROOT/data/IKEM_CEL"
IKEM_PRIVATE_CEL_DIR="${ARCHCON_IKEM_PRIVATE_CEL_DIR:-$IKEM_RAW_DIR/STADNIUK_LEGACY_CEL}"
IKEM_SAMPLE_METADATA="${ARCHCON_IKEM_SAMPLE_METADATA:-$PROJECT_ROOT/data/sample_metadata.csv}"
IKEM_EGFR_TABLE="${ARCHCON_IKEM_EGFR_TABLE:-$PROJECT_ROOT/data/egfr_data.xlsx}"
IKEM_CLASSIFIER_TABLE="${ARCHCON_IKEM_CLASSIFIER_TABLE:-$PROJECT_ROOT/data/Klasifikator_20_3_24_v2.xlsx}"
IKEM_CANDIDATE="$PERSIST_ROOT/IKEM_NUMPY_CANDIDATE_V5"
IKEM_EXPECTED_SAMPLES="${ARCHCON_IKEM_EXPECTED_SAMPLES:-288}"
IKEM_EXPECTED_NO_EGFR_TRAIN="${ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN:-34}"

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR is unset or unavailable. Run inside a PBS job." >&2
    exit 2
fi

case "${RMA_MODE,,}" in
    stream|streaming|disk|hdf5)
        RMA_MODE=stream
        ;;
    ram)
        RMA_MODE=ram
        ;;
    *)
        echo "ERROR: ARCHCON_RMA_MODE must be 'stream' or 'ram'." >&2
        exit 2
        ;;
esac

if [[ -n "${PBS_NCPUS:-}" ]]; then
    ALLOCATED_CPUS="$PBS_NCPUS"
elif [[ -n "${NCPUS:-}" ]]; then
    ALLOCATED_CPUS="$NCPUS"
elif [[ -n "${PBS_NODEFILE:-}" && -r "$PBS_NODEFILE" ]]; then
    ALLOCATED_CPUS="$(wc -l < "$PBS_NODEFILE" | tr -d '[:space:]')"
else
    ALLOCATED_CPUS="$(nproc)"
fi
# Stay conservative when the caller does not choose a worker count. The PBS
# file's built-in resource request is one CPU, and some clusters expose the
# whole node through nproc even when a smaller allocation was requested.
RMA_CPUS="${ARCHCON_RMA_CPUS:-1}"

if [[ ! "$ALLOCATED_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Could not determine the allocated CPU count: $ALLOCATED_CPUS" >&2
    exit 2
fi
if [[ ! "$RMA_CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ARCHCON_RMA_CPUS must be a positive integer." >&2
    exit 2
fi
if (( RMA_CPUS > ALLOCATED_CPUS )); then
    echo "ERROR: requested $RMA_CPUS RMA workers but PBS allocated only $ALLOCATED_CPUS CPU(s)." >&2
    echo "Resubmit with -l select=1:ncpus=$RMA_CPUS:mem=...:scratch_local=1gb" >&2
    exit 2
fi

set +u
source /cvmfs/software.metacentrum.cz/modulefiles/5.3.1/loadmodules
module load r/4.1.3-gcc-10.2.1-6xt26dl
set -u

export R_LIBS="$R_LIBRARY"
export R_LIBS_USER="$R_LIBRARY"
export R_MAKEVARS_USER=/dev/null
export ARCHCON_RMA_MODE="$RMA_MODE"
export ARCHCON_RMA_CPUS="$RMA_CPUS"
export R_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
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

# Only these small executable scripts use node-local scratch. CEL archives,
# matrices, checkpoints and R temporary files remain in the persistent project.
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
    "$IKEM_OUTPUT_STORE"
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
for required_ikem in "$IKEM_SAMPLE_METADATA" "$IKEM_EGFR_TABLE" "$IKEM_CLASSIFIER_TABLE"; do
    if [[ ! -f "$required_ikem" ]]; then
        echo "ERROR: required IKEM metadata file is missing: $required_ikem" >&2
        exit 2
    fi
done
if [[ ! -d "$IKEM_PRIVATE_CEL_DIR" ]]; then
    echo "ERROR: private IKEM CEL directory is missing: $IKEM_PRIVATE_CEL_DIR" >&2
    exit 2
fi

# Reuse historical per-GSE products if the persistent rebuild directory does
# not already contain them. This copies only compact RDS checkpoints, never RAW.
if [[ ! -d "$PERSIST_ROOT/GEO_RMA" && -d "$SOURCE_ROOT/GEO_RMA" ]]; then
    cp -a "$SOURCE_ROOT/GEO_RMA" "$PERSIST_ROOT/GEO_RMA"
fi

export ARCHCON_GEO_RAW_DIR="$SOURCE_ROOT/GEO_RAW"
export ARCHCON_IKEM_RAW_DIR="$IKEM_RAW_DIR"
export ARCHCON_IKEM_PRIVATE_CEL_DIR="$IKEM_PRIVATE_CEL_DIR"
export ARCHCON_IKEM_SAMPLE_METADATA="$IKEM_SAMPLE_METADATA"
export ARCHCON_IKEM_EGFR_TABLE="$IKEM_EGFR_TABLE"
export ARCHCON_IKEM_CLASSIFIER_TABLE="$IKEM_CLASSIFIER_TABLE"
export ARCHCON_IKEM_LEGACY_STORE="$IKEM_LEGACY_STORE"
export ARCHCON_IKEM_EXPECTED_SAMPLES="$IKEM_EXPECTED_SAMPLES"
export ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN="$IKEM_EXPECTED_NO_EGFR_TRAIN"

echo "Script execution root: $RUN_SCRIPTS"
echo "Persistent work root:  $PERSIST_ROOT"
echo "Persistent GEO CELs:   $ARCHCON_GEO_RAW_DIR"
echo "Private IKEM CELs:     $ARCHCON_IKEM_PRIVATE_CEL_DIR"
echo "Public CEL fallback:   $ARCHCON_IKEM_RAW_DIR"
echo "IKEM outcome table:    $ARCHCON_IKEM_EGFR_TABLE"
echo "IKEM output store:     $IKEM_OUTPUT_STORE"
echo "Frozen split:          $SWEEP_ROOT/prepared/sample_index.csv"
echo "R library:             $R_LIBS"
echo "R temporary directory: $TMPDIR"
echo "PASS 3 input mode:      $ARCHCON_RMA_MODE"
echo "PASS 3 worker count:    $ARCHCON_RMA_CPUS / $ALLOCATED_CPUS allocated CPU(s)"

"$RSCRIPT_BIN" --vanilla "$RUN_SCRIPTS/metacentrum_rebuild_geo_rma.R" \
    --work-root "$PERSIST_ROOT" \
    --frozen-split "$SWEEP_ROOT/prepared/sample_index.csv" \
    --numpy-store "$GEO_NUMPY_STORE" \
    --ikem-store "$IKEM_CANDIDATE" \
    --keep-raw true

# Validate the coherent 288-row candidate, record an optional intersection-only
# comparison with Stadniuk's historical matrix, and install it into a new store.
# The old IKEM_NUMPY_STORE is deliberately left untouched.
"$PYTHON_BIN" - \
    "$IKEM_LEGACY_STORE" \
    "$IKEM_CANDIDATE" \
    "$IKEM_OUTPUT_STORE" \
    "$GEO_NUMPY_STORE" \
    "$SWEEP_ROOT/prepared/sample_index.csv" \
    "$IKEM_PRIVATE_CEL_DIR" \
    "$IKEM_EGFR_TABLE" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

legacy = Path(sys.argv[1]).resolve()
candidate = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
geo_store = Path(sys.argv[4]).resolve()
split_path = Path(sys.argv[5]).resolve()
private_cel_dir = Path(sys.argv[6]).resolve()
egfr_table = Path(sys.argv[7]).resolve()
expected_samples = int(os.environ.get("ARCHCON_IKEM_EXPECTED_SAMPLES", "288"))
expected_train = int(
    os.environ.get("ARCHCON_IKEM_EXPECTED_NO_EGFR_TRAIN", "34")
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def matrix_stats(path: Path) -> dict[str, object]:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    for start in range(0, matrix.shape[0], 16):
        if not np.isfinite(matrix[start : start + 16]).all():
            raise RuntimeError(f"Non-finite value in {path} near row {start}.")
    return {
        "file": path.name,
        "sha256": sha256(path),
        "shape": [int(v) for v in matrix.shape],
        "dtype": str(matrix.dtype),
    }

def normalize_id(value: str) -> str:
    name = Path(str(value).strip()).name
    name = re.sub(r"\.CEL(?:\.gz)?$", "", name, flags=re.I)
    name = re.sub(r"^GSM\d+[_-]*", "", name, flags=re.I)
    name = re.sub(r"[_\s-]*\(?PrimeView\)?[_\s-]*$", "", name, flags=re.I)
    return re.sub(r"_+$", "", name).upper()


def read_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty sample index: {path}")
    for key in ("sample_id", "Sample_ID", "GSM", "sample", "id"):
        if key in rows[0]:
            ids = [normalize_id(row[key]) for row in rows]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate sample IDs in {path}")
            return ids
    raise RuntimeError(f"No sample-ID column in {path}")


def read_probe_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty probe index: {path}")
    for key in ("probe_id", "probe", "probeset_id", "ID", "id"):
        if key in rows[0]:
            ids = [str(row[key]).strip().upper() for row in rows]
            if any(not probe_id for probe_id in ids):
                raise RuntimeError(f"Blank probe ID in {path}")
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Duplicate probe IDs in {path}")
            return ids
    raise RuntimeError(f"No probe-ID column in {path}")


sample_path = candidate / "sample_index.csv"
probe_path = candidate / "probe_index.csv"
with sample_path.open(newline="", encoding="utf-8-sig") as handle:
    sample_rows = list(csv.DictReader(handle))
if len(sample_rows) != expected_samples:
    raise RuntimeError(
        f"IKEM candidate has {len(sample_rows)} samples; expected {expected_samples}."
    )
sample_ids = [normalize_id(row["sample_id"]) for row in sample_rows]
if len(sample_ids) != len(set(sample_ids)):
    raise RuntimeError("IKEM candidate sample IDs are not unique.")
train_rows = [row for row in sample_rows if row.get("training_role") == "train_no_measured_egfr"]
held_rows = [row for row in sample_rows if row.get("training_role") == "held_out_measured_egfr"]
crossing_donors = sorted(
    {
        row.get("donor_id", "")
        for row in sample_rows
        if row.get("donor_crosses_outcome_gate", "").strip().lower()
        in {"true", "t", "1", "yes"}
    }
    - {""}
)
if len(train_rows) != expected_train or len(train_rows) + len(held_rows) != expected_samples:
    raise RuntimeError(
        "IKEM outcome gate is inconsistent: "
        f"train={len(train_rows)}, held_out={len(held_rows)}, expected train={expected_train}."
    )

with probe_path.open(newline="", encoding="utf-8-sig") as handle:
    probe_count = sum(1 for _ in csv.DictReader(handle))
expected_shape = (expected_samples, probe_count)
method_stats = {}
for method in ("raw_original", "rma_per_gse", "rma_global"):
    path = candidate / f"{method}.npy"
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != expected_shape or matrix.dtype != np.dtype("<f4"):
        raise RuntimeError(
            f"{path.name} has shape/dtype {matrix.shape}/{matrix.dtype}; "
            f"expected {expected_shape}/float32."
        )
    method_stats[method] = matrix_stats(path)

# This comparison is descriptive only. The new local RMA is intentionally
# train-reference-fitted, whereas Stadniuk's old matrix was cohort-normalized.
comparison = {
    "available": False,
    "shared_samples": 0,
    "shared_probes": 0,
    "exact_probe_order": False,
    "n_values": 0,
    "max_abs": None,
    "mean_abs": None,
    "rmse": None,
    "note": "Historical comparison unavailable; production validation unaffected.",
}
try:
    old_matrix_path = legacy / "expression.npy"
    old_index_path = legacy / "sample_index.csv"
    old_probe_path = legacy / "probe_index.csv"
    if (
        old_matrix_path.is_file()
        and old_index_path.is_file()
        and old_probe_path.is_file()
    ):
        old_ids = read_ids(old_index_path)
        old_probe_ids = read_probe_ids(old_probe_path)
        new_probe_ids = read_probe_ids(probe_path)
        old = np.load(old_matrix_path, mmap_mode="r", allow_pickle=False)
        new = np.load(candidate / "rma_per_gse.npy", mmap_mode="r", allow_pickle=False)
        if (
            old.shape == (len(old_ids), len(old_probe_ids))
            and new.shape == (len(sample_ids), len(new_probe_ids))
        ):
            old_lookup = {sample_id: i for i, sample_id in enumerate(old_ids)}
            shared = [sample_id for sample_id in sample_ids if sample_id in old_lookup]
            new_sample_lookup = {
                sample_id: i for i, sample_id in enumerate(sample_ids)
            }
            old_probe_lookup = {
                probe_id: i for i, probe_id in enumerate(old_probe_ids)
            }
            shared_probes = [
                probe_id
                for probe_id in new_probe_ids
                if probe_id in old_probe_lookup
            ]
            old_probe_columns = [old_probe_lookup[p] for p in shared_probes]
            new_probe_lookup = {
                probe_id: i for i, probe_id in enumerate(new_probe_ids)
            }
            new_probe_columns = [new_probe_lookup[p] for p in shared_probes]
            sum_abs = sum_sq = max_abs = 0.0
            count = 0
            if shared and shared_probes:
                for start in range(0, len(shared), 8):
                    batch = shared[start : start + 8]
                    old_rows = [old_lookup[sample_id] for sample_id in batch]
                    new_rows = [new_sample_lookup[sample_id] for sample_id in batch]
                    old_block = np.asarray(old[old_rows, :], dtype=np.float64)
                    new_block = np.asarray(new[new_rows, :], dtype=np.float64)
                    delta = new_block[:, new_probe_columns]
                    delta -= old_block[:, old_probe_columns]
                    absolute = np.abs(delta)
                    max_abs = max(max_abs, float(absolute.max(initial=0.0)))
                    sum_abs += float(absolute.sum(dtype=np.float64))
                    sum_sq += float(np.square(delta).sum(dtype=np.float64))
                    count += int(delta.size)
            comparison = {
                "available": bool(count),
                "shared_samples": len(shared),
                "shared_probes": len(shared_probes),
                "exact_probe_order": old_probe_ids == new_probe_ids,
                "n_values": count,
                "max_abs": max_abs if count else None,
                "mean_abs": sum_abs / count if count else None,
                "rmse": (sum_sq / count) ** 0.5 if count else None,
                "note": (
                    "Descriptive only: probes were matched by ID and the two "
                    "matrices use different RMA fitting scopes."
                ),
            }
except Exception as error:
    comparison["note"] = f"Historical comparison skipped: {error}"

with (candidate / "legacy_rma_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparison))
    writer.writeheader()
    writer.writerow(comparison)

with (geo_store / "sample_index.csv").open(newline="", encoding="utf-8") as handle:
    geo_rows = list(csv.DictReader(handle))
geo_train_count = sum(
    str(row.get("pretraining_split", "")).strip().lower() == "train"
    for row in geo_rows
)
with (candidate / "ikem_rma_reference_samples.csv").open(
    newline="", encoding="utf-8"
) as handle:
    ikem_train_count = sum(1 for _ in csv.DictReader(handle))

geo_provenance = {
    "format": 1,
    "method": "cel_level_train_reference_rma",
    "exact_cel_level_rma": True,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_split": str(split_path),
    "frozen_split_sha256": sha256(split_path),
    "output": matrix_stats(geo_store / "rma_global.npy"),
    "fit_scope": "GEO TRAIN plus IKEM no-eGFR TRAIN arrays only",
    "geo_train_samples": geo_train_count,
    "ikem_no_egfr_train_samples": ikem_train_count,
    "fit_samples": geo_train_count + ikem_train_count,
}
(geo_store / "rma_global_provenance.json").write_text(
    json.dumps(geo_provenance, indent=2), encoding="utf-8"
)

ikem_provenance = {
    "format": 3,
    "primary_source": str(private_cel_dir),
    "public_fallback": "GSE290167",
    "public_source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290167",
    "platform": "GPL15207 PrimeView Human Gene Expression Array",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "outcome_availability_source": str(egfr_table),
    "outcome_availability_source_sha256": sha256(egfr_table),
    "outcome_gate": "train iff all of egfr_7d, egfr_3m, egfr_6m, egfr_12m are missing",
    "outcome_gate_unit": "biopsy",
    "train_no_measured_egfr_samples": len(train_rows),
    "held_out_measured_egfr_samples": len(held_rows),
    "donors_crossing_biopsy_level_outcome_gate": crossing_donors,
    "private_cel_samples": sum(row.get("cel_source") == "STADNIUK_LEGACY_CEL" for row in sample_rows),
    "public_fallback_samples": sum(row.get("cel_source") == "GSE290167" for row in sample_rows),
    "frozen_pretraining_split": str(split_path),
    "frozen_pretraining_split_sha256": sha256(split_path),
    "legacy_rma_comparison": comparison,
    "methods": {
        "raw_original": {
            **method_stats["raw_original"],
            "algorithm": "median of original PM CEL intensities within each common probe set",
            "uses_outcome_values_in_fit": False,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        "rma_per_gse": {
            **method_stats["rma_per_gse"],
            "algorithm": (
                "IKEM-specific quantile target and median-polish probe effects fitted "
                "only on biopsies without measured longitudinal eGFR; every "
                "measured-eGFR biopsy is transformed with frozen parameters"
            ),
            "fit_scope": "IKEM samples without measured longitudinal eGFR only",
            "reference_samples": "ikem_rma_reference_samples.csv",
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
            "uses_egfr_cv_fold": False,
            "transductive_across_egfr_folds": False,
        },
        "rma_global": {
            **method_stats["rma_global"],
            "algorithm": (
                "per-array RMA background correction followed by the frozen combined "
                "GEO TRAIN plus IKEM no-eGFR TRAIN quantile target and probe effects"
            ),
            "fit_scope": "frozen GEO TRAIN plus IKEM no-eGFR TRAIN samples only",
            "geo_train_samples": geo_train_count,
            "ikem_no_egfr_train_samples": ikem_train_count,
            "fit_samples": geo_train_count + ikem_train_count,
            "uses_outcome_values_in_fit": False,
            "uses_outcome_availability_for_partition": True,
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
    "ikem_cel_correspondence.csv",
    "ikem_gse290167_correspondence.csv",
    "ikem_rma_reference_samples.csv",
    "legacy_rma_comparison.csv",
    "preprocessing_provenance.json",
]
for optional_name in (
    "IKEM_LOCAL_RMA_COMPLETE.txt",
    "IKEM_CEL_PREPROCESSING_COMPLETE.txt",
):
    if candidate.joinpath(optional_name).is_file():
        install_names.append(optional_name)

output.mkdir(parents=True, exist_ok=True)
for name in install_names:
    source = candidate / name
    if not source.is_file():
        raise RuntimeError(f"Candidate output is missing: {source}")
    destination = output / name
    temporary = output / f".{name}.new"
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    os.replace(temporary, destination)

print("Verified and installed coherent IKEM CEL matrices:", output)
print(f"Outcome gate: {len(train_rows)} TRAIN, {len(held_rows)} HELD OUT")
if comparison["available"]:
    print(
        "Descriptive historical overlap: "
        f"RMSE={comparison['rmse']:.6g}, max_abs={comparison['max_abs']:.6g}"
    )
PY

echo "GEO and IKEM CEL/RMA pipeline completed successfully."

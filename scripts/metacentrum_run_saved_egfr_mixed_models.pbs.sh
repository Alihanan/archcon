#!/usr/bin/env bash
#PBS -N archcon-egfr-lme4
#PBS -l select=1:ncpus=1:mem=64gb:scratch_local=20gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

: "${SCRATCHDIR:?This job requires PBS scratch_local storage.}"

PROJECT_ROOT="${ARCHCON_PROJECT_ROOT:-/storage/praha1/home/anuarali/DP/ARCHCON}"
OUTPUT_ROOT="${ARCHCON_EGFR_OUTPUT_ROOT:-$PROJECT_ROOT/sweeps/archcon-pretrain-058/downstream/molecular_egfr_0511}"
PERSISTENT_MIXED_ROOT="$OUTPUT_ROOT/mixed_models"
PERSISTENT_EMBEDDING_METADATA="$OUTPUT_ROOT/embeddings/metadata.json"
R_BENCHMARK_SCRIPT="${ARCHCON_EGFR_R_SCRIPT:-$PROJECT_ROOT/scripts/metacentrum_run_saved_egfr_mixed_models.R}"
PYTHON_BIN="${ARCHCON_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
R_LIBS_DIR="${ARCHCON_R_LIBS:-/storage/praha1/home/anuarali/Rpackages}"
R_TIMEOUT="${ARCHCON_R_TIMEOUT:-23h}"

source /cvmfs/software.metacentrum.cz/modulefiles/5.3.1/loadmodules
module load r/4.1.3-gcc-10.2.1-6xt26dl
export R_LIBS="$R_LIBS_DIR"
export R_LIBS_USER="$R_LIBS_DIR"
export OMP_NUM_THREADS="${PBS_NCPUS:-1}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-1}"
export MKL_NUM_THREADS="${PBS_NCPUS:-1}"
export PYTHONNOUSERSITE=1

RSCRIPT_BIN="$(command -v Rscript || true)"
if [[ -z "$RSCRIPT_BIN" ]]; then
    echo "ERROR: Rscript was not added to PATH by the requested R module." >&2
    exit 2
fi

for required in \
    "$PYTHON_BIN" \
    "$R_BENCHMARK_SCRIPT" \
    "$PERSISTENT_MIXED_ROOT/nested_mixed_model_design.csv" \
    "$PERSISTENT_MIXED_ROOT/nested_mixed_model_specs.csv" \
    "$PERSISTENT_EMBEDDING_METADATA"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required input is missing: $required" >&2
        exit 2
    fi
done

STAGE_ROOT="$(mktemp -d "$SCRATCHDIR/archcon-egfr.${PBS_JOBID:-manual}.XXXXXX")"
STAGE_MIXED_ROOT="$STAGE_ROOT/mixed_models"
STAGE_EMBEDDING_ROOT="$STAGE_ROOT/embeddings"
STAGE_R_SCRIPT="$STAGE_ROOT/metacentrum_run_saved_egfr_mixed_models.R"
mkdir -p "$STAGE_MIXED_ROOT" "$STAGE_EMBEDDING_ROOT" "$PERSISTENT_MIXED_ROOT"

copy_atomic_to_persistent() {
    local source=$1
    local destination=$2
    local temporary
    temporary="$(dirname "$destination")/.${PBS_JOBID:-manual}.$(basename "$destination").incoming"
    cp -p -- "$source" "$temporary"
    mv -f -- "$temporary" "$destination"
}

persist_results() {
    local source destination
    if [[ ! -d "$STAGE_MIXED_ROOT" ]]; then
        return
    fi
    while IFS= read -r -d '' source; do
        destination="$PERSISTENT_MIXED_ROOT/$(basename "$source")"
        copy_atomic_to_persistent "$source" "$destination"
        echo "Persisted: $destination"
    done < <(
        find "$STAGE_MIXED_ROOT" -maxdepth 1 -type f \
            \( -name '*.csv' -o -name '*.txt' \) -print0
    )
}

job_exit() {
    local status=$?
    trap - EXIT
    set +e
    persist_results
    exit "$status"
}
trap job_exit EXIT

cp -p -- \
    "$PERSISTENT_MIXED_ROOT/nested_mixed_model_design.csv" \
    "$STAGE_MIXED_ROOT/nested_mixed_model_design.csv"
cp -p -- \
    "$PERSISTENT_MIXED_ROOT/nested_mixed_model_specs.csv" \
    "$STAGE_MIXED_ROOT/nested_mixed_model_specs.csv"
cp -p -- "$PERSISTENT_EMBEDDING_METADATA" "$STAGE_EMBEDDING_ROOT/metadata.json"
cp -p -- "$R_BENCHMARK_SCRIPT" "$STAGE_R_SCRIPT"
for reusable_output in \
    nested_all_fold_metrics.csv \
    nested_all_oof_predictions.csv; do
    if [[ -s "$PERSISTENT_MIXED_ROOT/$reusable_output" ]]; then
        cp -p -- \
            "$PERSISTENT_MIXED_ROOT/$reusable_output" \
            "$STAGE_MIXED_ROOT/$reusable_output"
        echo "Staged resumable output: $reusable_output"
    fi
done

cd "$STAGE_ROOT"
echo "Working directory: $PWD"
echo "Rscript: $RSCRIPT_BIN"
echo "R libraries: $R_LIBS"
echo "Python/venv (praha1, read-only): $PYTHON_BIN"
echo "Saved design staged in: $STAGE_MIXED_ROOT"
echo "R time limit: $R_TIMEOUT; PBS walltime: 24 hours"

"$RSCRIPT_BIN" -e '
cat(R.version.string, "\n")
print(.libPaths())
if (!requireNamespace("lme4", quietly = TRUE)) stop("lme4 is unavailable")
cat("lme4:", as.character(packageVersion("lme4")), "\n")
'

set +e
timeout --signal=TERM --kill-after=5m "$R_TIMEOUT" \
    "$RSCRIPT_BIN" "$STAGE_R_SCRIPT" \
    "$STAGE_MIXED_ROOT/nested_mixed_model_design.csv" \
    "$STAGE_MIXED_ROOT/nested_mixed_model_specs.csv" \
    "$STAGE_MIXED_ROOT/nested_all_fold_metrics.csv" \
    "$STAGE_MIXED_ROOT/nested_all_oof_predictions.csv" \
    > "$STAGE_MIXED_ROOT/nested_all_lme4_stdout.txt" \
    2> "$STAGE_MIXED_ROOT/nested_all_lme4_stderr.txt"
R_STATUS=$?
set -e

if (( R_STATUS != 0 )); then
    if (( R_STATUS == 124 )); then
        echo "ERROR: R reached its $R_TIMEOUT time limit; partial raw outputs will be persisted." >&2
    else
        echo "ERROR: R benchmark failed with status $R_STATUS; outputs and logs will be persisted." >&2
    fi
    exit "$R_STATUS"
fi

"$PYTHON_BIN" - "$STAGE_ROOT" <<'PY'
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from archcon.data.downstream import (
    finalize_nested_selection,
    summarize_mixed_model_results,
)

stage_root = Path(sys.argv[1]).resolve()
mixed_root = stage_root / "mixed_models"
metadata = json.loads(
    (stage_root / "embeddings" / "metadata.json").read_text(encoding="utf-8")
)

specs = pd.read_csv(mixed_root / "nested_mixed_model_specs.csv")
raw_metrics = pd.read_csv(mixed_root / "nested_all_fold_metrics.csv")
if len(raw_metrics) != len(specs):
    raise RuntimeError(
        f"Incomplete R benchmark: expected {len(specs)} fits, found {len(raw_metrics)}."
    )

embeddings = []
for item in metadata["models"]:
    embeddings.append(
        SimpleNamespace(
            model_id=item["model_id"],
            z=np.empty((0, int(item["latent_dim"])), dtype=np.float32),
            record=SimpleNamespace(
                run=item["run"],
                method=item["method"],
                architecture=item["architecture"],
                molecular_selection_mse=item["molecular_selection_mse"],
            ),
        )
    )

metrics_path, predictions_path, _selections, ranking = finalize_nested_selection(
    mixed_root / "nested_all_fold_metrics.csv",
    mixed_root / "nested_all_oof_predictions.csv",
    embeddings,
    mixed_root,
)
summary, pairwise, clinical = summarize_mixed_model_results(
    metrics_path,
    predictions_path,
    mixed_root,
)

print("\nEVALUATED ENCODERS")
print(
    ranking[
        [
            "model_label",
            "mean_cv_rmse",
            "nested_selection_count",
            "nested_selection_fraction",
            "run",
        ]
    ].to_string(index=False)
)
fixed = summary.loc[
    summary["model_id"].astype(str).str.startswith("candidate_")
]
print("\nALL FIXED MOLECULAR ENCODERS: OUTER-CV eGFR PERFORMANCE")
print(
    fixed[
        [
            "model_label",
            "mean_rmse",
            "pooled_rmse",
            "mean_delta_vs_time",
            "lcb_delta_vs_time",
            "positive_folds_vs_time",
        ]
    ].to_string(index=False)
)
print("\nNESTED-CV MIXED-MODEL SUMMARY")
print(
    summary[
        [
            "model_label",
            "mean_rmse",
            "pooled_rmse",
            "mean_delta_vs_time",
            "lcb_delta_vs_time",
            "positive_folds_vs_time",
        ]
    ].to_string(index=False)
)
print("\nNESTED-SELECTED Z AGAINST BASELINES")
print(
    pairwise[
        [
            "model_label",
            "mean_winner_gain",
            "lcb_winner_gain",
            "winner_better_fraction",
        ]
    ].to_string(index=False)
)
if not clinical.empty:
    print("\nINCREMENTAL VALUE BEYOND CLINICAL BASELINES")
    print(
        clinical[
            [
                "comparison_label",
                "mean_gain",
                "lcb_gain",
                "augmented_better_fraction",
            ]
        ].to_string(index=False)
    )
PY

echo "R benchmark and final summaries completed successfully."

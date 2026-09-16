#!/usr/bin/env bash
#PBS -N archcon-egfr-lme4
#PBS -l select=1:ncpus=1:mem=64gb:scratch_local=20gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

: "${SCRATCHDIR:?This job requires PBS scratch_local storage.}"

PROJECT_ROOT="${ARCHCON_PROJECT_ROOT:-/storage/brno2/home/anuarali/DP/ARCHCON}"
OUTPUT_ROOT="${ARCHCON_EGFR_OUTPUT_ROOT:-$PROJECT_ROOT/sweeps/archcon-pretrain-0516/downstream/molecular_egfr}"
PERSISTENT_MIXED_ROOT="$OUTPUT_ROOT/mixed_models"
PERSISTENT_EMBEDDING_METADATA="$OUTPUT_ROOT/embeddings/metadata.json"
R_BENCHMARK_SCRIPT="${ARCHCON_EGFR_R_SCRIPT:-$PROJECT_ROOT/scripts/metacentrum_run_saved_egfr_mixed_models.R}"
PYTHON_BIN="${ARCHCON_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
R_LIBS_DIR="${ARCHCON_R_LIBS:-/storage/brno2/home/anuarali/Rpackages-r413-bookworm:/storage/praha1/home/anuarali/Rpackages}"
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
    "$PERSISTENT_MIXED_ROOT/mixed_model_design.csv" \
    "$PERSISTENT_MIXED_ROOT/mixed_model_specs.csv" \
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
    "$PERSISTENT_MIXED_ROOT/mixed_model_design.csv" \
    "$STAGE_MIXED_ROOT/mixed_model_design.csv"
cp -p -- \
    "$PERSISTENT_MIXED_ROOT/mixed_model_specs.csv" \
    "$STAGE_MIXED_ROOT/mixed_model_specs.csv"
cp -p -- "$PERSISTENT_EMBEDDING_METADATA" "$STAGE_EMBEDDING_ROOT/metadata.json"
cp -p -- "$R_BENCHMARK_SCRIPT" "$STAGE_R_SCRIPT"
for reusable_output in \
    fixed_fold_metrics.csv \
    fixed_oof_predictions.csv; do
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
    "$STAGE_MIXED_ROOT/mixed_model_design.csv" \
    "$STAGE_MIXED_ROOT/mixed_model_specs.csv" \
    "$STAGE_MIXED_ROOT/fixed_fold_metrics.csv" \
    "$STAGE_MIXED_ROOT/fixed_oof_predictions.csv" \
    > "$STAGE_MIXED_ROOT/fixed_lme4_stdout.txt" \
    2> "$STAGE_MIXED_ROOT/fixed_lme4_stderr.txt"
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
    summarize_fixed_encoder_results,
)

stage_root = Path(sys.argv[1]).resolve()
mixed_root = stage_root / "mixed_models"
metadata = json.loads(
    (stage_root / "embeddings" / "metadata.json").read_text(encoding="utf-8")
)

specs = pd.read_csv(mixed_root / "mixed_model_specs.csv")
raw_metrics = pd.read_csv(mixed_root / "fixed_fold_metrics.csv")
if len(raw_metrics) != len(specs):
    raise RuntimeError(
        f"Incomplete R benchmark: expected {len(specs)} fits, found {len(raw_metrics)}."
    )

embeddings = []
for item in metadata["models"]:
    embeddings.append(
        SimpleNamespace(
            model_id=item["model_id"],
            label=item["label"],
            z=np.empty((0, int(item["latent_dim"])), dtype=np.float32),
            record=SimpleNamespace(
                run=item["run"],
                method=item["method"],
                architecture=item["architecture"],
                molecular_selection_mse=item["validation_selection_score"],
                molecular_test_mse=item["geo_test_mse_post_freeze"],
            ),
        )
    )

summary, fixed, comparisons = summarize_fixed_encoder_results(
    mixed_root / "fixed_fold_metrics.csv",
    mixed_root / "fixed_oof_predictions.csv",
    embeddings,
    mixed_root,
)

print("\nFIXED MOLECULAR ENCODERS: DONOR-GROUPED eGFR PERFORMANCE")
print(
    fixed[
        [
            "model_label",
            "mean_rmse",
            "mean_rmse_ci95_low",
            "mean_rmse_ci95_high",
            "pooled_rmse",
            "mean_delta_vs_time",
            "positive_folds_vs_time",
            "run",
        ]
    ].to_string(index=False)
)
print("\nNo eGFR-derived encoder winner is selected.")
if not comparisons.empty:
    print("\nMATCHED-FOLD BASELINE COMPARISONS")
    print(
        comparisons[
            [
                "candidate_id",
                "comparison",
                "mean_gain",
                "gain_ci95_low",
                "gain_ci95_high",
                "candidate_better_fraction",
            ]
        ].to_string(index=False)
    )
print(f"\nAll-model summary rows: {len(summary)}")
PY

echo "R benchmark and final summaries completed successfully."

#!/usr/bin/env bash

set -euo pipefail

SWEEP_DIR="${ARCHCON_SWEEP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULT_ROOT="${ARCHCON_RESULT_ROOT:-$SWEEP_DIR/results}"
LAUNCHER="${ARCHCON_ARRAY_LAUNCHER:-$SWEEP_DIR/run_array.pbs.sh}"
SUBMISSION_DIR="$SWEEP_DIR/submissions"

if [[ ! -f "$LAUNCHER" ]]; then
    echo "ERROR: PBS launcher is missing: $LAUNCHER" >&2
    exit 2
fi
if [[ ! -d "$SWEEP_DIR/jobs" ]]; then
    echo "ERROR: generated jobs directory is missing: $SWEEP_DIR/jobs" >&2
    exit 2
fi

mkdir -p "$SUBMISSION_DIR"
TASK_LIST="$SUBMISSION_DIR/pending_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
: > "$TASK_LIST"

submitted=0
skipped=0
shopt -s nullglob
jobs=("$SWEEP_DIR"/jobs/run_*.py)
if (( ${#jobs[@]} == 0 )); then
    echo "ERROR: no generated run_*.py jobs found in $SWEEP_DIR/jobs" >&2
    rm -f -- "$TASK_LIST"
    exit 2
fi

for job in "${jobs[@]}"; do
    run_name=$(basename "$job" .py)
    digits=${run_name#run_}
    if [[ ! "$digits" =~ ^[0-9]+$ ]]; then
        continue
    fi
    run_dir="$RESULT_ROOT/$run_name"
    if [[ -d "$run_dir" ]] &&
       find "$run_dir" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q .; then
        ((skipped += 1))
        continue
    fi
    printf '%d\n' "$((10#$digits))" >> "$TASK_LIST"
    ((submitted += 1))
done

if (( submitted == 0 )); then
    rm -f -- "$TASK_LIST"
    echo "Nothing submitted: all $skipped discovered runs already contain a .pt file."
    exit 0
fi

# Randomize the mapping from compact PBS array slots to the original run
# indices. This mixes preprocessing and architecture configurations early in
# a partially completed sweep instead of exhausting each configuration block
# sequentially.
shuf --output="$TASK_LIST" "$TASK_LIST"

QSUB_VARIABLES="ARCHCON_SWEEP_DIR=$SWEEP_DIR,ARCHCON_RUN_INDEX_FILE=$TASK_LIST"
if [[ -n "${ARCHCON_PROJECT_DIR:-}" ]]; then
    QSUB_VARIABLES+=",ARCHCON_PROJECT_DIR=$ARCHCON_PROJECT_DIR"
fi
if [[ -n "${ARCHCON_DATA_DIR:-}" ]]; then
    QSUB_VARIABLES+=",ARCHCON_DATA_DIR=$ARCHCON_DATA_DIR"
fi
if [[ -n "${ARCHCON_PYTHON:-}" ]]; then
    QSUB_VARIABLES+=",ARCHCON_PYTHON=$ARCHCON_PYTHON"
fi
if [[ -n "${ARCHCON_RESULT_ROOT:-}" ]]; then
    QSUB_VARIABLES+=",ARCHCON_RESULT_ROOT=$ARCHCON_RESULT_ROOT"
fi

set +e
JOB_ID=$(qsub -J "1-$submitted" -v "$QSUB_VARIABLES" "$LAUNCHER")
QSUB_STATUS=$?
set -e
if [[ $QSUB_STATUS -ne 0 ]]; then
    rm -f -- "$TASK_LIST"
    echo "ERROR: qsub failed with status $QSUB_STATUS." >&2
    exit "$QSUB_STATUS"
fi

echo "Submitted $submitted runs; skipped $skipped runs whose result folders contain .pt files."
echo "The pending run indices were randomized before submission."
echo "PBS job: $JOB_ID"
echo "Run-index map (keep until the array finishes): $TASK_LIST"


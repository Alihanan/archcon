#!/usr/bin/env bash
# Flat source-tree reference launcher for MetaCentrum.
#PBS -N archcon-pretrain-058
#PBS -l select=1:ncpus=1:mem=10gb:scratch_local=4gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail

: "${PBS_ARRAY_INDEX:?This script must be submitted as a PBS job array.}"
: "${ARCHCON_SWEEP_DIR:?submit.sh must export ARCHCON_SWEEP_DIR.}"
: "${SCRATCHDIR:?This launcher requires PBS scratch_local storage.}"

PROJECT_DIR="${ARCHCON_PROJECT_DIR:-/storage/praha1/home/anuarali/DP/ARCHCON}"
PERSISTENT_DATA_DIR="${ARCHCON_DATA_DIR:-$PROJECT_DIR/data}"
PYTHON_BIN="${ARCHCON_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
PERSISTENT_RESULT_ROOT="${ARCHCON_RESULT_ROOT:-$ARCHCON_SWEEP_DIR/results}"
TRAIN_TIMEOUT="${ARCHCON_TRAIN_TIMEOUT:-23h}"
TIMEOUT_KILL_AFTER="${ARCHCON_TIMEOUT_KILL_AFTER:-5m}"

# submit.sh creates a compact task list containing only runs without an
# existing .pt file. PBS_ARRAY_INDEX identifies a line in that list.
ARRAY_SLOT=$PBS_ARRAY_INDEX
if [[ -n "${ARCHCON_RUN_INDEX_FILE:-}" ]]; then
    if [[ ! -f "$ARCHCON_RUN_INDEX_FILE" ]]; then
        echo "ERROR: run-index file is missing: $ARCHCON_RUN_INDEX_FILE" >&2
        exit 2
    fi
    RUN_INDEX=$(sed -n "${ARRAY_SLOT}p" "$ARCHCON_RUN_INDEX_FILE")
else
    RUN_INDEX=$ARRAY_SLOT
fi

if [[ ! "$RUN_INDEX" =~ ^[0-9]+$ ]] || (( 10#$RUN_INDEX < 1 )); then
    echo "ERROR: invalid run index '$RUN_INDEX' for array slot $ARRAY_SLOT" >&2
    exit 2
fi

RUN_INDEX=$((10#$RUN_INDEX))
RUN_NAME=$(printf "run_%04d" "$RUN_INDEX")
PERSISTENT_CONFIG="$ARCHCON_SWEEP_DIR/configs/${RUN_NAME}.json"
PERSISTENT_JOB="$ARCHCON_SWEEP_DIR/jobs/${RUN_NAME}.py"
PERSISTENT_PREPARED="$ARCHCON_SWEEP_DIR/prepared"
PERSISTENT_RUN_DIR="$PERSISTENT_RESULT_ROOT/$RUN_NAME"

for required in \
    "$PYTHON_BIN" \
    "$PERSISTENT_CONFIG" \
    "$PERSISTENT_JOB" \
    "$PERSISTENT_PREPARED/prepared.json"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required input is missing: $required" >&2
        exit 2
    fi
done

# Recheck at job start in case another submission produced a checkpoint after
# submit.sh built its task list.
if [[ -d "$PERSISTENT_RUN_DIR" ]] &&
   find "$PERSISTENT_RUN_DIR" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q .; then
    echo "Skipping $RUN_NAME: persistent result folder already contains a .pt file."
    exit 0
fi

STAGE_ROOT=$(mktemp -d "$SCRATCHDIR/archcon.${PBS_JOBID:-manual}.${RUN_NAME}.XXXXXX")
STAGE_SWEEP="$STAGE_ROOT/sweep"
STAGE_DATA="$STAGE_ROOT/data"
STAGE_PREPARED="$STAGE_SWEEP/prepared"
STAGE_JOB="$STAGE_SWEEP/jobs/${RUN_NAME}.py"
STAGE_CONFIG="$STAGE_SWEEP/configs/${RUN_NAME}.json"
STAGE_RESULT_ROOT="$STAGE_ROOT/results"
STAGE_RUN_DIR="$STAGE_RESULT_ROOT/$RUN_NAME"
TRAIN_LOG="$STAGE_RUN_DIR/training.log"

cleanup_stage() {
    if [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]]; then
        rm -rf -- "$STAGE_ROOT"
    fi
}
trap cleanup_stage EXIT

mkdir -p \
    "$STAGE_SWEEP/jobs" \
    "$STAGE_SWEEP/configs" \
    "$STAGE_PREPARED" \
    "$STAGE_DATA" \
    "$STAGE_RUN_DIR"

copy_required() {
    local source=$1
    local destination=$2
    if [[ ! -f "$source" ]]; then
        echo "ERROR: required staging source is missing: $source" >&2
        exit 3
    fi
    mkdir -p "$(dirname "$destination")"
    cp -p -- "$source" "$destination"
}

copy_optional() {
    local source=$1
    local destination=$2
    if [[ -f "$source" ]]; then
        mkdir -p "$(dirname "$destination")"
        cp -p -- "$source" "$destination"
    fi
}

copy_atomic_to_persistent() {
    local source=$1
    local destination=$2
    local temporary
    temporary="$(dirname "$destination")/.${PBS_JOBID:-manual}.$(basename "$destination").incoming"
    cp -p -- "$source" "$temporary"
    mv -f -- "$temporary" "$destination"
}

echo "Staging $RUN_NAME in node-local scratch: $STAGE_ROOT"
copy_required "$PERSISTENT_JOB" "$STAGE_JOB"
copy_required "$PERSISTENT_CONFIG" "$STAGE_CONFIG"

METHOD=$(
    "$PYTHON_BIN" - "$PERSISTENT_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
method = config.get("method")
if not isinstance(method, str) or not method:
    raise SystemExit("Generated run configuration has no preprocessing method.")
print(method)
PY
)

mapfile -t PREPARED_FILES < <(
    "$PYTHON_BIN" - "$PERSISTENT_PREPARED/prepared.json" "$METHOD" <<'PY'
import json
import sys
from pathlib import PurePath

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
method = sys.argv[2]

rows = metadata.get("method_geo_rows", {})
columns = metadata.get("method_geo_columns", {})
supplemental = metadata.get("supplemental_matrices", {})
if method not in rows or method not in columns or method not in supplemental:
    raise SystemExit(f"No frozen prepared mapping exists for {method!r}.")

names = [
    "prepared.json",
    str(metadata["sample_index"]),
    str(supplemental[method]),
    str(metadata["train_rows"]),
    str(metadata["validation_rows"]),
    str(metadata["test_rows"]),
    str(rows[method]),
    str(columns[method]),
    "probe_index.csv",
]
extras = metadata.get("method_extra_files", {}).get(method, {})
if not isinstance(extras, dict):
    raise SystemExit(f"Invalid prepared extra-file mapping for {method!r}.")
names.extend(str(name) for name in extras.values())

seen = set()
for name in names:
    path = PurePath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe path in prepared metadata: {name!r}")
    if name not in seen:
        print(name)
        seen.add(name)
PY
)

for relative in "${PREPARED_FILES[@]}"; do
    if [[ "$relative" == "probe_index.csv" ]]; then
        copy_optional "$PERSISTENT_PREPARED/$relative" "$STAGE_PREPARED/$relative"
    else
        copy_required "$PERSISTENT_PREPARED/$relative" "$STAGE_PREPARED/$relative"
    fi
done

case "$METHOD" in
    "Per-dataset standardization"|"Per-dataset RMA"|"Global RMA")
        SOURCE_STORE="$PERSISTENT_DATA_DIR/GEO_NUMPY_STORE"
        STAGE_STORE="$STAGE_DATA/GEO_NUMPY_STORE"
        mkdir -p "$STAGE_STORE"

        for name in \
            sample_index.csv \
            source_sample_occurrences.csv \
            source_gse_occurrence_index.csv \
            gse_index.csv \
            probe_index.csv \
            cel_manifest.csv; do
            copy_required "$SOURCE_STORE/$name" "$STAGE_STORE/$name"
        done

        if [[ "$METHOD" == "Per-dataset standardization" ]]; then
            SELECTED_ARRAY="raw_original.npy"
        elif [[ "$METHOD" == "Per-dataset RMA" ]]; then
            SELECTED_ARRAY="rma_per_gse.npy"
        else
            SELECTED_ARRAY="rma_global.npy"
            copy_required \
                "$SOURCE_STORE/rma_global_provenance.json" \
                "$STAGE_STORE/rma_global_provenance.json"
        fi

        # GeoExpressionStore validates every variant when opened. Only the
        # selected matrix is copied and read during minibatch training; links
        # to unused variants incur only their small NumPy-header checks.
        for name in \
            raw_original.npy \
            rma_per_gse.npy \
            rma_global.npy \
            per_dataset_raw_original.npy \
            per_dataset_rma_per_gse.npy; do
            if [[ "$name" == "$SELECTED_ARRAY" ]]; then
                copy_required "$SOURCE_STORE/$name" "$STAGE_STORE/$name"
            else
                if [[ ! -f "$SOURCE_STORE/$name" ]]; then
                    echo "ERROR: required GEO store array is missing: $SOURCE_STORE/$name" >&2
                    exit 3
                fi
                ln -s -- "$SOURCE_STORE/$name" "$STAGE_STORE/$name"
            fi
        done
        STAGED_MATRIX="$STAGE_STORE/$SELECTED_ARRAY"
        ;;

    *)
        echo "ERROR: unsupported preprocessing method: $METHOD" >&2
        exit 4
        ;;
esac

if [[ -L "$STAGED_MATRIX" || ! -f "$STAGED_MATRIX" ]]; then
    echo "ERROR: selected training matrix was not copied to scratch: $STAGED_MATRIX" >&2
    exit 5
fi

export OMP_NUM_THREADS="${NCPUS:-1}"
export MKL_NUM_THREADS="${NCPUS:-1}"
export OPENBLAS_NUM_THREADS="${NCPUS:-1}"
export NUMEXPR_NUM_THREADS="${NCPUS:-1}"
export ARCHCON_CPU_THREADS="${NCPUS:-1}"
export TMPDIR="$STAGE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$STAGE_ROOT/torchinductor"
export PYTHONPYCACHEPREFIX="$STAGE_ROOT/pycache"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

SCRATCH_REAL=$(realpath -e "$SCRATCHDIR")
assert_in_scratch() {
    local label=$1
    local path=$2
    local resolved
    resolved=$(realpath -m "$path")
    case "$resolved" in
        "$SCRATCH_REAL"|"$SCRATCH_REAL"/*)
            ;;
        *)
            echo "ERROR: $label escaped SCRATCHDIR: $resolved" >&2
            exit 7
            ;;
    esac
}

# Fail closed if a future edit accidentally points any active training path
# back at persistent storage. PYTHON_BIN is intentionally exempt because the
# existing virtual environment is used read-only, like a software module.
assert_in_scratch "working directory" "$STAGE_ROOT"
assert_in_scratch "Python job" "$STAGE_JOB"
assert_in_scratch "training data" "$STAGE_DATA"
assert_in_scratch "selected expression matrix" "$STAGED_MATRIX"
assert_in_scratch "training result directory" "$STAGE_RUN_DIR"

echo "PBS array slot: $ARRAY_SLOT"
echo "ArchCon run index: $RUN_INDEX"
echo "Preprocessing: $METHOD"
echo "Python/venv (praha1, read-only during training): $PYTHON_BIN"
echo "Python job (scratch): $STAGE_JOB"
echo "Training data (scratch): $STAGE_DATA"
echo "Result directory during training (scratch): $STAGE_RUN_DIR"
echo "Persistent checkpoint destination after training: $PERSISTENT_RUN_DIR"
echo "Training timeout: $TRAIN_TIMEOUT; PBS walltime: 24 hours"
du -sh "$STAGE_PREPARED" "$STAGED_MATRIX" 2>/dev/null || true

cd "$STAGE_ROOT"
echo "Verified execution working directory: $(pwd -P)"
set +e
timeout --signal=TERM --kill-after="$TIMEOUT_KILL_AFTER" "$TRAIN_TIMEOUT" \
    "$PYTHON_BIN" "$STAGE_JOB" \
        --data-dir "$STAGE_DATA" \
        --output-root "$STAGE_RESULT_ROOT" \
        --run-directory "$STAGE_RUN_DIR" \
        >"$TRAIN_LOG" 2>&1
TRAIN_STATUS=$?
set -e

if [[ $TRAIN_STATUS -eq 124 || $TRAIN_STATUS -eq 137 ]]; then
    echo "Training reached its $TRAIN_TIMEOUT time budget (status $TRAIN_STATUS)."
elif [[ $TRAIN_STATUS -ne 0 ]]; then
    echo "WARNING: training exited with status $TRAIN_STATUS; copying any completed checkpoints." >&2
fi

# This is the first creation/write in the persistent result directory. Copy
# every complete checkpoint (including latest.pt and best.pt) only after Python
# has stopped, using a same-filesystem
# temporary name so readers never observe a partial .pt file.
shopt -s nullglob
CHECKPOINTS=("$STAGE_RUN_DIR"/*.pt)
mkdir -p "$PERSISTENT_RUN_DIR"
copy_atomic_to_persistent "$TRAIN_LOG" "$PERSISTENT_RUN_DIR/training.log"
if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "ERROR: $RUN_NAME produced no complete .pt checkpoint in scratch." >&2
    echo "Training log copied to: $PERSISTENT_RUN_DIR/training.log" >&2
    if [[ $TRAIN_STATUS -ne 0 ]]; then
        exit "$TRAIN_STATUS"
    fi
    exit 6
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
    destination="$PERSISTENT_RUN_DIR/$(basename "$checkpoint")"
    copy_atomic_to_persistent "$checkpoint" "$destination"
    echo "Copied checkpoint: $destination"
done

# These small files make completed runs immediately usable by selection tools.
for name in run_summary.json run_request.json; do
    if [[ -f "$STAGE_RUN_DIR/$name" ]]; then
        copy_atomic_to_persistent "$STAGE_RUN_DIR/$name" "$PERSISTENT_RUN_DIR/$name"
    fi
done

# Replace scratch provenance in copied JSON with durable paths.
if [[ -f "$PERSISTENT_RUN_DIR/run_summary.json" ]]; then
    "$PYTHON_BIN" - \
        "$PERSISTENT_RUN_DIR/run_summary.json" \
        "$PERSISTENT_RUN_DIR" \
        "$PERSISTENT_RESULT_ROOT" \
        "$PERSISTENT_JOB" \
        "$PERSISTENT_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
result_root = Path(sys.argv[3]).resolve()
persistent_job = Path(sys.argv[4]).resolve()
persistent_data = Path(sys.argv[5]).resolve()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["output_root"] = str(result_root)
summary["result_directory"] = str(run_dir)
summary["array_index"] = int(run_dir.name.split("_")[-1])
summary["config_path"] = str(persistent_job)
summary["data_dir"] = str(persistent_data)
for key, value in list(summary.items()):
    if isinstance(value, str) and value.endswith(".pt"):
        summary[key] = str(run_dir / Path(value).name)
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

request_path = run_dir / "run_request.json"
if request_path.is_file():
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["source_config"] = str(persistent_job)
    request_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8"
    )
PY
fi

echo "Persistent results: $PERSISTENT_RUN_DIR"
if [[ $TRAIN_STATUS -eq 0 || $TRAIN_STATUS -eq 124 || $TRAIN_STATUS -eq 137 ]]; then
    exit 0
fi
exit "$TRAIN_STATUS"

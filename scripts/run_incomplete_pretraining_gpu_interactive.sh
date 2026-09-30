#!/usr/bin/env bash
# Resume every incomplete ArchCon pretraining sweep run sequentially on one GPU.
#
# Designed for an already-running MetaCentrum interactive GPU allocation. The
# selected 1.9 GB expression matrix is copied to node-local scratch once, while
# checkpoints and completion summaries are copied atomically to persistent
# storage after every run.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  ARCHCON_CONFIRM_EXCLUSIVE_SWEEP=YES \
    bash run_incomplete_pretraining_gpu_interactive.sh [options]

Options:
  --strategy NAME       per-gse, standardized, or global; default: per-gse
  --project-root PATH   Default: current directory
  --sweep-root PATH     Override the strategy-specific sweep directory
  --data-dir PATH       Override the strategy-specific data directory
  --image PATH          Default: MetaCentrum NGC PyTorch 26.06 image
  --gpu-packages PATH   Default: PROJECT_ROOT/gpu-runtime/py312-post8-clean
  --budget-hours N      Runtime budget for this runner; default: 20
  --plan-only           Stage nothing and print the ordered incomplete-run plan
  -h, --help            Show this help

Safety:
  Before starting, stop or confirm the absence of other workers for this sweep.
  Concurrent workers must never write the same result folders. The explicit
  ARCHCON_CONFIRM_EXCLUSIVE_SWEEP=YES acknowledgement is therefore required.
EOF
}

PROJECT_ROOT=$PWD
STRATEGY="per-gse"
SWEEP_ROOT=""
DATA_DIR=""
IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:26.06-py3.SIF"
GPU_PACKAGES=""
BUDGET_HOURS=20
PLAN_ONLY=0

while (($#)); do
    case "$1" in
        --strategy)
            STRATEGY=${2:?Missing value for --strategy}
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT=${2:?Missing value for --project-root}
            shift 2
            ;;
        --sweep-root)
            SWEEP_ROOT=${2:?Missing value for --sweep-root}
            shift 2
            ;;
        --data-dir)
            DATA_DIR=${2:?Missing value for --data-dir}
            shift 2
            ;;
        --image)
            IMAGE=${2:?Missing value for --image}
            shift 2
            ;;
        --gpu-packages)
            GPU_PACKAGES=${2:?Missing value for --gpu-packages}
            shift 2
            ;;
        --budget-hours)
            BUDGET_HOURS=${2:?Missing value for --budget-hours}
            shift 2
            ;;
        --plan-only)
            PLAN_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT=$(realpath -e "$PROJECT_ROOT")
case "$STRATEGY" in
    per-gse)
        STRATEGY_SLUG="per-gse"
        STRATEGY_LABEL="Per-dataset RMA"
        METHOD="Per-dataset RMA"
        SELECTED_ARRAY="rma_per_gse.npy"
        DEFAULT_SWEEP_ROOT="$PROJECT_ROOT/sweeps/archcon-pretrain-0516-per-gse"
        DEFAULT_DATA_DIR="$PROJECT_ROOT/data-per-gse-ready"
        ;;
    standardized)
        STRATEGY_SLUG="standardized"
        STRATEGY_LABEL="Per-dataset standardization"
        METHOD="Per-dataset standardization"
        SELECTED_ARRAY="raw_original.npy"
        DEFAULT_SWEEP_ROOT="$PROJECT_ROOT/sweeps/archcon-pretrain-0516-standardized"
        DEFAULT_DATA_DIR="$PROJECT_ROOT/data"
        ;;
    global)
        STRATEGY_SLUG="global"
        STRATEGY_LABEL="Global RMA"
        METHOD="Global RMA"
        SELECTED_ARRAY="rma_global.npy"
        DEFAULT_SWEEP_ROOT="$PROJECT_ROOT/sweeps/archcon-pretrain-0516-global"
        DEFAULT_DATA_DIR="$PROJECT_ROOT/data"
        ;;
    *)
        echo "--strategy must be per-gse, standardized, or global; got: $STRATEGY" >&2
        exit 2
        ;;
esac

SWEEP_ROOT=$(realpath -e "${SWEEP_ROOT:-$DEFAULT_SWEEP_ROOT}")
DATA_DIR=$(realpath -e "${DATA_DIR:-$DEFAULT_DATA_DIR}")
IMAGE=$(realpath -e "$IMAGE")
GPU_PACKAGES=$(realpath -e "${GPU_PACKAGES:-$PROJECT_ROOT/gpu-runtime/py312-post8-clean}")

if [[ ! "$BUDGET_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--budget-hours must be a positive integer, got: $BUDGET_HOURS" >&2
    exit 2
fi

for command_name in singularity flock timeout realpath; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command_name" >&2
        exit 2
    fi
done

for required in \
    "$SWEEP_ROOT/prepared/prepared.json" \
    "$SWEEP_ROOT/jobs" \
    "$SWEEP_ROOT/results" \
    "$DATA_DIR/GEO_NUMPY_STORE/$SELECTED_ARRAY" \
    "$GPU_PACKAGES/archcon/__init__.py"; do
    if [[ ! -e "$required" ]]; then
        echo "Required input is missing: $required" >&2
        exit 2
    fi
done

if [[ "${ARCHCON_CONFIRM_EXCLUSIVE_SWEEP:-${ARCHCON_CONFIRM_EXCLUSIVE_PER_GSE:-}}" != "YES" ]]; then
    cat >&2 <<EOF
Refusing to start without an exclusivity acknowledgement.

First verify that no CPU/PBS worker is still processing:
  $SWEEP_ROOT

Then rerun with:
  export ARCHCON_CONFIRM_EXCLUSIVE_SWEEP=YES
EOF
    exit 3
fi

# Advisory process-level lock. It is released automatically even if PBS kills
# this shell. Existing legacy CPU launchers do not take this lock, hence the
# explicit exclusivity acknowledgement above remains necessary.
exec 9>"$SWEEP_ROOT/.gpu_interactive_runner.lock"
if ! flock -n 9; then
    echo "Another interactive GPU runner already holds the sweep lock." >&2
    exit 3
fi

export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH="$GPU_PACKAGES"
GPU_PYTHON=(singularity exec --nv "$IMAGE" python)

echo "Verifying the CUDA/ArchCon runtime..."
"${GPU_PYTHON[@]}" - "$GPU_PACKAGES" <<'PY'
from pathlib import Path
import sys

import archcon
import torch

expected_root = Path(sys.argv[1]).resolve()
loaded = Path(archcon.__file__).resolve()
if expected_root not in loaded.parents:
    raise RuntimeError(f"ArchCon loaded from {loaded}, expected it below {expected_root}")
if not torch.cuda.is_available():
    raise RuntimeError("The NGC container cannot see a CUDA device.")
print(f"ArchCon {archcon.__version__}: {loaded}")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
PY

PERSISTENT_RESULTS="$SWEEP_ROOT/results"
PREFLIGHT_RESULTS="$SWEEP_ROOT/gpu-preflight-results"
PLAN_FILE=$(mktemp "${TMPDIR:-/tmp}/archcon-${STRATEGY_SLUG}-plan.XXXXXX.tsv")
trap 'rm -f -- "$PLAN_FILE"' EXIT

# A saved GPU preflight checkpoint is eligible when it is newer than the main
# result checkpoint. Reading with mmap avoids materializing model tensors merely
# to inspect the scalar epoch.
"${GPU_PYTHON[@]}" - \
    "$SWEEP_ROOT/jobs" \
    "$PERSISTENT_RESULTS" \
    "$PREFLIGHT_RESULTS" >"$PLAN_FILE" <<'PY'
from pathlib import Path
import sys

import torch

jobs_dir = Path(sys.argv[1])
results_root = Path(sys.argv[2])
preflight_root = Path(sys.argv[3])


def checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        return int(checkpoint.get("epoch", 0))
    except Exception as error:
        print(f"WARNING: cannot inspect {path}: {error}", file=sys.stderr)
        return -1


pending = []
for job_path in sorted(jobs_dir.glob("run_[0-9][0-9][0-9][0-9].py")):
    run_name = job_path.stem
    persistent = results_root / run_name
    if (persistent / "run_summary.json").is_file():
        continue

    candidates = []
    for source in (persistent, preflight_root / run_name):
        latest = source / "latest.pt"
        best = source / "best.pt"
        checkpoint = latest if latest.is_file() else best
        if checkpoint.is_file():
            candidates.append((checkpoint_epoch(checkpoint), source, checkpoint))

    if candidates:
        epoch, source, checkpoint = max(candidates, key=lambda item: item[0])
        if epoch < 0:
            epoch = 0
            source = persistent
            checkpoint = Path("-")
    else:
        epoch = 0
        source = persistent
        checkpoint = Path("-")

    pending.append((epoch, run_name, source, checkpoint))

# Complete the closest-to-finished configurations first. This maximizes the
# number of usable completed models if the interactive allocation expires.
pending.sort(key=lambda item: (-item[0], item[1]))
for epoch, run_name, source, checkpoint in pending:
    print(f"{run_name}\t{epoch}\t{source}\t{checkpoint}")
PY

TOTAL_CONFIGS=$(find "$SWEEP_ROOT/jobs" -maxdepth 1 -type f \
    -name 'run_[0-9][0-9][0-9][0-9].py' | wc -l)
PENDING_COUNT=$(wc -l <"$PLAN_FILE")
echo "Preprocessing strategy: $STRATEGY_LABEL"
echo "Incomplete runs: $PENDING_COUNT"
if ((PENDING_COUNT == 0)); then
    echo "All runs already contain run_summary.json. Nothing to do."
    exit 0
fi

echo "Execution order (highest checkpoint epoch first):"
awk -F '\t' '{printf "  %s: epoch %s\n", $1, $2}' "$PLAN_FILE"

if ((PLAN_ONLY)); then
    exit 0
fi

: "${SCRATCHDIR:?This runner must execute inside a MetaCentrum job with SCRATCHDIR.}"
SCRATCH_REAL=$(realpath -e "$SCRATCHDIR")
STAGE_ROOT=$(mktemp -d "$SCRATCH_REAL/archcon-${STRATEGY_SLUG}-gpu.${PBS_JOBID:-interactive}.XXXXXX")
STAGE_SWEEP="$STAGE_ROOT/sweep"
STAGE_PREPARED="$STAGE_SWEEP/prepared"
STAGE_JOBS="$STAGE_SWEEP/jobs"
STAGE_DATA="$STAGE_ROOT/data"
STAGE_STORE="$STAGE_DATA/GEO_NUMPY_STORE"
STAGE_RESULTS="$STAGE_ROOT/results"
mkdir -p "$STAGE_PREPARED" "$STAGE_JOBS" "$STAGE_STORE" "$STAGE_RESULTS"

echo "Node-local stage: $STAGE_ROOT"

copy_required() {
    local source=$1
    local destination=$2
    if [[ ! -f "$source" ]]; then
        echo "Required file is missing: $source" >&2
        exit 4
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

copy_atomic() {
    local source=$1
    local destination=$2
    local temporary
    mkdir -p "$(dirname "$destination")"
    temporary="$(dirname "$destination")/.${PBS_JOBID:-interactive}.$(basename "$destination").incoming"
    cp -p -- "$source" "$temporary"
    mv -f -- "$temporary" "$destination"
}

mapfile -t PREPARED_FILES < <(
    "${GPU_PYTHON[@]}" - "$SWEEP_ROOT/prepared/prepared.json" "$METHOD" <<'PY'
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
names.extend(str(value) for value in extras.values())

seen = set()
for name in names:
    path = PurePath(name)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe prepared path: {name!r}")
    if name not in seen:
        print(name)
        seen.add(name)
PY
)

echo "Staging frozen prepared artifacts..."
for relative in "${PREPARED_FILES[@]}"; do
    if [[ "$relative" == "probe_index.csv" ]]; then
        copy_optional "$SWEEP_ROOT/prepared/$relative" "$STAGE_PREPARED/$relative"
    else
        copy_required "$SWEEP_ROOT/prepared/$relative" "$STAGE_PREPARED/$relative"
    fi
done

SOURCE_STORE="$DATA_DIR/GEO_NUMPY_STORE"
for name in \
    sample_index.csv \
    source_sample_occurrences.csv \
    source_gse_occurrence_index.csv \
    gse_index.csv \
    probe_index.csv \
    cel_manifest.csv; do
    copy_required "$SOURCE_STORE/$name" "$STAGE_STORE/$name"
done

echo "Staging $SELECTED_ARRAY once..."
copy_required "$SOURCE_STORE/$SELECTED_ARRAY" "$STAGE_STORE/$SELECTED_ARRAY"

if [[ "$SELECTED_ARRAY" == "rma_global.npy" ]]; then
    copy_required \
        "$SOURCE_STORE/rma_global_provenance.json" \
        "$STAGE_STORE/rma_global_provenance.json"
fi

# GeoExpressionStore validates the small headers of every declared variant.
# Only SELECTED_ARRAY is read for this sweep; unused matrices remain read-only
# links and never participate in minibatch training.
for name in \
    raw_original.npy \
    rma_per_gse.npy \
    rma_global.npy \
    per_dataset_raw_original.npy \
    per_dataset_rma_per_gse.npy; do
    if [[ "$name" == "$SELECTED_ARRAY" ]]; then
        continue
    fi
    if [[ ! -f "$SOURCE_STORE/$name" ]]; then
        echo "Required GEO variant is missing: $SOURCE_STORE/$name" >&2
        exit 4
    fi
    ln -s -- "$(realpath -e "$SOURCE_STORE/$name")" "$STAGE_STORE/$name"
done

export OMP_NUM_THREADS="${NCPUS:-2}"
export MKL_NUM_THREADS="${NCPUS:-2}"
export OPENBLAS_NUM_THREADS="${NCPUS:-2}"
export NUMEXPR_NUM_THREADS="${NCPUS:-2}"
export ARCHCON_CPU_THREADS="${NCPUS:-2}"
export TMPDIR="$STAGE_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export TORCHINDUCTOR_CACHE_DIR="$STAGE_ROOT/torchinductor"
export PYTHONPYCACHEPREFIX="$STAGE_ROOT/pycache"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

GPU_WRAPPER="$STAGE_ROOT/run_generated_job_on_cuda.py"
cat >"$GPU_WRAPPER" <<'PY'
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

job_path = Path(sys.argv[1]).resolve()
job_arguments = sys.argv[2:]
spec = importlib.util.spec_from_file_location("archcon_generated_gpu_job", job_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import generated job: {job_path}")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
job.TRAINING_CONFIG = replace(job.TRAINING_CONFIG, device="CUDA")
sys.argv = [str(job_path), *job_arguments]
job.main()
PY

REWRITE_SUMMARY="$STAGE_ROOT/rewrite_summary.py"
cat >"$REWRITE_SUMMARY" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
result_root = Path(sys.argv[3]).resolve()
persistent_job = Path(sys.argv[4]).resolve()
persistent_data = Path(sys.argv[5]).resolve()
image = str(Path(sys.argv[6]).resolve())

summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["output_root"] = str(result_root)
summary["result_directory"] = str(run_dir)
summary["array_index"] = int(run_dir.name.split("_")[-1])
summary["config_path"] = str(persistent_job)
summary["data_dir"] = str(persistent_data)
summary["execution_device_override"] = "CUDA"
summary["execution_container"] = image
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

START_UNIX=$(date +%s)
BUDGET_SECONDS=$((BUDGET_HOURS * 3600))
DEADLINE_UNIX=$((START_UNIX + BUDGET_SECONDS))
MIN_START_SECONDS=600
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_DIR="$SWEEP_ROOT/submissions"
REPORT="$REPORT_DIR/gpu_interactive_${STAMP}_${PBS_JOBID:-manual}.tsv"
mkdir -p "$REPORT_DIR"
printf 'run\tstart_epoch\texit_status\tcompleted\tcheckpoint_epoch\n' >"$REPORT"

echo "Runtime budget: $BUDGET_HOURS hours"
echo "Progress report: $REPORT"
echo "Starting sequential GPU continuation for $STRATEGY_LABEL..."

COMPLETED_NOW=0
FAILED_NOW=0
STOP_FOR_BUDGET=0
CONSECUTIVE_FAILURES=0

while IFS=$'\t' read -r RUN_NAME START_EPOCH SOURCE_RUN_DIR SOURCE_CHECKPOINT; do
    PERSISTENT_RUN="$PERSISTENT_RESULTS/$RUN_NAME"
    PERSISTENT_JOB="$SWEEP_ROOT/jobs/$RUN_NAME.py"

    # A different worker may have completed it since the plan was generated.
    if [[ -f "$PERSISTENT_RUN/run_summary.json" ]]; then
        echo "Skipping $RUN_NAME: run_summary.json appeared after planning."
        continue
    fi

    NOW=$(date +%s)
    REMAINING=$((DEADLINE_UNIX - NOW))
    if ((REMAINING < MIN_START_SECONDS)); then
        echo "Less than $MIN_START_SECONDS seconds remain in the runner budget; stopping safely."
        STOP_FOR_BUDGET=1
        break
    fi

    STAGE_JOB="$STAGE_JOBS/$RUN_NAME.py"
    STAGE_RUN="$STAGE_RESULTS/$RUN_NAME"
    CURRENT_LOG="$STAGE_RUN/training.current.log"
    PREVIOUS_LOG="$STAGE_RUN/training.previous.log"
    mkdir -p "$STAGE_RUN"
    copy_required "$PERSISTENT_JOB" "$STAGE_JOB"
    copy_optional "$PERSISTENT_RUN/training.log" "$PREVIOUS_LOG"

    # Use the source directory selected by the epoch-aware plan. This may be a
    # saved GPU preflight when it is newer than the production checkpoint.
    copy_optional "$SOURCE_RUN_DIR/best.pt" "$STAGE_RUN/best.pt"
    RESUME_ARGS=()
    if [[ "$SOURCE_CHECKPOINT" != "-" && -f "$SOURCE_CHECKPOINT" ]]; then
        copy_required "$SOURCE_CHECKPOINT" "$STAGE_RUN/latest.pt"
        RESUME_ARGS=(--resume-checkpoint "$STAGE_RUN/latest.pt")
    fi

    echo
    echo "======================================================================"
    echo "$RUN_NAME: resuming from epoch $START_EPOCH; budget remaining ${REMAINING}s"
    echo "======================================================================"

    set +e
    timeout --signal=TERM --kill-after=2m "${REMAINING}s" \
        singularity exec --nv "$IMAGE" \
        python "$GPU_WRAPPER" "$STAGE_JOB" \
            --data-dir "$STAGE_DATA" \
            --output-root "$STAGE_RESULTS" \
            --run-directory "$STAGE_RUN" \
            "${RESUME_ARGS[@]}" \
            2>&1 | tee "$CURRENT_LOG"
    TRAIN_STATUS=${PIPESTATUS[0]}
    set -e

    mkdir -p "$PERSISTENT_RUN"
    if [[ -f "$PREVIOUS_LOG" ]]; then
        {
            cat "$PREVIOUS_LOG"
            printf '\n===== GPU RESUME %s · PBS job %s =====\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PBS_JOBID:-interactive}"
            cat "$CURRENT_LOG"
        } >"$STAGE_RUN/training.merged.log"
        copy_atomic "$STAGE_RUN/training.merged.log" "$PERSISTENT_RUN/training.log"
    else
        copy_atomic "$CURRENT_LOG" "$PERSISTENT_RUN/training.log"
    fi

    shopt -s nullglob
    CHECKPOINTS=("$STAGE_RUN"/*.pt)
    if ((${#CHECKPOINTS[@]} == 0)); then
        echo "ERROR: $RUN_NAME produced no complete checkpoint." >&2
        CHECKPOINT_EPOCH=$START_EPOCH
    else
        for checkpoint in "${CHECKPOINTS[@]}"; do
            copy_atomic "$checkpoint" "$PERSISTENT_RUN/$(basename "$checkpoint")"
        done
        CHECKPOINT_EPOCH=$(
            "${GPU_PYTHON[@]}" - "$PERSISTENT_RUN/latest.pt" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
print(int(checkpoint.get("epoch", 0)))
PY
        )
    fi

    for name in run_summary.json run_request.json; do
        if [[ -f "$STAGE_RUN/$name" ]]; then
            copy_atomic "$STAGE_RUN/$name" "$PERSISTENT_RUN/$name"
        fi
    done

    if [[ -f "$PERSISTENT_RUN/run_summary.json" ]]; then
        "${GPU_PYTHON[@]}" "$REWRITE_SUMMARY" \
            "$PERSISTENT_RUN/run_summary.json" \
            "$PERSISTENT_RUN" \
            "$PERSISTENT_RESULTS" \
            "$PERSISTENT_JOB" \
            "$DATA_DIR" \
            "$IMAGE"
        COMPLETED=1
        COMPLETED_NOW=$((COMPLETED_NOW + 1))
        CONSECUTIVE_FAILURES=0
        echo "$RUN_NAME completed successfully at checkpoint epoch $CHECKPOINT_EPOCH."
    else
        COMPLETED=0
        if ((TRAIN_STATUS != 124 && TRAIN_STATUS != 137)); then
            FAILED_NOW=$((FAILED_NOW + 1))
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            echo "WARNING: $RUN_NAME stopped with status $TRAIN_STATUS." >&2
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_NAME" "$START_EPOCH" "$TRAIN_STATUS" "$COMPLETED" "$CHECKPOINT_EPOCH" \
        >>"$REPORT"

    # The path is constructed beneath this invocation's mktemp directory.
    # Remove only this completed staging directory so the shared matrix/cache
    # remain available to subsequent configurations.
    case "$STAGE_RUN" in
        "$STAGE_RESULTS"/run_[0-9][0-9][0-9][0-9])
            rm -rf -- "$STAGE_RUN"
            ;;
        *)
            echo "Refusing unsafe stage cleanup: $STAGE_RUN" >&2
            exit 7
            ;;
    esac

    if ((TRAIN_STATUS == 124 || TRAIN_STATUS == 137)); then
        echo "The overall runtime budget expired while processing $RUN_NAME."
        STOP_FOR_BUDGET=1
        break
    fi
    if ((CONSECUTIVE_FAILURES >= 3)); then
        echo "Stopping after three consecutive non-timeout failures." >&2
        echo "Inspect the copied training logs before rerunning." >&2
        break
    fi
done <"$PLAN_FILE"

FINAL_COMPLETE=$(find "$PERSISTENT_RESULTS" -mindepth 2 -maxdepth 2 \
    -name run_summary.json -type f | wc -l)

echo
echo "======================================================================"
echo "GPU CONTINUATION FINISHED"
echo "======================================================================"
echo "Completed during this invocation: $COMPLETED_NOW"
echo "Non-timeout failures:             $FAILED_NOW"
echo "Preprocessing strategy:           $STRATEGY_LABEL"
echo "Total completed summaries:        $FINAL_COMPLETE / $TOTAL_CONFIGS"
echo "Stopped for runtime budget:       $STOP_FOR_BUDGET"
echo "Progress report:                  $REPORT"
echo "Scratch stage:                    $STAGE_ROOT"
echo
echo "Rerunning the same command is safe: completed runs are skipped and the"
echo "remaining checkpoints resume from the latest atomically copied epoch."

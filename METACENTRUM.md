# ArchCon 0.5.12 on MetaCentrum

> **0.5.8 frozen-input fix.** The supervised store may contain 42,921 probes while the GEO experiment uses 42,917, and the Stadniuk-rescaled GEO store may contain the same 42,917 probes in a different native column order. ArchCon now resolves **all** of those mappings exactly once during sweep generation by probe ID, freezes method-specific GEO row and column arrays, writes the outcome-blind supervised subset into one canonical 42,917-probe space, and freezes final train/validation/test logical row arrays. Generated jobs only read those prepared artifacts. Persistent `latest.pt`/`best.pt` checkpointing from 0.5.6 is unchanged.


This release uses one fixed **molecular ~90/5/5 train/validation/test split** across all preprocessing/model combinations. GEO rows are grouped by connected GSE component. Supervised-dataset rows without eGFR are independently assigned using the same target fractions. Every sample with eGFR is kept outside molecular pretraining and reserved for downstream evaluation.

The default CPU sweep contains **900 runs**: for each of three preprocessing arms there are 180 Stadniuk-MLP and 120 ResNet-LN configurations. Every PBS subjob uses one CPU and executes its own readable `jobs/run_XXXX.py`. Per-study RMA is normalized inside each study. The legacy-named global-RMA arm requires a reference fitted only on the frozen GEO training rows; jobs refuse an unmarked all-sample matrix.

## 1. Data already on MetaCentrum

Your persistent project directory is:

```text
/storage/praha1/home/anuarali/DP/ARCHCON
```

Keep the transferred data here:

```text
/storage/praha1/home/anuarali/DP/ARCHCON/data/GEO_NUMPY_STORE/
```

The sweep reads `GEO_STADNIUK_STORE/expression.npy`, `GEO_NUMPY_STORE/rma_per_gse.npy`, the corrected `GEO_NUMPY_STORE/rma_global.npy`, and `IKEM_NUMPY_STORE/expression.npy`. After generating the sweep, run `archcon-rebuild-global-normalization --data-dir DATA --sweep-root SWEEP --replace`. It uses only frozen GEO training rows to fit a shared quantile reference. `raw_original.npy` is probe-set-level PM-median data, so the replacement is leakage-safe train-reference normalization, not exact CEL-level RMA.

Install ArchCon 0.5.12 first, then rebuild and replace the contaminated matrix:

```bash
/storage/praha1/home/anuarali/DP/ARCHCON/.venv/bin/archcon-rebuild-global-normalization \
  --data-dir /storage/praha1/home/anuarali/DP/ARCHCON/data \
  --sweep-root /storage/praha1/home/anuarali/DP/ARCHCON/sweeps/archcon-pretrain-058 \
  --replace
```

The replacement is atomic and the old matrix receives a timestamped
`rma_global.all_samples_backup_*.npy` name. The command also removes the stale
row-major training cache and writes `rma_global_provenance.json`; Global-RMA jobs refuse to
start unless that provenance matches the sweep's frozen `prepared/sample_index.csv`.

## 2. Build 0.5.12 locally

From the 0.5.12 source root on your workstation:

```bash
source .venv/bin/activate
python -m pip install --upgrade build
rm -rf build dist
python -m build
python scripts/verify_wheel.py dist/archcon-0.5.12-py3-none-any.whl
```

Expected artifacts:

```text
dist/archcon-0.5.12-py3-none-any.whl
dist/archcon-0.5.12.tar.gz
```

Upload them without touching the already transferred data:

```bash
rsync -avhP dist/archcon-0.5.12-py3-none-any.whl dist/archcon-0.5.12.tar.gz \
  anuarali@storage-praha1.metacentrum.cz:~/DP/ARCHCON/
```

## 3. Install into the existing CPU-PyTorch venv

On MetaCentrum:

```bash
cd /storage/praha1/home/anuarali/DP/ARCHCON
source .venv/bin/activate
python --version
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m pip install --upgrade ./archcon-0.5.12-py3-none-any.whl
python -m pip show archcon
```

The environment must use Python >= 3.10. `CUDA: False` is expected for this CPU sweep.

## 4. Generate the corrected split and sweep locally

Start the 0.5.12 web UI against your local `data/` directory. In **05 · Split**, keep seed 42. ArchCon now assigns complete connected source-GSE components to approximately:

```text
train       90%
validation   5%
test         5%
```

The exact sample fractions can differ slightly because whole GSE components are indivisible. Save/export that exact split once.

For command-line generation on MetaCentrum, use the data layout explicitly. This is important because generation itself freezes the probe alignment and final row indices:

```python
from pathlib import Path
from archcon.batch import build_run_request, generate_sweep_bundle, recommended_comparison_grid_json
from archcon.data.defaults import project_data_layout
from archcon.data.training import TrainingConfig
from archcon.data.training_sources import create_shared_preprocessing_split

ROOT = Path("/storage/praha1/home/anuarali/DP/ARCHCON")
DATA = ROOT / "data"
layout = project_data_layout(DATA)
split = create_shared_preprocessing_split(
    layout, seed=42, train_fraction=0.90, validation_fraction=0.05
)
base = build_run_request(
    method="Per-dataset RMA", split_seed=42, train_fraction=0.90,
    validation_fraction=0.05, training=TrainingConfig()
)
generate_sweep_bundle(
    base_request=base,
    grid_text=recommended_comparison_grid_json(),
    destination_root=ROOT / "sweeps",
    sweep_name="archcon-pretrain-057",
    project_dir=str(ROOT), data_dir=str(DATA),
    python_executable=str(ROOT / ".venv/bin/python"),
    ncpus=1, memory="10gb", scratch="4gb", walltime="96:00:00", ngpus=0,
    split_frame=split,
    data_layout=layout,
)
```

The important argument is `data_layout=layout`: before any `run_XXXX.py` executes, ArchCon writes `prepared/train_rows.npy`, `validation_rows.npy`, `test_rows.npy`, method-specific GEO row **and probe-column** maps, and a 42,917-column `supervised_no_egfr_common.npy`.

In **06 · Molecular AE → Headless / MetaCentrum sweep export**, use:

```text
Project directory: /storage/praha1/home/anuarali/DP/ARCHCON
Data directory:    /storage/praha1/home/anuarali/DP/ARCHCON/data
Python executable: /storage/praha1/home/anuarali/DP/ARCHCON/.venv/bin/python
PBS CPUs:          1
PBS RAM:           10gb
PBS scratch:       4gb
PBS walltime:      48:00:00
PBS GPUs:          0
```

The generated directory contains:

```text
archcon-pretrain/
├── jobs/
│   ├── run_0001.py
│   └── ... run_0900.py
├── configs/
│   ├── run_0001.json
│   └── ... run_0900.json
├── split.csv
├── prepared/
│   ├── prepared.json
│   ├── sample_index.csv
│   ├── probe_index.csv
│   ├── train_rows.npy
│   ├── validation_rows.npy
│   ├── test_rows.npy
│   ├── supervised_no_egfr_common.npy
│   ├── geo_rows_stadniuk.npy
│   ├── geo_columns_stadniuk.npy
│   ├── geo_rows_per_gse_rma.npy
│   ├── geo_columns_per_gse_rma.npy
│   ├── geo_rows_global_rma.npy
│   └── geo_columns_global_rma.npy
├── manifest.csv
├── run_array.pbs.sh
├── submit.sh
└── README.txt
```

Each Python job states the exact architecture, loss, L2, optimizer, cosine scheduler, batch size, seed, and split policy. **No job computes or modifies the data split.** `prepared/train_rows.npy`, `validation_rows.npy`, and `test_rows.npy` are the final logical indices created once during sweep generation. Jobs merely memory-map the selected GEO matrix, the already aligned `supervised_no_egfr_common.npy`, the saved method-specific GEO row/column maps, and those integer index arrays. Only train and validation rows are passed to PyTorch.

## 5. Copy the sweep

From the local machine:

```bash
rsync -avhP archcon-pretrain/ \
  anuarali@storage-praha1.metacentrum.cz:~/DP/ARCHCON/sweep/
```

## 6. Preflight one job

Inside an interactive PBS job:

```bash
cd /storage/praha1/home/anuarali/DP/ARCHCON
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARCHCON_CPU_THREADS=1

.venv/bin/python sweep/jobs/run_0001.py \
  --data-dir /storage/praha1/home/anuarali/DP/ARCHCON/data \
  --output-root /storage/praha1/home/anuarali/DP/ARCHCON/test-results
```

The startup log should report `Train/validation/test: ...` and state that samples with eGFR are excluded from molecular pretraining.

## 7. Submit the array

After the preflight succeeds:

```bash
cd /storage/praha1/home/anuarali/DP/ARCHCON/sweep
./submit.sh
qstat -t
```

`submit.sh` uses `qsub -J 1-900`.

## Persistent epoch checkpoints

Version 0.5.12 retains the 0.5.6 rule that model checkpoints are never staged in a hidden `.run_XXXX.work` directory. The PBS wrapper creates `results/run_XXXX/` **before training starts**. At the end of every completed epoch, ArchCon atomically replaces:

```text
results/run_XXXX/latest.pt
```

Whenever validation improves it also updates:

```text
results/run_XXXX/best.pt
```

Only temporary Torch/compilation caches use `SCRATCHDIR`. If PBS kills a job at walltime, the previous completed epoch remains valid on `praha1`. Resubmit the same array index; `run_array.pbs.sh` detects `latest.pt` and resumes optimizer, scheduler, history, and epoch state automatically.

## 8. Expanded grid

For **each preprocessing arm** (`Stadniuk rescaling`, `Per-dataset RMA`, `Global RMA`):

Stadniuk MLP: 180 model configurations.

```text
hidden widths       5: 256 / 256→64 / 256→128→64 / 256→192→128→64 / 256→224→192→128→64
latent dim          3: 3 / 8 / 16
objective           2: MSE / masked MSE
explicit L2         3: 0 / 1e-5 / 1e-4
BatchNorm           2: off / on
```

ResNet-LN: 120 model configurations.

```text
hidden widths       5: same five profiles
latent dim          3: 3 / 8 / 16
objective           2: MSE / masked MSE
residual blocks     2: 1 / 2 per stage
FFN expansion       2: 2x / 4x
LayerNorm           fixed on
```

Per preprocessing: **300**. Across three preprocessings: **900** jobs total.

The model-selection rule remains: train only on molecular train rows, monitor/rank on molecular validation rows, and leave molecular test rows untouched. All supervised-dataset samples with eGFR remain outside pretraining. The global reference is fitted only on frozen GEO training rows and its provenance must match the sweep.

## 9. Frozen z and clinical-baseline eGFR evaluation

After validation has selected the final checkpoint and its one-time frozen test evaluation is
recorded, run the downstream stage from the project root:

```bash
cd /storage/praha1/home/anuarali/DP/ARCHCON
source .venv/bin/activate

archcon-evaluate-egfr \
  --sweep-root sweeps/archcon-pretrain-058 \
  --data-dir data
```

The script never ranks encoders by test performance. It scans `best.pt`, selects by clean
validation MSE, aligns the complete supervised cohort using `prepared/probe_index.csv`, and
computes z before loading eGFR. It also selects the best non-Stadniuk checkpoint by validation
MSE when one is available. With a non-Stadniuk overall winner, the best Stadniuk checkpoint is
used as the architecture comparator instead.

The default R/lme4 evaluation is 5 repeats × 5 folds. Folds are grouped by donor, categorical
time is a fixed effect, and patient is a random intercept. Alongside time-only, z, and PCA, the
command fits KDRI, donor-age, cold-ischemia, and combined-clinical trajectory baselines. It also
fits matched KDRI/full-clinical models augmented with winner z or PCA. KDRI_8 stratifies folds
when enough donor groups exist in every quartile; otherwise the script falls back to shuffled
donor folds. PCA, every z standardization, and clinical median imputation/standardization are
trained within each fold. Use `--no-clinical` for the former molecular-only model set. Install the
R package `lme4` in advance; the script does not attempt a network installation inside a batch job.

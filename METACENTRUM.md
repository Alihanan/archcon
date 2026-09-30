# ArchCon 0.5.16 on MetaCentrum

This is the final-paper, donor-separated workflow. It preserves the existing GEO split and
changes only the IKEM eligibility/role contract, the preprocessing products that depend on those
IKEM roles, and the models selected with the new target-domain validation signal.

## Frozen paper contract

| Domain | Train | Validation | Test | Split unit |
|---|---:|---:|---:|---|
| GEO | 10,522 | 585 | 584 | connected source-GSE component |
| IKEM, donor-clean and no eGFR | 24 | 6 | 0 | donor, stratified by L/P biopsy pattern |
| Combined molecular rows | 10,546 | 591 | 584 | as above |

IKEM eligibility is determined before splitting: if either biopsy from a donor has any finite
value in `egfr_7d`, `egfr_3m`, `egfr_6m`, or `egfr_12m`, every biopsy from that donor is excluded
from molecular pretraining. In the canonical 288-biopsy cohort this gives:

- 254 biopsies with measured eGFR, held out;
- four no-eGFR siblings of measured-eGFR biopsies, held out: `D006_L`, `D037_P`, `D106_P`,
  `D118_L`;
- 30 donor-clean no-eGFR biopsies, all used: 24 train and six validation;
- validation donors `D076`, `D097`, `D184`, and `D204`, giving samples `D076_L`, `D076_P`,
  `D097_L`, `D097_P`, `D184_P`, and `D204_L`.

The IKEM split is deterministic with seed `20260915` and validation fraction `0.20`. The software
fails closed if any canonical count, donor, sample, role, or membership hash changes.

Checkpoint and hyperparameter selection occurs separately inside each of the six
preprocessing×architecture groups. The predeclared score is

```text
0.50 × GEO clean-validation MSE
+ 0.50 × donor-balanced IKEM clean-validation MSE
```

The IKEM term first averages probes within a biopsy, then biopsies within a donor, then the four
donors. The 584 GEO test rows are evaluated only after the six winners are frozen. eGFR outcomes
never choose an encoder; downstream CV reports all six fixed encoders as separate analyses.

## 1. Project and input layout

The examples assume:

```text
/storage/brno2/home/anuarali/DP/ARCHCON/
├── .venv/
├── scripts/
├── data/
│   ├── GEO_NUMPY_STORE/
│   ├── GEO_DWNLD/
│   ├── GEO_DWNLD_TRAIN_REFERENCE_REBUILD/
│   ├── IKEM_CEL/STADNIUK_LEGACY_CEL/
│   ├── IKEM_NUMPY_STORE/                 # historical; left untouched
│   ├── common_probes.pkl
│   ├── sample_metadata.csv
│   ├── egfr_data.xlsx
│   └── Klasifikator_20_3_24_v2.xlsx
└── sweeps/
```

`sample_metadata.csv` must contain all 288 canonical biopsy IDs. The private-CEL directory may
contain all available original Stadniuk CELs; for canonical samples without a private CEL, the R
pipeline uses the matching public GSE290167 CEL. The classifier workbook is used only to
cross-check identities. Numeric eGFR values are read only to form the donor-level availability
gate and never enter normalization or autoencoder fitting.

The final coherent 288-row output is installed into `data/IKEM_CEL_NUMPY_STORE/`. The historical
`IKEM_NUMPY_STORE/` is retained for audit comparison and is never overwritten.

## 2. Build and install 0.5.16.post9

Build into a version-specific directory so older distributions need not be deleted:

```bash
cd /path/to/archcon
source .venv/bin/activate
python -m pip install --upgrade build
python -m build --outdir dist-post9
python -m zipfile -l dist-post9/archcon-0.5.16.post9-py3-none-any.whl \
  | grep 'archcon/data/training_sources.py'
```

Copy the wheel and updated `scripts/` directory to MetaCentrum, then install into the existing
CPU-PyTorch environment:

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON
source .venv/bin/activate
python -m pip install --upgrade ./archcon-0.5.16.post9-py3-none-any.whl
python -m pip show archcon
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Python 3.10 or newer is required. `CUDA: False` is expected for the CPU sweep.

The R rebuild needs `affxparser`, `preprocessCore`, `rhdf5`, `R.utils`, and `readxl` in the
configured MetaCentrum R library.

## 3. Rebuild only data products affected by IKEM membership

The rebuild needs the already frozen GEO membership. For the first 0.5.16 bootstrap, point it at
the preceding sweep's `prepared/sample_index.csv`; the wrapper immediately filters that file to
GEO rows and asserts the exact 10,522/585/584 counts. Old IKEM assignments in that file are
ignored. If a valid 0.5.16 sweep already exists, it can be used instead.

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON

qsub \
  -v ARCHCON_PROJECT_ROOT=/storage/brno2/home/anuarali/DP/ARCHCON,ARCHCON_SWEEP_ROOT=/storage/brno2/home/anuarali/DP/ARCHCON/sweeps/archcon-pretrain-0515 \
  scripts/metacentrum_rebuild_geo_rma.pbs.sh
```

For a parallel Phase-3 run, override both the PBS allocation and worker count, for example:

```bash
qsub \
  -l select=1:ncpus=8:mem=128gb:scratch_local=1gb \
  -v ARCHCON_PROJECT_ROOT=/storage/brno2/home/anuarali/DP/ARCHCON,ARCHCON_SWEEP_ROOT=/storage/brno2/home/anuarali/DP/ARCHCON/sweeps/archcon-pretrain-0515,ARCHCON_RMA_CPUS=8 \
  scripts/metacentrum_rebuild_geo_rma.pbs.sh
```

The wrapper is resumable. Its signatures deliberately preserve unaffected work:

| Artifact | Reused? | Reason |
|---|---|---|
| downloaded GEO archives and validated TAR markers | yes | GEO membership is unchanged |
| per-GSE raw PM summaries and per-GSE RMA matrices | yes | no IKEM sample participates |
| final `raw_original.npy` and `rma_per_gse.npy` GEO matrices | yes | no IKEM-dependent fit |
| IKEM raw PM extraction | yes when CEL/metadata signatures match | sample-level CEL work is cached |
| IKEM-local RMA target/effects and transforms | recomputed as needed | reference is now exactly 24 IKEM train biopsies |
| combined global target, probe effects, and `rma_global.npy` | recomputed | the 24-member IKEM contribution changed |

Changing only IKEM roles cannot preserve the old global matrix: the paper's global arm fits one
reference on 10,522 GEO training CELs plus 24 IKEM training CELs. A different IKEM training set
changes the quantile target and probe effects, so every global transform is mathematically
affected. The script invalidates only those combined-reference products; it does not redownload
or redo per-GSE work.

Rerun the same `qsub` command after interruption. Completion markers and block checkpoints make
the pipeline continue from the last compatible unit rather than restart.

After completion, verify the installed roles:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/storage/brno2/home/anuarali/DP/ARCHCON/data/IKEM_CEL_NUMPY_STORE')
rows = pd.read_csv(root / 'sample_index.csv')
print(rows['training_role'].value_counts())
print(sorted(rows.loc[rows['pretraining_split'].eq('validation'), 'sample_id']))
assert rows['training_role'].value_counts().to_dict() == {
    'held_out_measured_egfr': 254,
    'pretrain_train_no_egfr': 24,
    'pretrain_validation_no_egfr': 6,
    'held_out_related_to_measured_egfr': 4,
}
assert sorted(rows.loc[rows['pretraining_split'].eq('validation'), 'sample_id']) == [
    'D076_L', 'D076_P', 'D097_L', 'D097_P', 'D184_P', 'D204_L'
]
PY
```

## 4. Generate a new final-paper sweep

Do not regenerate the old sweep directory: its checkpoints used different IKEM training and
validation membership and a different checkpoint-selection rule. Keep it as an audit archive and
create `archcon-pretrain-0516`.

```python
from pathlib import Path

from archcon.batch import (
    build_run_request,
    generate_sweep_bundle,
    recommended_comparison_grid_json,
)
from archcon.data.defaults import project_data_layout
from archcon.data.training import TrainingConfig
from archcon.data.training_sources import create_shared_preprocessing_split

ROOT = Path('/storage/brno2/home/anuarali/DP/ARCHCON')
DATA = ROOT / 'data'
layout = project_data_layout(DATA)

split = create_shared_preprocessing_split(
    layout,
    seed=42,
    train_fraction=0.90,
    validation_fraction=0.05,
)
base = build_run_request(
    method='Per-dataset RMA',
    split_seed=42,
    train_fraction=0.90,
    validation_fraction=0.05,
    training=TrainingConfig(),
)
result = generate_sweep_bundle(
    base_request=base,
    grid_text=recommended_comparison_grid_json(),
    destination_root=ROOT / 'sweeps',
    sweep_name='archcon-pretrain-0516',
    project_dir=str(ROOT),
    data_dir=str(DATA),
    python_executable=str(ROOT / '.venv/bin/python'),
    ncpus=1,
    memory='10gb',
    scratch='4gb',
    walltime='24:00:00',
    ngpus=0,
    split_frame=split,
    data_layout=layout,
)
print(result)
```

Generation freezes all data-dependent decisions in `prepared/`. Before any job is written it
recomputes the donor gate from `egfr_data.xlsx`, compares it with the R-produced role manifest,
checks exact membership hashes/provenance, aligns probes, materializes only the 30 eligible IKEM
rows for each preprocessing arm, and writes the final integer row arrays. Generated jobs never
reclassify outcomes or recompute a split.

Expected prepared counts are 10,546 train, 591 validation, and 584 test. `prepared.json` records
24/6 IKEM membership and the 50/50 selection policy.

### Start a per-study-RMA pilot before Global RMA finishes

Once the rebuild has printed both `[IKEM PASS3] block 336/336` and
`IKEM local RMA ... ready`, the completed local-RMA matrix can be exported into an isolated data
overlay. This does not write into the running rebuild or replace production data:

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON
source .venv/bin/activate
python -m pip install 'h5py>=3.10,<4'

.venv/bin/python scripts/export_ready_per_gse_training_data.py \
  --project-root /storage/brno2/home/anuarali/DP/ARCHCON \
  --work-root /storage/brno2/home/anuarali/DP/ARCHCON/data/GEO_DWNLD_TRAIN_REFERENCE_REBUILD
```

Generate a fresh 480-run sweep containing only the per-study-RMA arm:

```bash
.venv/bin/python scripts/generate_final_paper_sweep.py \
  --project-root /storage/brno2/home/anuarali/DP/ARCHCON \
  --data-dir /storage/brno2/home/anuarali/DP/ARCHCON/data-per-gse-ready \
  --sweep-name archcon-pretrain-0516-per-gse \
  --methods per-gse
```

This pilot is already final-method compatible for the per-study arm: it contains 180 dropout-0.1
Stadniuk-MLP runs, 180 dropout-0 Stadniuk-MLP runs, and 120 ResNet-LN runs. Do not call it the
complete three-preprocessing sweep. After Stage 4 installs the production stores, generate the
complete 1,440-run sweep with:

```bash
.venv/bin/python scripts/generate_final_paper_sweep.py \
  --project-root /storage/brno2/home/anuarali/DP/ARCHCON \
  --sweep-name archcon-pretrain-0516 \
  --methods all
```

## 5. Preflight and submit

Run one generated job inside an interactive PBS allocation:

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON
source .venv/bin/activate

.venv/bin/python sweeps/archcon-pretrain-0516/jobs/run_0001.py \
  --data-dir data \
  --output-root sweeps/archcon-pretrain-0516/preflight-results
```

The startup output must report 10,546/591/584 rows and six IKEM validation samples from four
donors. During training, `best.pt` is replaced only when the 50/50 validation score improves;
learning-rate plateau detection and convergence use that same score.

Submit the full 1,440-run CPU grid:

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON/sweeps/archcon-pretrain-0516
./submit.sh
qstat -t
```

Each array job stages only its selected GEO matrix, one small method-matched IKEM matrix, frozen
mappings, job, and checkpoints under `SCRATCHDIR`. Training has a 23-hour timeout inside the
24-hour PBS allocation. Complete checkpoints and small provenance files are copied atomically to
the persistent `results/run_XXXX/` directory. Resubmitting with `./submit.sh` randomizes and
submits only unfinished run indices.

Old checkpoints cannot be reused as final-paper results: both their molecular training rows and
their validation/checkpoint criterion differ. Within the new sweep, `latest.pt` resumes only when
the exact matrix, train/validation row arrays, validation domains, and IKEM donor IDs match.

## 6. Validation diagnostics and final evaluation

While the sweep is incomplete, validation-only diagnostics are allowed:

```bash
archcon-evaluate-egfr \
  --sweep-root sweeps/archcon-pretrain-0516 \
  --data-dir data \
  --allow-incomplete-sweep
```

This mode writes current molecular-validation scores and stops before touching GEO test or eGFR.
Add `--include-running-checkpoints` only when explicitly inspecting best-so-far snapshots.

After all 1,440 run summaries are complete, run the final workflow without the incomplete flag:

```bash
archcon-evaluate-egfr \
  --sweep-root sweeps/archcon-pretrain-0516 \
  --data-dir data
```

The command:

1. verifies the saved 50/50 validation score and role provenance for every completed checkpoint;
2. freezes one winner in each of the six preprocessing×architecture groups;
3. evaluates the 584 GEO test rows for those six winners only;
4. creates method-matched IKEM embeddings without refitting on eGFR samples;
5. evaluates every fixed encoder with 5×5 repeated donor-grouped eGFR CV, matched fold-fitted PCA,
   KDRI, donor-age, cold-ischemia, and combined-clinical baselines.

No eGFR-derived encoder rank or deployment winner is written. Fold-level uncertainty and matched
baseline gains are descriptive for each pre-frozen encoder.

If the long R/lme4 phase is interrupted after its design files have been created, resume it as a
separate MetaCentrum job:

```bash
qsub scripts/metacentrum_run_saved_egfr_mixed_models.pbs.sh
```

The launcher stages any existing `fixed_fold_metrics.csv` and `fixed_oof_predictions.csv`, and the
R helper skips complete `fit_id` values. Partial results are copied back on timeout or failure.

Final outputs are under
`sweeps/archcon-pretrain-0516/downstream/molecular_egfr/`, including:

```text
molecular_selection/
├── molecular_selection_scores.csv
└── molecular_group_winners.csv
embeddings/
├── embeddings.csv
├── embeddings.npz
├── candidate_<preprocessing>_<architecture>_z.npy
└── metadata.json
mixed_models/
├── fixed_fold_metrics.csv
├── fixed_oof_predictions.csv
├── fold_metrics_with_deltas.csv
├── all_model_cv_summary.csv
├── fixed_encoder_cv_summary.csv
└── fixed_encoder_comparisons.csv
```

## 7. Nested eGFR selection: reconstruction control versus soft InfoNCE

ArchCon 0.5.16.post9 adds a separate, resumable final evaluation command. It does not change or
refit the CEL preprocessing references. It first freezes the same molecular-validation winner in
each of the six preprocessing×architecture groups, then creates one persisted donor manifest with
these columns:

```text
outer_repeat, outer_fold, patient, donor, outer_role, inner_role
```

Within every outer fold, outer-training donors are divided into one inner-training and one
inner-validation partition. All biopsies from a donor stay together. The same identities select
the encoder fine-tuning loss and PCA dimension. The chosen configuration is reloaded from the
original molecular checkpoint, refitted on all outer-training donors, and evaluated once on the
untouched outer-test donors. Population-level lme4 predictions exclude unseen patient random
effects.

The encoder candidates are one reconstruction-only control (`none`) and the Cartesian product of
the requested soft-InfoNCE weights and temperatures. Both arms receive otherwise identical IKEM
fine-tuning and preserve the winning checkpoint's MSE or masked-MSE reconstruction objective.
Soft targets are Gaussian similarities in training-fold-scaled clinical space; the
default columns are Stadniuk's prespecified eight-feature pool (`Donor_age`, `AKI`, `don_bio_cv`,
`don_bio_ah`, `don_bio_ifta`, `DCD`, `ECD_1`, and `BMI_donor`) from the classifier workbook. They
can be explicitly changed with `--contrastive-clinical-columns`. The bandwidth is the median
nonzero clinical distance in measured inner training. No eGFR value enters the neural loss. The
24 outcome-blind molecular-training biopsies are added to reconstruction/contrastive fitting in
every fold; the six molecular-validation biopsies remain excluded. Inner-validation RMSE for
`full clinical + z` chooses the encoder candidate. PCA dimensions are selected using
`full clinical + PCA` on those same inner donors.

Load the R module before entering the GPU container and retain the absolute `Rscript` path. The
container must be able to execute that path because each completed inner and outer fold invokes
the packaged lme4 helper:

```bash
cd /storage/brno2/home/anuarali/DP/ARCHCON

source /cvmfs/software.metacentrum.cz/modulefiles/5.3.1/loadmodules
module load r/4.1.3-gcc-10.2.1-6xt26dl
export R_LIBS="/storage/praha1/home/anuarali/Rpackages"
export ARCHCON_RSCRIPT="$(command -v Rscript)"

export ARCHCON_GPU_IMAGE="/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:26.06-py3.SIF"
export ARCHCON_GPU_PACKAGES="$PWD/gpu-runtime/py312-post9-clean"
export SINGULARITYENV_PYTHONPATH="$ARCHCON_GPU_PACKAGES"
export SINGULARITYENV_R_LIBS="$R_LIBS"

mkdir -p "$ARCHCON_GPU_PACKAGES"
singularity exec --nv "$ARCHCON_GPU_IMAGE" \
  python -m pip install --upgrade --target "$ARCHCON_GPU_PACKAGES" \
  "$PWD/archcon-0.5.16.post9-py3-none-any.whl"

singularity exec --nv "$ARCHCON_GPU_IMAGE" "$ARCHCON_RSCRIPT" --version
singularity exec --nv "$ARCHCON_GPU_IMAGE" python - <<'PY'
import archcon
import torch
print("ArchCon:", archcon.__version__, archcon.__file__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Run all six frozen encoder arms and all three nested-selected PCA arms with the default 5×5
outer CV and one donor-grouped inner holdout per outer fold:

```bash
singularity exec --nv "$ARCHCON_GPU_IMAGE" \
  python -m archcon.evaluate_egfr_nested \
  --project-root "$PWD" \
  --data-dir "$PWD/data" \
  --sweep-root "$PWD/sweeps/archcon-pretrain-0516-standardized" \
  --sweep-root "$PWD/sweeps/archcon-pretrain-0516-per-gse" \
  --sweep-root "$PWD/sweeps/archcon-pretrain-0516-global" \
  --output-root "$PWD/evaluations/egfr-nested-infonce-seed-0" \
  --device cuda \
  --folds 5 \
  --repeats 5 \
  --inner-folds 5 \
  --cv-seed 0 \
  --contrastive-weights 0.01,0.05,0.1 \
  --temperatures 0.05,0.1,0.2 \
  --finetune-epochs 50 \
  --finetune-batch-size 32 \
  --finetune-learning-rate 0.0001 \
  --projection-dim 16 \
  --pca-dimensions 3,8,16 \
  --rscript "$ARCHCON_RSCRIPT" \
  --reuse-lasso-root "$PWD/evaluations/egfr-baselines-post7-all-methods-seed-0"
```

`--reuse-lasso-root` is accepted only if every saved repeat/fold/partition/patient/donor row is
identical to the new manifest. To refit LASSO-AIC and LASSO-BIC instead, replace that option with
`--run-lasso`; both models then use the same persisted outer folds, while lambda remains selected
inside each outer-training fold by AIC or BIC. Omitting both options evaluates the neural, PCA,
time-only, KDRI, and full-clinical mixed models without touching LASSO.

The command is restart-safe. Completed outer folds are skipped, each neural candidate embedding
has a contract-checked cache, and a changed checkpoint, split, grid, or fine-tuning control is
rejected rather than mixed into an existing result directory. Principal outputs are:

```text
evaluations/egfr-nested-infonce-seed-0/
├── nested_split_manifest.csv
├── nested_split_manifest.csv.json
├── nested_command_summary.json
├── nested_complete_model_summary.csv
└── nested_models/
    ├── nested_evaluation_contract.json
    ├── nested_inner_candidate_scores.csv
    ├── nested_encoder_selections.csv
    ├── nested_pca_selections.csv
    ├── nested_outer_fold_metrics.csv
    ├── nested_outer_oof_predictions.csv
    ├── nested_model_summary.csv
    └── parts/rXXX_fXXX/
```

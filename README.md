# ArchCon

ArchCon is a local Python/Gradio research application for rebuilding, inspecting, and modelling the gene-expression pipeline developed around **Modern Alternatives to Archetypal Analysis of Gene Expression Data**.

**Current scope (0.5.13):** a beginner-readable molecular pipeline with a large public **unsupervised dataset**, a smaller kidney-donor **supervised dataset**, reproducible outcome-blind pretraining splits, configurable reconstruction-only autoencoders, non-blocking visual exploration, and a leakage-safe molecular/clinical eGFR benchmark.

**0.5.13 memory and MetaCentrum update:** `archcon-evaluate-egfr` scans checkpoints on PyTorch's metadata device, retains no model/optimizer tensors in its checkpoint index, memory-maps and evaluates only one checkpoint at a time, and streams R fit results directly to CSV. Newly generated sweep launchers stage the selected matrix, prepared mappings, Python job, results, caches, and checkpoints under node-local `SCRATCHDIR`; after the 12-hour training budget stops, complete `.pt` files are copied atomically to persistent storage. Submission randomizes unfinished run indices and skips result folders that already contain a checkpoint. The existing persistent virtual environment is used read-only. Standalone reference launchers live under `scripts/metacentrum/`; executable scripts are not placed in the repository root.

**0.5.8 frozen-input update:** the complete molecular training universe is prepared **once when the sweep is generated**. ArchCon aligns the supervised 42,921-probe store to the exact 42,917 canonical GEO probe IDs/order, freezes method-specific GEO row **and probe-column** mappings (so Stadniuk/RMA stores may have different native order), materializes only supervised samples already classified as having no eGFR, freezes one canonical logical sample order, and saves final `train_rows.npy`, `validation_rows.npy`, and `test_rows.npy`. Generated `run_XXXX.py` jobs only memory-map these prepared arrays; they do **not** reclassify outcomes, compare/reorder probes, remap samples, or recompute a split. Version 0.5.13 retains these frozen inputs while moving active job data and checkpoint writes into node-local scratch.

**0.5.12 train-reference correction:** `archcon-rebuild-global-normalization` fits a quantile reference using only the frozen GEO training rows, maps every GEO row independently to that reference, and atomically replaces `rma_global.npy`. Global-arm batch jobs refuse matrices without provenance matching their frozen split. Since `raw_original.npy` is already probe-set PM-median summarized, this leakage-safe replacement is not exact CEL-level RMA.

**0.5.9 downstream update:** `archcon-evaluate-egfr` selects encoders by clean validation MSE, freezes them, aligns every supervised expression row to the prepared 42,917-probe order, and exports exact latent arrays plus sample/donor metadata. It then compares a time-only mixed model, the winning z, the best validation-selected alternative architecture, and a fold-fitted PCA baseline under repeated donor-grouped CV. The fitted formulas contain molecular features only; KDRI may stratify folds but never enters the model. The optional source-tree compatibility wrapper is kept under `scripts/evaluate_molecular_egfr.py`.

**0.5.10 clinical-baseline update:** the same command now also fits KDRI, donor-age, cold-ischemia, and combined-clinical trajectory baselines, plus matched clinical+winner-z and clinical+PCA models. Clinical imputation and standardization are fitted independently inside every training fold. The new incremental summary directly tests whether z improves RMSE beyond KDRI or the full clinical baseline.

## Main web workflow

```text
01 Unsupervised data
   public GEO studies and molecular samples
        ↓
02 Supervised data
   kidney-donor molecular samples; visibly separate eGFR / no-eGFR groups
        ↓
03 Preprocessing
   Stadniuk representation / per-study RMA / global RMA
        ↓
04 Matrix
   samples × 42,917 probes
        ↓
05 Split
   GEO connected-study 90/5/5 + no-eGFR supervised 90/5/5
        ↓
06 Molecular AE
   reconstruction-only representation z + persistent checkpoints
        ↓
07 Downstream
   eGFR evaluation / AA later; outcome-bearing samples enter only here
```

Plot-heavy GEO views are submitted to a **single background worker**. The browser immediately displays a spinner and remains responsive; while one view is being calculated the page overlay blocks additional plot clicks, so repeated clicking cannot build an unbounded PCA/histogram queue.

Stage 06 now focuses the architecture experiment on two interpretable MLP families:

- **Stadniuk MLP:** dense ReLU autoencoder, default `256 → 64`, with the final no-normalization form as the faithful preset. The sweep also tests the earlier/experimental BatchNorm variant.
- **ResNet-LN:** dense MLP with LayerNorm same-width residual FFN blocks, configurable hidden depth, residual blocks per stage, and FFN expansion.

The MetaCentrum comparison remains a **900-run CPU grid**. For each of three GEO preprocessing arms — **Stadniuk rescaling**, **per-study RMA**, and the legacy-named **global RMA** arm — it evaluates 180 Stadniuk-MLP and 120 ResNet-LN configurations. Public GEO is split by whole connected GSE components at approximately 90/5/5. The supervised kidney-donor dataset is shown separately in the web UI: only molecular samples with **no eGFR** are added to pretraining, and those samples are independently assigned to ~90/5/5 using the same seed. Samples with any eGFR are reserved for downstream evaluation and never enter reconstruction training. The global arm requires a reference fitted only from its frozen GEO training rows.

Training uses memory-mapped NumPy matrices, optional one-batch background prefetch, CUDA pinned transfers, mixed precision, optional `torch.compile`, configurable validation cadence, live loss/MSE/R² plots, and asynchronous validation latent PCA. Model/training controls are locked during an active run. Checkpoints save the architecture, optimizer/scheduler state, preprocessing choice, and exact train/validation indices.

Stage 07 deliberately **does not train archetypes during molecular pretraining**. It keeps eGFR evaluation and later archetypal/clinical modelling separate from representation learning. The recommended first extension is shared-membership multiview AA,

```text
molecular latent:  Z ≈ W A_Z
clinical view:     F ≈ W A_F
```

so clinical information is explained by the same archetypal mixture instead of merely pushing/pulling molecular latent vectors. The UI also keeps the original contrastive formulation, a conditional-membership prior, a hybrid formulation, and a strictly nested-CV outcome-guided variant as planned ablations.

## Requirements

- Python 3.10+
- modern web browser
- optional: R + Bioconductor `affy` for raw CEL → RMA
- optional: PyArrow for the multi-GB GEO Parquet inspector

## Install for development

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

For GEO Parquet support:

```bash
python -m pip install --editable ".[dev,parquet]"
```

Start the web application:

```bash
archcon
```

or:

```python
from archcon import start

start()
```

By default it opens at `http://127.0.0.1:7860` and is accessible only from the same computer.

## Headless training and MetaCentrum sweeps

A step-by-step deployment and PBS-array tutorial is included in [`METACENTRUM.md`](METACENTRUM.md).

Stage 06 can export the current configuration plus an architecture-specific branch grid as a portable
MetaCentrum sweep bundle. Every run gets both a machine-readable JSON record and a separate
human-readable executable Python program. The Python program contains the exact generated
PyTorch architecture, objective, explicit L2 penalty, optimizer/scheduler and all configuration
values. Sweep generation first creates a `prepared/` directory containing the aligned no-eGFR supervised matrix, method-specific frozen GEO row/probe-column maps, and final train/validation/test integer indices. Each job then only memory-maps those artifacts and the selected GEO matrix. Outcome-bearing samples are never placed in the prepared training universe:

```bash
/storage/.../ArchCon/.venv/bin/python jobs/run_0001.py \
  --data-dir /storage/.../ArchCon/data \
  --output-root /storage/.../ArchCon/results
```

The corrected comparison uses the already-built Stadniuk-rescaled store and `rma_per_gse.npy`, plus a replacement `rma_global.npy` derived after the split is frozen. Run `archcon-rebuild-global-normalization --data-dir DATA --sweep-root SWEEP --replace` before any global-arm jobs. This fits a training-only quantile target from `raw_original.npy` and applies it independently to all rows. Since `raw_original.npy` is a pre-RMA probe-set summary rather than CEL/probe-level data, this representation must not be presented as exact RMA.

The matching `configs/run_0001.json` is retained for programmatic bookkeeping. The exported
bundle also contains `prepared/`, `run_array.pbs.sh`, `submit.sh`, and `manifest.csv`.
After copying the entire sweep directory to MetaCentrum and checking the path/resource
defaults, submit all combinations with:

```bash
./submit.sh
qstat -t
```

`submit.sh` uses a PBS job array; every `PBS_ARRAY_INDEX` executes exactly one
`jobs/run_XXXX.py` program. The generated PBS script creates `results/run_XXXX/` before training starts. `latest.pt` is atomically replaced there after every completed epoch and `best.pt` is updated on validation improvement; no model checkpoint waits for job completion or scratch-to-persistent copying. The
CLI imports Gradio only for browser mode, so PBS batch jobs do not start a web server.

For an interactive MetaCentrum UI, allocate a compute node first (do not run training on a
frontend), then start ArchCon there with `--no-browser`. Open OnDemand/Interactive Desktop
or forward the selected port over SSH to your workstation.

## Optional CEL → RMA setup

The browser UI calls a bundled R script that follows the source thesis path `affy::ReadAffy()` → `affy::rma()`.

Install R, then run inside R:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("affy")
```

The CEL tab contains a collapsed **R backend** panel that makes this dependency explicit. It provides:

- **Install R** → opens the official R Project download page in a new browser tab;
- **Bioconductor setup** and **affy documentation** links;
- **Check R backend** → reports the `Rscript` path/version and whether `BiocManager` and `affy` are installed;
- **Install / repair R packages** → after an explicit click, runs R to install `BiocManager` (if needed) and `affy`.

ArchCon deliberately **does not install R itself** and does not call `sudo`, `apt`, `brew`, or another system package manager. If the R package installation fails because the R library is not writable or system build dependencies are missing, the UI shows the captured R error.

The CEL/RMA tab provides dropdowns for both sides of the operation. **CEL input source** can use a detected `data/CEL/` directory, browser uploads, or a custom local directory. **RMA output** can use a temporary directory, `data/rma`, or a custom local directory. For large cohorts, prefer a local CEL directory instead of browser uploads to avoid an unnecessary temporary copy.


## Conventional local data directory

When ArchCon is launched from the project root, it automatically scans `./data`. You can
override this location with the `ARCHCON_DATA_DIR` environment variable. Large/private
data remain external to the Python package and are ignored by Git.

Recommended local layout is:

```text
data/
├── GEO_NUMPY_STORE/
│   ├── raw_original.npy
│   ├── rma_per_gse.npy
│   ├── rma_global.npy
│   ├── per_dataset_raw_original.npy
│   ├── per_dataset_rma_per_gse.npy
│   ├── sample_index.csv
│   ├── source_sample_occurrences.csv
│   ├── source_gse_occurrence_index.csv
│   ├── gse_index.csv
│   ├── probe_index.csv
│   ├── cel_manifest.csv
│   └── ... optional provenance / validation files
├── IKEM_NUMPY_STORE/           # optional compact IKEM matrix
├── GEO_STADNIUK_STORE/        # optional compact original-Stadniuk GEO matrix
├── common_probes.pkl
├── gene_annotations.csv
├── sample_metadata.csv
├── Klasifikator_20_3_24_v2.xlsx
├── egfr_data.xlsx
└── CEL/                       # optional new CEL input
```

The legacy large `expression_matrix.csv` and `geo_expr_normalized_to_ikem.parquet` are not required by the new GEO-pretraining page once their compact stores have been created. External matrix inspection remains available in the stage-01 advanced accordion.

ArchCon resolves its data directory in this order: an explicit path, the `ARCHCON_DATA_DIR` environment variable, a recognizable `./data` under the launch directory, then `data/` beside the editable ArchCon source tree. This means an editable install can be launched from another working directory while still using the project's existing `data/`; trained checkpoints remain separate under `./models/` in the launch directory.

For the reconstructed GEO normalization explorer, the simplest setup is to copy the **entire** `GEO_NUMPY_STORE/` directory produced by `repair_geo_python_store_from_v4_v2.R` into ArchCon's `data/` directory. The five `.npy` matrices and six core CSV indexes listed above are required; provenance files such as `series_alias_occurrences.csv`, `stadniuk_gsm_to_gse_mapping.csv`, `raw_archive_health.csv`, `global_rma_streaming_validation.csv`, `numpy_validation.json`, `README.txt`, and `COMPLETE.txt` are optional but should normally be copied too. `GEO_RAW/`, `GEO_RMA/`, and `.GLOBAL_RMA_WORK/` are **not** required by the web interface.

```bash
export ARCHCON_DATA_DIR=/path/to/my/data
archcon
```

## Supported data forms

Expression matrix:

- CSV / TSV / TXT
- Excel
- Parquet (requires `archcon[parquet]`)

The library immediately converts expression data to the internal convention:

```text
rows    = samples
columns = probes/features
```

The original thesis code encountered both orientations, so the UI provides `auto`, `samples_rows`, and `samples_columns` modes.

Clinical/eGFR tables:

- CSV / TSV / TXT
- Excel
- Parquet

Expected thesis-style identifiers are `Sample_ID` for clinical data and `patient` for eGFR data. CEL suffixes such as `_(PrimeView).CEL` are canonicalized automatically.

## Python API

```python
from archcon import load_expression_matrix

expr = load_expression_matrix(
    "expression_matrix.csv",
    orientation="auto",
)

print(expr.frame.shape)  # samples × probes
```

## Tests and local checks

```bash
ruff format --check .
ruff check .
pytest
```

Build and validate distributions:

```bash
rm -rf build dist
python -m build
python -m twine check dist/*
```

## Ship to another user directly

After building, send the wheel from `dist/`, for example:

```text
dist/archcon-0.5.1-py3-none-any.whl
```

The user installs it with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install archcon-0.5.1-py3-none-any.whl
archcon
```

If that user needs GEO Parquet inspection, PyArrow must also be installed:

```bash
python -m pip install pyarrow
```

CEL → RMA additionally requires the R/`affy` setup above.

## Initialize and upload to GitHub

Create an empty GitHub repository named `archcon`, then from the project root:

```bash
git init
git add .
git commit -m "Initial ArchCon data explorer"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/archcon.git
git push -u origin main
```

For browser authentication with GitHub CLI:

```bash
gh auth login --web
```

For later commits:

```bash
git add .
git commit -m "Describe the change"
git push
```

## Publish to PyPI

Before the first public release, replace the author/repository placeholders in `pyproject.toml` and the copyright placeholder in `LICENSE`.

Run the release checks:

```bash
python -m pip install --editable ".[dev,parquet]"
ruff format --check .
ruff check .
pytest
rm -rf build dist
python -m build
python -m twine check dist/*
```

Test the wheel in a clean environment:

```bash
python -m venv wheel-test
source wheel-test/bin/activate
python -m pip install dist/archcon-0.5.1-py3-none-any.whl
python -c "import archcon; print(archcon.__version__)"
archcon --help
```

For manual publication:

```bash
python -m twine upload dist/*
```

After publication, users install with:

```bash
python -m pip install archcon
archcon
```

The repository also contains GitHub Actions workflows for CI and PyPI publication. Configure PyPI Trusted Publishing before using the automatic publication workflow.

## Project structure

```text
src/archcon/
├── app.py                  # Gradio web UI
├── cli.py                  # `archcon` command
├── data/
│   ├── loading.py          # expression/table loading + ID alignment
│   ├── qc.py               # deterministic QC visualizations
│   ├── cel.py              # optional R/affy bridge
│   ├── geo.py              # lazy legacy GEO Parquet inspection
│   ├── geo_rma.py          # memory-mapped reconstructed GEO store
│   ├── pretraining.py      # split + architecture presets/diagrams
│   ├── training.py         # streaming PyTorch pretraining/checkpoints
│   └── archetypal_design.py # future IKEM AA design alternatives
└── assets/
    ├── rma_preprocess.R
    └── geo_normalization_check.png
```

The data/QC modules remain usable from Python without launching the web interface. Model code should later depend on these data-layer APIs, never on Gradio components.


## GEO autoencoder pretraining and architecture experiments (0.5.x)

Stage **06 · Molecular AE** contains a streaming PyTorch pretraining loop focused on the **Stadniuk MLP** and **ResNet-LN** model families. The architecture diagram, generated PyTorch audit code, and actual model builder share one explicit execution plan; in ResNet-LN each width-changing projection is followed by same-width `x + F(x)` blocks. Install the optional backend with:

```bash
python -m pip install -e ".[training,dev]"
```

Training reads GEO through memory mapping and virtually appends the much smaller outcome-blind supervised subset, so the multi-GB GEO matrix is not duplicated. The current combined train/validation/test split from stage 05 is used. For exported sweeps, the final logical row indices, method-specific GEO probe mappings, and the supervised 42,917-probe subset are frozen once into `prepared/` before any batch job is generated. Test rows and all samples with eGFR are never passed to the training backend.

The UI streams running training loss every configurable number of batches, evaluates validation loss/MSE/R² at a configurable epoch cadence, and schedules validation latent-space PCA in a one-worker background executor every configurable number of epochs. Model/training controls are locked while a run is active. Batch jobs write `latest.pt` directly into their persistent `run_XXXX/` directory after each completed epoch; `best.pt` is updated immediately when validation improves. The **Stop** button is cooperative and saves a recovery checkpoint after the current batch.

Checkpoints are written outside the data/library tree, relative to the directory where ArchCon is launched:

```text
./models/geo_ae_<run-id>/
├── latest.pt
├── best.pt
└── stopped.pt   # created only after a cooperative stop
```

The large row-major matrix cache remains under `data/training_cache/`; moving checkpoints does not duplicate or relocate the expression data.

`latest.pt` is replaced after every completed epoch; `best.pt` is replaced only when the validation objective improves. Uploading a trusted ArchCon checkpoint restores its architecture/training controls. It can then be used either as weights-only initialization or to resume optimizer/scheduler/history state. Exact resume additionally requires the same stage-02 preprocessing matrix and the same train/validation indices; weights-only loading is allowed for a new split or normalization experiment.

The old generic text/log viewer and its dedicated text-reader dependency have been removed; ArchCon now focuses on the GEO preprocessing/training workflow.

## Molecular and clinical eGFR benchmark

Run this only after the sweep has enough completed checkpoints and the validation-selected
winner has been evaluated once on the frozen molecular test rows:

```bash
cd /storage/praha1/home/anuarali/DP/ARCHCON
source .venv/bin/activate

python -m pip install --editable ".[training,parquet]"
# R must provide lme4, e.g. install.packages("lme4") in your MetaCentrum R library.

archcon-evaluate-egfr \
  --sweep-root sweeps/archcon-pretrain-058 \
  --data-dir data
```

Default comparisons are:

- `time_only`: `eGFR ~ time + (1 | patient)`;
- `winner_z`: `eGFR ~ time + z1 + ... + zd + (1 | patient)`;
- the best validation-selected alternative architecture, when available;
- PCA with the winner's latent dimension, fitted anew inside each molecular training fold.
- separate `KDRI_8 × time`, donor-age × time, and cold-ischemia × time baselines;
- a combined `(KDRI_8 + donor age + cold ischemia) × time` baseline;
- winner-z and PCA models augmented with KDRI or all three clinical predictors.

Time is categorical (`7d`, `3m`, `6m`, `12m`). The default is 5 repeats × 5 folds,
grouped by donor so sibling kidneys cannot cross train/test. z standardization, PCA, clinical
median imputation, and clinical standardization are training-fold only. Test patients receive population-level predictions (`re.form = NA`) because
their patient random intercepts are unseen. No eGFR value participates in encoder selection or z
construction. Use `--preprocessing-winners` to add the validation winner from every preprocessing
arm, `--no-clinical` to reproduce the molecular-only comparison, or `--embeddings-only` to stop after exporting z.

Outputs are written under `sweeps/archcon-pretrain-058/downstream/molecular_egfr/`:

```text
embeddings/
├── embeddings.csv
├── embeddings.parquet       # when PyArrow is installed
├── embeddings.npz
├── winner_z.npy
├── best_non_stadniuk_z.npy  # when available
└── metadata.json
mixed_models/
├── fold_metrics.csv
├── fold_metrics_with_deltas.csv
├── oof_predictions.csv
├── summary.csv
├── winner_pairwise_summary.csv
└── clinical_incremental_summary.csv
```

`mean_delta_vs_time = RMSE(time-only) - RMSE(model)` is positive when molecular features
improve held-out prediction. `mean_winner_gain = RMSE(baseline) - RMSE(winner z)` is positive
when the selected winner's z outperforms that baseline on matched folds.
`mean_gain` in `clinical_incremental_summary.csv` is
`RMSE(clinical baseline) - RMSE(clinical + molecular features)`; positive values mean the
molecular representation adds predictive value beyond the matched clinical model.


### Reconstruction and latent-space objectives

The architecture presets now treat normalization and regularization explicitly. The faithful **Stadniuk MLP** preset uses ReLU, hidden dropout `0.10`, no hidden normalization, and exposes explicit L2; the MetaCentrum grid tests the same MLP both with and without BatchNorm. **ResNet-LN** fixes LayerNorm inside residual FFN blocks and defaults to dropout `0.00`.

Stage 06 separates the following representation-learning questions that are easy to conflate:

1. **What self-supervised reconstruction task should GEO pretraining solve?** Plain full-profile reconstruction is the thesis baseline; masked denoising hides part of the profile and asks the encoder to infer it from the rest.
2. **How should reconstruction error be measured?** MSE, Huber, and MSE+cosine are continuous-value reconstruction objectives.
3. **Should latent space have an explicit distributional geometry?** β-VAE adds a per-sample KL prior; WAE/InfoVAE-style training adds an aggregate MMD prior.

Available objectives:

- `MSE / L2 reconstruction` — thesis reproduction baseline.
- `Smooth L1 / Huber` — robust reconstruction when large errors/outliers dominate MSE.
- `MSE + profile cosine` — preserves both absolute expression and whole-profile direction.
- `β-VAE · MSE + KL` — variational encoder with configurable KL strength and linear warm-up.
- `WAE / InfoVAE · MSE + MMD` — deterministic encoder with aggregate latent-prior matching.
- `Masked denoising reconstruction · masked MSE` — randomly hides a configurable fraction of probes and scores reconstruction only at hidden positions, while reporting clean-input validation MSE/R² separately. This adapts the masked-value pretraining idea used in recent transcriptomic models without changing ArchCon into a transformer.

The earlier softmax-distribution experiment is no longer offered for new runs. Its implementation remains only for legacy checkpoint compatibility because converting RMA profiles into a probe-wise probability distribution changes the target without a strong biological/statistical justification.

The live training chart now distinguishes a batch-level exponential moving average from the exact epoch training mean. Validation is plotted using the same selected objective. Plain validation MSE and global element-wise R² are shown in separate panels, and KL/MMD/cosine auxiliary terms receive a third panel when relevant. This avoids interpreting a cumulative within-epoch running average as an epoch loss or comparing a latent regularizer directly with MSE.

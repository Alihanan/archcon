"""Local ArchCon web interface for GEO preprocessing and autoencoder pretraining."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd

from .batch import (
    build_run_request,
    generate_sweep_bundle,
    recommended_comparison_grid_json,
)
from .data.archetypal_design import (
    AA_DESIGN_OPTIONS,
    AA_SHARED,
    aa_design_markdown,
    aa_design_svg,
    aa_research_markdown,
)
from .data.cel import (
    discover_cel_files,
    install_rma_dependencies,
    rma_environment_status,
    run_rma,
)
from .data.defaults import (
    data_directory_status,
    detected_default_paths,
    project_data_layout,
)
from .data.geo import inspect_geo_parquet
from .data.geo_rma import (
    METHOD_GLOBAL_RMA,
    METHOD_PER_GSE_RMA,
    METHOD_RAW,
    SCOPE_AGGREGATE,
    SCOPE_SERIES,
    geo_store_status,
    load_geo_expression_store,
    normalization_pipeline_markdown,
    raw_geo_overview_markdown,
)
from .data.geo_rma import (
    dataset_heading as geo_dataset_heading,
)
from .data.geo_rma import (
    dataset_metadata_markdown as geo_dataset_metadata_markdown,
)
from .data.geo_rma import (
    plot_normalization_comparison as plot_geo_normalization_comparison,
)
from .data.geo_rma import (
    plot_sample_boxplots as plot_geo_sample_boxplots,
)
from .data.geo_rma import (
    plot_scope_histogram as plot_geo_scope_histogram,
)
from .data.geo_rma import (
    plot_scope_pca as plot_geo_scope_pca,
)
from .data.geo_rma import (
    plot_selected_sample_method as plot_geo_selected_sample_method,
)
from .data.geo_rma import (
    scope_summary as geo_scope_summary,
)
from .data.geo_rma import (
    selected_sample_metadata as geo_selected_sample_metadata,
)
from .data.geo_rma import (
    selected_sample_plot as plot_geo_selected_sample,
)
from .data.geo_rma import (
    source_links_markdown as geo_source_links_markdown,
)
from .data.loading import DataWorkspace, align_workspace, load_expression_matrix, load_table
from .data.pretraining import (
    ACTIVATION_OPTIONS,
    ARCH_DENSE,
    ARCH_STADNIUK,
    ARCH_RESNET_LN,
    ARCHITECTURE_OPTIONS,
    ARCHITECTURE_FAMILIES,
    ARCHITECTURE_PRESET_OPTIONS,
    LOSS_MSE,
    LOSS_OPTIONS,
    architecture_preset_from_label,
    architecture_research_markdown,
    architecture_summary_markdown,
    autoencoder_architecture_svg,
    create_train_validation_split,
    loss_theory_markdown,
    normalize_loss_name,
    parse_hidden_widths,
    save_split_csv,
    split_summary_markdown,
    validate_loaded_split,
)
from .data.qc import (
    plot_clinical_variable,
    plot_dataset_vs_reference,
    plot_distribution_overlay,
    plot_egfr_trajectories,
    plot_missingness,
    plot_pca,
    plot_raw_vs_rma,
    plot_sample_boxplots,
    plot_sample_vs_reference,
)
from .data.training import (
    CHECKPOINT_MODES,
    CHECKPOINT_WEIGHTS,
    DEVICE_OPTIONS,
    OPTIMIZER_OPTIONS,
    PRECISION_OPTIONS,
    LR_SCHEDULE_COSINE,
    LR_SCHEDULE_OPTIONS,
    TrainingConfig,
    inspect_checkpoint,
    model_execution_markdown,
    plot_training_history,
    plot_validation_metrics,
    pytorch_model_code,
    request_training_stop,
    train_autoencoder_stream,
    training_backend_status,
)
from .data.training import (
    plot_latent_pca as plot_training_latent_pca,
)
from .data.training_sources import (
    TRAINING_PREPROCESSING_OPTIONS,
    create_shared_preprocessing_split,
    load_pretraining_source,
    load_training_source,
    load_ikem_source,
    split_rows_for_source,
    validate_pretraining_split,
)
from .data.supervised import (
    classify_supervised_samples,
    plot_outcome_group_counts,
    plot_supervised_expression_pca,
    supervised_overview_markdown,
)

LOGGER = logging.getLogger(__name__)

APP_UPLOAD_LIMIT = "2gb"

CEL_SOURCE_UPLOAD = "Upload CEL files"
CEL_SOURCE_DETECTED = "Detected data/CEL directory"
CEL_SOURCE_CUSTOM = "Custom local CEL directory"

RMA_OUTPUT_TEMPORARY = "Temporary directory"
RMA_OUTPUT_DATA = "data/rma"
RMA_OUTPUT_CUSTOM = "Custom local directory"

# One background worker is enough for the local data browser.  The guard makes
# repeated button mashing a no-op instead of building an unbounded queue of
# expensive PCA/histogram jobs. Training uses its own independent worker path.
_VIEW_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archcon-view")
_VIEW_TASK_LOCK = threading.Lock()
_VIEW_ACTIVE = None

_LOADING_IDLE_HTML = '<div class="archcon-busy-backdrop"></div>'
_LOADING_ACTIVE_HTML = """
<div class="archcon-busy-backdrop active" role="status" aria-live="polite">
  <div class="archcon-busy-card">
    <div class="archcon-spinner" aria-label="Loading"></div>
    <div><b>Loading new view…</b><br><span>Keep using the current page while it finishes.</span></div>
  </div>
</div>
"""


def _pending_button_js(elem_id: str) -> str:
    """Client-side pending style for an async navigation button.

    The currently active button stays fully selected. The requested button gets
    a translucent accent immediately, before Python starts expensive work.
    While that request is pending, the other buttons in the same async group
    ignore pointer clicks so repeated clicks cannot build a request backlog.
    """
    return f"""(...args) => {{
      if (!document.documentElement.classList.contains('archcon-view-pending')) {{
        document.querySelectorAll('.archcon-pending-selection').forEach((el) =>
          el.classList.remove('archcon-pending-selection')
        );
        const target = document.getElementById('{elem_id}');
        if (target) target.classList.add('archcon-pending-selection');
        document.documentElement.classList.add('archcon-view-pending');
      }}
      return args;
    }}"""


_CLEAR_PENDING_JS = """() => {
  document.querySelectorAll('.archcon-pending-selection').forEach((el) =>
    el.classList.remove('archcon-pending-selection')
  );
  document.documentElement.classList.remove('archcon-view-pending');
}"""


def _async_view_result(fn, n_outputs: int, *args):
    """Run one expensive dashboard callback off the Gradio event thread.

    At most one data-view task can exist at a time.  A second click while the
    first task is running is ignored, which prevents a backlog of PCA/plot jobs.
    """
    global _VIEW_ACTIVE
    with _VIEW_TASK_LOCK:
        if _VIEW_ACTIVE is not None and not _VIEW_ACTIVE.done():
            yield (_LOADING_ACTIVE_HTML, *[gr.skip() for _ in range(n_outputs)])
            return
        future = _VIEW_EXECUTOR.submit(fn, *args)
        _VIEW_ACTIVE = future

    yield (_LOADING_ACTIVE_HTML, *[gr.skip() for _ in range(n_outputs)])
    try:
        while not future.done():
            time.sleep(0.08)
        result = future.result()
        if not isinstance(result, tuple):
            result = (result,)
        if len(result) != n_outputs:
            raise RuntimeError(f"Async view returned {len(result)} outputs; expected {n_outputs}.")
        yield (_LOADING_IDLE_HTML, *result)
    finally:
        with _VIEW_TASK_LOCK:
            if _VIEW_ACTIVE is future:
                _VIEW_ACTIVE = None


PIPELINE_DETAILS = {
    "input": r"""
### 01 · Unsupervised data · public GEO

**GEO** is NCBI's public Gene Expression Omnibus. In this project every array uses the same Affymetrix PrimeView platform.

$$
\text{GSE study}\;\supset\;\text{GSM samples}\;\supset\;\text{CEL intensities}.
$$

ArchCon starts from the reconstructed **before-RMA** signal for the common probe sets. Pick a GSE in the table below to inspect that study; no normalization choice is made on this page.
""",
    "supervised": r"""
### 02 · Supervised dataset

This is the smaller kidney-donor cohort used later for outcome prediction. The important first separation is very simple:

- samples **with eGFR** are reserved for downstream evaluation;
- samples **without eGFR** contain molecular information but no target outcome, so they may join outcome-blind autoencoder pretraining.

The web page shows this separation explicitly before any model is trained.
""",
    "rma": r"""
### 03 · GEO preprocessing

RMA makes multiple arrays comparable through three operations:

$$
\text{CEL}\xrightarrow{\text{background correction}}X^{(b)}
\xrightarrow{\text{quantile normalization}}X^{(q)}
\xrightarrow{\text{median polish}}X^{(\mathrm{RMA})}.
$$

A compact view of quantile normalization is

$$
\bar x_{(r)}=\frac{1}{m}\sum_{j=1}^{m}x_{(r)j},
\qquad
x'_{ij}=\bar x_{(\operatorname{rank}(x_{ij}))}.
$$

Median polish can be interpreted through the robust additive model

$$
y_{ij}=\mu+\alpha_i+\beta_j+\varepsilon_{ij}.
$$

Only this stage currently crosses the language boundary:

`ArchCon (Python) → Rscript → Bioconductor affy → normalized matrix → Python`

[Install R](https://cran.r-project.org/) · [Bioconductor](https://bioconductor.org/install/) · [`affy`](https://bioconductor.org/packages/affy/) · [RMA paper](https://doi.org/10.1093/biostatistics/4.2.249)
""",
    "matrix": r"""
### 04 · Canonical expression matrix

Regardless of how the source file is stored, ArchCon immediately converts expression data to one internal convention:

$$
X\in\mathbb{R}^{n\times p},
\qquad
\text{rows}=\text{samples},
\qquad
\text{columns}=\text{probes}.
$$

This keeps every later operation explicit and removes silent transpose mistakes. Automatic orientation is only a convenience; it can always be overridden during import.
""",
    "align": r"""
### 04 · Align samples and metadata

Expression, clinical information, and post-transplant outcomes remain separate tables but are matched through canonical sample/patient identifiers:

$$
I_{XC}=I_X\cap I_C,
\qquad
I_{XE}=I_X\cap I_{\mathrm{eGFR}}.
$$

ArchCon reports IDs present on only one side instead of silently dropping them. This distinction matters because a sample may be valid for representation learning while lacking an outcome required for evaluation.
""",
    "split": r"""
### 05 · Molecular train / validation / test split

Pretraining is performed on the **canonical unique-GSM matrix**, but assignment is made by **source GEO study**, not by individual sample. Related SubSeries/SuperSeries connected through a shared physical GSM are kept in the same component.

With seed $s$, ArchCon creates disjoint study-level partitions

$$
I_{\mathrm{train}}\cap I_{\mathrm{val}}=
I_{\mathrm{train}}\cap I_{\mathrm{test}}=
I_{\mathrm{val}}\cap I_{\mathrm{test}}=\varnothing.
$$

The default is approximately **90% train / 5% validation / 5% test**. GEO is split by whole connected study components. Supervised-dataset samples without eGFR are independently split 90/5/5 and appended to the same molecular partitions. Samples with eGFR never enter this pretraining split. Validation selects checkpoints and hyperparameters; test rows remain blinded during the sweep.
""",
    "latent": r"""
### 06 · Molecular representation · autoencoder pretraining

Before archetypal constraints are introduced, the molecular pretraining pool is used to learn a compact representation with a standard encoder-decoder. It contains public GEO plus only supervised-dataset samples that have no eGFR:

$$
z=E_\phi(x),
\qquad
\hat x=D_\theta(z).
$$

The thesis pretraining stage uses a reconstruction-only autoencoder. Hidden layers are dense, the decoder mirrors the encoder, ReLU is used in the hidden layers, and no activation is applied at the latent code. In the reported ArchCon experiments the hidden widths were $256\rightarrow64$ and the latent dimension was treated separately.

The comparison keeps the thesis-style **256 → 64** hidden encoder as one anchor while expanding depth and latent-size choices. All model configurations use the same combined molecular 90/5/5 assignment: GEO is grouped by connected source-GSE component, and supervised samples without eGFR are split independently by sample. Per-study RMA is performed within each source GSE. The legacy-named Global RMA arm now requires a reference fitted only on the frozen GEO training rows and applied independently to held-out rows. Because `raw_original.npy` is already summarized to probe-set PM medians, this corrected representation is train-reference quantile normalization followed by log2, not exact CEL-level RMA. Validation is tracked live and checkpoints are saved automatically. Public GEO plus supervised-dataset samples without eGFR form the molecular-only pretraining pool; samples with eGFR are not evaluated or trained on here.
""",
    "aa": r"""
### 07 · Downstream outcome / archetypal design · configuration only

Molecular pretraining remains a plain autoencoder. The downstream stage uses the **supervised dataset**, where frozen molecular latent codes can later be evaluated against eGFR and extended with archetypal/clinical models.

No archetype class labels are required for ordinary AA. Outcome-bearing samples stay outside the molecular-pretraining stage and enter only here under a leakage-safe cross-validation protocol.
""",
    "qc": r"""
### 05 · Quality control

QC visualizations inspect the data; they are **not inserted into the modelling pipeline**. Distribution plots check scale and array consistency, while PCA provides a deterministic low-dimensional diagnostic view:

$$
Z=(X-\bar X)V_{1:2}.
$$

For outcome data, longitudinal kidney function is inspected directly as

$$
y_i(t),
\qquad
t\in\{7\mathrm{d},3\mathrm{m},6\mathrm{m},12\mathrm{m}\}.
$$

These views are intended to expose outliers, scale shifts, missingness, unexpected groups, and alignment problems before any ArchCon model is fitted.
""",
}

CSS = r"""
:root {
  --archcon-content: 1360px;
  --archcon-reading: 820px;
}

.gradio-container {
  max-width: var(--archcon-content) !important;
  margin: 0 auto !important;
  padding: 28px 28px 56px !important;
  font-size: 16px !important;
}

.archcon-shell {
  width: 100%;
}

.archcon-hero {
  max-width: 760px;
  margin: 4px auto 28px auto !important;
  text-align: center;
}

.archcon-hero h1 {
  margin-bottom: 0.2rem !important;
  font-size: 2.15rem !important;
  letter-spacing: -0.035em;
}

.archcon-hero p {
  margin: 0.35rem auto 0 auto !important;
  max-width: 650px;
  font-size: 1.02rem;
  line-height: 1.65;
  opacity: 0.76;
}

.section-intro {
  max-width: var(--archcon-reading);
  margin: 0 auto 18px auto !important;
  text-align: center;
  line-height: 1.65;
}

.section-intro h2,
.section-intro h3 {
  text-align: center;
  letter-spacing: -0.02em;
}

.pipeline-strip {
  max-width: 1040px;
  margin: 14px auto 8px auto !important;
  gap: 9px !important;
  display: grid !important;
  grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
}

.pipeline-step,
.pipeline-step button {
  min-width: 0 !important;
  min-height: 92px !important;
  white-space: pre-line !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
  line-height: 1.25 !important;
  transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
}

.pipeline-step button {
  font-size: 0.96rem !important;
}

.pipeline-step button:hover {
  transform: translateY(-2px);
}

.pipeline-step button.primary,
.normalization-choice button.primary {
  box-shadow: 0 0 0 3px rgba(255, 112, 18, 0.20) !important;
  transform: translateY(-1px);
}

.beginner-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  max-width: 1040px;
  margin: 0 auto 16px auto;
}

.beginner-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 12px 14px;
  background: var(--background-fill-secondary);
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: center;
}

.beginner-card b {
  font-size: 1.05rem;
}

.beginner-card span {
  font-size: 0.84rem;
  opacity: 0.74;
  line-height: 1.35;
}

.normalization-choice-strip {
  max-width: 980px;
  margin: 8px auto 18px auto !important;
  display: grid !important;
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
  gap: 10px !important;
}

.normalization-choice,
.normalization-choice button {
  min-height: 72px !important;
  white-space: pre-line !important;
  border-radius: 12px !important;
  font-weight: 650 !important;
}

.workflow-divider {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 30px 0 12px 0;
  color: var(--body-text-color-subdued);
  font-weight: 650;
}

.workflow-divider::before,
.workflow-divider::after {
  content: "";
  height: 1px;
  background: var(--border-color-primary);
  flex: 1;
}

.workflow-divider span {
  white-space: nowrap;
}

.full-width-scroll-table {
  width: 100% !important;
  max-width: none !important;
}

.pipeline-later,
.pipeline-later button {
  opacity: 0.52;
}

.reference-status {
  max-width: var(--archcon-reading);
  margin: 10px auto 14px auto !important;
  text-align: center;
  padding: 8px 12px;
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
  background: var(--background-fill-secondary);
}

.reference-actions {
  max-width: var(--archcon-reading);
  margin-left: auto !important;
  margin-right: auto !important;
}

.pipeline-hint {
  margin: 6px auto 0 auto !important;
  text-align: center;
  font-size: 0.9rem;
  opacity: 0.66;
}

.pipeline-detail {
  max-width: var(--archcon-reading);
  margin: 22px auto 0 auto !important;
  padding: 4px 18px 6px 18px;
  font-size: 1.04rem;
  line-height: 1.75;
}

.pipeline-detail h3 {
  text-align: center;
  font-size: 1.35rem !important;
  margin-bottom: 1rem !important;
}

.pipeline-detail p,
.reading-width p {
  line-height: 1.75 !important;
}

.pipeline-detail .katex,
.reading-width .katex {
  font-size: 1.18em !important;
}

.pipeline-detail .katex-display > .katex,
.reading-width .katex-display > .katex {
  font-size: 1.38em !important;
}

.pipeline-detail .katex-display,
.reading-width .katex-display {
  margin: 1.45rem auto !important;
  text-align: center !important;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.2rem 0;
}

.reading-width {
  max-width: var(--archcon-reading);
  margin-left: auto !important;
  margin-right: auto !important;
  font-size: 1.02rem;
  line-height: 1.72;
}

.compact-note {
  max-width: var(--archcon-reading);
  margin: 10px auto !important;
  padding: 11px 14px;
  border-left: 3px solid var(--color-accent);
  border-radius: 6px;
  background: var(--background-fill-secondary);
  line-height: 1.55;
}

.backend-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
  max-width: 900px;
  margin: 12px auto 18px auto;
}

.backend-box {
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  padding: 9px 12px;
  background: var(--background-fill-secondary);
  text-align: center;
  min-width: 112px;
}

.backend-arrow {
  font-size: 1.1rem;
  opacity: 0.55;
}

.backend-pill {
  display: inline-block;
  padding: 2px 8px;
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 600;
  opacity: 0.82;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
  gap: 10px;
  margin: 16px 0 10px 0;
}

.metric {
  border: 1px solid var(--border-color-primary);
  border-radius: 11px;
  padding: 11px 12px;
  background: var(--background-fill-secondary);
  text-align: center;
}

.metric .value {
  font-size: 1.4rem;
  font-weight: 650;
  letter-spacing: -0.025em;
}

.metric .label {
  margin-top: 2px;
  font-size: 0.79rem;
  opacity: 0.72;
}

.note {
  border-left: 3px solid var(--color-accent);
  border-radius: 6px;
  padding: 9px 12px;
  margin: 10px 0;
  background: var(--background-fill-secondary);
}

.result-area {
  margin-top: 12px !important;
}

.minimal-accordion {
  margin-top: 10px !important;
}


.geo-rma-toolbar {
  max-width: 1040px;
  margin: 0 auto 10px auto !important;
}

.geo-rma-store-status {
  max-width: 1040px;
  margin: 10px auto 14px auto !important;
}

.geo-rma-table {
  font-size: 0.9rem;
}

.geo-rma-help {
  max-width: 920px;
  margin-left: auto !important;
  margin-right: auto !important;
}

.geo-dashboard-intro {
  max-width: 980px;
}

.geo-dashboard {
  align-items: flex-start !important;
  gap: 18px !important;
}

.geo-browser-panel {
  position: sticky;
  top: 12px;
  align-self: flex-start;
  border: 1px solid var(--border-color-primary);
  border-radius: 16px;
  padding: 14px !important;
  background: var(--background-fill-secondary);
}

.geo-browser-heading h3 {
  margin-top: 0.35rem !important;
  margin-bottom: 0.25rem !important;
  letter-spacing: -0.02em;
}

.geo-browser-heading p {
  margin-top: 0 !important;
  opacity: 0.72;
  font-size: 0.9rem;
}

.geo-aggregate-button button {
  justify-content: flex-start !important;
  min-height: 42px !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
}

.geo-dataset-browser {
  margin-top: 4px !important;
}

.geo-detail-panel {
  min-width: 0;
}

.geo-selection-heading {
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 8px !important;
}

.geo-selection-heading h2 {
  margin-bottom: 0.35rem !important;
  letter-spacing: -0.03em;
}

.geo-method-switch {
  margin: 4px 0 10px 0 !important;
}

.geo-metadata-card,
.geo-pipeline-card,
.geo-sample-metadata {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 12px 14px;
  background: var(--background-fill-secondary);
  min-height: 100%;
}

.geo-metadata-card h3,
.geo-pipeline-card h3,
.geo-sample-metadata h3 {
  margin-top: 0 !important;
}

.geo-section-heading {
  margin-top: 14px !important;
}

.geo-section-heading h3 {
  margin-bottom: 0.25rem !important;
  letter-spacing: -0.02em;
}

.geo-section-heading p {
  margin-top: 0 !important;
  opacity: 0.72;
}

.geo-sample-browser {
  margin-bottom: 10px !important;
}


.pipeline-stage-panel {
  max-width: 1280px;
  margin: 20px auto 0 auto !important;
}

.pipeline-stage-theory {
  max-width: 920px;
  margin: 0 auto 22px auto !important;
  line-height: 1.65;
}

.pipeline-stage-theory h3 {
  text-align: center;
  letter-spacing: -0.02em;
}

.stage-subheading {
  margin-top: 18px !important;
  margin-bottom: 8px !important;
}

.split-controls,
.model-controls {
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  padding: 14px;
  background: var(--background-fill-secondary);
}

.architecture-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  padding: 14px 18px;
  font-size: 1.02rem;
}

.architecture-scroll-shell {
  width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  border: 1px solid var(--border-color-primary);
  border-radius: 16px;
  background: var(--background-fill-secondary);
  padding: 20px 16px 8px 16px;
  scrollbar-gutter: stable;
}

.architecture-scroll-shell svg {
  display: block;
  max-width: none !important;
  min-width: 1580px;
  margin: 0 auto;
}

.architecture-viewport {
  width: 100% !important;
  font-size: 1.08rem !important;
}

.preset-note {
  margin: 2px 4px 12px 4px !important;
  opacity: 0.84;
}

.model-config-row,
.training-config-row {
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  padding: 12px;
  background: var(--background-fill-secondary);
  margin-bottom: 10px !important;
}

.training-actions {
  align-items: end !important;
  gap: 10px !important;
}

.training-live-grid {
  gap: 14px !important;
}

.training-status-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 12px;
  padding: 12px 14px;
  background: var(--background-fill-secondary);
}

.checkpoint-grid {
  align-items: start !important;
}

.utility-accordion {
  max-width: 1040px;
  margin: 28px auto 0 auto !important;
}

#archcon-loading-overlay {
  position: relative;
  z-index: 9999;
}

.archcon-busy-backdrop {
  display: none;
}

/* Non-blocking toast: the old view remains fully usable while a new one loads. */
.archcon-busy-backdrop.active {
  position: fixed;
  right: 24px;
  bottom: 24px;
  display: block;
  z-index: 99999;
  pointer-events: none;
  animation: archcon-toast-in .16s ease-out;
}

.archcon-busy-card {
  display: flex;
  align-items: center;
  gap: 13px;
  min-width: 285px;
  max-width: min(390px, calc(100vw - 32px));
  border: 1px solid var(--border-color-primary);
  border-radius: 14px;
  background: color-mix(in srgb, var(--background-fill-primary) 96%, transparent);
  padding: 13px 16px;
  box-shadow: 0 12px 34px rgba(0,0,0,.18);
  backdrop-filter: blur(10px);
  font-size: .96rem;
}

.archcon-busy-card span { opacity: .68; }

.archcon-spinner {
  width: 28px;
  height: 28px;
  border: 3px solid var(--border-color-primary);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: archcon-spin .8s linear infinite;
  flex: 0 0 auto;
}

/* Pending selection = the destination requested by the user, but not loaded yet.
   The old primary button deliberately stays fully orange until the swap finishes. */
.normalization-choice.archcon-pending-selection button {
  background: color-mix(in srgb, var(--color-accent) 38%, transparent) !important;
  border-color: var(--color-accent) !important;
  color: var(--body-text-color) !important;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--color-accent) 50%, transparent);
  transition: background .12s ease, border-color .12s ease, box-shadow .12s ease;
}

html.archcon-view-pending .normalization-choice button {
  cursor: progress !important;
  pointer-events: none !important;
}

@keyframes archcon-toast-in {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@keyframes archcon-spin {
  to { transform: rotate(360deg); }
}

.aa-design-shell svg { min-width: 1500px; }

@media (max-width: 760px) {
  .gradio-container {
    padding: 18px 14px 40px !important;
  }

  .archcon-hero h1 {
    font-size: 1.85rem !important;
  }

  .pipeline-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 7px !important;
  }

  .beginner-grid,
  .normalization-choice-strip {
    grid-template-columns: 1fr !important;
  }

  .workflow-divider span {
    white-space: normal;
    text-align: center;
  }

  .pipeline-detail {
    padding-left: 4px;
    padding-right: 4px;
  }

  .geo-browser-panel {
    position: static;
  }
}
"""


def _pipeline_detail_callback(step: str) -> str:
    """Return the expandable explanation for one pipeline stage."""
    return PIPELINE_DETAILS.get(step, "")


def _resolve_path(uploaded: str | None, local_path: str | None) -> str | None:
    if uploaded:
        return uploaded
    if local_path and local_path.strip():
        return str(Path(local_path.strip()).expanduser())
    return None


def _cel_source_choices(detected_directory: str) -> list[str]:
    """Return CEL-source choices appropriate for the detected project data."""
    choices = [CEL_SOURCE_UPLOAD]
    if detected_directory:
        choices.append(CEL_SOURCE_DETECTED)
    choices.append(CEL_SOURCE_CUSTOM)
    return choices


def _default_cel_source(detected_directory: str) -> str:
    return CEL_SOURCE_DETECTED if detected_directory else CEL_SOURCE_UPLOAD


def _cel_source_mode_callback(mode: str, detected_directory: str):
    """Show only the input control relevant to the selected CEL source."""
    if mode == CEL_SOURCE_DETECTED and detected_directory:
        return (
            gr.update(visible=False),
            gr.update(value=detected_directory, visible=True, interactive=False),
        )
    if mode == CEL_SOURCE_CUSTOM:
        return (
            gr.update(visible=False),
            gr.update(value="", visible=True, interactive=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
    )


def _rma_output_mode_callback(mode: str, data_dir: str):
    """Update the optional output-path field for the selected RMA destination."""
    if mode == RMA_OUTPUT_CUSTOM:
        return gr.update(value="", visible=True, interactive=True)
    if mode == RMA_OUTPUT_DATA:
        path = project_data_layout(data_dir).root / "rma"
        return gr.update(value=str(path), visible=True, interactive=False)
    return gr.update(value="", visible=False, interactive=False)


def _resolve_rma_output_directory(mode: str, data_dir: str, custom_path: str) -> str | None:
    if mode == RMA_OUTPUT_TEMPORARY:
        return None
    if mode == RMA_OUTPUT_DATA:
        return str(project_data_layout(data_dir).root / "rma")
    if custom_path and custom_path.strip():
        return str(Path(custom_path.strip()).expanduser())
    raise ValueError("Choose an RMA output directory or select another output option.")


def _scan_data_directory_callback(data_dir: str, rma_output_mode: str):
    """Rescan a conventional data directory and update all path defaults."""
    layout = project_data_layout(data_dir)
    defaults = detected_default_paths(layout)
    cel_default = _default_cel_source(defaults["cel"])
    return (
        data_directory_status(layout),
        defaults["expression"],
        defaults["clinical"],
        defaults["egfr"],
        defaults["reference"],
        defaults["geo"],
        defaults["geo_rma"],
        defaults["cel"],
        gr.update(choices=_cel_source_choices(defaults["cel"]), value=cel_default),
        gr.update(
            visible=cel_default == CEL_SOURCE_UPLOAD,
        ),
        gr.update(
            value=defaults["cel"],
            visible=cel_default == CEL_SOURCE_DETECTED,
            interactive=False,
        ),
        _rma_output_mode_callback(rma_output_mode, str(layout.root)),
    )


def _workspace_metrics(workspace: DataWorkspace, summary: dict[str, object] | None = None) -> str:
    if workspace.expression is None:
        return '<div class="note">No expression matrix loaded yet.</div>'
    summary = summary or align_workspace(workspace)
    expression = workspace.expression
    clinical_match = summary.get("clinical_matched", "—")
    egfr_match = summary.get("egfr_matched", "—")
    egfr_valid = summary.get("egfr_valid_matched", "—")
    return f"""
<div class="metric-row">
  <div class="metric"><div class="value">{expression.n_samples:,}</div><div class="label">expression samples</div></div>
  <div class="metric"><div class="value">{expression.n_probes:,}</div><div class="label">probes / features</div></div>
  <div class="metric"><div class="value">{clinical_match}</div><div class="label">matched clinical IDs</div></div>
  <div class="metric"><div class="value">{egfr_match}</div><div class="label">matched eGFR IDs</div></div>
  <div class="metric"><div class="value">{egfr_valid}</div><div class="label">matched with any eGFR</div></div>
</div>
"""


def _alignment_table(summary: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, key_a, key_b in (
        ("Clinical", "clinical_unmatched_expression", "clinical_unmatched_table"),
        ("eGFR", "egfr_unmatched_expression", "egfr_unmatched_table"),
    ):
        if key_a not in summary and key_b not in summary:
            continue
        expression_only = list(summary.get(key_a, []))
        table_only = list(summary.get(key_b, []))
        max_len = max(len(expression_only), len(table_only), 1)
        for index in range(max_len):
            rows.append(
                {
                    "source": source,
                    "expression only": expression_only[index]
                    if index < len(expression_only)
                    else "",
                    "table only": table_only[index] if index < len(table_only) else "",
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source", "expression only", "table only"])
    return pd.DataFrame(rows)


def _load_workspace_callback(
    expression_upload: str | None,
    expression_path: str,
    orientation: str,
    clinical_upload: str | None,
    clinical_path: str,
    egfr_upload: str | None,
    egfr_path: str,
    workspace: DataWorkspace,
    reference_expression,
):
    try:
        expression_file = _resolve_path(expression_upload, expression_path)
        if expression_file is None:
            raise ValueError("Choose an expression matrix or enter its local path.")

        workspace = DataWorkspace()
        workspace.expression = load_expression_matrix(expression_file, orientation=orientation)

        clinical_file = _resolve_path(clinical_upload, clinical_path)
        if clinical_file is not None:
            workspace.clinical = load_table(clinical_file)

        egfr_file = _resolve_path(egfr_upload, egfr_path)
        if egfr_file is not None:
            workspace.egfr = load_table(egfr_file)

        summary = align_workspace(workspace)
        expression = workspace.expression.frame
        sample_ids = [str(value) for value in expression.index]
        first_sample = sample_ids[0] if sample_ids else None

        clinical_choices: list[str] = []
        if workspace.clinical is not None:
            clinical_choices = [
                str(column)
                for column in workspace.clinical.columns
                if str(column) != workspace.clinical_id_column
            ]

        orientation_text = workspace.expression.source_orientation.replace("_", " ")
        status = (
            f"✅ **Loaded** `{workspace.expression.source_path.name}` as "
            f"**{workspace.expression.n_samples:,} samples × {workspace.expression.n_probes:,} probes**.  "
            f"Detected source orientation: **{orientation_text}**."
        )
        if workspace.expression.duplicate_sample_ids_removed:
            status += (
                f" Removed {workspace.expression.duplicate_sample_ids_removed} duplicate sample ID(s) "
                "using the first occurrence, matching the thesis preprocessing behavior."
            )

        preview = expression.iloc[: min(8, len(expression)), : min(12, expression.shape[1])]
        return (
            workspace,
            status,
            _workspace_metrics(workspace, summary),
            _alignment_table(summary),
            preview,
            gr.update(choices=sample_ids, value=first_sample),
            gr.update(choices=["(none)"] + clinical_choices, value="(none)"),
            gr.update(
                choices=clinical_choices,
                value=clinical_choices[0] if clinical_choices else None,
            ),
            plot_sample_vs_reference(
                expression,
                first_sample,
                None if reference_expression is None else reference_expression.frame,
            ),
            plot_dataset_vs_reference(
                expression,
                None if reference_expression is None else reference_expression.frame,
            ),
            plot_distribution_overlay(expression),
            plot_sample_boxplots(expression),
            plot_pca(expression),
            plot_missingness(workspace.clinical),
            plot_egfr_trajectories(workspace.egfr, workspace.egfr_id_column),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            workspace,
            f"❌ **Could not load data:** {exc}",
            _workspace_metrics(workspace),
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            gr.update(choices=["(none)"], value="(none)"),
            gr.update(choices=[], value=None),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _sample_hist_callback(
    sample_id: str | None,
    workspace: DataWorkspace,
    reference_expression,
):
    if workspace.expression is None:
        return None
    reference_frame = None if reference_expression is None else reference_expression.frame
    return plot_sample_vs_reference(
        workspace.expression.frame,
        sample_id,
        reference_frame,
    )


def _reference_status(reference_expression) -> str:
    if reference_expression is None:
        return (
            "**Supervised reference:** not set. Load the original supervised expression matrix, or load "
            "the supervised cohort as the current dataset and click **Set current as supervised reference**."
        )
    return (
        f"**Supervised reference:** `{reference_expression.source_path.name}` · "
        f"**{reference_expression.n_samples:,} samples × "
        f"{reference_expression.n_probes:,} probes**"
    )


def _reference_plots(workspace: DataWorkspace, reference_expression):
    if workspace.expression is None:
        return None, None
    expression = workspace.expression.frame
    reference_frame = None if reference_expression is None else reference_expression.frame
    sample_ids = [str(value) for value in expression.index]
    first_sample = sample_ids[0] if sample_ids else None
    return (
        plot_sample_vs_reference(expression, first_sample, reference_frame),
        plot_dataset_vs_reference(expression, reference_frame),
    )


def _set_current_reference_callback(workspace: DataWorkspace):
    if workspace.expression is None:
        return None, "⚠️ Load an expression matrix before setting a supervised reference.", None, None
    reference_expression = workspace.expression
    sample_plot, cohort_plot = _reference_plots(workspace, reference_expression)
    return reference_expression, _reference_status(reference_expression), sample_plot, cohort_plot


def _load_reference_callback(
    reference_upload: str | None,
    reference_path: str,
    reference_orientation: str,
    workspace: DataWorkspace,
):
    try:
        source = _resolve_path(reference_upload, reference_path)
        if source is None:
            raise ValueError("Choose a supervised reference matrix or enter its local path.")
        reference_expression = load_expression_matrix(source, orientation=reference_orientation)
        sample_plot, cohort_plot = _reference_plots(workspace, reference_expression)
        return (
            reference_expression,
            _reference_status(reference_expression),
            sample_plot,
            cohort_plot,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ **Could not load supervised reference:** {exc}", None, None


def _clear_reference_callback(workspace: DataWorkspace):
    sample_plot, cohort_plot = _reference_plots(workspace, None)
    return None, _reference_status(None), sample_plot, cohort_plot


def _pca_callback(color_variable: str | None, workspace: DataWorkspace):
    if workspace.expression is None:
        return None
    variable = None if color_variable in {None, "(none)"} else color_variable
    return plot_pca(
        workspace.expression.frame,
        workspace.clinical,
        workspace.clinical_id_column,
        variable,
    )


def _clinical_plot_callback(variable: str | None, workspace: DataWorkspace):
    return plot_clinical_variable(workspace.clinical, variable)


def _format_rma_status() -> str:
    status = rma_environment_status()
    ready = bool(status["ready"])
    rscript = status["rscript"] or "not found"
    r_version = status["r_version"] or "—"
    biocmanager = "✅ installed" if status["biocmanager"] else "❌ missing"
    affy = "✅ installed" if status["affy"] else "❌ missing"
    headline = "✅ **RMA backend ready**" if ready else "⚠️ **RMA backend not ready**"
    return f"""
{headline}

| Component | Status |
|---|---|
| `Rscript` | `{rscript}` |
| R version | {r_version} |
| `BiocManager` | {biocmanager} |
| `affy` | {affy} |

{status["message"]}

> This backend is used **only for raw CEL → RMA**. Loading normalized matrices, QC, alignment, GEO inspection, and later ArchCon modelling remain Python-side.
"""


def _rma_environment_callback() -> str:
    return _format_rma_status()


def _install_rma_dependencies_callback():
    yield "⏳ **Installing/repairing R packages...** This may take several minutes on the first run."
    success, message = install_rma_dependencies()
    prefix = "✅" if success else "❌"
    yield f"{prefix} **R package setup finished.**\n\n{message}\n\n---\n\n{_format_rma_status()}"


def _collect_cel_paths(uploaded: list[str] | None, directory: str) -> list[str]:
    paths: list[str] = []
    if uploaded:
        paths.extend(str(path) for path in uploaded)
    if directory and directory.strip():
        paths.extend(str(path) for path in discover_cel_files(directory.strip()))
    # Deduplicate while preserving deterministic order.
    return list(dict.fromkeys(paths))


def _run_rma_callback(
    cel_source_mode: str,
    cel_uploads: list[str] | None,
    cel_directory: str,
    detected_cel_directory: str,
    output_mode: str,
    output_directory: str,
    data_directory: str,
    workspace: DataWorkspace,
    reference_expression,
):
    try:
        if cel_source_mode == CEL_SOURCE_UPLOAD:
            cel_paths = _collect_cel_paths(cel_uploads, "")
        elif cel_source_mode == CEL_SOURCE_DETECTED:
            cel_paths = _collect_cel_paths(None, detected_cel_directory)
        elif cel_source_mode == CEL_SOURCE_CUSTOM:
            cel_paths = _collect_cel_paths(None, cel_directory)
        else:
            raise ValueError(f"Unknown CEL source option: {cel_source_mode}")

        if not cel_paths:
            raise ValueError("No CEL files were found for the selected input option.")

        out_dir = _resolve_rma_output_directory(output_mode, data_directory, output_directory)
        expression, raw_sample, actual_out = run_rma(cel_paths, output_directory=out_dir)

        workspace = DataWorkspace(
            expression=expression,
            clinical=workspace.clinical,
            egfr=workspace.egfr,
        )
        summary = align_workspace(workspace)
        sample_ids = [str(value) for value in expression.frame.index]
        first_sample = sample_ids[0] if sample_ids else None
        preview = expression.frame.iloc[: min(8, len(expression.frame)), :12]

        status = (
            f"✅ RMA completed for **{len(cel_paths)} CEL file(s)**. Output saved in `{actual_out}`. "
            "The normalized matrix has been loaded into the Data Explorer."
        )
        return (
            workspace,
            status,
            plot_raw_vs_rma(raw_sample, expression.frame),
            _workspace_metrics(workspace, summary),
            _alignment_table(summary),
            preview,
            gr.update(choices=sample_ids, value=first_sample),
            plot_sample_vs_reference(
                expression.frame,
                first_sample,
                None if reference_expression is None else reference_expression.frame,
            ),
            plot_dataset_vs_reference(
                expression.frame,
                None if reference_expression is None else reference_expression.frame,
            ),
            plot_distribution_overlay(expression.frame),
            plot_sample_boxplots(expression.frame),
            plot_pca(expression.frame),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            workspace,
            f"❌ **RMA failed:** {exc}",
            None,
            _workspace_metrics(workspace),
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(),
            None,
            None,
            None,
            None,
            None,
        )


def _geo_callback(path: str):
    try:
        metadata, figure = inspect_geo_parquet(path)
        summary = (
            f"✅ **Parquet inspected without loading the full dataset.**  \n"
            f"Rows: **{metadata['rows']:,}** · columns: **{metadata['columns']:,}** · "
            f"sample columns: **{metadata['sample_columns']:,}** · row groups: "
            f"**{metadata['row_groups']:,}**."
        )
        return summary, figure
    except Exception as exc:  # noqa: BLE001
        return f"❌ **Could not inspect GEO Parquet:** {exc}", None


def _resolve_geo_rma_gse(store, scope: str, gse: str | None) -> str | None:
    if scope == SCOPE_AGGREGATE:
        return None
    choices = store.gses
    if not choices:
        raise ValueError("The GEO store contains no source GSEs.")
    if gse in choices:
        return gse
    return choices[0]


def _geo_rma_dashboard_outputs(
    store_path: str,
    scope: str,
    method: str,
    gse: str | None,
):
    store = load_geo_expression_store(store_path)
    gse = _resolve_geo_rma_gse(store, scope, gse)
    return (
        geo_dataset_heading(store, scope, gse),
        store.sample_table(scope, gse),
        geo_scope_summary(store, method, scope, gse),
        normalization_pipeline_markdown(method),
        plot_geo_scope_histogram(store, method, scope, gse),
        plot_geo_normalization_comparison(store, scope, gse),
        plot_geo_sample_boxplots(store, method, scope, gse),
        plot_geo_scope_pca(store, method, scope, gse),
        geo_dataset_metadata_markdown(store, method, scope, gse),
        geo_source_links_markdown(store, method, scope, gse),
        plot_geo_selected_sample(store, scope, gse, None),
        geo_selected_sample_metadata(store, scope, gse, None),
    )


def _geo_rma_empty_dashboard(message: str, method: str):
    return (
        f"## GEO datasets\n\n{message}",
        pd.DataFrame(),
        f'<div class="note">{message}</div>',
        normalization_pipeline_markdown(method),
        None,
        None,
        None,
        None,
        message,
        message,
        None,
        "Click a sample row to see provenance and its distributions.",
    )


def _geo_rma_load_callback(store_path: str, method: str):
    try:
        store = load_geo_expression_store(store_path)
        dashboard = _geo_rma_dashboard_outputs(store_path, SCOPE_AGGREGATE, method, None)
        return (
            geo_store_status(store),
            store.browser_catalog(),
            SCOPE_AGGREGATE,
            None,
            *dashboard,
        )
    except Exception as exc:  # noqa: BLE001
        message = (
            "The precomputed GEO store could not be loaded. Put `GEO_NUMPY_STORE/` "
            f"inside the ArchCon data directory, then refresh. Details: {exc}"
        )
        return (
            f"❌ **GEO dataset store unavailable:** {exc}",
            pd.DataFrame(),
            SCOPE_AGGREGATE,
            None,
            *_geo_rma_empty_dashboard(message, method),
        )


def _geo_rma_method_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    method: str,
):
    try:
        store = load_geo_expression_store(store_path)
        gse = _resolve_geo_rma_gse(store, scope, gse)
        return (
            geo_dataset_heading(store, scope, gse),
            geo_scope_summary(store, method, scope, gse),
            normalization_pipeline_markdown(method),
            plot_geo_scope_histogram(store, method, scope, gse),
            plot_geo_normalization_comparison(store, scope, gse),
            plot_geo_sample_boxplots(store, method, scope, gse),
            plot_geo_scope_pca(store, method, scope, gse),
            geo_dataset_metadata_markdown(store, method, scope, gse),
            geo_source_links_markdown(store, method, scope, gse),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Could not update the selected normalization view: {exc}"
        return (
            "## GEO datasets",
            f'<div class="note">{message}</div>',
            normalization_pipeline_markdown(method),
            None,
            None,
            None,
            None,
            message,
            message,
        )


def _geo_rma_aggregate_callback(store_path: str, method: str):
    try:
        dashboard = _geo_rma_dashboard_outputs(store_path, SCOPE_AGGREGATE, method, None)
        return SCOPE_AGGREGATE, None, gr.update(value=method), *dashboard
    except Exception as exc:  # noqa: BLE001
        message = f"Could not open the aggregate GEO collection: {exc}"
        return (
            SCOPE_AGGREGATE,
            None,
            gr.update(value=method),
            *_geo_rma_empty_dashboard(message, method),
        )


def _geo_rma_catalog_select_callback(
    store_path: str,
    method: str,
    evt: gr.SelectData,
):
    try:
        if not evt.row_value:
            raise ValueError("No dataset row was selected.")
        gse = str(evt.row_value[0])
        dashboard = _geo_rma_dashboard_outputs(store_path, SCOPE_SERIES, method, gse)
        return SCOPE_SERIES, gse, *dashboard
    except Exception as exc:  # noqa: BLE001
        message = f"Could not open the selected GEO Series: {exc}"
        return (
            SCOPE_AGGREGATE,
            None,
            *_geo_rma_empty_dashboard(message, method),
        )


def _geo_rma_sample_select_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    evt: gr.SelectData,
):
    try:
        if not evt.row_value:
            raise ValueError("No sample row was selected.")
        gsm = str(evt.row_value[0])
        store = load_geo_expression_store(store_path)
        gse = _resolve_geo_rma_gse(store, scope, gse)
        return (
            plot_geo_selected_sample(store, scope, gse, gsm),
            geo_selected_sample_metadata(store, scope, gse, gsm),
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not inspect the selected sample: {exc}"


def _simple_expression_import_callback(
    expression_upload: str | None,
    expression_path: str,
    orientation: str,
):
    try:
        source = _resolve_path(expression_upload, expression_path)
        if source is None:
            raise ValueError("Choose an expression matrix or enter a local path.")
        expression = load_expression_matrix(source, orientation=orientation)
        preview = expression.frame.iloc[
            : min(10, expression.n_samples), : min(12, expression.n_probes)
        ]
        message = (
            f"✅ Loaded `{expression.source_path}` as **{expression.n_samples:,} samples × "
            f"{expression.n_probes:,} probes**. This external matrix is kept separate from the "
            "precomputed GEO store used by the pretraining pipeline."
        )
        return message, preview
    except Exception as exc:  # noqa: BLE001
        return f"❌ **Could not import expression matrix:** {exc}", pd.DataFrame()


def _run_rma_stage_callback(
    cel_source_mode: str,
    cel_uploads: list[str] | None,
    cel_directory: str,
    detected_cel_directory: str,
    output_mode: str,
    output_directory: str,
    data_directory: str,
):
    try:
        if cel_source_mode == CEL_SOURCE_UPLOAD:
            cel_paths = _collect_cel_paths(cel_uploads, "")
        elif cel_source_mode == CEL_SOURCE_DETECTED:
            cel_paths = _collect_cel_paths(None, detected_cel_directory)
        elif cel_source_mode == CEL_SOURCE_CUSTOM:
            cel_paths = _collect_cel_paths(None, cel_directory)
        else:
            raise ValueError(f"Unknown CEL source option: {cel_source_mode}")
        if not cel_paths:
            raise ValueError("No CEL files were found for the selected input option.")
        out_dir = _resolve_rma_output_directory(output_mode, data_directory, output_directory)
        expression, raw_sample, actual_out = run_rma(cel_paths, output_directory=out_dir)
        status = (
            f"✅ RMA completed for **{len(cel_paths)} CEL file(s)** → "
            f"**{expression.n_samples:,} samples × {expression.n_probes:,} probes**. "
            f"Output: `{actual_out}`."
        )
        return status, plot_raw_vs_rma(raw_sample, expression.frame)
    except Exception as exc:  # noqa: BLE001
        return f"❌ **RMA failed:** {exc}", None


def _stage_visibility_callback(selected_stage: str):
    """Show exactly one of the six main pipeline panels."""
    stages = ("input", "supervised", "rma", "matrix", "split", "latent", "aa")
    return tuple(gr.update(visible=stage == selected_stage) for stage in stages)


def _stage_navigation_callback(selected_stage: str):
    """Show one stage and make its pipeline button visibly selected."""
    stages = ("input", "supervised", "rma", "matrix", "split", "latent", "aa")
    panels = tuple(gr.update(visible=stage == selected_stage) for stage in stages)
    buttons = tuple(
        gr.update(variant="primary" if stage == selected_stage else "secondary") for stage in stages
    )
    return (*panels, *buttons)


def _raw_geo_dashboard_outputs(
    store_path: str,
    scope: str,
    gse: str | None,
):
    """RAW-only dashboard used by stage 01."""
    store = load_geo_expression_store(store_path)
    gse = _resolve_geo_rma_gse(store, scope, gse)
    return (
        geo_dataset_heading(store, scope, gse),
        store.sample_table(scope, gse),
        geo_scope_summary(store, METHOD_RAW, scope, gse),
        plot_geo_scope_histogram(store, METHOD_RAW, scope, gse),
        plot_geo_sample_boxplots(store, METHOD_RAW, scope, gse),
        plot_geo_scope_pca(store, METHOD_RAW, scope, gse),
        geo_dataset_metadata_markdown(store, METHOD_RAW, scope, gse),
        geo_source_links_markdown(store, METHOD_RAW, scope, gse),
        plot_geo_selected_sample_method(store, METHOD_RAW, scope, gse, None),
        geo_selected_sample_metadata(store, scope, gse, None),
    )


def _raw_geo_empty_dashboard(message: str):
    return (
        f"## GEO raw data\n\n{message}",
        pd.DataFrame(),
        f'<div class="note">{message}</div>',
        None,
        None,
        None,
        message,
        message,
        None,
        "Click a GSM row to see its GEO metadata.",
    )


def _raw_geo_load_callback(store_path: str):
    try:
        store = load_geo_expression_store(store_path)
        dashboard = _raw_geo_dashboard_outputs(store_path, SCOPE_AGGREGATE, None)
        return (
            raw_geo_overview_markdown(store),
            store.browser_catalog(),
            SCOPE_AGGREGATE,
            None,
            *dashboard,
        )
    except Exception as exc:  # noqa: BLE001
        message = (
            "Put `GEO_NUMPY_STORE/` inside the ArchCon data directory, then refresh. "
            f"Details: {exc}"
        )
        return (
            f"❌ **GEO store unavailable:** {exc}",
            pd.DataFrame(),
            SCOPE_AGGREGATE,
            None,
            *_raw_geo_empty_dashboard(message),
        )


def _raw_geo_catalog_callback(store_path: str, evt: gr.SelectData):
    if not evt.row_value:
        return SCOPE_AGGREGATE, None, *_raw_geo_empty_dashboard("No GSE row was selected.")
    gse = str(evt.row_value[0])
    try:
        return SCOPE_SERIES, gse, *_raw_geo_dashboard_outputs(store_path, SCOPE_SERIES, gse)
    except Exception as exc:  # noqa: BLE001
        return (
            SCOPE_AGGREGATE,
            None,
            *_raw_geo_empty_dashboard(f"Could not open {gse}: {exc}"),
        )


def _raw_geo_all_callback(store_path: str):
    try:
        return SCOPE_AGGREGATE, None, *_raw_geo_dashboard_outputs(store_path, SCOPE_AGGREGATE, None)
    except Exception as exc:  # noqa: BLE001
        return (
            SCOPE_AGGREGATE,
            None,
            *_raw_geo_empty_dashboard(f"Could not open the full GEO collection: {exc}"),
        )


def _raw_geo_sample_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    evt: gr.SelectData,
):
    try:
        if not evt.row_value:
            raise ValueError("No GSM row was selected.")
        gsm = str(evt.row_value[0])
        store = load_geo_expression_store(store_path)
        gse = _resolve_geo_rma_gse(store, scope, gse)
        return (
            plot_geo_selected_sample_method(store, METHOD_RAW, scope, gse, gsm),
            geo_selected_sample_metadata(store, scope, gse, gsm),
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not inspect the selected sample: {exc}"


def _rma_method_button_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    method: str,
):
    """Select a preprocessing method, update button emphasis, and refresh stage 02."""
    button_updates = tuple(
        gr.update(variant="primary" if candidate == method else "secondary")
        for candidate in (METHOD_RAW, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA)
    )
    return method, *button_updates, *_geo_stage_refresh_callback(store_path, scope, gse, method)


def _geo_stage_refresh_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    method: str,
):
    """Refresh the compact GEO view used by the RMA and matrix pipeline stages."""
    try:
        store = load_geo_expression_store(store_path)
        gse = _resolve_geo_rma_gse(store, scope, gse)
        return (
            geo_dataset_heading(store, scope, gse),
            geo_scope_summary(store, method, scope, gse),
            normalization_pipeline_markdown(method),
            geo_dataset_metadata_markdown(store, method, scope, gse),
            geo_source_links_markdown(store, method, scope, gse),
            plot_geo_scope_histogram(store, method, scope, gse),
            plot_geo_normalization_comparison(store, scope, gse),
            plot_geo_sample_boxplots(store, method, scope, gse),
            plot_geo_scope_pca(store, method, scope, gse),
            store.sample_table(scope, gse),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Could not load the selected GEO view: {exc}"
        return (
            "## GEO expression matrix",
            f'<div class="note">{message}</div>',
            normalization_pipeline_markdown(method),
            message,
            message,
            None,
            None,
            None,
            None,
            pd.DataFrame(),
        )


def _geo_method_stage_callback(
    store_path: str,
    scope: str,
    gse: str | None,
    method: str,
):
    return method, *_geo_stage_refresh_callback(store_path, scope, gse, method)


# Background wrappers for plot-heavy UI actions.  Each generator emits the
# loading overlay immediately, performs the work on _VIEW_EXECUTOR, then swaps
# in the finished outputs.
def _raw_geo_load_async(store_path: str):
    yield from _async_view_result(_raw_geo_load_callback, 14, store_path)


def _raw_geo_catalog_async(store_path: str, evt: gr.SelectData):
    yield from _async_view_result(_raw_geo_catalog_callback, 12, store_path, evt)


def _raw_geo_all_async(store_path: str):
    yield from _async_view_result(_raw_geo_all_callback, 12, store_path)


def _raw_geo_sample_async(store_path: str, scope: str, gse: str | None, evt: gr.SelectData):
    yield from _async_view_result(_raw_geo_sample_callback, 2, store_path, scope, gse, evt)


def _rma_method_button_async(store_path: str, scope: str, gse: str | None, method: str):
    yield from _async_view_result(_rma_method_button_callback, 14, store_path, scope, gse, method)


def _geo_stage_refresh_async(store_path: str, scope: str, gse: str | None, method: str):
    yield from _async_view_result(_geo_stage_refresh_callback, 10, store_path, scope, gse, method)


def _supervised_dataset_outputs(store_path: str):
    """Load the kidney-donor molecular cohort and separate it by eGFR availability."""
    try:
        store_root = Path(store_path).expanduser().resolve()
        layout = project_data_layout(store_root.parent)
        source = load_ikem_source(layout)
        if source is None:
            raise FileNotFoundError(
                "The supervised molecular store was not found in the configured data directory."
            )
        ids = source.sample_index[source.sample_id_column].astype(str)
        status = classify_supervised_samples(layout, ids)
        table = status.table.copy()
        table.insert(0, "row_index", range(len(table)))
        return (
            supervised_overview_markdown(status, str(source.root)),
            table,
            plot_outcome_group_counts(status),
            plot_supervised_expression_pca(source.matrix, status),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"❌ **Could not open the supervised dataset:** {exc}",
            pd.DataFrame(),
            None,
            None,
        )


def _supervised_dataset_async(store_path: str):
    yield from _async_view_result(_supervised_dataset_outputs, 4, store_path)


def _geo_catalog_stage_callback(
    store_path: str,
    method: str,
    evt: gr.SelectData,
):
    """Open a GSE from the input-stage browser."""
    if not evt.row_value:
        return (
            SCOPE_AGGREGATE,
            None,
            *_geo_rma_empty_dashboard("No dataset row was selected.", method),
        )
    gse = str(evt.row_value[0])
    try:
        return (
            SCOPE_SERIES,
            gse,
            *_geo_rma_dashboard_outputs(store_path, SCOPE_SERIES, method, gse),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            SCOPE_AGGREGATE,
            None,
            *_geo_rma_empty_dashboard(f"Could not open {gse}: {exc}", method),
        )


def _geo_aggregate_stage_callback(store_path: str, method: str):
    try:
        return (
            SCOPE_AGGREGATE,
            None,
            method,
            *_geo_rma_dashboard_outputs(store_path, SCOPE_AGGREGATE, method, None),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Could not open the aggregate GEO collection: {exc}"
        return (
            SCOPE_AGGREGATE,
            None,
            method,
            *_geo_rma_empty_dashboard(message, method),
        )


def _comparison_split_from_store_path(
    store_path: str,
    seed: int,
    train_fraction: float,
) -> tuple[pd.DataFrame, Path]:
    """Build the fixed 90/5/5 split by leakage-safe source-GSE component."""
    store_root = Path(store_path).expanduser().resolve()
    layout = project_data_layout(store_root.parent)
    split = create_shared_preprocessing_split(
        layout,
        int(seed),
        float(train_fraction),
        methods=TRAINING_PREPROCESSING_OPTIONS,
    )
    path = save_split_csv(split, store_root)
    return split, path


def _split_initial_callback(
    store_path: str,
    seed: int = 42,
    train_fraction: float = 0.9,
):
    """Build the single fixed comparison split immediately when all stores exist."""
    try:
        split, path = _comparison_split_from_store_path(store_path, seed, train_fraction)
        return (
            split,
            split_summary_markdown(
                split,
                source="corrected per-study RMA · connected source-GSE components",
            ),
            split,
            str(path),
        )
    except Exception as comparison_exc:  # noqa: BLE001
        try:
            store = load_geo_expression_store(store_path)
            split = create_train_validation_split(store, int(seed), float(train_fraction))
            path = save_split_csv(split, store.root)
            summary = split_summary_markdown(
                split, source="canonical GEO store · corrected GSE-grouped fallback"
            )
            summary += (
                "\n\n⚠️ The formal per-study-RMA source could not be opened: "
                f"`{comparison_exc}`. This fallback is for split inspection only; formal "
                "sweep export requires `rma_per_gse.npy`."
            )
            return split, summary, split, str(path)
        except Exception as exc:  # noqa: BLE001
            message = f"Could not initialize the GEO split: {exc}"
            return None, message, pd.DataFrame(), None


def _split_open_callback(
    store_path: str,
    seed: int,
    train_fraction: float,
    current_split,
):
    """Keep the one session split when revisiting stage 04; create it only once."""
    if isinstance(current_split, pd.DataFrame) and not current_split.empty:
        try:
            path = save_split_csv(current_split, Path(store_path).expanduser().resolve())
            return (
                current_split,
                split_summary_markdown(current_split, source="current fixed session split"),
                current_split,
                str(path),
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "Could not persist the current split while reopening stage 04; "
                "falling back to the default split initialization.",
                exc_info=True,
            )
    return _split_initial_callback(store_path, seed, train_fraction)


def _generate_split_callback(store_path: str, seed: int, train_fraction: float):
    """Explicitly resample the one shared comparison split."""
    try:
        split, path = _comparison_split_from_store_path(store_path, seed, train_fraction)
        return (
            split,
            split_summary_markdown(split, source="corrected GSE-disjoint split generated in ArchCon"),
            split,
            str(path),
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ **Could not generate shared comparison split:** {exc}", pd.DataFrame(), None


def _load_split_callback(store_path: str, split_file: str | None):
    try:
        if not split_file:
            raise ValueError("Choose a previously saved split CSV or JSON file.")
        store_root = Path(store_path).expanduser().resolve()
        layout = project_data_layout(store_root.parent)
        split = validate_pretraining_split(
            layout, split_file, methods=TRAINING_PREPROCESSING_OPTIONS
        )
        path = save_split_csv(split, store_root)
        return (
            split,
            split_summary_markdown(split, source=f"loaded from `{Path(split_file).name}`"),
            split,
            str(path),
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"❌ **Could not load split:** {exc}", pd.DataFrame(), None


def _architecture_callback(
    store_path: str,
    architecture_family: str,
    hidden_widths_text: str,
    activation: str,
    latent_dim: int,
    loss_name: str,
    dropout: float,
    l2_lambda: float,
    stadniuk_batch_norm: bool,
    low_rank_dim: int,
    residual_blocks: int,
    residual_expansion: int,
    compile_model: bool,
):
    try:
        input_dim = 42917
        if store_path and Path(store_path).expanduser().is_dir():
            input_dim = load_geo_expression_store(store_path).n_probes
        hidden_widths = parse_hidden_widths(hidden_widths_text)
        architecture_html = autoencoder_architecture_svg(
            input_dim,
            hidden_widths,
            int(latent_dim),
            activation,
            loss_name,
            architecture_family,
            int(low_rank_dim),
            int(residual_blocks),
            int(residual_expansion),
            bool(stadniuk_batch_norm),
        )
        summary = architecture_summary_markdown(
            input_dim,
            hidden_widths,
            int(latent_dim),
            activation,
            loss_name,
            float(dropout),
            float(l2_lambda),
            architecture_family,
            int(low_rank_dim),
            int(residual_blocks),
            int(residual_expansion),
            bool(stadniuk_batch_norm),
        )
        preview_config = TrainingConfig(
            hidden_widths=tuple(hidden_widths),
            latent_dim=int(latent_dim),
            activation=activation,
            dropout=float(dropout),
            weight_decay=0.0,
            l2_lambda=float(l2_lambda),
            architecture_family=architecture_family,
            low_rank_dim=int(low_rank_dim),
            residual_blocks=int(residual_blocks),
            residual_expansion=int(residual_expansion),
            stadniuk_batch_norm=bool(stadniuk_batch_norm),
            loss_name=loss_name,
            compile_model=bool(compile_model),
        )
        return (
            summary,
            loss_theory_markdown(loss_name),
            architecture_html,
            model_execution_markdown(input_dim, preview_config),
            pytorch_model_code(input_dim, preview_config),
        )
    except Exception as exc:  # noqa: BLE001
        return f"❌ **Could not build architecture preview:** {exc}", "", "", "", ""


def _architecture_preset_callback(label: str):
    """Apply one complete preset to model and a few speed-sensitive controls."""
    preset = architecture_preset_from_label(label)
    return (
        preset.family,
        ", ".join(str(value) for value in preset.hidden_widths),
        preset.activation,
        preset.latent_dim,
        preset.dropout,
        preset.l2_lambda,
        preset.stadniuk_batch_norm,
        preset.low_rank_dim,
        preset.residual_blocks,
        preset.residual_expansion,
        preset.batch_size,
        preset.learning_rate,
        f"**Preset:** {preset.note}",
    )


def _aa_design_callback(mode: str):
    return aa_design_markdown(mode), aa_design_svg(mode)


def _checkpoint_clear_callback():
    return None, "No checkpoint selected. New training starts from a fresh initialization."


def _checkpoint_load_callback(store_path: str, checkpoint_file: str | None):
    """Validate a checkpoint and restore its model/training controls into the UI."""
    control_count = 37
    if not checkpoint_file:
        return (
            None,
            "No checkpoint selected. New training starts from a fresh initialization.",
            *[gr.update() for _ in range(control_count)],
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    try:
        metadata = inspect_checkpoint(checkpoint_file)
        config = metadata["config"]
        input_dim = int(metadata["input_dim"])
        if store_path and Path(store_path).expanduser().is_dir():
            expected = load_geo_expression_store(store_path).n_probes
            if input_dim != expected:
                raise ValueError(
                    f"Checkpoint expects {input_dim:,} probes, but this store has {expected:,}."
                )

        hidden_widths = [int(value) for value in config.get("hidden_widths", [256, 64])]
        architecture_family = str(config.get("architecture_family", ARCH_DENSE))
        activation = str(config.get("activation", "ReLU"))
        latent_dim = int(config.get("latent_dim", 3))
        dropout = float(config.get("dropout", 0.1))
        l2_lambda = float(config.get("l2_lambda", config.get("weight_decay", 0.0)))
        stadniuk_batch_norm = bool(config.get("stadniuk_batch_norm", False))
        low_rank_dim = int(config.get("low_rank_dim", 64))
        residual_blocks = int(config.get("residual_blocks", 1))
        residual_expansion = int(config.get("residual_expansion", 1))
        loss_name = normalize_loss_name(str(config.get("loss_name", LOSS_MSE)))
        legacy_loss_note = ""
        if loss_name not in LOSS_OPTIONS:
            legacy_loss_note = (
                f" Legacy objective `{loss_name}` is no longer offered for new runs; "
                "controls were restored to the MSE baseline for safe weights-only reuse."
            )
            loss_name = LOSS_MSE
        epochs = int(config.get("epochs", 1000))
        batch_size = int(config.get("batch_size", 32))
        learning_rate = float(config.get("learning_rate", 1e-3))
        lr_schedule = str(config.get("lr_schedule", LR_SCHEDULE_COSINE))
        optimizer = str(config.get("optimizer", OPTIMIZER_OPTIONS[0]))
        early_patience = int(config.get("convergence_window", 5))
        lr_patience = int(config.get("lr_decay_epochs", 500))
        gradient_clip = float(config.get("gradient_clip", 1.0))
        seed = int(config.get("seed", 42))
        device = str(config.get("device", DEVICE_OPTIONS[0]))
        precision = str(config.get("precision", PRECISION_OPTIONS[0]))
        compile_model = bool(config.get("compile_model", False))
        background_prefetch = bool(config.get("background_prefetch", True))
        deterministic = bool(config.get("deterministic", True))
        validation_every = int(config.get("validation_every_epochs", 1))
        ui_update = int(config.get("ui_update_batches", 10))
        pca_every = int(config.get("latent_pca_every_epochs", 5))
        huber_delta = float(config.get("huber_delta", 1.0))
        cosine_weight = float(config.get("cosine_weight", 0.25))
        kl_beta = float(config.get("kl_beta", 1e-3))
        kl_warmup_epochs = int(config.get("kl_warmup_epochs", 10))
        mmd_weight = float(config.get("mmd_weight", 0.1))
        mask_fraction = float(config.get("mask_fraction", 0.15))

        architecture_html = autoencoder_architecture_svg(
            input_dim,
            hidden_widths,
            latent_dim,
            activation,
            loss_name,
            architecture_family,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            stadniuk_batch_norm,
        )
        summary = architecture_summary_markdown(
            input_dim,
            hidden_widths,
            latent_dim,
            activation,
            loss_name,
            dropout,
            l2_lambda,
            architecture_family,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            stadniuk_batch_norm,
        )
        preview_config = TrainingConfig(
            hidden_widths=tuple(hidden_widths),
            latent_dim=latent_dim,
            activation=activation,
            dropout=dropout,
            weight_decay=0.0,
            l2_lambda=l2_lambda,
            architecture_family=architecture_family,
            low_rank_dim=low_rank_dim,
            residual_blocks=residual_blocks,
            residual_expansion=residual_expansion,
            stadniuk_batch_norm=stadniuk_batch_norm,
            loss_name=loss_name,
            lr_schedule=lr_schedule,
            compile_model=compile_model,
        )
        status = (
            f"✅ **Checkpoint ready:** `{Path(checkpoint_file).name}` · epoch "
            f"{metadata['epoch']} · method `{metadata['method']}` · best validation "
            f"{metadata['best_val_loss']:.6g}. Controls were restored from the checkpoint."
            f"{legacy_loss_note}"
        )
        # No preset is asserted after loading: the checkpoint may be a custom architecture.
        checkpoint_method = str(metadata.get("method", TRAINING_PREPROCESSING_OPTIONS[1]))
        controls = (
            gr.update(),
            checkpoint_method if checkpoint_method in TRAINING_PREPROCESSING_OPTIONS else TRAINING_PREPROCESSING_OPTIONS[1],
            architecture_family,
            ", ".join(str(value) for value in hidden_widths),
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule if lr_schedule in LR_SCHEDULE_OPTIONS else LR_SCHEDULE_COSINE,
            optimizer,
            early_patience,
            lr_patience,
            gradient_clip,
            seed,
            device if device in DEVICE_OPTIONS else DEVICE_OPTIONS[0],
            precision if precision in PRECISION_OPTIONS else PRECISION_OPTIONS[0],
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update,
            pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
            gr.update(),
        )
        return (
            str(Path(checkpoint_file).resolve()),
            status,
            *controls,
            summary,
            loss_theory_markdown(loss_name),
            architecture_html,
            model_execution_markdown(input_dim, preview_config),
            pytorch_model_code(input_dim, preview_config),
        )
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            f"❌ **Could not load checkpoint:** {exc}",
            *[gr.update() for _ in range(control_count)],
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )


def _training_interactivity(running: bool):
    """Return component updates that lock all training/model controls during a run."""
    editable = not running
    updates = [gr.update(interactive=editable) for _ in range(38)]
    start = gr.update(interactive=editable)
    stop = gr.update(interactive=running)
    return (*updates, start, stop)


def _training_status_with_source(status: str, method: str) -> str:
    return (
        f"{status}\n\n**Training preprocessing:** **{method}**. "
        "The molecular training pool is public GEO plus supervised-dataset samples without eGFR. "
        "Validation alone controls convergence/checkpoint selection; samples with eGFR remain excluded."
    )


def _checkpoint_download_html(path: str | Path | None, label: str) -> str:
    """Render a direct Gradio file-route link without copying a checkpoint to its cache."""
    if not path:
        return f'<div class="checkpoint-download-empty">{label}: not available yet.</div>'
    resolved = Path(path).resolve()
    href = f"/gradio_api/file={quote(str(resolved), safe='/')}"
    return (
        '<div class="checkpoint-download-card">'
        f'<strong>{label}</strong><br>'
        f'<a href="{href}" download="{resolved.name}">⬇️ Download {resolved.name}</a><br>'
        f'<code>{resolved}</code>'
        '</div>'
    )


def _training_ui_snapshot(update, method: str):
    return (
        update.run_id,
        _training_status_with_source(update.status, method),
        plot_training_history(update),
        plot_validation_metrics(update),
        plot_training_latent_pca(update),
        _checkpoint_download_html(update.latest_checkpoint, "Latest checkpoint"),
        _checkpoint_download_html(update.best_checkpoint, "Best validation checkpoint"),
        *_training_interactivity(update.running),
    )


def _model_output_root() -> Path:
    """Return the launch-directory workspace used for trained model checkpoints."""
    return Path.cwd().resolve() / "models"


def _gradio_temp_root() -> Path:
    """Keep unavoidable Gradio uploads/cache on the launch filesystem, not system /tmp."""
    return Path.cwd().resolve() / ".archcon-gradio"


def _configure_gradio_storage() -> Path:
    """Route Gradio temporary storage to the current ArchCon launch directory.

    Respect an explicit GRADIO_TEMP_DIR supplied by the user. Otherwise ArchCon
    uses ./.archcon-gradio so a large external launch volume also carries UI
    uploads/cache instead of filling the operating-system drive.
    """
    configured = os.environ.get("GRADIO_TEMP_DIR")
    root = Path(configured).expanduser().resolve() if configured else _gradio_temp_root()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GRADIO_TEMP_DIR", str(root))
    return root



def _loss_specific_control_updates(loss_name: str):
    """Show only controls that affect the currently selected objective."""
    from .data.pretraining import LOSS_COSINE, LOSS_DVIB, LOSS_HUBER, LOSS_MASKED, LOSS_MMD

    return (
        gr.update(visible=loss_name == LOSS_HUBER),
        gr.update(visible=loss_name == LOSS_COSINE),
        gr.update(visible=loss_name == LOSS_DVIB),
        gr.update(visible=loss_name == LOSS_DVIB),
        gr.update(visible=loss_name == LOSS_MMD),
        gr.update(visible=loss_name == LOSS_MASKED),
    )


def _training_config_from_controls(
    architecture_family: str,
    hidden_widths_text: str,
    activation: str,
    latent_dim: int,
    dropout: float,
    l2_lambda: float,
    stadniuk_batch_norm: bool,
    low_rank_dim: int,
    residual_blocks: int,
    residual_expansion: int,
    loss_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_schedule: str,
    optimizer_name: str,
    early_stopping_patience: int,
    lr_patience: int,
    gradient_clip: float,
    model_seed: int,
    training_device: str,
    precision: str,
    compile_model: bool,
    background_prefetch: bool,
    deterministic: bool,
    validation_every: int,
    ui_update_batches: int,
    latent_pca_every: int,
    huber_delta: float,
    cosine_weight: float,
    kl_beta: float,
    kl_warmup_epochs: int,
    mmd_weight: float,
    mask_fraction: float,
) -> TrainingConfig:
    """Build the exact TrainingConfig shared by web training and batch export."""
    return TrainingConfig(
        hidden_widths=tuple(parse_hidden_widths(hidden_widths_text)),
        latent_dim=int(latent_dim),
        activation=str(activation),
        dropout=float(dropout),
        weight_decay=0.0,
        l2_lambda=float(l2_lambda),
        stadniuk_batch_norm=bool(stadniuk_batch_norm),
        architecture_family=str(architecture_family),
        low_rank_dim=int(low_rank_dim),
        residual_blocks=int(residual_blocks),
        residual_expansion=int(residual_expansion),
        loss_name=str(loss_name),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        lr_schedule=str(lr_schedule),
        optimizer=str(optimizer_name),
        lr_decay_epochs=int(lr_patience),
        convergence_tolerance=1e-5,
        convergence_window=int(early_stopping_patience),
        early_stopping_patience=0,
        gradient_clip=float(gradient_clip),
        seed=int(model_seed),
        device=str(training_device),
        precision=str(precision),
        compile_model=bool(compile_model),
        background_prefetch=bool(background_prefetch),
        deterministic=bool(deterministic),
        validation_every_epochs=int(validation_every),
        ui_update_batches=int(ui_update_batches),
        latent_pca_every_epochs=int(latent_pca_every),
        huber_delta=float(huber_delta),
        cosine_weight=float(cosine_weight),
        kl_beta=float(kl_beta),
        kl_warmup_epochs=int(kl_warmup_epochs),
        mmd_weight=float(mmd_weight),
        mask_fraction=float(mask_fraction),
    )


def _sweep_output_root() -> Path:
    return Path.cwd().resolve() / "sweeps"


def _sweep_export_callback(
    store_path: str,
    method: str,
    split_seed: int,
    train_fraction: float,
    current_split,
    architecture_family: str,
    hidden_widths_text: str,
    activation: str,
    latent_dim: int,
    dropout: float,
    l2_lambda: float,
    stadniuk_batch_norm: bool,
    low_rank_dim: int,
    residual_blocks: int,
    residual_expansion: int,
    loss_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_schedule: str,
    optimizer_name: str,
    early_stopping_patience: int,
    lr_patience: int,
    gradient_clip: float,
    model_seed: int,
    training_device: str,
    precision: str,
    compile_model: bool,
    background_prefetch: bool,
    deterministic: bool,
    validation_every: int,
    ui_update_batches: int,
    latent_pca_every: int,
    huber_delta: float,
    cosine_weight: float,
    kl_beta: float,
    kl_warmup_epochs: int,
    mmd_weight: float,
    mask_fraction: float,
    sweep_grid_json: str,
    sweep_name: str,
    meta_project_dir: str,
    meta_data_dir: str,
    meta_python: str,
    pbs_ncpus: int,
    pbs_memory: str,
    pbs_scratch: str,
    pbs_walltime: str,
    pbs_ngpus: int,
    pbs_gpu_memory: str,
):
    try:
        config = _training_config_from_controls(
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule,
            optimizer_name,
            early_stopping_patience,
            lr_patience,
            gradient_clip,
            model_seed,
            training_device,
            precision,
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update_batches,
            latent_pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
        )
        base_request = build_run_request(
            method=str(method),
            split_seed=int(split_seed),
            train_fraction=float(train_fraction),
            training=config,
        )
        layout = project_data_layout(Path(store_path).expanduser().resolve().parent)
        fixed_split = create_shared_preprocessing_split(
            layout,
            int(split_seed),
            float(train_fraction),
            methods=TRAINING_PREPROCESSING_OPTIONS,
        )
        bundle = generate_sweep_bundle(
            base_request=base_request,
            grid_text=sweep_grid_json,
            destination_root=_sweep_output_root(),
            sweep_name=sweep_name,
            project_dir=meta_project_dir,
            data_dir=meta_data_dir,
            python_executable=meta_python,
            ncpus=int(pbs_ncpus),
            memory=pbs_memory,
            scratch=pbs_scratch,
            walltime=pbs_walltime,
            ngpus=int(pbs_ngpus),
            gpu_memory=pbs_gpu_memory,
            split_frame=fixed_split,
            data_layout=layout,
        )
        root = Path(bundle["root"])
        archive = Path(shutil.make_archive(str(root), "zip", root_dir=root))
        run_script = Path(bundle["run_script"]).read_text(encoding="utf-8")
        href = f"/gradio_api/file={quote(str(archive.resolve()), safe='/')}"
        download = (
            '<div class="checkpoint-download-card">'
            f'<strong>Portable sweep bundle · {bundle["count"]} runs</strong><br>'
            f'<a href="{href}" download="{archive.name}">⬇️ Download {archive.name}</a><br>'
            f'<code>{archive}</code>'
            '</div>'
        )
        status = (
            f"✅ Generated **{bundle['count']}** executable Python jobs plus matching JSON records "
            f"using one frozen prepared molecular split under `{root}`. Probe/sample alignment and final row indices are saved once before any job runs. Copy the ZIP to MetaCentrum, unpack it, "
            "inspect any `jobs/run_XXXX.py` you want, check paths/resources in `run_array.pbs.sh`, "
            "then execute `./submit.sh`."
        )
        return status, run_script, download
    except Exception as exc:  # noqa: BLE001
        return f"❌ **Could not generate sweep:** {exc}", "", ""


def _train_autoencoder_callback(
    store_path: str,
    method: str,
    split_seed: int,
    train_fraction: float,
    current_split,
    architecture_family: str,
    hidden_widths_text: str,
    activation: str,
    latent_dim: int,
    dropout: float,
    l2_lambda: float,
    stadniuk_batch_norm: bool,
    low_rank_dim: int,
    residual_blocks: int,
    residual_expansion: int,
    loss_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_schedule: str,
    optimizer_name: str,
    early_stopping_patience: int,
    lr_patience: int,
    gradient_clip: float,
    model_seed: int,
    training_device: str,
    precision: str,
    compile_model: bool,
    background_prefetch: bool,
    deterministic: bool,
    validation_every: int,
    ui_update_batches: int,
    latent_pca_every: int,
    huber_delta: float,
    cosine_weight: float,
    kl_beta: float,
    kl_warmup_epochs: int,
    mmd_weight: float,
    mask_fraction: float,
    checkpoint_path: str | None,
    checkpoint_mode: str,
):
    """Gradio generator: stream live training plots and lock controls while running."""
    try:
        store_root = Path(store_path).expanduser().resolve()
        layout = project_data_layout(store_root.parent)
        source = load_pretraining_source(layout, method)
        split = current_split
        if not isinstance(split, pd.DataFrame) or split.empty:
            split = create_shared_preprocessing_split(
                layout,
                int(split_seed),
                float(train_fraction),
                methods=TRAINING_PREPROCESSING_OPTIONS,
            )
        train_rows, validation_rows = split_rows_for_source(split, source)
        matrix = source.matrix

        config = _training_config_from_controls(
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule,
            optimizer_name,
            early_stopping_patience,
            lr_patience,
            gradient_clip,
            model_seed,
            training_device,
            precision,
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update_batches,
            latent_pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
        )
        # Keep trained models outside the data/library tree.  The launch directory
        # is the natural experiment workspace when running `archcon` locally.
        output_root = _model_output_root()
        for update in train_autoencoder_stream(
            matrix,
            train_rows,
            validation_rows,
            config,
            output_root,
            method=method,
            checkpoint_path=checkpoint_path or None,
            checkpoint_mode=checkpoint_mode,
        ):
            yield _training_ui_snapshot(update, method)
    except Exception as exc:  # noqa: BLE001
        placeholder_updates = _training_interactivity(False)
        yield (
            None,
            f"❌ **Training failed:** {exc}",
            None,
            None,
            None,
            None,
            None,
            *placeholder_updates,
        )


def _stop_training_callback(run_id: str | None):
    if request_training_stop(run_id):
        return (
            "⏹ **Stop requested.** ArchCon will finish the current batch and save a recovery checkpoint.",
            gr.update(interactive=False),
        )
    return "No active training run was found.", gr.update(interactive=False)


def build_app() -> gr.Blocks:
    """Construct the single-page molecular representation workflow."""
    _configure_gradio_storage()
    initial_layout = project_data_layout()
    initial_defaults = detected_default_paths(initial_layout)

    with gr.Blocks(
        title="ArchCon",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as app:
        detected_cel_directory = gr.State(initial_defaults["cel"])
        geo_scope_state = gr.State(SCOPE_AGGREGATE)
        geo_gse_state = gr.State(None)
        rma_method_state = gr.State(METHOD_PER_GSE_RMA)
        split_state = gr.State(None)
        gr.HTML(f"<style>{CSS}</style>")
        loading_overlay = gr.HTML(
            _LOADING_IDLE_HTML,
            elem_id="archcon-loading-overlay",
        )

        with gr.Column(elem_classes=["archcon-shell"]):
            gr.Markdown(
                """
# ArchCon

**Molecular representation pipeline**  
Learn from a large public **unsupervised dataset**, add only outcome-blind samples from the smaller **supervised dataset**, then reserve outcome-bearing samples for downstream eGFR evaluation.
""",
                elem_classes=["archcon-hero"],
            )

            gr.Markdown(
                """
## Pipeline

Click a step. Each page answers one simple question; the selected step stays **orange**.
""",
                elem_classes=["section-intro"],
            )

            with gr.Row(elem_classes=["pipeline-strip"]):
                stage_input_button = gr.Button(
                    "🌍\n01 · Unsupervised data\npublic GEO",
                    variant="primary",
                    elem_classes=["pipeline-step"],
                )
                stage_supervised_button = gr.Button(
                    "🧬\n02 · Supervised data\nkidney cohort",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )
                stage_rma_button = gr.Button(
                    "🧪\n03 · Preprocessing\nexpression scale",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )
                stage_matrix_button = gr.Button(
                    "▦\n04 · Matrix\nsamples × probes",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )
                stage_split_button = gr.Button(
                    "✂️\n05 · Split\ntrain / val / test",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )
                stage_latent_button = gr.Button(
                    "🧠\n06 · Molecular AE\nlearn z",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )
                stage_aa_button = gr.Button(
                    "📈\n07 · Downstream\neGFR / AA later",
                    variant="secondary",
                    elem_classes=["pipeline-step"],
                )

            gr.Markdown(
                "**Molecular pretraining:** GEO + supervised-dataset samples with no eGFR. **Held back:** every sample that has eGFR, until downstream evaluation.",
                elem_classes=["pipeline-hint"],
            )

            # -----------------------------------------------------------------
            # 01 · RAW GEO data
            # -----------------------------------------------------------------
            with gr.Column(
                visible=True, elem_classes=["pipeline-stage-panel"]
            ) as stage_input_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["input"],
                    elem_classes=["pipeline-stage-theory"],
                )

                gr.HTML(
                    """
<div class="beginner-grid">
  <div class="beginner-card"><b>🌍 GEO</b><span>NCBI's public gene-expression database</span></div>
  <div class="beginner-card"><b>📚 GSE</b><span>one study / Series</span></div>
  <div class="beginner-card"><b>🧪 GSM</b><span>one biological sample / array</span></div>
  <div class="beginner-card"><b>📄 CEL</b><span>raw Affymetrix chip intensities</span></div>
</div>
"""
                )

                geo_store_status_md = gr.Markdown(
                    "Loading `data/GEO_NUMPY_STORE`…"
                    if initial_defaults["geo_rma"]
                    else "⚠️ `data/GEO_NUMPY_STORE` was not detected.",
                    elem_classes=["geo-rma-store-status"],
                )

                with gr.Row():
                    show_all_geo_button = gr.Button(
                        "🌐 Show whole GEO collection",
                        variant="secondary",
                    )

                gr.Markdown(
                    "### 📚 GEO studies\nSearch by **GSE** and click a row. This table is only a study browser; the values shown below are always the **raw / before-RMA** view.",
                    elem_classes=["geo-section-heading"],
                )
                geo_catalog = gr.Dataframe(
                    headers=[
                        "GSE",
                        "unique GSMs",
                        "memberships",
                        "shared",
                        "overlap",
                        "GEO",
                        "RAW archive",
                    ],
                    label="GEO Series browser",
                    interactive=False,
                    wrap=False,
                    max_height=360,
                    show_search="filter",
                    pinned_columns=1,
                    show_row_numbers=False,
                    datatype=[
                        "str",
                        "number",
                        "number",
                        "number",
                        "str",
                        "markdown",
                        "markdown",
                    ],
                    elem_classes=["geo-rma-table", "full-width-scroll-table"],
                )

                with gr.Accordion(
                    "⚙️ Data location / refresh",
                    open=False,
                    elem_classes=["minimal-accordion", "utility-accordion"],
                ):
                    gr.Markdown(
                        "Usually you do not need this. Use it only if the store was moved.",
                        elem_classes=["reading-width"],
                    )
                    geo_store_path = gr.Textbox(
                        label="GEO NumPy store",
                        value=initial_defaults["geo_rma"],
                        placeholder="data/GEO_NUMPY_STORE",
                    )
                    geo_refresh_button = gr.Button("↻ Refresh store", variant="secondary")

                geo_heading = gr.Markdown(
                    "## 🌐 All unique GEO samples · raw view",
                    elem_classes=["geo-selection-heading"],
                )
                geo_summary = gr.Markdown('<div class="note">Loading raw-data statistics…</div>')

                with gr.Row(equal_height=False):
                    geo_dataset_metadata = gr.Markdown(
                        "Dataset metadata will appear here.",
                        elem_classes=["geo-metadata-card"],
                    )
                    geo_source_links = gr.Markdown(
                        "Files and GEO links will appear here.",
                        elem_classes=["geo-pipeline-card"],
                    )

                gr.Markdown(
                    "### 📊 Raw-data checks\nThese plots summarize the **before-RMA intensity values**. They do not change the data.",
                    elem_classes=["geo-section-heading"],
                )
                with gr.Row(equal_height=True):
                    geo_hist = gr.Plot(label="Raw-value histogram")
                    geo_boxplot = gr.Plot(label="Raw distribution by sample")
                geo_pca = gr.Plot(label="Raw-data PCA overview")

                gr.Markdown(
                    "### 🧪 Samples\nSearch by **GSM** and click a row. The table is full width but scrolls vertically so thousands of samples do not stretch the page.",
                    elem_classes=["geo-section-heading"],
                )
                geo_samples = gr.Dataframe(
                    label="Samples in the current GEO view",
                    interactive=False,
                    wrap=False,
                    max_height=360,
                    show_search="filter",
                    pinned_columns=1,
                    show_row_numbers=True,
                    elem_classes=["geo-rma-table", "full-width-scroll-table"],
                )
                with gr.Row(equal_height=False):
                    geo_sample_plot = gr.Plot(label="Selected GSM · raw values")
                    geo_sample_metadata = gr.Markdown(
                        "Click a GSM row to see its GEO page and Series memberships.",
                        elem_classes=["geo-sample-metadata"],
                    )

                gr.HTML(
                    '<div class="workflow-divider"><span>🧰 Optional: inspect another matrix</span></div>'
                )
                gr.Markdown(
                    "The default GEO store above is already loaded. The controls below are only for a different file you want to inspect temporarily.",
                    elem_classes=["reading-width"],
                )
                with gr.Accordion(
                    "Open an external expression matrix",
                    open=False,
                    elem_classes=["minimal-accordion", "utility-accordion"],
                ):
                    with gr.Row():
                        external_expression_upload = gr.File(
                            label="Expression matrix",
                            file_count="single",
                            file_types=[".csv", ".tsv", ".txt", ".xlsx", ".parquet"],
                            type="filepath",
                        )
                        external_expression_path = gr.Textbox(
                            label="or local path",
                            placeholder="/path/to/expression.csv",
                        )
                        external_orientation = gr.Radio(
                            choices=["auto", "samples_rows", "samples_columns"],
                            value="auto",
                            label="Orientation",
                        )
                    external_import_button = gr.Button("Inspect external matrix")
                    external_import_status = gr.Markdown("No external matrix loaded.")
                    external_import_preview = gr.Dataframe(
                        label="Small preview",
                        interactive=False,
                        wrap=False,
                    )

            # -----------------------------------------------------------------
            # 02 · Supervised dataset
            # -----------------------------------------------------------------
            with gr.Column(visible=False, elem_classes=["pipeline-stage-panel"]) as stage_supervised_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["supervised"],
                    elem_classes=["pipeline-stage-theory"],
                )
                supervised_overview = gr.Markdown(
                    "Open this stage to load the supervised dataset and separate samples by eGFR availability.",
                    elem_classes=["reading-width"],
                )
                with gr.Row():
                    supervised_counts_plot = gr.Plot(label="Outcome availability")
                    supervised_pca_plot = gr.Plot(label="Expression overview")
                supervised_table = gr.Dataframe(
                    label="Molecular samples → outcome availability → pretraining eligibility",
                    interactive=False,
                    wrap=False,
                    max_height=430,
                    show_search="filter",
                    pinned_columns=3,
                    show_row_numbers=False,
                    elem_classes=["full-width-scroll-table"],
                )

            # -----------------------------------------------------------------
            # 03 · GEO preprocessing choices
            # -----------------------------------------------------------------
            with gr.Column(visible=False, elem_classes=["pipeline-stage-panel"]) as stage_rma_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["rma"],
                    elem_classes=["pipeline-stage-theory"],
                )
                gr.Markdown(
                    "### 🧪 Choose public-data preprocessing\nThe same selected GEO samples are shown three ways. Supervised-dataset samples keep their existing RMA representation when they are added to molecular pretraining.",
                    elem_classes=["stage-subheading"],
                )
                with gr.Row(elem_classes=["normalization-choice-strip"]):
                    rma_raw_button = gr.Button(
                        "① No RMA\nraw PM median",
                        variant="secondary",
                        elem_id="rma-method-raw",
                        elem_classes=["normalization-choice"],
                    )
                    rma_per_gse_button = gr.Button(
                        "② Per-study RMA\nnormalize each GSE",
                        variant="primary",
                        elem_id="rma-method-per-gse",
                        elem_classes=["normalization-choice"],
                    )
                    rma_global_button = gr.Button(
                        "③ Global RMA\none shared normalization",
                        variant="secondary",
                        elem_id="rma-method-global",
                        elem_classes=["normalization-choice"],
                    )

                rma_heading = gr.Markdown("## GEO expression")
                rma_summary = gr.Markdown("Open this stage to load statistics.")
                with gr.Row(equal_height=False):
                    rma_pipeline = gr.Markdown(
                        normalization_pipeline_markdown(METHOD_PER_GSE_RMA),
                        elem_classes=["geo-pipeline-card"],
                    )
                    rma_metadata = gr.Markdown(
                        "Dataset metadata will appear here.",
                        elem_classes=["geo-metadata-card"],
                    )
                    rma_links = gr.Markdown(
                        "Files and GEO links will appear here.",
                        elem_classes=["geo-pipeline-card"],
                    )
                with gr.Row(equal_height=True):
                    rma_hist = gr.Plot(label="Selected value distribution")
                    rma_compare = gr.Plot(label="Raw vs per-study vs global RMA")
                with gr.Row(equal_height=True):
                    rma_boxplot = gr.Plot(label="Sample distributions")
                    rma_pca = gr.Plot(label="PCA")
                rma_samples = gr.Dataframe(
                    label="Samples used in this view",
                    interactive=False,
                    wrap=False,
                    max_height=360,
                    show_search="filter",
                    pinned_columns=1,
                    elem_classes=["full-width-scroll-table"],
                )

                gr.HTML(
                    '<div class="workflow-divider"><span>🧰 Optional: normalize new CEL files</span></div>'
                )
                gr.Markdown(
                    "The three views above are **already computed**. Use this section only when you bring in new Affymetrix CEL files.",
                    elem_classes=["reading-width"],
                )
                with gr.Accordion(
                    "Run CEL → RMA on another dataset",
                    open=False,
                    elem_classes=["minimal-accordion", "utility-accordion"],
                ):
                    rma_data_directory = gr.Textbox(
                        label="ArchCon data directory",
                        value=str(initial_layout.root),
                    )
                    with gr.Row():
                        cel_source_mode = gr.Dropdown(
                            choices=_cel_source_choices(initial_defaults["cel"]),
                            value=_default_cel_source(initial_defaults["cel"]),
                            label="CEL input source",
                        )
                        rma_output_mode = gr.Dropdown(
                            choices=[RMA_OUTPUT_TEMPORARY, RMA_OUTPUT_DATA, RMA_OUTPUT_CUSTOM],
                            value=RMA_OUTPUT_TEMPORARY,
                            label="Output directory",
                        )
                    cel_uploads = gr.File(
                        label="CEL files",
                        file_count="multiple",
                        file_types=[".cel", ".CEL", ".gz"],
                        type="filepath",
                        visible=not bool(initial_defaults["cel"]),
                    )
                    cel_directory = gr.Textbox(
                        label="Custom CEL directory",
                        visible=False,
                    )
                    rma_output_directory = gr.Textbox(
                        label="Custom RMA output directory",
                        visible=False,
                    )
                    run_rma_button = gr.Button("Run CEL → RMA", variant="primary")
                    rma_run_status = gr.Markdown("RMA has not been run in this session.")
                    rma_run_plot = gr.Plot(label="Raw example vs RMA distribution")
                    with gr.Row():
                        check_rma_button = gr.Button("Check R/Bioconductor backend")
                        install_rma_button = gr.Button("Install / repair R packages")
                    rma_backend_status = gr.Markdown(
                        "Backend is checked only on request.",
                        elem_classes=["reading-width"],
                    )

            # -----------------------------------------------------------------
            # 04 · Matrix
            # -----------------------------------------------------------------
            with gr.Column(
                visible=False, elem_classes=["pipeline-stage-panel"]
            ) as stage_matrix_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["matrix"],
                    elem_classes=["pipeline-stage-theory"],
                )
                gr.Markdown(
                    "### ▦ Matrix produced by step 02\nThis page does not normalize again. It shows the **currently selected preprocessing output** as rows = samples and columns = probes.",
                    elem_classes=["stage-subheading"],
                )
                matrix_heading = gr.Markdown("## GEO expression matrix")
                matrix_summary = gr.Markdown("Open this stage to inspect the matrix.")
                with gr.Row(equal_height=False):
                    matrix_pipeline = gr.Markdown(
                        normalization_pipeline_markdown(METHOD_PER_GSE_RMA),
                        elem_classes=["geo-pipeline-card"],
                    )
                    matrix_metadata = gr.Markdown(
                        "Matrix metadata will appear here.",
                        elem_classes=["geo-metadata-card"],
                    )
                    matrix_links = gr.Markdown(
                        "Backing files and GEO links will appear here.",
                        elem_classes=["geo-pipeline-card"],
                    )
                with gr.Row(equal_height=True):
                    matrix_hist = gr.Plot(label="Matrix-value histogram")
                    matrix_compare = gr.Plot(label="Same samples under all pipelines")
                with gr.Row(equal_height=True):
                    matrix_boxplot = gr.Plot(label="Sample distributions")
                    matrix_pca = gr.Plot(label="PCA")
                matrix_samples = gr.Dataframe(
                    label="Rows represented in this matrix view",
                    interactive=False,
                    wrap=False,
                    max_height=360,
                    show_search="filter",
                    pinned_columns=1,
                    show_row_numbers=True,
                    elem_classes=["full-width-scroll-table"],
                )

            # -----------------------------------------------------------------
            # 05 · Split
            # -----------------------------------------------------------------
            with gr.Column(
                visible=False, elem_classes=["pipeline-stage-panel"]
            ) as stage_split_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["split"],
                    elem_classes=["pipeline-stage-theory"],
                )
                gr.Markdown(
                    "### ✂️ One study-level split is sampled once\nThe formal protocol is **~90% train / ~5% validation / ~5% test**, assigned by connected source-GSE component. No GSE (or related SubSeries/SuperSeries sharing a GSM) can cross partitions. Seed 42 is reused for every architecture and hyperparameter combination.",
                    elem_classes=["stage-subheading"],
                )
                with gr.Row(elem_classes=["split-controls"]):
                    split_seed = gr.Number(
                        label="Random seed",
                        value=42,
                        precision=0,
                    )
                    train_fraction = gr.Number(
                        value=0.90,
                        precision=2,
                        label="Train fraction (fixed; val/test = 0.05/0.05)",
                        interactive=False,
                    )
                split_summary = gr.Markdown(
                    "Open this stage to create the default split.",
                    elem_classes=["reading-width"],
                )
                split_table = gr.Dataframe(
                    label="Molecular sample → train / validation / test assignment",
                    interactive=False,
                    wrap=False,
                    max_height=420,
                    show_search="filter",
                    pinned_columns=3,
                    show_row_numbers=False,
                    elem_classes=["full-width-scroll-table"],
                )

                gr.HTML(
                    '<div class="workflow-divider"><span>💾 Save or reuse the exact same split</span></div>'
                )
                gr.Markdown(
                    "**Save:** download the current CSV. It records both public-GEO identities and the outcome-blind supervised samples. Reusing the same file keeps every model on exactly the same partition.",
                    elem_classes=["reading-width"],
                )
                with gr.Row():
                    split_download = gr.File(
                        label="⬇️ Current split CSV",
                        interactive=False,
                    )
                    split_upload = gr.File(
                        label="⬆️ Load saved split CSV / JSON",
                        file_count="single",
                        file_types=[".csv", ".json"],
                        type="filepath",
                    )

            # -----------------------------------------------------------------
            # 06 · Molecular autoencoder + real training
            # -----------------------------------------------------------------
            with gr.Column(
                visible=False, elem_classes=["pipeline-stage-panel"]
            ) as stage_latent_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["latent"],
                    elem_classes=["pipeline-stage-theory"],
                )
                gr.Markdown(
                    "### 🧠 Choose a starting architecture\nStart with a preset, then edit any field. **Presets are hypotheses, not winners** — validation will decide.",
                    elem_classes=["stage-subheading"],
                )
                gr.Markdown(
                    training_backend_status(),
                    elem_classes=["training-status-card"],
                )

                with gr.Row(equal_height=True, elem_classes=["model-config-row"]):
                    architecture_preset = gr.Dropdown(
                        choices=ARCHITECTURE_PRESET_OPTIONS,
                        value=ARCHITECTURE_PRESET_OPTIONS[0],
                        label="Architecture preset",
                        scale=2,
                    )
                    architecture_family = gr.Dropdown(
                        choices=ARCHITECTURE_OPTIONS,
                        value=ARCH_STADNIUK,
                        label="Layer family",
                        scale=2,
                    )
                    hidden_widths_text = gr.Textbox(
                        value="256, 64",
                        label="Hidden widths",
                        info="comma separated",
                        scale=2,
                    )
                    activation = gr.Dropdown(
                        choices=ACTIVATION_OPTIONS,
                        value="ReLU",
                        label="Activation",
                    )
                    latent_dim = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=3,
                        step=1,
                        label="Latent dim",
                    )

                architecture_preset_note = gr.Markdown(
                    "**Preset:** faithful final Stadniuk MLP without normalization; the sweep also tests BatchNorm.",
                    elem_classes=["preset-note"],
                )

                training_preprocessing = gr.Dropdown(
                    choices=TRAINING_PREPROCESSING_OPTIONS,
                    value=METHOD_PER_GSE_RMA,
                    label="Training preprocessing",
                    info=(
                        "The comparison sweep evaluates Stadniuk rescaling, per-study RMA, and the legacy-named "
                        "Global RMA arm on the same GSE-disjoint 90/5/5 identities. Global preprocessing must "
                        "carry provenance proving that its reference was fitted only on frozen GEO training rows."
                    ),
                )

                with gr.Row(equal_height=True, elem_classes=["model-config-row"]):
                    low_rank_dim = gr.Slider(
                        8,
                        256,
                        value=64,
                        step=8,
                        label="Legacy low-rank adapter rank",
                        info="checkpoint compatibility only; not used by the two comparison families",
                        visible=False,
                    )
                    residual_blocks = gr.Slider(
                        0,
                        4,
                        value=1,
                        step=1,
                        label="Same-width residual blocks after each projection",
                        info="Each shortcut is local x + F(x); it does not jump across width-changing stages.",
                    )
                    residual_expansion = gr.Slider(
                        1,
                        4,
                        value=1,
                        step=1,
                        label="Residual FFN expansion ×",
                        info="ResNet only: hidden branch d → r·d → d; comparison grid uses 2× and 4×.",
                    )
                    dropout = gr.Slider(
                        minimum=0.0,
                        maximum=0.8,
                        value=0.1,
                        step=0.01,
                        label="Hidden dropout · optional regularization",
                        info=(
                            "0 = disabled. The thesis preset keeps 0.10; newer ArchCon "
                            "architectures default to 0 so dropout can be tested separately."
                        ),
                    )
                    l2_lambda = gr.Number(
                        label="Explicit L2 λ",
                        value=1e-5,
                        minimum=0.0,
                        info="Keras-style kernel L2 penalty; Stadniuk comparison grid uses 0, 1e-5, 1e-4.",
                    )
                    stadniuk_batch_norm = gr.Checkbox(
                        value=False,
                        label="Stadniuk BatchNorm",
                        info=(
                            "Off reproduces Stadniuk's final no-normalization MLP; the comparison "
                            "grid also tests the earlier/experimental BatchNorm variant. Ignored by ResNet-LN."
                        ),
                    )

                loss_name = gr.Radio(
                    choices=LOSS_OPTIONS,
                    value=LOSS_MSE,
                    label="Combined training objective",
                    info=(
                        "MSE/Huber/cosine change the reconstruction term; masked denoising changes "
                        "the self-supervised reconstruction task. β-VAE and MMD use MSE reconstruction "
                        "plus a latent regularizer. The explanation below updates when you choose one."
                    ),
                    elem_classes=["model-config-row"],
                )

                with gr.Accordion("🎛 Loss-specific controls", open=True):
                    gr.Markdown(
                        "Only the controls relevant to the selected objective are used; the others "
                        "are kept so you can switch losses without losing settings."
                    )
                    with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                        huber_delta = gr.Number(
                            value=1.0,
                            minimum=1e-6,
                            label="Huber δ",
                            info="Smooth-L1 transition point",
                            visible=False,
                        )
                        cosine_weight = gr.Number(
                            value=0.25,
                            minimum=0.0,
                            label="Cosine λ",
                            info="weight of profile-shape term",
                            visible=False,
                        )
                        kl_beta = gr.Number(
                            value=1e-3,
                            minimum=0.0,
                            label="β-VAE KL β",
                            info="latent prior strength",
                            visible=False,
                        )
                        kl_warmup_epochs = gr.Slider(
                            0,
                            100,
                            value=10,
                            step=1,
                            label="KL warm-up epochs",
                            visible=False,
                        )
                        mmd_weight = gr.Number(
                            value=0.1,
                            minimum=0.0,
                            label="MMD λ",
                            info="aggregate latent-prior strength",
                            visible=False,
                        )
                        mask_fraction = gr.Slider(
                            0.05,
                            0.60,
                            value=0.15,
                            step=0.05,
                            label="Masked-probe fraction",
                            info="used only by masked denoising reconstruction",
                            visible=False,
                        )

                gr.Markdown(
                    "### ⚙️ Training controls\nThe expensive matrix is memory-mapped. GPU mixed precision and a one-batch background prefetch are enabled automatically when useful. Controls lock during a run.",
                    elem_classes=["stage-subheading"],
                )
                with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                    epochs = gr.Slider(1, 5000, value=1000, step=1, label="Max epochs")
                    batch_size = gr.Dropdown(
                        choices=[8, 16, 32, 64, 128, 256],
                        value=64,
                        label="Batch size",
                    )
                    learning_rate = gr.Number(
                        value=1e-3,
                        minimum=1e-7,
                        label="Starting learning rate",
                    )
                    lr_schedule = gr.Dropdown(
                        choices=LR_SCHEDULE_OPTIONS,
                        value=LR_SCHEDULE_COSINE,
                        label="Learning-rate schedule",
                        info="The comparison sweep fixes 1e-3 and decreases it deterministically with cosine annealing.",
                    )
                    optimizer_name = gr.Dropdown(
                        choices=OPTIMIZER_OPTIONS,
                        value=OPTIMIZER_OPTIONS[0],
                        label="Optimizer",
                    )
                    model_seed = gr.Number(value=42, precision=0, label="Model seed")
                    training_device = gr.Dropdown(
                        choices=DEVICE_OPTIONS,
                        value=DEVICE_OPTIONS[0],
                        label="Device",
                    )

                with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                    precision = gr.Dropdown(
                        choices=PRECISION_OPTIONS,
                        value=PRECISION_OPTIONS[0],
                        label="Compute precision",
                    )
                    compile_model = gr.Checkbox(
                        value=False,
                        label="torch.compile",
                        info="optional; first epoch may compile slowly",
                    )
                    background_prefetch = gr.Checkbox(
                        value=True,
                        label="Background batch prefetch",
                    )
                    deterministic = gr.Checkbox(
                        value=True,
                        label="Deterministic kernels",
                        info="disable for maximum GPU speed",
                    )
                    validation_every = gr.Slider(
                        1,
                        20,
                        value=1,
                        step=1,
                        label="Validate every N epochs",
                    )

                with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                    early_stopping_patience = gr.Slider(
                        1, 20, value=5, step=1, label="Convergence window",
                        info="Stop only when every relative validation-objective change in this window is <= 1e-5 and LR is already at its floor.",
                    )
                    lr_patience = gr.Slider(
                        10, 2000, value=500, step=10, label="Cosine decay epochs",
                        info="Learning rate decays from the starting LR to the minimum LR over this many epochs, then stays at the floor.",
                    )
                    gradient_clip = gr.Number(value=1.0, minimum=0.0, label="Gradient clip")
                    ui_update_batches = gr.Slider(
                        1, 100, value=10, step=1, label="Live update every N batches"
                    )
                    latent_pca_every = gr.Slider(
                        1, 50, value=5, step=1, label="Validation PCA every N epochs"
                    )

                training_run_state = gr.State(None)
                checkpoint_state = gr.State(None)

                with gr.Row(elem_classes=["training-actions"]):
                    start_training_button = gr.Button(
                        "▶ Start training with current settings",
                        variant="primary",
                        scale=4,
                    )
                    stop_training_button = gr.Button(
                        "⏹ Stop after current batch",
                        variant="secondary",
                        interactive=False,
                        scale=2,
                    )

                architecture_plot = gr.HTML(
                    value=autoencoder_architecture_svg(
                        42917,
                        [256, 64],
                        3,
                        "ReLU",
                        LOSS_MSE,
                        ARCH_STADNIUK,
                        64,
                        0,
                    ),
                    elem_classes=["architecture-viewport"],
                )
                architecture_summary = gr.Markdown(
                    architecture_summary_markdown(
                        42917,
                        [256, 64],
                        3,
                        "ReLU",
                        LOSS_MSE,
                        0.1,
                        1e-5,
                        ARCH_STADNIUK,
                        64,
                        0,
                    ),
                    elem_classes=["architecture-card"],
                )
                loss_theory = gr.Markdown(
                    loss_theory_markdown(LOSS_MSE),
                    elem_classes=["pipeline-stage-theory"],
                )
                with gr.Accordion("📚 Why these architecture presets?", open=False):
                    gr.Markdown(architecture_research_markdown())

                gr.HTML('<div class="workflow-divider"><span>📈 Live training</span></div>')
                gr.Markdown(
                    "The faint training curve is a **batch EMA**, while the solid training points are exact epoch means. Validation uses the **same selected objective**; plain MSE and R² are shown separately. Held-out molecular test rows and outcome-bearing supervised samples are deliberately **not evaluated here**, preventing visual/manual leakage into model selection. Hollow circles mark the validation-selected best checkpoint epoch.",
                    elem_classes=["reading-width"],
                )
                training_status = gr.Markdown(
                    "Ready. Choose one of the three public-data preprocessing representations above. Molecular training uses public GEO plus supervised-dataset samples without eGFR; validation selects checkpoints. Outcome-bearing samples stay outside this stage. The legacy-named Global RMA arm requires a frozen train-only reference.",
                    elem_classes=["training-status-card"],
                )
                with gr.Row(equal_height=True, elem_classes=["training-live-grid"]):
                    training_loss_plot = gr.Plot(label="Live train / validation loss")
                    validation_metrics_plot = gr.Plot(label="Validation MSE / R²")
                validation_latent_plot = gr.Plot(
                    label="Validation latent-space PCA · updated asynchronously"
                )

                gr.HTML(
                    '<div class="workflow-divider"><span>💾 Checkpoints · automatic save and resume</span></div>'
                )
                gr.Markdown(
                    "ArchCon automatically writes **latest** after every completed epoch and **best** whenever validation improves. Runs are stored under `./models/` in the directory where ArchCon was launched. The download links below serve those exact files in place — ArchCon does **not** duplicate checkpoints into Gradio's cache. Other unavoidable Gradio temporary files/uploads are kept under `./.archcon-gradio/` on the same launch filesystem unless `GRADIO_TEMP_DIR` is explicitly set. To continue later, upload a trusted ArchCon `.pt` checkpoint; its architecture/settings are restored automatically before the next run.",
                    elem_classes=["reading-width"],
                )
                with gr.Row(equal_height=False, elem_classes=["checkpoint-grid"]):
                    latest_checkpoint_file = gr.HTML(
                        _checkpoint_download_html(None, "Latest checkpoint"),
                    )
                    best_checkpoint_file = gr.HTML(
                        _checkpoint_download_html(None, "Best validation checkpoint"),
                    )
                    checkpoint_upload = gr.File(
                        label="⬆️ Load trusted ArchCon checkpoint",
                        file_count="single",
                        file_types=[".pt", ".pth"],
                        type="filepath",
                    )
                    checkpoint_mode = gr.Radio(
                        choices=CHECKPOINT_MODES,
                        value=CHECKPOINT_WEIGHTS,
                        label="How to use loaded checkpoint",
                    )
                checkpoint_status = gr.Markdown(
                    "No checkpoint selected. New training starts from a fresh initialization.",
                    elem_classes=["training-status-card"],
                )

                initial_preview_config = TrainingConfig(
                    hidden_widths=(256, 64),
                    latent_dim=3,
                    activation="ReLU",
                    dropout=0.1,
                    weight_decay=0.0,
                    l2_lambda=1e-5,
                    architecture_family=ARCH_STADNIUK,
                    low_rank_dim=64,
                    residual_blocks=0,
                    residual_expansion=1,
                    stadniuk_batch_norm=False,
                    lr_schedule=LR_SCHEDULE_COSINE,
                    loss_name=LOSS_MSE,
                    compile_model=False,
                )
                with gr.Accordion(
                    "🧾 PyTorch model code / executable architecture plan", open=False
                ):
                    gr.Markdown(
                        "This audit view is generated from the **same execution plan used by the "
                        "actual PyTorch model builder**. It is intentionally explicit so the diagram, "
                        "layer order, residual widths, and optional `torch.compile` step can be checked "
                        "before training.",
                        elem_classes=["reading-width"],
                    )
                    model_execution_plan = gr.Markdown(
                        model_execution_markdown(42917, initial_preview_config),
                        elem_classes=["architecture-card"],
                    )
                    model_code_preview = gr.Code(
                        value=pytorch_model_code(42917, initial_preview_config),
                        language="python",
                        label="Generated PyTorch structural audit",
                        interactive=False,
                        lines=28,
                        max_lines=80,
                        wrap_lines=False,
                    )

                with gr.Accordion("🧪 Headless / MetaCentrum sweep export", open=False):
                    gr.Markdown(
                        "Export the **current Stage 06 setup** as one readable executable Python "
                        "program per run, with a matching JSON record and a MetaCentrum PBS array "
                        "script. Every generated `jobs/run_XXXX.py` contains the exact PyTorch model "
                        "definition, loss, L2 penalty, optimizer, scheduler and all configuration "
                        "values; its `main()` loads the selected GEO preprocessing and outcome-blind supervised rows. "
                        "The expanded default grid has 900 runs and uses one shared molecular 90/5/5 split. "
                        "Every job loads test row identities only to report their count; it never evaluates them.",
                        elem_classes=["reading-width"],
                    )
                    gr.Markdown(
                        "The grid contains **540 Stadniuk MLP** runs (180 model configurations × 3 preprocessings) and "
                        "**360 ResNet-LN** runs (120 model configurations × 3 preprocessings). It varies five hidden-depth profiles, "
                        "latent dimensions 3/8/16, MSE/masked MSE, plus architecture-specific L2/BatchNorm or residual-block/expansion axes. "
                        "The preprocessing axis is Stadniuk rescaling / per-study RMA / global RMA. Seed 42, batch size 64, starting LR 1e-3 "
                        "and one-way cosine decay are fixed. Molecular test rows and all samples with eGFR remain blinded during the sweep. "
                        "The legacy-named global arm must be rebuilt against this frozen split before its jobs run; "
                        "held-out GEO arrays cannot contribute to its reference.",
                        elem_classes=["reading-width"],
                    )
                    sweep_grid_json = gr.Code(
                        value=recommended_comparison_grid_json(),
                        language="json",
                        label="Architecture + preprocessing hyperparameter grid · JSON · 900 runs",
                        interactive=True,
                        lines=28,
                        max_lines=80,
                    )
                    with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                        sweep_name = gr.Textbox(value="archcon-pretrain", label="Sweep name")
                        meta_project_dir = gr.Textbox(
                            value="$HOME/ArchCon",
                            label="MetaCentrum project directory",
                        )
                        meta_data_dir = gr.Textbox(
                            value="$HOME/ArchCon/data",
                            label="MetaCentrum data directory",
                        )
                        meta_python = gr.Textbox(
                            value="$HOME/ArchCon/.venv/bin/python",
                            label="MetaCentrum Python executable",
                        )
                    with gr.Row(equal_height=True, elem_classes=["training-config-row"]):
                        pbs_ncpus = gr.Number(value=1, precision=0, minimum=1, label="PBS CPUs")
                        pbs_memory = gr.Textbox(value="10gb", label="PBS RAM")
                        pbs_scratch = gr.Textbox(value="4gb", label="PBS scratch_local")
                        pbs_walltime = gr.Textbox(value="48:00:00", label="PBS walltime")
                        pbs_ngpus = gr.Number(value=0, precision=0, minimum=0, label="PBS GPUs")
                        pbs_gpu_memory = gr.Textbox(value="12gb", label="Minimum GPU memory")
                    generate_sweep_button = gr.Button(
                        "Generate Python jobs + JSON + PBS array",
                        variant="secondary",
                    )
                    sweep_status = gr.Markdown(
                        "No sweep generated yet. `{}` produces one headless run with the current controls.",
                        elem_classes=["training-status-card"],
                    )
                    sweep_script_preview = gr.Code(
                        value="",
                        language="shell",
                        label="Generated run_array.pbs.sh preview",
                        interactive=False,
                        lines=18,
                        max_lines=60,
                    )
                    sweep_bundle_download = gr.HTML(
                        '<div class="checkpoint-download-empty">Sweep ZIP: not generated yet.</div>'
                    )

            # -----------------------------------------------------------------
            # 07 · Downstream outcome / archetypal design
            # -----------------------------------------------------------------
            with gr.Column(visible=False, elem_classes=["pipeline-stage-panel"]) as stage_aa_panel:
                gr.Markdown(
                    PIPELINE_DETAILS["aa"],
                    elem_classes=["pipeline-stage-theory"],
                )
                gr.Markdown(
                    "### 📈 Downstream evaluation and AA\nThis stage is deliberately separate from molecular pretraining. Samples with eGFR enter here only after the molecular representation is frozen.",
                    elem_classes=["stage-subheading"],
                )
                aa_design_mode = gr.Radio(
                    choices=AA_DESIGN_OPTIONS,
                    value=AA_SHARED,
                    label="Future AA design",
                )
                aa_design_plot = gr.HTML(
                    aa_design_svg(AA_SHARED),
                    elem_classes=["architecture-viewport"],
                )
                aa_design_theory = gr.Markdown(
                    aa_design_markdown(AA_SHARED),
                    elem_classes=["pipeline-stage-theory"],
                )
                with gr.Accordion("🔬 Recent AA / multi-omics directions", open=True):
                    gr.Markdown(aa_research_markdown())
                gr.Button(
                    "🔒 Downstream modelling · next milestone",
                    interactive=False,
                    variant="secondary",
                )

        stage_panels = [
            stage_input_panel,
            stage_supervised_panel,
            stage_rma_panel,
            stage_matrix_panel,
            stage_split_panel,
            stage_latent_panel,
            stage_aa_panel,
        ]
        stage_buttons = [
            stage_input_button,
            stage_supervised_button,
            stage_rma_button,
            stage_matrix_button,
            stage_split_button,
            stage_latent_button,
            stage_aa_button,
        ]

        for button, stage_name in (
            (stage_input_button, "input"),
            (stage_supervised_button, "supervised"),
            (stage_rma_button, "rma"),
            (stage_matrix_button, "matrix"),
            (stage_split_button, "split"),
            (stage_latent_button, "latent"),
            (stage_aa_button, "aa"),
        ):
            button.click(
                fn=lambda name=stage_name: _stage_navigation_callback(name),
                outputs=[*stage_panels, *stage_buttons],
                api_visibility="private",
            )

        stage_supervised_button.click(
            fn=_supervised_dataset_async,
            inputs=[geo_store_path],
            outputs=[
                loading_overlay,
                supervised_overview,
                supervised_table,
                supervised_counts_plot,
                supervised_pca_plot,
            ],
            show_progress="hidden",
            concurrency_limit=4,
            api_visibility="private",
        )

        # RAW GEO stage -------------------------------------------------------
        raw_geo_dashboard_outputs = [
            geo_heading,
            geo_samples,
            geo_summary,
            geo_hist,
            geo_boxplot,
            geo_pca,
            geo_dataset_metadata,
            geo_source_links,
            geo_sample_plot,
            geo_sample_metadata,
        ]
        load_raw_geo_outputs = [
            geo_store_status_md,
            geo_catalog,
            geo_scope_state,
            geo_gse_state,
            *raw_geo_dashboard_outputs,
        ]
        app.load(
            fn=_raw_geo_load_async,
            inputs=[geo_store_path],
            outputs=[loading_overlay, *load_raw_geo_outputs],
            show_progress="hidden",
            concurrency_limit=4,
            api_visibility="private",
        )
        geo_refresh_button.click(
            fn=_raw_geo_load_async,
            inputs=[geo_store_path],
            outputs=[loading_overlay, *load_raw_geo_outputs],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=4,
            api_visibility="private",
        )
        show_all_geo_button.click(
            fn=_raw_geo_all_async,
            inputs=[geo_store_path],
            outputs=[
                loading_overlay,
                geo_scope_state,
                geo_gse_state,
                *raw_geo_dashboard_outputs,
            ],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=4,
            api_visibility="private",
        )
        geo_catalog.select(
            fn=_raw_geo_catalog_async,
            inputs=[geo_store_path],
            outputs=[
                loading_overlay,
                geo_scope_state,
                geo_gse_state,
                *raw_geo_dashboard_outputs,
            ],
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=4,
            api_visibility="private",
        )
        geo_samples.select(
            fn=_raw_geo_sample_async,
            inputs=[geo_store_path, geo_scope_state, geo_gse_state],
            outputs=[loading_overlay, geo_sample_plot, geo_sample_metadata],
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=4,
            api_visibility="private",
        )
        external_import_button.click(
            fn=_simple_expression_import_callback,
            inputs=[
                external_expression_upload,
                external_expression_path,
                external_orientation,
            ],
            outputs=[external_import_status, external_import_preview],
            api_visibility="private",
        )

        # RMA stage -----------------------------------------------------------
        rma_stage_outputs = [
            rma_heading,
            rma_summary,
            rma_pipeline,
            rma_metadata,
            rma_links,
            rma_hist,
            rma_compare,
            rma_boxplot,
            rma_pca,
            rma_samples,
        ]
        stage_rma_button.click(
            fn=_geo_stage_refresh_async,
            inputs=[geo_store_path, geo_scope_state, geo_gse_state, rma_method_state],
            outputs=[loading_overlay, *rma_stage_outputs],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=4,
            api_visibility="private",
        )
        for method_button, method, elem_id in (
            (rma_raw_button, METHOD_RAW, "rma-method-raw"),
            (rma_per_gse_button, METHOD_PER_GSE_RMA, "rma-method-per-gse"),
            (rma_global_button, METHOD_GLOBAL_RMA, "rma-method-global"),
        ):
            dependency = method_button.click(
                fn=partial(_rma_method_button_async, method=method),
                inputs=[geo_store_path, geo_scope_state, geo_gse_state],
                outputs=[
                    loading_overlay,
                    rma_method_state,
                    rma_raw_button,
                    rma_per_gse_button,
                    rma_global_button,
                    *rma_stage_outputs,
                ],
                js=_pending_button_js(elem_id),
                show_progress="hidden",
                trigger_mode="once",
                concurrency_limit=4,
                api_visibility="private",
            )
            dependency.then(
                fn=None,
                js=_CLEAR_PENDING_JS,
                queue=False,
                api_visibility="private",
            )

        cel_source_mode.change(
            fn=_cel_source_mode_callback,
            inputs=[cel_source_mode, detected_cel_directory],
            outputs=[cel_uploads, cel_directory],
            api_visibility="private",
        )
        rma_output_mode.change(
            fn=_rma_output_mode_callback,
            inputs=[rma_output_mode, rma_data_directory],
            outputs=[rma_output_directory],
            api_visibility="private",
        )
        run_rma_button.click(
            fn=_run_rma_stage_callback,
            inputs=[
                cel_source_mode,
                cel_uploads,
                cel_directory,
                detected_cel_directory,
                rma_output_mode,
                rma_output_directory,
                rma_data_directory,
            ],
            outputs=[rma_run_status, rma_run_plot],
            api_visibility="private",
        )
        check_rma_button.click(
            fn=_rma_environment_callback,
            outputs=[rma_backend_status],
            api_visibility="private",
        )
        install_rma_button.click(
            fn=_install_rma_dependencies_callback,
            outputs=[rma_backend_status],
            api_visibility="private",
        )

        # Matrix stage --------------------------------------------------------
        matrix_stage_outputs = [
            matrix_heading,
            matrix_summary,
            matrix_pipeline,
            matrix_metadata,
            matrix_links,
            matrix_hist,
            matrix_compare,
            matrix_boxplot,
            matrix_pca,
            matrix_samples,
        ]
        stage_matrix_button.click(
            fn=_geo_stage_refresh_async,
            inputs=[geo_store_path, geo_scope_state, geo_gse_state, rma_method_state],
            outputs=[loading_overlay, *matrix_stage_outputs],
            show_progress="hidden",
            trigger_mode="once",
            concurrency_limit=4,
            api_visibility="private",
        )

        # Split stage ---------------------------------------------------------
        split_outputs = [split_state, split_summary, split_table, split_download]
        stage_split_button.click(
            fn=_split_open_callback,
            inputs=[geo_store_path, split_seed, train_fraction, split_state],
            outputs=split_outputs,
            api_visibility="private",
        )
        for split_control in (split_seed, train_fraction):
            split_control.change(
                fn=_generate_split_callback,
                inputs=[geo_store_path, split_seed, train_fraction],
                outputs=split_outputs,
                api_visibility="private",
            )
        split_upload.change(
            fn=_load_split_callback,
            inputs=[geo_store_path, split_upload],
            outputs=split_outputs,
            api_visibility="private",
        )

        # Latent stage --------------------------------------------------------
        architecture_outputs = [
            architecture_summary,
            loss_theory,
            architecture_plot,
            model_execution_plan,
            model_code_preview,
        ]
        architecture_inputs = [
            geo_store_path,
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            loss_name,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            compile_model,
        ]
        stage_latent_button.click(
            fn=_architecture_callback,
            inputs=architecture_inputs,
            outputs=architecture_outputs,
            api_visibility="private",
        )

        # A preset fills the controls first; the chained callback redraws the
        # architecture from the resulting values.  No explicit "apply" button
        # is needed and every custom edit below also redraws immediately.
        architecture_preset.change(
            fn=_architecture_preset_callback,
            inputs=[architecture_preset],
            outputs=[
                architecture_family,
                hidden_widths_text,
                activation,
                latent_dim,
                dropout,
                l2_lambda,
                stadniuk_batch_norm,
                low_rank_dim,
                residual_blocks,
                residual_expansion,
                batch_size,
                learning_rate,
                architecture_preset_note,
            ],
            api_visibility="private",
        ).then(
            fn=_architecture_callback,
            inputs=architecture_inputs,
            outputs=architecture_outputs,
            api_visibility="private",
        )
        for control in (
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            loss_name,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            compile_model,
        ):
            control.change(
                fn=_architecture_callback,
                inputs=architecture_inputs,
                outputs=architecture_outputs,
                api_visibility="private",
            )

        loss_name.change(
            fn=_loss_specific_control_updates,
            inputs=[loss_name],
            outputs=[
                huber_delta,
                cosine_weight,
                kl_beta,
                kl_warmup_epochs,
                mmd_weight,
                mask_fraction,
            ],
            api_visibility="private",
        )

        sweep_training_inputs = [
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule,
            optimizer_name,
            early_stopping_patience,
            lr_patience,
            gradient_clip,
            model_seed,
            training_device,
            precision,
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update_batches,
            latent_pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
        ]
        generate_sweep_button.click(
            fn=_sweep_export_callback,
            inputs=[
                geo_store_path,
                training_preprocessing,
                split_seed,
                train_fraction,
                split_state,
                *sweep_training_inputs,
                sweep_grid_json,
                sweep_name,
                meta_project_dir,
                meta_data_dir,
                meta_python,
                pbs_ncpus,
                pbs_memory,
                pbs_scratch,
                pbs_walltime,
                pbs_ngpus,
                pbs_gpu_memory,
            ],
            outputs=[sweep_status, sweep_script_preview, sweep_bundle_download],
            concurrency_limit=1,
            api_visibility="private",
        )

        checkpoint_restore_controls = [
            architecture_preset,
            training_preprocessing,
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule,
            optimizer_name,
            early_stopping_patience,
            lr_patience,
            gradient_clip,
            model_seed,
            training_device,
            precision,
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update_batches,
            latent_pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
            architecture_preset_note,
        ]
        checkpoint_upload.change(
            fn=_checkpoint_load_callback,
            inputs=[geo_store_path, checkpoint_upload],
            outputs=[
                checkpoint_state,
                checkpoint_status,
                *checkpoint_restore_controls,
                architecture_summary,
                loss_theory,
                architecture_plot,
                model_execution_plan,
                model_code_preview,
            ],
            api_visibility="private",
        )
        checkpoint_upload.clear(
            fn=_checkpoint_clear_callback,
            outputs=[checkpoint_state, checkpoint_status],
            api_visibility="private",
        )

        # Markdown note is restored by checkpoints but is not an interactive
        # control. Everything else is frozen while a model is running so the
        # live configuration cannot drift underneath the optimizer.
        training_lock_controls = [
            architecture_preset,
            training_preprocessing,
            architecture_family,
            hidden_widths_text,
            activation,
            latent_dim,
            dropout,
            l2_lambda,
            stadniuk_batch_norm,
            low_rank_dim,
            residual_blocks,
            residual_expansion,
            loss_name,
            epochs,
            batch_size,
            learning_rate,
            lr_schedule,
            optimizer_name,
            early_stopping_patience,
            lr_patience,
            gradient_clip,
            model_seed,
            training_device,
            precision,
            compile_model,
            background_prefetch,
            deterministic,
            validation_every,
            ui_update_batches,
            latent_pca_every,
            huber_delta,
            cosine_weight,
            kl_beta,
            kl_warmup_epochs,
            mmd_weight,
            mask_fraction,
            checkpoint_upload,
            checkpoint_mode,
        ]
        training_outputs = [
            training_run_state,
            training_status,
            training_loss_plot,
            validation_metrics_plot,
            validation_latent_plot,
            latest_checkpoint_file,
            best_checkpoint_file,
            *training_lock_controls,
            start_training_button,
            stop_training_button,
        ]
        start_training_button.click(
            fn=_train_autoencoder_callback,
            inputs=[
                geo_store_path,
                training_preprocessing,
                split_seed,
                train_fraction,
                split_state,
                architecture_family,
                hidden_widths_text,
                activation,
                latent_dim,
                dropout,
                l2_lambda,
                stadniuk_batch_norm,
                low_rank_dim,
                residual_blocks,
                residual_expansion,
                loss_name,
                epochs,
                batch_size,
                learning_rate,
                lr_schedule,
                optimizer_name,
                early_stopping_patience,
                lr_patience,
                gradient_clip,
                model_seed,
                training_device,
                precision,
                compile_model,
                background_prefetch,
                deterministic,
                validation_every,
                ui_update_batches,
                latent_pca_every,
                huber_delta,
                cosine_weight,
                kl_beta,
                kl_warmup_epochs,
                mmd_weight,
                mask_fraction,
                checkpoint_state,
                checkpoint_mode,
            ],
            outputs=training_outputs,
            concurrency_limit=1,
            concurrency_id="archcon-autoencoder-training",
            api_visibility="private",
        )
        stop_training_button.click(
            fn=_stop_training_callback,
            inputs=[training_run_state],
            outputs=[training_status, stop_training_button],
            concurrency_limit=4,
            api_visibility="private",
        )

        # Future AA design ---------------------------------------------------
        aa_design_mode.change(
            fn=_aa_design_callback,
            inputs=[aa_design_mode],
            outputs=[aa_design_theory, aa_design_plot],
            api_visibility="private",
        )

    app.queue(default_concurrency_limit=4)

    return app


def start(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    open_browser: bool = True,
) -> None:
    """Start the local web UI and block until it is stopped."""
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
        show_error=True,
        max_file_size=APP_UPLOAD_LIMIT,
        allowed_paths=[str(_model_output_root()), str(_sweep_output_root())],
        enable_monitoring=False,
        footer_links=[],
    )

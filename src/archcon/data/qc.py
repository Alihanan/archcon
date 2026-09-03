"""Deterministic quality-control visualizations for expression and clinical data."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def _finite_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _deterministic_pooled_values(
    expression: pd.DataFrame,
    *,
    max_samples: int = 24,
    max_values: int = 600_000,
) -> np.ndarray:
    """Return a bounded deterministic sample of values for cohort-level QC plots."""
    if expression.empty:
        return np.array([], dtype=float)

    n = len(expression)
    row_positions = np.unique(np.linspace(0, n - 1, min(n, max_samples), dtype=int))
    values = _finite_vector(expression.iloc[row_positions].to_numpy(dtype=float))
    if values.size > max_values:
        positions = np.linspace(0, values.size - 1, max_values, dtype=int)
        values = values[positions]
    return values


def _shared_bins(*vectors: np.ndarray, count: int = 70) -> np.ndarray | None:
    finite = [vector for vector in vectors if vector.size]
    if not finite:
        return None
    merged = np.concatenate(finite)
    if merged.size < 2:
        return None
    lo, hi = np.quantile(merged, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return np.linspace(lo, hi, count)


def plot_sample_histogram(expression: pd.DataFrame, sample_id: str | None):
    if expression.empty or sample_id not in expression.index:
        return _empty_figure("Select a valid sample after loading expression data.")
    values = _finite_vector(expression.loc[sample_id].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(values, bins=60)
    ax.set_title(f"Expression distribution — {sample_id}")
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Probe count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_sample_vs_reference(
    expression: pd.DataFrame,
    sample_id: str | None,
    reference: pd.DataFrame | None = None,
    *,
    reference_label: str = "Supervised reference",
):
    """Compare one selected sample against a persistent reference cohort."""
    if expression.empty or sample_id not in expression.index:
        return _empty_figure("Select a valid sample after loading expression data.")

    sample_values = _finite_vector(expression.loc[sample_id].to_numpy(dtype=float))
    if sample_values.size == 0:
        return _empty_figure("The selected sample contains no finite expression values.")

    reference_values = (
        _deterministic_pooled_values(reference) if reference is not None else np.array([], dtype=float)
    )
    bins = _shared_bins(sample_values, reference_values)
    if bins is None:
        bins = 60

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(
        sample_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.2,
        label=f"Selected sample: {sample_id}",
    )
    if reference_values.size:
        ax.hist(
            reference_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            label=reference_label,
        )
        title = "Selected sample vs supervised reference"
    else:
        title = "Selected sample distribution"
        ax.text(
            0.5,
            0.93,
            "No supervised reference is set yet",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    ax.set_title(title)
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_dataset_vs_reference(
    expression: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    *,
    current_label: str = "Current dataset",
    reference_label: str = "Supervised reference",
):
    """Compare a deterministic pooled current-data distribution with the supervised reference."""
    if expression.empty:
        return _empty_figure("Load expression data first.")

    current_values = _deterministic_pooled_values(expression)
    reference_values = (
        _deterministic_pooled_values(reference) if reference is not None else np.array([], dtype=float)
    )
    if current_values.size == 0:
        return _empty_figure("Expression matrix contains no finite values.")

    bins = _shared_bins(current_values, reference_values)
    if bins is None:
        bins = 70

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(
        current_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.3,
        label=current_label,
    )
    if reference_values.size:
        ax.hist(
            reference_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.3,
            label=reference_label,
        )
        title = "Current dataset vs supervised reference"
    else:
        title = "Current dataset distribution"
        ax.text(
            0.5,
            0.93,
            "Load or set a supervised reference to enable comparison",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    ax.set_title(title)
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_distribution_overlay(expression: pd.DataFrame, max_samples: int = 16):
    if expression.empty:
        return _empty_figure("Load expression data first.")
    n = len(expression)
    positions = np.unique(np.linspace(0, n - 1, min(n, max_samples), dtype=int))
    selected = expression.iloc[positions]

    finite = selected.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return _empty_figure("Expression matrix contains no finite values.")
    lo, hi = np.quantile(finite, [0.005, 0.995])
    bins = np.linspace(lo, hi, 70)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for sample_id, row in selected.iterrows():
        values = row.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        hist, edges = np.histogram(values, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, alpha=0.6, linewidth=1, label=str(sample_id))
    ax.set_title(f"Expression distributions — {len(selected)} deterministic sample(s)")
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    if len(selected) <= 8:
        ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_sample_boxplots(expression: pd.DataFrame, max_samples: int = 24):
    if expression.empty:
        return _empty_figure("Load expression data first.")
    n = len(expression)
    positions = np.unique(np.linspace(0, n - 1, min(n, max_samples), dtype=int))
    selected = expression.iloc[positions]
    data = [row[np.isfinite(row)] for row in selected.to_numpy(dtype=float)]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.boxplot(data, tick_labels=[str(x) for x in selected.index], showfliers=False)
    ax.set_title("Per-sample expression distributions")
    ax.set_ylabel("Expression value")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def _join_color_values(
    sample_ids: pd.Index,
    clinical: pd.DataFrame | None,
    id_column: str | None,
    variable: str | None,
) -> pd.Series | None:
    if clinical is None or id_column is None or variable is None:
        return None
    if variable not in clinical.columns:
        return None
    table = clinical[[id_column, variable]].drop_duplicates(id_column).set_index(id_column)
    return table[variable].reindex(sample_ids)


def plot_pca(
    expression: pd.DataFrame,
    clinical: pd.DataFrame | None = None,
    id_column: str | None = None,
    color_variable: str | None = None,
):
    if expression.empty or len(expression) < 3:
        return _empty_figure("At least three samples are required for PCA.")

    X = expression.to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        column_means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(column_means, np.where(bad)[1])

    model = PCA(n_components=2, svd_solver="randomized", random_state=0)
    coords = model.fit_transform(X)

    colors = _join_color_values(expression.index, clinical, id_column, color_variable)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    if colors is None or colors.isna().all():
        ax.scatter(coords[:, 0], coords[:, 1], s=36, alpha=0.8)
    else:
        numeric = pd.to_numeric(colors, errors="coerce")
        if numeric.notna().sum() >= max(3, int(0.7 * len(colors))):
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=numeric.to_numpy(dtype=float),
                s=40,
                alpha=0.85,
            )
            fig.colorbar(scatter, ax=ax, label=str(color_variable))
        else:
            categorical = colors.astype("string").fillna("Missing")
            for category in sorted(categorical.unique()):
                mask = categorical.to_numpy() == category
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=40,
                    alpha=0.8,
                    label=str(category),
                )
            if categorical.nunique() <= 12:
                ax.legend(title=str(color_variable), fontsize="small")

    variance = model.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    ax.set_title("Expression-space PCA (QC only)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_clinical_variable(frame: pd.DataFrame | None, variable: str | None):
    if frame is None or variable is None or variable not in frame.columns:
        return _empty_figure("Load a clinical table and choose a variable.")

    values = frame[variable]
    numeric = pd.to_numeric(values, errors="coerce")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if numeric.notna().sum() >= max(3, int(0.7 * len(values))) and numeric.nunique() > 10:
        ax.hist(numeric.dropna().to_numpy(dtype=float), bins=min(30, max(8, int(math.sqrt(len(values))))))
        ax.set_xlabel(str(variable))
        ax.set_ylabel("Patient count")
    else:
        counts = values.astype("string").fillna("Missing").value_counts(dropna=False)
        counts.iloc[:25].plot(kind="bar", ax=ax)
        ax.set_xlabel(str(variable))
        ax.set_ylabel("Patient count")
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(f"Clinical distribution — {variable}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_missingness(frame: pd.DataFrame | None, max_variables: int = 30):
    if frame is None or frame.empty:
        return _empty_figure("Load a clinical table first.")
    missing = frame.isna().mean().mul(100).sort_values(ascending=False).head(max_variables)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    missing[::-1].plot(kind="barh", ax=ax)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Clinical-data missingness")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_egfr_trajectories(
    frame: pd.DataFrame | None,
    id_column: str | None,
    max_patients: int = 35,
):
    if frame is None or id_column is None:
        return _empty_figure("Load an eGFR table first.")

    candidates = [
        ("egfr_7d", "7 d"),
        ("egfr_3m", "3 m"),
        ("egfr_6m", "6 m"),
        ("egfr_12m", "12 m"),
    ]
    available = [(column, label) for column, label in candidates if column in frame.columns]
    if len(available) < 2:
        return _empty_figure(
            "Expected at least two of: egfr_7d, egfr_3m, egfr_6m, egfr_12m."
        )

    columns = [column for column, _ in available]
    labels = [label for _, label in available]
    patient_table = frame.drop_duplicates(id_column).set_index(id_column)
    n = len(patient_table)
    positions = np.unique(np.linspace(0, n - 1, min(n, max_patients), dtype=int))
    selected = patient_table.iloc[positions]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(columns))
    for _, row in selected.iterrows():
        y = pd.to_numeric(row[columns], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, alpha=0.22, linewidth=1)

    median = patient_table[columns].apply(pd.to_numeric, errors="coerce").median(axis=0)
    ax.plot(x, median.to_numpy(dtype=float), marker="o", linewidth=3, label="Cohort median")
    ax.set_xticks(x, labels)
    ax.set_ylabel("eGFR")
    ax.set_title("Post-transplant eGFR trajectories")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_raw_vs_rma(raw_sample: pd.DataFrame | None, normalized: pd.DataFrame | None):
    if raw_sample is None or normalized is None or raw_sample.empty or normalized.empty:
        return _empty_figure("Run CEL → RMA preprocessing to compare distributions.")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    raw_values = np.log2(np.clip(raw_sample.to_numpy(dtype=float), 1e-6, None))
    raw_values = raw_values[np.isfinite(raw_values)]
    norm_values = normalized.to_numpy(dtype=float)
    norm_values = norm_values[np.isfinite(norm_values)]
    if not raw_values.size or not norm_values.size:
        return _empty_figure("No finite raw/normalized values were available.")
    ax.hist(raw_values, bins=70, density=True, histtype="step", linewidth=2, label="Raw CEL (log2 sample)")
    ax.hist(norm_values, bins=70, density=True, histtype="step", linewidth=2, label="RMA expression")
    ax.set_xlabel("log2 intensity / expression")
    ax.set_ylabel("Density")
    ax.set_title("Before and after RMA")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

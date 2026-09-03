"""Disk-backed exploration of reconstructed GEO normalization variants.

The store is produced by the GEO preprocessing/recovery scripts and intentionally
keeps large expression matrices outside the Python package. NumPy ``.npy`` files
are opened with memory mapping, so interactive summaries can inspect deterministic
subsets without loading the full 10k+ x 40k+ matrices into RAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCOPE_AGGREGATE: Final = "All unique GSMs"
SCOPE_SERIES: Final = "GEO Series (source GSE)"

METHOD_RAW: Final = "Raw original (PM median)"
METHOD_PER_GSE_RMA: Final = "Per-dataset RMA"
METHOD_GLOBAL_RMA: Final = "Global RMA"

GEO_RMA_SCOPES: Final = [SCOPE_AGGREGATE, SCOPE_SERIES]
GEO_RMA_METHODS: Final = [METHOD_RAW, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA]

_METHOD_AGGREGATE_FILE: Final = {
    METHOD_RAW: "raw_original.npy",
    METHOD_PER_GSE_RMA: "rma_per_gse.npy",
    METHOD_GLOBAL_RMA: "rma_global.npy",
}

_METHOD_SERIES_FILE: Final = {
    METHOD_RAW: "per_dataset_raw_original.npy",
    METHOD_PER_GSE_RMA: "per_dataset_rma_per_gse.npy",
}

REQUIRED_STORE_FILES: Final = (
    "raw_original.npy",
    "rma_per_gse.npy",
    "rma_global.npy",
    "per_dataset_raw_original.npy",
    "per_dataset_rma_per_gse.npy",
    "sample_index.csv",
    "source_sample_occurrences.csv",
    "source_gse_occurrence_index.csv",
    "gse_index.csv",
    "probe_index.csv",
    "cel_manifest.csv",
)

OPTIONAL_STORE_FILES: Final = (
    "store_manifest.csv",
    "sample_occurrences.csv",
    "series_alias_occurrences.csv",
    "source_gse_index.csv",
    "stadniuk_gsm_to_gse_mapping.csv",
    "raw_archive_health.csv",
    "global_rma_streaming_validation.csv",
    "numpy_validation.json",
    "README.txt",
    "COMPLETE.txt",
)

_PIPELINE_TEXT: Final = {
    METHOD_RAW: """### ① No RMA · reconstructed raw signal

`CEL → PM intensities → median for each common probe set`

This is the **before-RMA** reference. Values are still on the intensity scale: no background correction, no quantile normalization, and no RMA log2 summarization.
""",
    METHOD_PER_GSE_RMA: """### ② RMA inside each GEO study · leakage-safe comparison arm

`CELs from one GSE → background correction → quantile normalization → median polish + log2`

Each **GSE study is normalized separately**. ArchCon holds out complete GSE components, so validation/test studies never contribute to preprocessing of training studies. A newly arriving external GSE is treated the same way: RMA is performed within that new study before inference. This is the leakage-safe deployment-style arm of the three-preprocessing comparison. The reconstructed `raw_original.npy` is already summarized to probe sets, so it is **not** sufficient to refit exact probe-level train/add-on RMA; that alternative requires the original CEL/probe-level files.
""",
    METHOD_GLOBAL_RMA: """### ③ One global RMA · transductive comparison arm

`every unique GSM once → background correction → one shared quantile target → median polish + log2`

All available arrays contribute to one RMA solution. A GSM that appears in both a SubSeries and SuperSeries is counted only once. The expanded comparison sweep includes this matrix deliberately as a **transductive benchmark**, but validation/test studies contributed to the shared normalization. Therefore its downstream held-out scores must not be interpreted as an unbiased preprocessing-generalization estimate; the neural network still never trains on validation/test rows.
""",
}



def _natural_gse_key(value: str) -> tuple[int, str]:
    text = str(value)
    try:
        return int(text.upper().replace("GSE", "", 1)), text
    except ValueError:
        return 10**18, text


def _geo_url(accession: str) -> str:
    accession = str(accession).strip().upper()
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={quote(accession)}"


def _geo_raw_tar_url(gse: str) -> str:
    gse = str(gse).strip().upper()
    digits = gse.removeprefix("GSE")
    if not digits.isdigit() or len(digits) < 3:
        return _geo_url(gse)
    bucket = f"GSE{digits[:-3]}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{gse}/suppl/{gse}_RAW.tar"


def _human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _deterministic_positions(length: int, maximum: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    if length <= maximum:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def _empty_figure(message: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


@dataclass
class GeoExpressionStore:
    """Memory-mapped access to the reconstructed GEO expression store."""

    root: Path
    sample_index: pd.DataFrame
    source_occurrences: pd.DataFrame
    source_gse_index: pd.DataFrame
    gse_index: pd.DataFrame
    probe_index: pd.DataFrame
    cel_manifest: pd.DataFrame
    _arrays: dict[str, np.memmap] = field(default_factory=dict, repr=False)

    @classmethod
    def open(cls, root: str | Path) -> "GeoExpressionStore":
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"GEO NumPy store not found: {path}")

        missing = [name for name in REQUIRED_STORE_FILES if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "GEO NumPy store is incomplete. Missing: " + ", ".join(missing)
            )

        sample_index = pd.read_csv(path / "sample_index.csv", low_memory=False)
        source_occurrences = pd.read_csv(
            path / "source_sample_occurrences.csv", low_memory=False
        )
        source_gse_index = pd.read_csv(path / "source_gse_occurrence_index.csv")
        gse_index = pd.read_csv(path / "gse_index.csv")
        probe_index = pd.read_csv(path / "probe_index.csv", low_memory=False)
        cel_manifest = pd.read_csv(path / "cel_manifest.csv", low_memory=False)

        for frame in (sample_index, source_occurrences, cel_manifest):
            for column in ("GSM", "GSE", "source_GSE", "canonical_GSE", "cel_source_GSE"):
                if column in frame.columns:
                    frame[column] = frame[column].astype("string")

        store = cls(
            root=path,
            sample_index=sample_index,
            source_occurrences=source_occurrences,
            source_gse_index=source_gse_index,
            gse_index=gse_index,
            probe_index=probe_index,
            cel_manifest=cel_manifest,
        )
        store._validate_shapes()
        return store

    @property
    def n_samples(self) -> int:
        return len(self.sample_index)

    @property
    def n_occurrences(self) -> int:
        return len(self.source_occurrences)

    @property
    def n_probes(self) -> int:
        return len(self.probe_index)

    @property
    def gses(self) -> list[str]:
        values = self.source_occurrences["source_GSE"].dropna().astype(str).unique().tolist()
        return sorted(values, key=_natural_gse_key)

    def array(self, filename: str) -> np.memmap:
        if filename not in self._arrays:
            self._arrays[filename] = np.load(
                self.root / filename,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._arrays[filename]

    def _validate_shapes(self) -> None:
        canonical_shape = (self.n_samples, self.n_probes)
        occurrence_shape = (self.n_occurrences, self.n_probes)
        for filename in _METHOD_AGGREGATE_FILE.values():
            shape = self.array(filename).shape
            if shape != canonical_shape:
                raise ValueError(f"{filename}: shape {shape} != {canonical_shape}")
        for filename in _METHOD_SERIES_FILE.values():
            shape = self.array(filename).shape
            if shape != occurrence_shape:
                raise ValueError(f"{filename}: shape {shape} != {occurrence_shape}")

    def selection(self, scope: str, gse: str | None) -> tuple[pd.DataFrame, np.ndarray]:
        """Return display metadata and canonical/source occurrence row indices."""
        if scope == SCOPE_AGGREGATE:
            frame = self.sample_index.copy()
            if "global_row_python" in frame.columns:
                rows = pd.to_numeric(frame["global_row_python"], errors="raise").to_numpy(dtype=int)
            else:
                rows = np.arange(len(frame), dtype=int)
            return frame, rows

        if scope != SCOPE_SERIES:
            raise ValueError(f"Unknown GEO scope: {scope}")
        if not gse:
            raise ValueError("Choose a source GSE for the GEO Series view.")

        frame = self.source_occurrences[
            self.source_occurrences["source_GSE"].astype(str) == str(gse)
        ].copy()
        if frame.empty:
            raise ValueError(f"No source-GSE rows found for {gse}.")
        if "occurrence_row_python" in frame.columns:
            rows = pd.to_numeric(frame["occurrence_row_python"], errors="raise").to_numpy(dtype=int)
        else:
            rows = frame.index.to_numpy(dtype=int)
        order = np.argsort(rows)
        return frame.iloc[order].reset_index(drop=True), rows[order]

    def matrix_rows(
        self,
        method: str,
        scope: str,
        gse: str | None,
    ) -> tuple[np.memmap, np.ndarray, pd.DataFrame]:
        metadata, scope_rows = self.selection(scope, gse)

        if scope == SCOPE_AGGREGATE:
            filename = _METHOD_AGGREGATE_FILE[method]
            return self.array(filename), scope_rows, metadata

        if method in _METHOD_SERIES_FILE:
            filename = _METHOD_SERIES_FILE[method]
            return self.array(filename), scope_rows, metadata

        # Global RMA exists once per unique GSM. Map exact source-GSE membership
        # rows to the canonical unique-GSM matrix.
        global_row_by_gsm = pd.Series(
            pd.to_numeric(self.sample_index["global_row_python"], errors="raise").to_numpy(dtype=int),
            index=self.sample_index["GSM"].astype(str),
        )
        global_rows = metadata["GSM"].astype(str).map(global_row_by_gsm)
        if global_rows.isna().any():
            missing = metadata.loc[global_rows.isna(), "GSM"].astype(str).head(10).tolist()
            raise ValueError("Global-RMA rows missing for GSM(s): " + ", ".join(missing))
        return (
            self.array(_METHOD_AGGREGATE_FILE[METHOD_GLOBAL_RMA]),
            global_rows.to_numpy(dtype=int),
            metadata,
        )

    def dataset_catalog(self) -> pd.DataFrame:
        occurrences = self.source_occurrences.copy()
        multiplicity = occurrences.groupby("GSM", dropna=False).size()
        occurrences["shared_GSM"] = occurrences["GSM"].map(multiplicity).fillna(1).gt(1)

        grouped = occurrences.groupby("source_GSE", sort=False, dropna=False)
        catalog = grouped.agg(
            sample_memberships=("GSM", "size"),
            unique_GSMs=("GSM", "nunique"),
            shared_with_other_series=("shared_GSM", "sum"),
            canonical_GSEs=("canonical_GSE", "nunique"),
        ).reset_index()
        catalog = catalog.rename(columns={"source_GSE": "GSE"})
        catalog["GEO"] = catalog["GSE"].map(
            lambda value: f"[{value}]({_geo_url(str(value))})"
        )
        catalog["RAW archive"] = catalog["GSE"].map(
            lambda value: f"[RAW.tar]({_geo_raw_tar_url(str(value))})"
        )
        catalog = catalog.sort_values(
            "GSE", key=lambda series: series.map(lambda x: _natural_gse_key(str(x))[0])
        ).reset_index(drop=True)
        return catalog[
            [
                "GSE",
                "sample_memberships",
                "unique_GSMs",
                "shared_with_other_series",
                "canonical_GSEs",
                "GEO",
                "RAW archive",
            ]
        ]

    def browser_catalog(self) -> pd.DataFrame:
        """Return the compact, clickable catalog used by the web dashboard."""
        catalog = self.dataset_catalog().copy()
        catalog["overlap"] = np.where(
            catalog["shared_with_other_series"].astype(int).gt(0),
            "shared GSMs",
            "unique to Series",
        )
        return catalog[
            [
                "GSE",
                "unique_GSMs",
                "sample_memberships",
                "shared_with_other_series",
                "overlap",
                "GEO",
                "RAW archive",
            ]
        ]

    def sample_table(self, scope: str, gse: str | None) -> pd.DataFrame:
        frame, _ = self.selection(scope, gse)
        wanted = [
            "GSM",
            "source_GSE",
            "GSE",
            "canonical_GSE",
            "stadniuk_gse",
            "is_multi_series_gsm",
            "series_relation",
            "selection_reason",
        ]
        columns = [column for column in wanted if column in frame.columns]
        result = frame[columns].copy()
        if "GSM" in result.columns:
            result["GEO sample"] = result["GSM"].astype(str).map(
                lambda value: f"[{value}]({_geo_url(value)})"
            )
        return result


def load_geo_expression_store(path: str | Path) -> GeoExpressionStore:
    """Open a store, caching metadata/memmaps by resolved path."""
    return _load_geo_expression_store_cached(str(Path(path).expanduser().resolve()))


@lru_cache(maxsize=4)
def _load_geo_expression_store_cached(path: str) -> GeoExpressionStore:
    return GeoExpressionStore.open(path)


def raw_geo_overview_markdown(store: GeoExpressionStore) -> str:
    """Beginner-friendly summary for the first, raw-data pipeline stage."""
    raw_path = store.root / _METHOD_AGGREGATE_FILE[METHOD_RAW]
    return f"""
<div class="metric-row">
  <div class="metric"><div class="value">{len(store.gses):,}</div><div class="label">📚 GSE studies</div></div>
  <div class="metric"><div class="value">{store.n_samples:,}</div><div class="label">🧪 unique GSM samples</div></div>
  <div class="metric"><div class="value">{store.n_probes:,}</div><div class="label">🧬 common probe sets</div></div>
  <div class="metric"><div class="value">{_human_size(raw_path)}</div><div class="label">💾 raw matrix on disk</div></div>
</div>

**GEO** = NCBI Gene Expression Omnibus. **GSE** = one study/Series. **GSM** = one sample/array. **CEL** = the original Affymetrix chip file.  
ArchCon shows a compact **before-RMA** matrix here: one row per GSM and one column per common PrimeView probe set.
"""


def plot_selected_sample_method(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
    gsm: str | None,
):
    """Plot one sample under one preprocessing method, without cross-method overlay."""
    if not gsm:
        return _empty_figure("Click a GSM row to inspect one sample.")

    matrix, rows, metadata = store.matrix_rows(method, scope, gse)
    matches = np.flatnonzero(metadata["GSM"].astype(str).to_numpy() == str(gsm))
    if matches.size == 0:
        return _empty_figure(f"{gsm} was not found in this view.")

    values = np.asarray(matrix[rows[matches[0]], :], dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_figure(f"{gsm} has no finite values.")

    lo, hi = np.quantile(values, [0.002, 0.998])
    clipped = values[(values >= lo) & (values <= hi)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(clipped, bins=85, density=True, histtype="stepfilled", alpha=0.45)
    ax.axvline(np.median(values), linewidth=1.5, linestyle="--", label="Median")
    ax.set_title(f"{gsm} · {method}")
    ax.set_xlabel("Intensity / expression value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def geo_store_status(store: GeoExpressionStore) -> str:
    multi_series = 0
    if "is_multi_series_gsm" in store.sample_index.columns:
        multi_series = int(
            store.sample_index["is_multi_series_gsm"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            .sum()
        )
    extra_memberships = store.n_occurrences - store.n_samples

    validation_notes: list[str] = []
    complete = store.root / "COMPLETE.txt"
    validation_notes.append(
        "✅ final NumPy store marked complete"
        if complete.is_file()
        else "⚠️ COMPLETE.txt not present"
    )

    streaming_validation = store.root / "global_rma_streaming_validation.csv"
    if streaming_validation.is_file():
        try:
            validation = pd.read_csv(streaming_validation)
            if not validation.empty:
                row = validation.iloc[0]
                validation_notes.append(
                    "streaming-RMA validation "
                    f"max |Δ|={float(row.get('max_abs', float('nan'))):.3g}, "
                    f"RMSE={float(row.get('rmse', float('nan'))):.3g}"
                )
        except Exception:  # noqa: BLE001
            validation_notes.append("streaming-RMA validation file present")

    archive_health = store.root / "raw_archive_health.csv"
    if archive_health.is_file():
        try:
            health = pd.read_csv(archive_health)
            if "tar_ok" in health.columns:
                ok = health["tar_ok"].astype(str).str.lower().isin({"true", "1", "yes"})
                validation_notes.append(
                    f"RAW archive check: {int(ok.sum()):,}/{len(health):,} valid"
                )
        except Exception:  # noqa: BLE001
            validation_notes.append("RAW archive health file present")

    validation_text = " · ".join(validation_notes)
    return f"""
✅ **GEO normalization store ready** · `{store.root}`

<div class="metric-row">
  <div class="metric"><div class="value">{store.n_samples:,}</div><div class="label">unique GSMs</div></div>
  <div class="metric"><div class="value">{store.n_occurrences:,}</div><div class="label">GSM–GSE memberships</div></div>
  <div class="metric"><div class="value">{extra_memberships:,}</div><div class="label">extra Series memberships</div></div>
  <div class="metric"><div class="value">{len(store.gses):,}</div><div class="label">source GSEs</div></div>
  <div class="metric"><div class="value">{store.n_probes:,}</div><div class="label">common probes</div></div>
</div>

Canonical matrices: **{_human_size(Path(store.root / 'raw_original.npy'))} raw**, **{_human_size(Path(store.root / 'rma_per_gse.npy'))} per-GSE RMA**, **{_human_size(Path(store.root / 'rma_global.npy'))} global RMA**.  
Exact per-Series matrices occupy **{_human_size(Path(store.root / 'per_dataset_raw_original.npy'))} + {_human_size(Path(store.root / 'per_dataset_rma_per_gse.npy'))}**.  
{multi_series:,} unique GSMs are explicitly marked as belonging to multiple processed GEO Series.  
**Validation:** {validation_text}
"""


def normalization_pipeline_markdown(method: str) -> str:
    return _PIPELINE_TEXT.get(method, "")


def _sampled_values(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
    *,
    max_samples: int = 96,
    max_probes: int = 4096,
) -> tuple[np.ndarray, int, int]:
    matrix, rows, _ = store.matrix_rows(method, scope, gse)
    sample_positions = _deterministic_positions(len(rows), max_samples)
    probe_positions = _deterministic_positions(store.n_probes, max_probes)
    if sample_positions.size == 0 or probe_positions.size == 0:
        return np.array([], dtype=np.float32), 0, 0
    chosen_rows = rows[sample_positions]
    values = np.asarray(matrix[np.ix_(chosen_rows, probe_positions)], dtype=np.float32)
    finite = values[np.isfinite(values)]
    return finite, len(chosen_rows), len(probe_positions)


def scope_summary(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
) -> str:
    values, sampled_samples, sampled_probes = _sampled_values(
        store, method, scope, gse
    )
    _, rows, _ = store.matrix_rows(method, scope, gse)
    matrix = store.array(
        _METHOD_AGGREGATE_FILE[method]
        if scope == SCOPE_AGGREGATE or method == METHOD_GLOBAL_RMA
        else _METHOD_SERIES_FILE[method]
    )

    scope_name = "all unique GSMs" if scope == SCOPE_AGGREGATE else str(gse)
    if values.size:
        q01, q25, median, q75, q99 = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
        mean = float(np.mean(values))
        std = float(np.std(values))
        sampled_note = (
            f"Statistics use a deterministic subset of **{sampled_samples:,} samples × "
            f"{sampled_probes:,} probes** so the multi-GB matrices stay memory-mapped."
        )
    else:
        q01 = q25 = median = q75 = q99 = mean = std = float("nan")
        sampled_note = "No finite values were available for the selected view."

    return f"""
<div class="metric-row">
  <div class="metric"><div class="value">{len(rows):,}</div><div class="label">rows in view</div></div>
  <div class="metric"><div class="value">{store.n_probes:,}</div><div class="label">probe sets</div></div>
  <div class="metric"><div class="value">{median:.3f}</div><div class="label">sampled median</div></div>
  <div class="metric"><div class="value">{q25:.3f}–{q75:.3f}</div><div class="label">sampled IQR</div></div>
  <div class="metric"><div class="value">{mean:.3f} ± {std:.3f}</div><div class="label">sampled mean ± SD</div></div>
</div>

**View:** {scope_name} · **pipeline:** {method} · **matrix dtype:** `{matrix.dtype}` · **shape on disk:** `{matrix.shape}`  
Sampled 1st–99th percentile: **{q01:.3f}–{q99:.3f}**. {sampled_note}
"""


def plot_scope_histogram(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
):
    values, n_samples, n_probes = _sampled_values(store, method, scope, gse)
    if values.size == 0:
        return _empty_figure("No finite expression values in this view.")

    lo, hi = np.quantile(values, [0.001, 0.999])
    clipped = values[(values >= lo) & (values <= hi)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.hist(clipped, bins=80, density=True, histtype="stepfilled", alpha=0.45)
    ax.axvline(np.median(values), linewidth=1.5, linestyle="--", label="Median")
    ax.set_title(f"{method} · value distribution")
    ax.set_xlabel("Expression / intensity value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend()
    ax.text(
        0.99,
        0.97,
        f"deterministic sample: {n_samples:,} arrays × {n_probes:,} probes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        alpha=0.7,
    )
    fig.tight_layout()
    return fig


def plot_normalization_comparison(
    store: GeoExpressionStore,
    scope: str,
    gse: str | None,
):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    plotted = 0
    for method in GEO_RMA_METHODS:
        values, _, _ = _sampled_values(
            store,
            method,
            scope,
            gse,
            max_samples=72,
            max_probes=3072,
        )
        if values.size == 0:
            continue
        label = method
        if method == METHOD_RAW:
            values = np.log2(np.clip(values, 1e-6, None))
            label = "Raw original · log2 for visual comparison"
        lo, hi = np.quantile(values, [0.002, 0.998])
        values = values[(values >= lo) & (values <= hi)]
        hist, edges = np.histogram(values, bins=80, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, linewidth=1.7, label=label)
        plotted += 1

    if not plotted:
        plt.close(fig)
        return _empty_figure("No finite values available for comparison.")

    ax.set_title("Normalization comparison")
    ax.set_xlabel("log2 intensity / RMA expression")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_sample_boxplots(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
    *,
    max_samples: int = 28,
    max_probes: int = 6000,
):
    matrix, rows, metadata = store.matrix_rows(method, scope, gse)
    positions = _deterministic_positions(len(rows), max_samples)
    probes = _deterministic_positions(store.n_probes, max_probes)
    if positions.size == 0:
        return _empty_figure("No samples in this view.")
    chosen_rows = rows[positions]
    block = np.asarray(matrix[np.ix_(chosen_rows, probes)], dtype=np.float32)
    labels = metadata.iloc[positions]["GSM"].astype(str).tolist()
    data = [row[np.isfinite(row)] for row in block]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(f"{method} · per-sample distributions")
    ax.set_ylabel("Expression / intensity value")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def plot_scope_pca(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
    *,
    max_samples: int = 420,
    max_probes: int = 1800,
):
    matrix, rows, metadata = store.matrix_rows(method, scope, gse)
    if len(rows) < 3:
        return _empty_figure("At least three samples are required for PCA.")
    sample_positions = _deterministic_positions(len(rows), max_samples)
    probe_positions = _deterministic_positions(store.n_probes, max_probes)
    chosen_rows = rows[sample_positions]
    X = np.asarray(matrix[np.ix_(chosen_rows, probe_positions)], dtype=np.float32)
    if not np.isfinite(X).all():
        means = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
        bad = ~np.isfinite(X)
        X[bad] = np.take(means, np.where(bad)[1])

    model = PCA(n_components=2, svd_solver="randomized", random_state=0)
    coords = model.fit_transform(X)
    variance = model.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.72)
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    ax.set_title(f"{method} · PCA of deterministic subset")
    ax.grid(alpha=0.2)
    ax.text(
        0.99,
        0.02,
        f"{len(sample_positions):,} samples × {len(probe_positions):,} probes",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.7,
    )
    fig.tight_layout()
    return fig


def selected_sample_plot(
    store: GeoExpressionStore,
    scope: str,
    gse: str | None,
    gsm: str | None,
):
    if not gsm:
        return _empty_figure("Choose a GSM to inspect one array across pipelines.")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    plotted = 0
    for method in GEO_RMA_METHODS:
        matrix, rows, metadata = store.matrix_rows(method, scope, gse)
        matches = np.flatnonzero(metadata["GSM"].astype(str).to_numpy() == str(gsm))
        if matches.size == 0:
            continue
        row = rows[matches[0]]
        values = np.asarray(matrix[row, :], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        label = method
        if method == METHOD_RAW:
            values = np.log2(np.clip(values, 1e-6, None))
            label = "Raw original · log2 for visual comparison"
        lo, hi = np.quantile(values, [0.002, 0.998])
        values = values[(values >= lo) & (values <= hi)]
        hist, edges = np.histogram(values, bins=85, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, linewidth=1.7, label=label)
        plotted += 1

    if not plotted:
        plt.close(fig)
        return _empty_figure(f"{gsm} was not found in the selected view.")

    ax.set_title(f"{gsm} · normalization pipelines")
    ax.set_xlabel("log2 intensity / RMA expression")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def selected_sample_metadata(
    store: GeoExpressionStore,
    scope: str,
    gse: str | None,
    gsm: str | None,
) -> str:
    if not gsm:
        return "Choose a GSM to see provenance and GEO links."

    canonical = store.sample_index[store.sample_index["GSM"].astype(str) == str(gsm)]
    occurrences = store.source_occurrences[
        store.source_occurrences["GSM"].astype(str) == str(gsm)
    ]
    if canonical.empty and occurrences.empty:
        return f"`{gsm}` was not found in the store."

    canonical_gse = "—"
    selected_source = "—"
    global_row = "—"
    selection_reason = "—"
    if not canonical.empty:
        row = canonical.iloc[0]
        canonical_gse = str(row.get("GSE", row.get("canonical_GSE", "—")))
        selected_source = str(row.get("source_GSE", "—"))
        global_row = str(row.get("global_row_python", "—"))
        selection_reason = str(row.get("selection_reason", "—"))

    all_sources = sorted(
        occurrences["source_GSE"].dropna().astype(str).unique().tolist(),
        key=_natural_gse_key,
    )
    source_links = ", ".join(f"[{value}]({_geo_url(value)})" for value in all_sources) or "—"
    current_view = "all unique GSMs" if scope == SCOPE_AGGREGATE else str(gse)

    return f"""
### {gsm}

[GEO sample page]({_geo_url(str(gsm))}) · current view: **{current_view}**

| Field | Value |
|---|---|
| canonical GSE | [{canonical_gse}]({_geo_url(canonical_gse)}) |
| selected expression source GSE | {selected_source} |
| all processed source GSE memberships | {source_links} |
| global matrix row (0-based) | {global_row} |
| canonical-selection rule | `{selection_reason}` |
| number of processed GSE memberships | {len(occurrences):,} |
"""


def dataset_heading(
    store: GeoExpressionStore,
    scope: str,
    gse: str | None,
) -> str:
    """Compact heading for the currently selected aggregate or GEO Series."""
    if scope == SCOPE_AGGREGATE:
        return (
            f"## All unique GEO samples\n\n"
            f"**{store.n_samples:,} unique GSMs** · **{len(store.gses):,} source GSEs** · "
            f"**{store.n_probes:,} common probes**"
        )

    if not gse:
        return "## GEO Series\n\nChoose a Series from the dataset browser."

    catalog = store.dataset_catalog()
    row = catalog[catalog["GSE"].astype(str) == str(gse)]
    if row.empty:
        return f"## {gse}"
    item = row.iloc[0]
    shared = int(item["shared_with_other_series"])
    overlap = (
        f"{shared:,} memberships are shared with another processed Series"
        if shared
        else "no shared GSM memberships in the processed collection"
    )
    return (
        f"## {gse}\n\n"
        f"**{int(item['unique_GSMs']):,} unique GSMs** · "
        f"**{int(item['sample_memberships']):,} sample-Series memberships** · {overlap}"
    )


def dataset_metadata_markdown(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
) -> str:
    """Show human-readable dataset provenance beside the plots."""
    matrix_filename = (
        _METHOD_AGGREGATE_FILE[method]
        if scope == SCOPE_AGGREGATE or method == METHOD_GLOBAL_RMA
        else _METHOD_SERIES_FILE[method]
    )
    lines = [
        "### Dataset metadata",
        "",
        f"**Normalization:** {method}",
        f"**Memory-mapped matrix:** `{matrix_filename}`",
        f"**Local store:** `{store.root}`",
    ]

    if scope == SCOPE_AGGREGATE:
        lines.extend(
            [
                f"**Rows:** {store.n_samples:,} unique GSMs (each physical array once)",
                f"**Columns:** {store.n_probes:,} common PrimeView probe sets",
                f"**Source Series represented:** {len(store.gses):,}",
            ]
        )
    elif gse:
        catalog = store.dataset_catalog()
        row = catalog[catalog["GSE"].astype(str) == str(gse)]
        if not row.empty:
            item = row.iloc[0]
            lines.extend(
                [
                    f"**Unique GSMs in Series:** {int(item['unique_GSMs']):,}",
                    f"**Series memberships:** {int(item['sample_memberships']):,}",
                    f"**Shared with another processed Series:** "
                    f"{int(item['shared_with_other_series']):,}",
                ]
            )
        lines.extend(
            [
                f"**GEO:** [{gse}]({_geo_url(gse)})",
                f"**Original RAW archive:** [{gse}_RAW.tar]({_geo_raw_tar_url(gse)})",
            ]
        )

    lines.append(
        "\nValues are read on demand with `numpy.load(..., mmap_mode='r')`; "
        "the full matrix is not copied into RAM."
    )
    return "  \n".join(lines)


def source_links_markdown(
    store: GeoExpressionStore,
    method: str,
    scope: str,
    gse: str | None,
) -> str:
    matrix_filename = (
        _METHOD_AGGREGATE_FILE[method]
        if scope == SCOPE_AGGREGATE or method == METHOD_GLOBAL_RMA
        else _METHOD_SERIES_FILE[method]
    )
    lines = [
        "### Files and provenance",
        "",
        f"**Matrix file:** `{store.root / matrix_filename}`",
        f"**Sample index:** `{store.root / 'sample_index.csv'}`",
        f"**Probe index:** `{store.root / 'probe_index.csv'}`",
    ]

    if scope == SCOPE_SERIES and gse:
        occurrences = store.source_occurrences[
            store.source_occurrences["source_GSE"].astype(str) == str(gse)
        ]
        raw_rds = "—"
        rma_rds = "—"
        if not occurrences.empty:
            if "raw_rds" in occurrences.columns:
                values = occurrences["raw_rds"].dropna().astype(str).unique().tolist()
                if values:
                    raw_rds = values[0]
            if "rma_rds" in occurrences.columns:
                values = occurrences["rma_rds"].dropna().astype(str).unique().tolist()
                if values:
                    rma_rds = values[0]

        lines.extend(
            [
                "",
                f"**GEO Series:** [{gse}]({_geo_url(gse)})",
                f"**Original RAW archive:** [{gse}_RAW.tar]({_geo_raw_tar_url(gse)})",
                f"**Per-GSE reconstructed raw RDS provenance:** `{raw_rds}`",
                f"**Per-GSE reconstructed RMA RDS provenance:** `{rma_rds}`",
            ]
        )

    lines.extend(
        [
            "",
            "The web view reads the `.npy` matrix with `numpy.load(..., mmap_mode='r')`; it does not copy the full matrix into RAM.",
        ]
    )
    return "\n".join(lines)


def view_sample_choices(store: GeoExpressionStore, scope: str, gse: str | None) -> list[str]:
    frame, _ = store.selection(scope, gse)
    return frame["GSM"].dropna().astype(str).drop_duplicates().tolist()

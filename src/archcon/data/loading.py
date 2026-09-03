"""Load and align expression, clinical, and eGFR data.

The public library uses one unambiguous internal convention:

    rows = samples
    columns = probes/features

The original thesis code used both orientations at different stages. The loader
therefore accepts either orientation and converts it immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ExpressionOrientation = Literal["auto", "samples_rows", "samples_columns"]


@dataclass
class ExpressionData:
    """Expression matrix plus information about how it was interpreted."""

    frame: pd.DataFrame
    source_path: Path
    source_orientation: str
    duplicate_sample_ids_removed: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_probes(self) -> int:
        return int(self.frame.shape[1])


@dataclass
class DataWorkspace:
    """In-memory dataset used by the local web explorer."""

    expression: ExpressionData | None = None
    clinical: pd.DataFrame | None = None
    egfr: pd.DataFrame | None = None
    clinical_id_column: str | None = None
    egfr_id_column: str | None = None
    messages: list[str] = field(default_factory=list)


def normalize_sample_id(value: object) -> str:
    """Canonicalize sample identifiers used by the thesis source code."""
    text = Path(str(value).strip()).name
    replacements = (
        "_(PrimeView).CEL.gz",
        "_(PrimeView).CEL",
        "_(PrimeView)",
        ".CEL.gz",
        ".CEL",
        ".cel.gz",
        ".cel",
    )
    for suffix in replacements:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def _read_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=0)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, index_col=0)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                'Parquet support is optional. Install it with: pip install "archcon[parquet]"'
            ) from exc
    raise ValueError(f"Unsupported expression-matrix format: {suffix or '<none>'}")


def _infer_orientation(frame: pd.DataFrame) -> Literal["samples_rows", "samples_columns"]:
    """Infer orientation using the smaller-axis-as-samples gene-expression heuristic."""
    rows, columns = frame.shape

    column_labels = [normalize_sample_id(x) for x in frame.columns[: min(columns, 30)]]
    index_labels = [normalize_sample_id(x) for x in frame.index[: min(rows, 30)]]

    def sample_like(labels: list[str]) -> int:
        score = 0
        for label in labels:
            low = label.lower()
            if ".cel" in low or "primeview" in low:
                score += 3
            if "_" in label and len(label) < 80:
                score += 1
        return score

    column_score = sample_like(column_labels)
    index_score = sample_like(index_labels)
    if column_score > index_score:
        return "samples_columns"
    if index_score > column_score:
        return "samples_rows"

    # In the thesis data, samples are far fewer than probes. This also works for
    # the 13,940 x 42,917 GEO setting when samples are rows.
    return "samples_rows" if rows <= columns else "samples_columns"


def load_expression_matrix(
    file_path: str | Path,
    *,
    orientation: ExpressionOrientation = "auto",
) -> ExpressionData:
    """Load an expression matrix and convert it to samples x probes."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expression matrix not found: {path}")

    frame = _read_matrix(path)
    if frame.empty:
        raise ValueError("Expression matrix is empty.")

    inferred = _infer_orientation(frame) if orientation == "auto" else orientation
    if inferred == "samples_columns":
        frame = frame.T

    frame.index = [normalize_sample_id(x) for x in frame.index]
    duplicate_count = int(frame.index.duplicated(keep="first").sum())
    if duplicate_count:
        frame = frame.loc[~frame.index.duplicated(keep="first")]

    try:
        frame = frame.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Expression matrix must contain numeric values after the row/column labels."
        ) from exc

    return ExpressionData(
        frame=frame,
        source_path=path,
        source_orientation=inferred,
        duplicate_sample_ids_removed=duplicate_count,
    )


def load_table(file_path: str | Path) -> pd.DataFrame:
    """Load a clinical/eGFR table from CSV, TSV, Excel, or Parquet."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError(
                'Parquet support is optional. Install it with: pip install "archcon[parquet]"'
            ) from exc
    raise ValueError(f"Unsupported table format: {suffix or '<none>'}")


def _choose_id_column(frame: pd.DataFrame, preferred: tuple[str, ...]) -> str:
    by_lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in preferred:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        "Could not identify the sample-ID column. Expected one of: " + ", ".join(preferred)
    )


def _canonicalize_table_ids(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    result = frame.copy()
    result[id_column] = result[id_column].map(normalize_sample_id)
    return result


def align_workspace(workspace: DataWorkspace) -> dict[str, object]:
    """Align IDs conceptually and return a summary without discarding rows."""
    if workspace.expression is None:
        return {"status": "No expression matrix loaded."}

    expression_ids = set(workspace.expression.frame.index.astype(str))
    summary: dict[str, object] = {
        "samples": workspace.expression.n_samples,
        "probes": workspace.expression.n_probes,
        "expression_ids": expression_ids,
    }

    if workspace.clinical is not None:
        clinical_id = _choose_id_column(
            workspace.clinical,
            ("Sample_ID", "sample_id", "patient", "Patient"),
        )
        workspace.clinical = _canonicalize_table_ids(workspace.clinical, clinical_id)
        workspace.clinical_id_column = clinical_id
        clinical_ids = set(workspace.clinical[clinical_id].dropna().astype(str))
        summary.update(
            clinical_rows=len(workspace.clinical),
            clinical_ids=clinical_ids,
            clinical_matched=len(expression_ids & clinical_ids),
            clinical_unmatched_expression=sorted(expression_ids - clinical_ids),
            clinical_unmatched_table=sorted(clinical_ids - expression_ids),
        )

    if workspace.egfr is not None:
        egfr_id = _choose_id_column(
            workspace.egfr,
            ("patient", "Patient", "Sample_ID", "sample_id"),
        )
        workspace.egfr = _canonicalize_table_ids(workspace.egfr, egfr_id)
        workspace.egfr_id_column = egfr_id
        egfr_ids = set(workspace.egfr[egfr_id].dropna().astype(str))
        known_egfr_columns = [
            column
            for column in ("egfr_7d", "egfr_3m", "egfr_6m", "egfr_12m")
            if column in workspace.egfr.columns
        ]
        if known_egfr_columns:
            valid_rows = workspace.egfr.dropna(subset=known_egfr_columns, how="all")
            valid_egfr_ids = set(valid_rows[egfr_id].dropna().astype(str))
        else:
            valid_egfr_ids = egfr_ids
        summary.update(
            egfr_rows=len(workspace.egfr),
            egfr_ids=egfr_ids,
            egfr_valid_ids=valid_egfr_ids,
            egfr_matched=len(expression_ids & egfr_ids),
            egfr_valid_matched=len(expression_ids & valid_egfr_ids),
            egfr_unmatched_expression=sorted(expression_ids - egfr_ids),
            egfr_unmatched_table=sorted(egfr_ids - expression_ids),
        )

    return summary

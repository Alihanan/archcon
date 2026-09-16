"""Discovery of the optional local ``data/`` directory used by ArchCon.

Large or private thesis datasets are intentionally kept outside the installed
Python package.  This module only discovers conventional local paths; it never
copies or bundles the data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ARCHCON_DATA_DIR_ENV = "ARCHCON_DATA_DIR"


@dataclass(frozen=True)
class ProjectDataLayout:
    """Resolved conventional paths inside one ArchCon data directory."""

    root: Path
    expression_matrix: Path
    clinical_table: Path
    egfr_table: Path
    common_probes: Path
    gene_annotations: Path
    sample_metadata: Path
    geo_parquet: Path
    geo_rma_store: Path
    geo_stadniuk_store: Path
    ikem_store: Path
    ikem_cel_store: Path
    ikem_legacy_store: Path
    cel_directory: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectDataLayout":
        path = Path(root).expanduser().resolve()
        ikem_cel_store = path / "IKEM_CEL_NUMPY_STORE"
        ikem_legacy_store = path / "IKEM_NUMPY_STORE"
        return cls(
            root=path,
            expression_matrix=path / "expression_matrix.csv",
            clinical_table=path / "Klasifikator_20_3_24_v2.xlsx",
            egfr_table=path / "egfr_data.xlsx",
            common_probes=path / "common_probes.pkl",
            gene_annotations=path / "gene_annotations.csv",
            sample_metadata=path / "sample_metadata.csv",
            geo_parquet=path / "geo_expr_normalized_to_ikem.parquet",
            geo_rma_store=path / "GEO_NUMPY_STORE",
            geo_stadniuk_store=path / "GEO_STADNIUK_STORE",
            ikem_store=(
                ikem_cel_store if ikem_cel_store.is_dir() else ikem_legacy_store
            ),
            ikem_cel_store=ikem_cel_store,
            ikem_legacy_store=ikem_legacy_store,
            cel_directory=path / "CEL",
        )


def _looks_like_archcon_data_directory(path: Path) -> bool:
    """Return whether ``path`` contains at least one canonical ArchCon data resource."""
    markers = (
        "GEO_NUMPY_STORE",
        "IKEM_NUMPY_STORE",
        "GEO_STADNIUK_STORE",
        "IKEM_CEL_NUMPY_STORE",
        "splits",
        "common_probes.pkl",
        "gene_annotations.csv",
        "sample_metadata.csv",
        "egfr_data.xlsx",
        "Klasifikator_20_3_24_v2.xlsx",
    )
    return path.is_dir() and any((path / marker).exists() for marker in markers)


def _source_tree_data_directory() -> Path | None:
    """Find ``data/`` beside a source-tree ``pyproject.toml`` when available.

    Editable installs keep ``__file__`` inside the checked-out project, which lets
    ArchCon find the project's large local data even when the CLI is launched from
    another working directory.  A normal wheel install usually has no enclosing
    ``pyproject.toml`` and therefore falls back to the launch directory or the
    explicit ``ARCHCON_DATA_DIR`` override.
    """
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "pyproject.toml").is_file():
            return (parent / "data").resolve()
    return None


def resolve_data_directory(data_dir: str | Path | None = None) -> Path:
    """Resolve the local ArchCon data directory without requiring it to exist.

    Resolution order is: an explicit value, ``ARCHCON_DATA_DIR``, a recognizable
    ``./data`` under the launch directory, then ``data/`` beside the editable
    ArchCon source tree.  If nothing can be detected, ``./data`` remains the
    conventional fallback.
    """
    if data_dir is not None and str(data_dir).strip():
        return Path(data_dir).expanduser().resolve()

    environment_value = os.environ.get(ARCHCON_DATA_DIR_ENV, "").strip()
    if environment_value:
        return Path(environment_value).expanduser().resolve()

    launch_data = (Path.cwd() / "data").resolve()
    if _looks_like_archcon_data_directory(launch_data):
        return launch_data

    source_data = _source_tree_data_directory()
    if source_data is not None and _looks_like_archcon_data_directory(source_data):
        return source_data

    if launch_data.is_dir():
        return launch_data
    if source_data is not None and source_data.is_dir():
        return source_data
    return launch_data


def project_data_layout(data_dir: str | Path | None = None) -> ProjectDataLayout:
    """Return the conventional thesis-data paths for a resolved data directory."""
    return ProjectDataLayout.from_root(resolve_data_directory(data_dir))


def _human_size(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "not present"

    size = float(path.stat().st_size)
    units = ("B", "KB", "MB", "GB", "TB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if size < 1024 or candidate == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def data_directory_status(layout: ProjectDataLayout) -> str:
    """Build a compact Markdown status report for the conventional data files."""
    items = (
        ("expression_matrix.csv", layout.expression_matrix, "file"),
        ("Klasifikator_20_3_24_v2.xlsx", layout.clinical_table, "file"),
        ("egfr_data.xlsx", layout.egfr_table, "file"),
        ("common_probes.pkl", layout.common_probes, "file"),
        ("gene_annotations.csv", layout.gene_annotations, "file"),
        ("sample_metadata.csv", layout.sample_metadata, "file"),
        ("geo_expr_normalized_to_ikem.parquet", layout.geo_parquet, "file"),
        ("GEO_NUMPY_STORE/", layout.geo_rma_store, "directory"),
        ("GEO_STADNIUK_STORE/", layout.geo_stadniuk_store, "directory"),
        ("IKEM_CEL_NUMPY_STORE/", layout.ikem_cel_store, "directory"),
        ("IKEM_NUMPY_STORE/ (historical)", layout.ikem_legacy_store, "directory"),
        ("CEL/", layout.cel_directory, "directory"),
    )

    lines = [f"**Data directory:** `{layout.root}`", "", "| Resource | Status |", "|---|---|"]
    for label, path, kind in items:
        exists = path.is_dir() if kind == "directory" else path.is_file()
        marker = "✅" if exists else "○"
        detail = "found" if kind == "directory" and exists else _human_size(path)
        lines.append(f"| `{label}` | {marker} {detail} |")

    if not layout.root.is_dir():
        lines.extend(
            [
                "",
                "The directory does not exist yet. Create it beside `pyproject.toml`, or set "
                f"`{ARCHCON_DATA_DIR_ENV}` to another location.",
            ]
        )
    return "\n".join(lines)


def detected_default_paths(layout: ProjectDataLayout) -> dict[str, str]:
    """Return UI defaults only for conventional resources that actually exist."""
    return {
        "expression": str(layout.expression_matrix) if layout.expression_matrix.is_file() else "",
        "clinical": str(layout.clinical_table) if layout.clinical_table.is_file() else "",
        "egfr": str(layout.egfr_table) if layout.egfr_table.is_file() else "",
        "reference": str(layout.expression_matrix) if layout.expression_matrix.is_file() else "",
        "geo": str(layout.geo_parquet) if layout.geo_parquet.is_file() else "",
        "geo_rma": str(layout.geo_rma_store) if layout.geo_rma_store.is_dir() else "",
        "geo_stadniuk": str(layout.geo_stadniuk_store) if layout.geo_stadniuk_store.is_dir() else "",
        "ikem": str(layout.ikem_store) if layout.ikem_store.is_dir() else "",
        "cel": str(layout.cel_directory) if layout.cel_directory.is_dir() else "",
    }

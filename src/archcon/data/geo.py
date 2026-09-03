"""Memory-efficient inspection of the large GEO Parquet used for pretraining."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def inspect_geo_parquet(file_path: str | Path, max_samples: int = 12):
    """Read Parquet metadata and only a small deterministic subset of sample columns."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GEO Parquet file not found: {path}")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'Large Parquet inspection requires PyArrow. Install: pip install "archcon[parquet]"'
        ) from exc

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    sample_columns = [column for column in columns if column != "ID_REF"]
    if not sample_columns:
        raise ValueError("No sample columns found. Expected an ID_REF column plus GEO samples.")

    positions = np.unique(
        np.linspace(0, len(sample_columns) - 1, min(max_samples, len(sample_columns)), dtype=int)
    )
    chosen = [sample_columns[index] for index in positions]
    table = pq.read_table(path, columns=chosen)
    frame = table.to_pandas()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    finite = frame.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = np.quantile(finite, [0.005, 0.995])
        bins = np.linspace(lo, hi, 70)
        for column in chosen:
            values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
            hist, edges = np.histogram(values, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, alpha=0.65, linewidth=1)
    ax.set_title(f"GEO Parquet expression — {len(chosen)} sampled columns")
    ax.set_xlabel("Expression value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    metadata = {
        "path": str(path),
        "rows": parquet.metadata.num_rows,
        "columns": parquet.metadata.num_columns,
        "row_groups": parquet.metadata.num_row_groups,
        "sample_columns": len(sample_columns),
        "sampled_columns": chosen,
    }
    return metadata, fig

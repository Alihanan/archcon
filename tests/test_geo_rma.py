from pathlib import Path

import numpy as np
import pandas as pd

from archcon.data.geo_rma import (
    METHOD_GLOBAL_RMA,
    METHOD_PER_GSE_RMA,
    SCOPE_AGGREGATE,
    SCOPE_SERIES,
    load_geo_expression_store,
)


def _write_store(root: Path) -> Path:
    root.mkdir()
    probes = [f"probe_{i}" for i in range(5)]

    sample_index = pd.DataFrame(
        {
            "GSM": ["GSM1", "GSM2", "GSM3", "GSM4"],
            "GSE": ["GSE10", "GSE10", "GSE20", "GSE20"],
            "source_GSE": ["GSE10", "GSE10", "GSE20", "GSE20"],
            "global_row_python": [0, 1, 2, 3],
            "is_multi_series_gsm": [False, True, True, False],
            "selection_reason": ["only_occurrence", "preferred", "preferred", "only_occurrence"],
        }
    )
    occurrences = pd.DataFrame(
        {
            "GSM": ["GSM1", "GSM2", "GSM2", "GSM3", "GSM3", "GSM4"],
            "source_GSE": ["GSE10", "GSE10", "GSE11", "GSE20", "GSE21", "GSE20"],
            "canonical_GSE": ["GSE10", "GSE10", "GSE10", "GSE20", "GSE20", "GSE20"],
            "occurrence_row_python": [0, 1, 2, 3, 4, 5],
            "raw_rds": ["raw10", "raw10", "raw11", "raw20", "raw21", "raw20"],
            "rma_rds": ["rma10", "rma10", "rma11", "rma20", "rma21", "rma20"],
        }
    )
    source_index = pd.DataFrame(
        {
            "source_GSE": ["GSE10", "GSE11", "GSE20", "GSE21"],
            "n_samples": [2, 1, 2, 1],
            "start_row_python": [0, 2, 3, 4],
            "stop_row_python": [2, 3, 5, 5],
        }
    )
    gse_index = pd.DataFrame(
        {
            "GSE": ["GSE10", "GSE20"],
            "n_samples": [2, 2],
            "start_row_python": [0, 2],
            "stop_row_python": [2, 4],
        }
    )
    probe_index = pd.DataFrame({"probe_index_python": range(5), "probe_id": probes})
    cel_manifest = pd.DataFrame(
        {
            "GSM": ["GSM1", "GSM2", "GSM3", "GSM4"],
            "GSE": ["GSE10", "GSE10", "GSE20", "GSE20"],
            "cel_source_GSE": ["GSE10", "GSE10", "GSE20", "GSE20"],
        }
    )

    sample_index.to_csv(root / "sample_index.csv", index=False)
    occurrences.to_csv(root / "source_sample_occurrences.csv", index=False)
    source_index.to_csv(root / "source_gse_occurrence_index.csv", index=False)
    gse_index.to_csv(root / "gse_index.csv", index=False)
    probe_index.to_csv(root / "probe_index.csv", index=False)
    cel_manifest.to_csv(root / "cel_manifest.csv", index=False)

    canonical = np.arange(20, dtype=np.float32).reshape(4, 5)
    occurrences_matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
    np.save(root / "raw_original.npy", canonical + 100)
    np.save(root / "rma_per_gse.npy", canonical + 10)
    np.save(root / "rma_global.npy", canonical + 20)
    np.save(root / "per_dataset_raw_original.npy", occurrences_matrix + 100)
    np.save(root / "per_dataset_rma_per_gse.npy", occurrences_matrix + 10)
    return root


def test_store_opens_and_uses_unique_aggregate(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))
    matrix, rows, metadata = store.matrix_rows(METHOD_PER_GSE_RMA, SCOPE_AGGREGATE, None)

    assert matrix.shape == (4, 5)
    assert rows.tolist() == [0, 1, 2, 3]
    assert metadata["GSM"].tolist() == ["GSM1", "GSM2", "GSM3", "GSM4"]


def test_source_gse_view_preserves_membership_and_global_maps_by_gsm(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))

    per_gse_matrix, per_gse_rows, metadata = store.matrix_rows(
        METHOD_PER_GSE_RMA, SCOPE_SERIES, "GSE11"
    )
    assert metadata["GSM"].tolist() == ["GSM2"]
    assert per_gse_rows.tolist() == [2]
    assert float(per_gse_matrix[2, 0]) == 20.0

    global_matrix, global_rows, metadata = store.matrix_rows(
        METHOD_GLOBAL_RMA, SCOPE_SERIES, "GSE11"
    )
    assert metadata["GSM"].tolist() == ["GSM2"]
    assert global_rows.tolist() == [1]
    assert float(global_matrix[1, 0]) == 25.0


def test_dataset_catalog_marks_shared_series_memberships(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))
    catalog = store.dataset_catalog().set_index("GSE")

    assert catalog.loc["GSE10", "sample_memberships"] == 2
    assert catalog.loc["GSE10", "shared_with_other_series"] == 1
    assert catalog.loc["GSE11", "shared_with_other_series"] == 1


def test_browser_catalog_is_compact_and_clickable(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))
    catalog = store.browser_catalog()

    assert catalog.columns.tolist() == [
        "GSE",
        "unique_GSMs",
        "sample_memberships",
        "shared_with_other_series",
        "overlap",
        "GEO",
        "RAW archive",
    ]
    assert catalog.iloc[0]["GSE"] == "GSE10"
    assert "shared GSMs" in catalog["overlap"].tolist()


def test_dataset_heading_distinguishes_aggregate_and_series(tmp_path: Path) -> None:
    from archcon.data.geo_rma import dataset_heading

    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))

    aggregate = dataset_heading(store, SCOPE_AGGREGATE, None)
    series = dataset_heading(store, SCOPE_SERIES, "GSE10")

    assert "All unique GEO samples" in aggregate
    assert "4 unique GSMs" in aggregate
    assert "GSE10" in series
    assert "2 unique GSMs" in series

from pathlib import Path

import pandas as pd

from archcon import normalize_sample_id
from archcon.data.loading import DataWorkspace, align_workspace, load_expression_matrix


def test_normalize_sample_id() -> None:
    assert normalize_sample_id("103_1_(PrimeView).CEL") == "103_1"
    assert normalize_sample_id("/tmp/103_2.CEL") == "103_2"


def test_load_expression_transposes_probe_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "101_1_(PrimeView).CEL": [4.0, 5.0, 6.0],
            "102_1_(PrimeView).CEL": [7.0, 8.0, 9.0],
        },
        index=["probe_a", "probe_b", "probe_c"],
    )
    path = tmp_path / "expression.csv"
    frame.to_csv(path)

    loaded = load_expression_matrix(path, orientation="samples_columns")

    assert loaded.frame.shape == (2, 3)
    assert list(loaded.frame.index) == ["101_1", "102_1"]
    assert list(loaded.frame.columns) == ["probe_a", "probe_b", "probe_c"]


def test_align_workspace_counts_matches(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"probe_a": [4.0, 5.0], "probe_b": [6.0, 7.0]},
        index=["101_1", "102_1"],
    )
    path = tmp_path / "expression.csv"
    frame.to_csv(path)

    workspace = DataWorkspace(expression=load_expression_matrix(path, orientation="samples_rows"))
    workspace.clinical = pd.DataFrame({"Sample_ID": ["101_1", "999_1"], "age": [45, 60]})
    workspace.egfr = pd.DataFrame(
        {
            "patient": ["101_1", "102_1"],
            "egfr_7d": [50.0, None],
            "egfr_3m": [55.0, None],
        }
    )

    summary = align_workspace(workspace)

    assert summary["clinical_matched"] == 1
    assert summary["egfr_matched"] == 2
    assert summary["egfr_valid_matched"] == 1

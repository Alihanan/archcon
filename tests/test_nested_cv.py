from pathlib import Path

import pandas as pd
import pytest

from archcon.data.nested_cv import (
    NestedSplitConfig,
    build_nested_split_manifest,
    create_or_load_nested_split_manifest,
    nested_folds_from_manifest,
)


def _patients() -> pd.DataFrame:
    rows = []
    for donor_index in range(24):
        donor = f"D{donor_index:03d}"
        for biopsy_index in range(1 + int(donor_index % 4 == 0)):
            rows.append(
                {
                    "patient": f"{donor}_P{biopsy_index}",
                    "donor": donor,
                    "KDRI_8": 0.5 + donor_index / 20.0,
                }
            )
    return pd.DataFrame(rows)


def test_nested_manifest_keeps_every_donor_in_one_role() -> None:
    patients = _patients()
    config = NestedSplitConfig(
        outer_splits=4,
        outer_repeats=2,
        inner_splits=3,
        seed=17,
    )
    manifest = build_nested_split_manifest(patients, config)
    assert list(manifest.columns) == [
        "outer_repeat",
        "outer_fold",
        "patient",
        "donor",
        "outer_role",
        "inner_role",
    ]
    assert len(manifest) == len(patients) * 8

    folds = nested_folds_from_manifest(patients, manifest)
    assert len(folds) == 8
    for fold in folds:
        outer_train = set(patients.iloc[fold.outer_train]["donor"])
        outer_test = set(patients.iloc[fold.outer_test]["donor"])
        inner_train = set(patients.iloc[fold.inner_train]["donor"])
        inner_validation = set(patients.iloc[fold.inner_validation]["donor"])
        assert outer_train.isdisjoint(outer_test)
        assert inner_train.isdisjoint(inner_validation)
        assert inner_train | inner_validation == outer_train


def test_persisted_nested_manifest_is_immutable(tmp_path: Path) -> None:
    patients = _patients()
    config = NestedSplitConfig(outer_splits=4, outer_repeats=1, inner_splits=3)
    path = tmp_path / "nested_split_manifest.csv"
    first = create_or_load_nested_split_manifest(patients, path, config)
    second = create_or_load_nested_split_manifest(patients, path, config)
    pd.testing.assert_frame_equal(first, second)

    changed = patients.copy()
    changed.loc[0, "donor"] = "CHANGED"
    with pytest.raises(RuntimeError, match="different cohort or CV contract"):
        create_or_load_nested_split_manifest(changed, path, config)


def test_manifest_loader_rejects_manual_donor_leakage() -> None:
    patients = _patients()
    config = NestedSplitConfig(outer_splits=4, outer_repeats=1, inner_splits=3)
    manifest = build_nested_split_manifest(patients, config)
    block = manifest.loc[
        manifest["outer_repeat"].eq(0) & manifest["outer_fold"].eq(0)
    ]
    repeated_donor = block["donor"].value_counts().loc[lambda values: values > 1].index[0]
    rows = block.index[block["donor"].eq(repeated_donor)].tolist()
    assert len(rows) > 1
    manifest.loc[rows[0], ["outer_role", "inner_role"]] = ["test", "not_applicable"]
    manifest.loc[rows[1], ["outer_role", "inner_role"]] = ["train", "train"]
    with pytest.raises(ValueError, match="Donor leakage"):
        nested_folds_from_manifest(patients, manifest)

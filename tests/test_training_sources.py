from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archcon.data.defaults import project_data_layout
from archcon.data.geo_rma import METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from archcon.data.training_sources import (
    METHOD_STADNIUK_RESCALED,
    assert_probe_alignment,
    create_shared_preprocessing_split,
    load_ikem_source,
    load_prepared_pretraining_source,
    load_prepared_split_rows,
    load_training_source,
    prepare_pretraining_assets,
    split_rows_for_source,
    load_pretraining_source,
)


def _write_geo_rma_store(root: Path, samples: list[str], probes: list[str]) -> None:
    root.mkdir(parents=True)
    n = len(samples)
    p = len(probes)
    gses = [f"GSE{i+1}" for i in range(n)]
    sample_index = pd.DataFrame(
        {
            "GSM": samples,
            "GSE": gses,
            "source_GSE": gses,
            "global_row_python": range(n),
        }
    )
    sample_index.to_csv(root / "sample_index.csv", index=False)
    pd.DataFrame(
        {
            "GSM": samples,
            "source_GSE": gses,
            "canonical_GSE": gses,
            "occurrence_row_python": range(n),
        }
    ).to_csv(root / "source_sample_occurrences.csv", index=False)
    pd.DataFrame(
        {"GSE": gses, "n_samples": [1] * n, "start_row_python": list(range(n)), "stop_row_python": list(range(1, n + 1))}
    ).to_csv(root / "gse_index.csv", index=False)
    pd.DataFrame(
        {"source_GSE": gses, "n_samples": [1] * n, "start_row_python": list(range(n)), "stop_row_python": list(range(1, n + 1))}
    ).to_csv(root / "source_gse_occurrence_index.csv", index=False)
    pd.DataFrame({"probe_index_python": range(p), "probe_id": probes}).to_csv(
        root / "probe_index.csv", index=False
    )
    pd.DataFrame({"GSM": samples, "GSE": gses, "cel_source_GSE": gses}).to_csv(
        root / "cel_manifest.csv", index=False
    )
    matrix = np.arange(n * p, dtype=np.float32).reshape(n, p)
    np.save(root / "raw_original.npy", matrix)
    np.save(root / "rma_per_gse.npy", matrix + 10)
    np.save(root / "rma_global.npy", matrix + 20)
    np.save(root / "per_dataset_raw_original.npy", matrix)
    np.save(root / "per_dataset_rma_per_gse.npy", matrix + 10)


def _write_simple_store(root: Path, samples: list[str], probes: list[str], offset: float) -> None:
    root.mkdir(parents=True)
    matrix = np.arange(len(samples) * len(probes), dtype=np.float32).reshape(len(samples), len(probes))
    np.save(root / "expression.npy", matrix + offset)
    pd.DataFrame({"GSM": samples, "row_index_python": range(len(samples))}).to_csv(
        root / "sample_index.csv", index=False
    )
    pd.DataFrame({"probe_id": probes}).to_csv(root / "probe_index.csv", index=False)


def _layout(tmp_path: Path):
    samples = [f"GSM{i}" for i in range(8)]
    probes = [f"probe_{i}" for i in range(5)]
    _write_geo_rma_store(tmp_path / "GEO_NUMPY_STORE", samples, probes)
    # Deliberately reverse row order: the fixed split must map by GSM, not row number.
    # Deliberately reverse both row and probe order: prepared sweeps must freeze
    # mappings by identity rather than requiring native stores to share order.
    _write_simple_store(
        tmp_path / "GEO_STADNIUK_STORE",
        list(reversed(samples)),
        list(reversed(probes)),
        100,
    )
    _write_simple_store(tmp_path / "IKEM_NUMPY_STORE", ["IKEM1", "IKEM2", "IKEM3", "IKEM4"], probes, 200)
    pd.DataFrame(
        {
            "patient": ["IKEM1", "IKEM2", "IKEM3", "IKEM4"],
            "egfr_7d": [50.0, None, None, None],
            "egfr_3m": [55.0, None, None, None],
        }
    ).to_excel(tmp_path / "egfr_data.xlsx", index=False)
    return project_data_layout(tmp_path), samples, probes


def test_shared_split_is_sampled_once_and_maps_by_gsm(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    first = create_shared_preprocessing_split(layout, seed=42, train_fraction=0.75)
    second = create_shared_preprocessing_split(layout, seed=42, train_fraction=0.75)
    assert first[["GSM", "split"]].equals(second[["GSM", "split"]])
    assert set(first.loc[first["dataset_role"] == "unsupervised data · GEO", "GSM"]) == set(samples)
    assert (first["dataset_role"] == "supervised dataset · no eGFR").sum() == 3
    assert (first["split"] == "train").sum() >= 5
    assert (first["split"] == "validation").sum() >= 1
    assert (first["split"] == "test").sum() >= 1

    rma = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    stadniuk = load_pretraining_source(layout, METHOD_STADNIUK_RESCALED)
    rma_train, rma_val, rma_test = split_rows_for_source(first, rma, include_test=True)
    stad_train, stad_val, stad_test = split_rows_for_source(first, stadniuk, include_test=True)
    rma_ids = set(rma.sample_index.iloc[rma_train]["sample_key"])
    stad_ids = set(stadniuk.sample_index.iloc[stad_train]["sample_key"])
    assert rma_ids == stad_ids
    assert len(rma_val) == len(stad_val)
    assert len(rma_test) == len(stad_test)


def test_all_three_training_preprocessings_and_ikem_are_memory_mapped(tmp_path: Path) -> None:
    layout, _, _ = _layout(tmp_path)
    for method in (METHOD_STADNIUK_RESCALED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA):
        source = load_training_source(layout, method)
        assert isinstance(source.matrix, np.memmap)
        assert source.matrix.shape == (8, 5)
    ikem = load_ikem_source(layout)
    assert ikem is not None
    assert isinstance(ikem.matrix, np.memmap)
    assert ikem.matrix.shape == (4, 5)


def test_ikem_probe_mismatch_is_rejected(tmp_path: Path) -> None:
    layout, _, probes = _layout(tmp_path)
    pd.DataFrame({"probe_id": list(reversed(probes))}).to_csv(
        layout.ikem_store / "probe_index.csv", index=False
    )
    training = load_training_source(layout, METHOD_GLOBAL_RMA)
    ikem = load_ikem_source(layout)
    assert ikem is not None
    with pytest.raises(ValueError, match="Probe order differs"):
        assert_probe_alignment(training, ikem)


def test_comparison_pretraining_offers_all_three_preprocessings() -> None:
    from archcon.data.training_sources import TRAINING_PREPROCESSING_OPTIONS

    assert TRAINING_PREPROCESSING_OPTIONS == [
        METHOD_STADNIUK_RESCALED,
        METHOD_PER_GSE_RMA,
        METHOD_GLOBAL_RMA,
    ]


def test_pretraining_source_appends_only_supervised_samples_without_egfr(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    source = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    assert source.matrix.shape == (len(samples) + 3, 5)
    roles = source.sample_index["dataset_role"].value_counts().to_dict()
    assert roles["unsupervised data · GEO"] == len(samples)
    assert roles["supervised dataset · no eGFR"] == 3
    assert "SUPERVISED:IKEM1" not in set(source.sample_index["sample_key"])
    assert {"SUPERVISED:IKEM2", "SUPERVISED:IKEM3", "SUPERVISED:IKEM4"}.issubset(
        set(source.sample_index["sample_key"])
    )


def test_shared_split_adds_supervised_no_egfr_rows_and_maps_to_combined_source(tmp_path: Path) -> None:
    layout, samples, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    geo = split.loc[split["dataset_role"] == "unsupervised data · GEO"]
    supervised = split.loc[split["dataset_role"] == "supervised dataset · no eGFR"]
    assert set(geo["GSM"].dropna()) == set(samples)
    assert len(supervised) == 3
    assert set(supervised["split"]) == {"train", "validation", "test"}
    source = load_pretraining_source(layout, METHOD_PER_GSE_RMA)
    train, validation, test = split_rows_for_source(split, source, include_test=True)
    assert len(train) + len(validation) + len(test) == len(samples) + 3



def test_prepared_assets_align_extra_supervised_probes_and_freeze_rows(tmp_path: Path) -> None:
    layout, samples, probes = _layout(tmp_path)
    supervised_samples = ["IKEM1", "IKEM2", "IKEM3", "IKEM4"]
    extra_probes = ["unused_extra_probe", *probes]
    values = np.arange(
        len(supervised_samples) * len(extra_probes), dtype=np.float32
    ).reshape(len(supervised_samples), len(extra_probes))
    np.save(layout.ikem_store / "expression.npy", values)
    pd.DataFrame({"probe_id": extra_probes}).to_csv(
        layout.ikem_store / "probe_index.csv", index=False
    )

    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    prepared = prepare_pretraining_assets(
        layout,
        split,
        tmp_path / "prepared",
        methods=(METHOD_STADNIUK_RESCALED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA),
    )

    supplemental = np.load(prepared / "supervised_no_egfr_common.npy", mmap_mode="r")
    assert supplemental.shape == (3, len(probes))
    # IKEM1 has eGFR, so frozen supplemental rows are IKEM2..4; the extra probe is dropped.
    np.testing.assert_array_equal(supplemental, values[1:, 1:])

    train, validation, test = load_prepared_split_rows(prepared)
    all_rows = np.sort(np.concatenate([train, validation, test]))
    np.testing.assert_array_equal(all_rows, np.arange(len(samples) + 3))

    for method in (METHOD_STADNIUK_RESCALED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA):
        source = load_prepared_pretraining_source(layout, method, prepared)
        assert source.matrix.shape == (len(samples) + 3, len(probes))
        assert source.probe_ids == tuple(probes)
        assert source.sample_index["row_index_python"].tolist() == list(range(len(samples) + 3))
        # Fetch the three supplemental logical rows through the prepared read-only stack.
        np.testing.assert_array_equal(
            source.matrix[np.arange(len(samples), len(samples) + 3), :],
            values[1:, 1:],
        )


def test_prepared_assets_freeze_different_geo_probe_orders(tmp_path: Path) -> None:
    layout, samples, probes = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    prepared = prepare_pretraining_assets(
        layout,
        split,
        tmp_path / "prepared_reordered_geo",
        methods=(METHOD_STADNIUK_RESCALED, METHOD_PER_GSE_RMA, METHOD_GLOBAL_RMA),
    )

    # Native Stadniuk columns are reversed, but the prepared logical matrix must
    # expose the canonical per-GSE-RMA probe order.
    source = load_prepared_pretraining_source(
        layout, METHOD_STADNIUK_RESCALED, prepared
    )
    assert source.probe_ids == tuple(probes)
    gsm0_row = int(
        source.sample_index.index[source.sample_index["sample_key"].eq("GEO:GSM0")][0]
    )
    # GSM0 is native Stadniuk row 7; native columns are probe_4..probe_0.
    expected = np.asarray([139, 138, 137, 136, 135], dtype=np.float32)
    np.testing.assert_array_equal(source.matrix[[gsm0_row], :][0], expected)

    metadata = __import__("json").loads((prepared / "prepared.json").read_text())
    assert metadata["format"] == 2
    assert set(metadata["method_geo_columns"]) == {
        METHOD_STADNIUK_RESCALED,
        METHOD_PER_GSE_RMA,
        METHOD_GLOBAL_RMA,
    }
    assert (prepared / "geo_columns_stadniuk.npy").is_file()
    np.testing.assert_array_equal(
        np.load(prepared / "geo_columns_stadniuk.npy"),
        np.asarray([4, 3, 2, 1, 0], dtype=np.int64),
    )


def test_formal_sweep_bundle_embeds_frozen_prepared_plan(tmp_path: Path) -> None:
    import json

    from archcon.batch import build_run_request, generate_sweep_bundle
    from archcon.data.training import TrainingConfig

    layout, _, _ = _layout(tmp_path)
    split = create_shared_preprocessing_split(
        layout, seed=42, train_fraction=0.6, validation_fraction=0.2
    )
    base = build_run_request(
        method=METHOD_PER_GSE_RMA,
        split_seed=42,
        train_fraction=0.6,
        validation_fraction=0.2,
        training=TrainingConfig(hidden_widths=(8,), latent_dim=3, epochs=2),
    )
    bundle = generate_sweep_bundle(
        base_request=base,
        grid_text=json.dumps({"method": [METHOD_PER_GSE_RMA]}),
        destination_root=tmp_path / "sweeps",
        sweep_name="prepared",
        data_dir=str(layout.root),
        data_layout=layout,
        split_frame=split,
    )
    root = Path(bundle["root"])
    assert (root / "prepared" / "prepared.json").is_file()
    assert (root / "prepared" / "train_rows.npy").is_file()
    assert (root / "prepared" / "validation_rows.npy").is_file()
    assert (root / "prepared" / "test_rows.npy").is_file()
    request = json.loads((root / "configs" / "run_0001.json").read_text())
    assert request["prepared_dir"] == "../prepared"
    job = (root / "jobs" / "run_0001.py").read_text()
    assert "load_prepared_pretraining_source" in job
    assert "load_prepared_split_rows" in job
    assert "create_shared_preprocessing_split" not in job
    assert "load_pretraining_source" not in job
    assert "pd.read_csv" not in job

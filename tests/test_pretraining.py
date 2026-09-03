from pathlib import Path

import numpy as np
import pandas as pd

from archcon.data.geo_rma import load_geo_expression_store
from archcon.data.pretraining import (
    ARCH_DENSE,
    ARCH_LOW_RANK,
    ARCH_STADNIUK,
    ARCH_RESNET_LN,
    ARCHITECTURE_PRESET_OPTIONS,
    architecture_preset_from_label,
    autoencoder_architecture_svg,
    autoencoder_parameter_count_advanced,
    create_train_validation_split,
    default_hidden_widths,
    parse_hidden_widths,
    validate_loaded_split,
)


def _write_store(root: Path) -> Path:
    root.mkdir()
    samples = [f"GSM{i}" for i in range(10)]
    probes = [f"probe_{i}" for i in range(4)]
    sample_index = pd.DataFrame(
        {
            "GSM": samples,
            "GSE": [f"GSE{i+1}" for i in range(10)],
            "source_GSE": [f"GSE{i+1}" for i in range(10)],
            "global_row_python": range(10),
        }
    )
    occurrences = pd.DataFrame(
        {
            "GSM": samples,
            "source_GSE": [f"GSE{i+1}" for i in range(10)],
            "canonical_GSE": [f"GSE{i+1}" for i in range(10)],
            "occurrence_row_python": range(10),
        }
    )
    pd.DataFrame(
        {
            "source_GSE": [f"GSE{i+1}" for i in range(10)],
            "n_samples": [1] * 10,
            "start_row_python": list(range(10)),
            "stop_row_python": list(range(1, 11)),
        }
    ).to_csv(root / "source_gse_occurrence_index.csv", index=False)
    pd.DataFrame(
        {
            "GSE": [f"GSE{i+1}" for i in range(10)],
            "n_samples": [1] * 10,
            "start_row_python": list(range(10)),
            "stop_row_python": list(range(1, 11)),
        }
    ).to_csv(root / "gse_index.csv", index=False)
    pd.DataFrame({"probe_index_python": range(4), "probe_id": probes}).to_csv(
        root / "probe_index.csv", index=False
    )
    pd.DataFrame(
        {
            "GSM": samples,
            "GSE": [f"GSE{i+1}" for i in range(10)],
            "cel_source_GSE": [f"GSE{i+1}" for i in range(10)],
        }
    ).to_csv(root / "cel_manifest.csv", index=False)
    sample_index.to_csv(root / "sample_index.csv", index=False)
    occurrences.to_csv(root / "source_sample_occurrences.csv", index=False)

    canonical = np.arange(40, dtype=np.float32).reshape(10, 4)
    np.save(root / "raw_original.npy", canonical + 100)
    np.save(root / "rma_per_gse.npy", canonical + 10)
    np.save(root / "rma_global.npy", canonical + 20)
    np.save(root / "per_dataset_raw_original.npy", canonical + 100)
    np.save(root / "per_dataset_rma_per_gse.npy", canonical + 10)
    return root


def test_random_split_is_reproducible(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))
    first = create_train_validation_split(store, seed=17, train_fraction=0.8)
    second = create_train_validation_split(store, seed=17, train_fraction=0.8)

    assert first["split"].tolist() == second["split"].tolist()
    assert (first["split"] == "train").sum() == 8
    assert (first["split"] == "validation").sum() == 1
    assert (first["split"] == "test").sum() == 1
    assert first["GSM"].nunique() == 10
    assert first.groupby("split_group")["split"].nunique().max() == 1


def test_saved_split_can_be_loaded_and_reordered(tmp_path: Path) -> None:
    store = load_geo_expression_store(_write_store(tmp_path / "GEO_NUMPY_STORE"))
    split = create_train_validation_split(store, seed=4, train_fraction=0.7)
    source = tmp_path / "split.csv"
    split.sample(frac=1.0, random_state=2).to_csv(source, index=False)

    loaded = validate_loaded_split(store, source)
    assert loaded["row_index_python"].tolist() == list(range(10))
    assert set(loaded["split"]) == {"train", "validation", "test"}


def test_default_hidden_widths_match_thesis_default() -> None:
    assert default_hidden_widths(2) == [256, 64]


def test_architecture_presets_and_width_parser() -> None:
    assert len(ARCHITECTURE_PRESET_OPTIONS) == 2
    assert parse_hidden_widths("512 -> 256 → 64") == [512, 256, 64]
    stadniuk = architecture_preset_from_label(ARCHITECTURE_PRESET_OPTIONS[0])
    resnet = architecture_preset_from_label(ARCHITECTURE_PRESET_OPTIONS[1])
    assert stadniuk.family == ARCH_STADNIUK
    assert stadniuk.l2_lambda == 1e-5
    assert resnet.family == ARCH_RESNET_LN
    assert resnet.residual_expansion == 2


def test_stadniuk_preset_keeps_pretraining_dropout() -> None:
    presets = [architecture_preset_from_label(label) for label in ARCHITECTURE_PRESET_OPTIONS]
    assert presets[0].key == "stadniuk"
    assert presets[0].dropout == 0.10
    assert presets[0].stadniuk_batch_norm is False
    assert presets[1].dropout == 0.0


def test_low_rank_edges_reduce_parameter_count() -> None:
    dense = autoencoder_parameter_count_advanced(
        42917, [192, 64], 8, ARCH_DENSE, low_rank_dim=48, residual_blocks=1
    )
    low_rank = autoencoder_parameter_count_advanced(
        42917, [192, 64], 8, ARCH_LOW_RANK, low_rank_dim=48, residual_blocks=1
    )
    assert low_rank < dense


def test_residual_svg_draws_explicit_identity_paths() -> None:
    from archcon.data.pretraining import ARCH_RESIDUAL

    svg = autoencoder_architecture_svg(
        42917,
        [512, 256, 128],
        16,
        "ReLU",
        architecture_family=ARCH_RESIDUAL,
        residual_blocks=2,
    )
    assert "residual skip ×2" not in svg
    assert "identity skip · block 1" in svg
    assert "identity skip · block 2" in svg
    assert "512→512" in svg
    assert "F(x)" in svg
    assert 'class="merge"' in svg


def test_execution_plan_places_residuals_after_width_change() -> None:
    from archcon.data.pretraining import ARCH_RESIDUAL, autoencoder_execution_plan

    plan = autoencoder_execution_plan(
        16,
        [12, 8],
        3,
        ARCH_RESIDUAL,
        residual_blocks=2,
    )
    assert [(stage.in_features, stage.out_features) for stage in plan.encoder_stages] == [
        (16, 12),
        (12, 8),
    ]
    assert [stage.residual_blocks for stage in plan.encoder_stages] == [2, 2]
    assert [stage.out_features for stage in plan.encoder_stages] == [12, 8]
    assert [(stage.in_features, stage.out_features) for stage in plan.decoder_stages] == [
        (3, 8),
        (8, 12),
    ]


def test_related_source_gses_cannot_cross_partitions(tmp_path: Path) -> None:
    root = _write_store(tmp_path / "GEO_NUMPY_STORE")
    occurrences = pd.read_csv(root / "source_sample_occurrences.csv")
    # Expose GSM0 through a second Series; this connects GSE1 and GSE2 into one
    # leakage-safe study component even though their canonical GSE labels differ.
    extra = pd.DataFrame(
        {
            "GSM": ["GSM0"],
            "source_GSE": ["GSE2"],
            "canonical_GSE": ["GSE1"],
            "occurrence_row_python": [len(occurrences)],
        }
    )
    occurrences = pd.concat([occurrences, extra], ignore_index=True)
    occurrences.to_csv(root / "source_sample_occurrences.csv", index=False)
    # The occurrence-level matrices must match the enlarged occurrence metadata.
    for filename in ("per_dataset_raw_original.npy", "per_dataset_rma_per_gse.npy"):
        matrix = np.load(root / filename)
        np.save(root / filename, np.vstack([matrix, matrix[[0]]]).astype(np.float32))
    pd.DataFrame(
        {
            "source_GSE": [f"GSE{i+1}" for i in range(10)],
            "n_samples": [1, 2, *([1] * 8)],
            "start_row_python": list(range(10)),
            "stop_row_python": list(range(1, 11)),
        }
    ).to_csv(root / "source_gse_occurrence_index.csv", index=False)

    store = load_geo_expression_store(root)
    split = create_train_validation_split(store, seed=42, train_fraction=0.8)
    labels = split.set_index("GSM")["split"]
    assert labels["GSM0"] == labels["GSM1"]

from dataclasses import asdict, replace

import numpy as np
import pytest

from archcon.data.contrastive_egfr import (
    CONTRASTIVE_NONE,
    CONTRASTIVE_SOFT_INFONCE,
    contrastive_candidate_grid,
    fit_clinical_geometry,
)


def test_candidate_grid_has_one_control_and_full_soft_cartesian_product() -> None:
    candidates = contrastive_candidate_grid((0.01, 0.05, 0.1), (0.05, 0.1, 0.2))
    assert len(candidates) == 10
    assert candidates[0].mode == CONTRASTIVE_NONE
    assert candidates[0].weight == 0.0
    assert len({candidate.candidate_id for candidate in candidates}) == 10
    assert all(
        candidate.mode == CONTRASTIVE_SOFT_INFONCE for candidate in candidates[1:]
    )


def test_clinical_geometry_is_train_fitted_and_handles_missing_values() -> None:
    train = np.asarray(
        [[1.0, 10.0], [2.0, np.nan], [4.0, 14.0], [8.0, 18.0]],
        dtype=np.float64,
    )
    geometry = fit_clinical_geometry(train)
    transformed = geometry.transform(np.asarray([[np.nan, 12.0], [3.0, np.nan]]))
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()
    assert geometry.bandwidth > 0.0


def test_soft_infonce_prefers_clinically_aligned_neighbours() -> None:
    torch = pytest.importorskip("torch")
    from archcon.data.contrastive_egfr import soft_infonce_loss

    clinical = torch.tensor(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    aligned = clinical.clone().requires_grad_(True)
    shuffled = clinical[[0, 2, 1, 3]].clone().requires_grad_(True)
    projection = torch.nn.Identity()
    aligned_loss = soft_infonce_loss(
        aligned,
        clinical,
        projection,
        temperature=0.1,
        bandwidth=1.0,
    )
    shuffled_loss = soft_infonce_loss(
        shuffled,
        clinical,
        projection,
        temperature=0.1,
        bandwidth=1.0,
    )
    assert aligned_loss.item() < shuffled_loss.item()
    aligned_loss.backward()
    assert torch.isfinite(aligned.grad).all()


def test_none_and_soft_infonce_fine_tuning_execute_from_same_checkpoint(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    from archcon.data.contrastive_egfr import (
        ContrastiveCandidate,
        FineTuneControls,
        fine_tune_checkpoint,
    )
    from archcon.data.downstream import ValidationCheckpoint
    from archcon.data.pretraining import LOSS_MASKED
    from archcon.data.training import TrainingConfig, build_autoencoder

    config = TrainingConfig(
        hidden_widths=(8,),
        latent_dim=3,
        epochs=2,
        batch_size=4,
        device="CPU",
        background_prefetch=False,
    )
    path = tmp_path / "best.pt"
    model = build_autoencoder(6, config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": 6,
            "config": asdict(config),
        },
        path,
    )
    record = ValidationCheckpoint(
        run="run_0001",
        path=path,
        checkpoint=None,
        method="test",
        architecture=config.architecture_family,
        latent_dim=3,
        validation_mse=0.1,
        validation_r2=0.2,
        validation_objective=0.1,
        best_epoch=1,
        config=config,
        input_dim=6,
        molecular_selection_mse=0.1,
    )
    rng = np.random.default_rng(1)
    expression = rng.normal(size=(12, 6)).astype(np.float32)
    clinical = rng.normal(size=(12, 3)).astype(np.float32)
    to_encode = rng.normal(size=(5, 6)).astype(np.float32)
    controls = FineTuneControls(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        projection_dim=4,
    )
    none = fine_tune_checkpoint(
        record,
        expression,
        clinical,
        clinical[:8],
        to_encode,
        ContrastiveCandidate("none"),
        controls,
        device="cpu",
        seed=7,
    )
    soft = fine_tune_checkpoint(
        record,
        expression,
        clinical,
        clinical[:8],
        to_encode,
        ContrastiveCandidate("soft_infonce", 0.05, 0.1),
        controls,
        device="cpu",
        seed=7,
    )
    assert none.z.shape == soft.z.shape == (5, 3)
    assert np.isfinite(none.z).all() and np.isfinite(soft.z).all()
    assert all(value == 0.0 for value in none.contrastive_history)
    assert all(value > 0.0 for value in soft.contrastive_history)

    masked_config = TrainingConfig(
        hidden_widths=(8,),
        latent_dim=3,
        loss_name=LOSS_MASKED,
        mask_fraction=0.25,
        epochs=2,
        batch_size=4,
        device="CPU",
        background_prefetch=False,
    )
    masked_path = tmp_path / "masked_best.pt"
    masked_model = build_autoencoder(6, masked_config)
    torch.save(
        {
            "model_state": masked_model.state_dict(),
            "input_dim": 6,
            "config": asdict(masked_config),
        },
        masked_path,
    )
    masked_record = replace(record, path=masked_path, config=masked_config)
    masked = fine_tune_checkpoint(
        masked_record,
        expression,
        clinical,
        clinical[:8],
        to_encode,
        ContrastiveCandidate("none"),
        controls,
        device="cpu",
        seed=7,
    )
    assert masked.reconstruction_loss_name == LOSS_MASKED
    assert np.isfinite(masked.z).all()

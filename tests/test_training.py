from pathlib import Path

import numpy as np
import pytest

from archcon.data.pretraining import (
    ARCH_DENSE,
    ARCH_GATED_RESIDUAL,
    ARCH_LOW_RANK,
    ARCH_RESIDUAL,
    LOSS_COSINE,
    LOSS_DVIB,
    LOSS_HUBER,
    LOSS_MMD,
    LOSS_MASKED,
    LOSS_MSE,
    LOSS_SOFTMAX,
)
from archcon.data.training import (
    DEVICE_CPU,
    TrainingConfig,
    build_autoencoder,
    inspect_checkpoint,
    train_autoencoder_stream,
)

torch = pytest.importorskip("torch")


def _tiny_config(loss_name: str, epochs: int = 2) -> TrainingConfig:
    return TrainingConfig(
        hidden_widths=(6, 4),
        latent_dim=2,
        activation="ReLU",
        dropout=0.0,
        weight_decay=0.0,
        loss_name=loss_name,
        epochs=epochs,
        batch_size=4,
        learning_rate=1e-3,
        early_stopping_patience=5,
        lr_patience=2,
        seed=7,
        device=DEVICE_CPU,
        ui_update_batches=2,
        latent_pca_every_epochs=1,
        latent_pca_max_samples=8,
    )


def test_streaming_training_saves_valid_checkpoint(tmp_path: Path) -> None:
    matrix = np.random.default_rng(2).normal(size=(20, 8)).astype(np.float32)
    updates = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            _tiny_config(LOSS_MSE),
            tmp_path / "models",
            method="unit-test-rma",
        )
    )

    final = updates[-1]
    assert final.done
    assert not final.running
    assert len(final.val_epoch) == 2
    assert final.latest_checkpoint is not None
    assert final.best_checkpoint is not None
    assert Path(final.latest_checkpoint).is_file()
    assert Path(final.best_checkpoint).is_file()

    metadata = inspect_checkpoint(final.best_checkpoint)
    assert metadata["input_dim"] == 8
    assert metadata["method"] == "unit-test-rma"
    assert metadata["config"]["hidden_widths"] == (6, 4)


@pytest.mark.parametrize(
    "loss_name",
    [LOSS_HUBER, LOSS_COSINE, LOSS_DVIB, LOSS_MMD, LOSS_MASKED],
)
def test_alternative_losses_complete_one_epoch(tmp_path: Path, loss_name: str) -> None:
    matrix = np.random.default_rng(3).normal(size=(12, 6)).astype(np.float32)
    config = TrainingConfig(
        hidden_widths=(4,),
        latent_dim=2,
        dropout=0.0,
        weight_decay=0.0,
        loss_name=loss_name,
        epochs=1,
        batch_size=3,
        device=DEVICE_CPU,
        ui_update_batches=3,
        latent_pca_every_epochs=1,
        early_stopping_patience=2,
        lr_patience=1,
    )
    updates = list(
        train_autoencoder_stream(
            matrix,
            np.arange(9),
            np.arange(9, 12),
            config,
            tmp_path / loss_name.split()[0],
            method="test",
        )
    )
    assert updates[-1].done
    assert np.isfinite(updates[-1].val_loss[-1])
    assert np.isfinite(updates[-1].val_mse[-1])


def test_masked_reconstruction_records_fraction_and_clean_validation_mse(tmp_path: Path) -> None:
    matrix = np.random.default_rng(31).uniform(3.0, 12.0, size=(16, 8)).astype(np.float32)
    config = TrainingConfig(
        hidden_widths=(6,),
        latent_dim=2,
        dropout=0.0,
        weight_decay=0.0,
        loss_name=LOSS_MASKED,
        mask_fraction=0.25,
        epochs=1,
        batch_size=4,
        device=DEVICE_CPU,
        ui_update_batches=4,
        latent_pca_every_epochs=1,
        early_stopping_patience=2,
        lr_patience=1,
        seed=13,
    )
    updates = list(
        train_autoencoder_stream(
            matrix,
            np.arange(12),
            np.arange(12, 16),
            config,
            tmp_path / "masked",
            method="test",
        )
    )
    final = updates[-1]
    assert final.done
    assert np.isfinite(final.val_loss[-1])
    assert np.isfinite(final.val_mse[-1])
    metadata = inspect_checkpoint(final.latest_checkpoint)
    assert metadata["config"]["mask_fraction"] == pytest.approx(0.25)


def test_stop_request_saves_recovery_checkpoint(tmp_path: Path) -> None:
    from archcon.data.training import request_training_stop

    matrix = np.random.default_rng(4).normal(size=(16, 6)).astype(np.float32)
    config = TrainingConfig(
        hidden_widths=(4,),
        latent_dim=2,
        epochs=5,
        batch_size=4,
        device=DEVICE_CPU,
        ui_update_batches=1,
        latent_pca_every_epochs=2,
        early_stopping_patience=10,
        lr_patience=2,
    )
    stream = train_autoencoder_stream(
        matrix,
        np.arange(12),
        np.arange(12, 16),
        config,
        tmp_path / "models",
        method="test",
    )
    first = next(stream)
    assert request_training_stop(first.run_id)
    final = list(stream)[-1]
    assert final.stopped
    assert final.latest_checkpoint is not None
    assert Path(final.latest_checkpoint).is_file()


def test_exact_resume_rejects_different_split(tmp_path: Path) -> None:
    from archcon.data.training import CHECKPOINT_RESUME

    matrix = np.random.default_rng(6).normal(size=(20, 8)).astype(np.float32)
    first = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            _tiny_config(LOSS_MSE, epochs=1),
            tmp_path / "first",
            method="per-gse",
        )
    )[-1]
    resumed_config = _tiny_config(LOSS_MSE, epochs=2)
    stream = train_autoencoder_stream(
        matrix,
        np.arange(15),
        np.arange(15, 20),
        resumed_config,
        tmp_path / "second",
        method="per-gse",
        checkpoint_path=first.latest_checkpoint,
        checkpoint_mode=CHECKPOINT_RESUME,
    )
    with pytest.raises(ValueError, match="same train/validation split"):
        next(stream)


@pytest.mark.parametrize("family", [ARCH_DENSE, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL, ARCH_LOW_RANK])
def test_advanced_architecture_families_forward(family: str) -> None:
    config = TrainingConfig(
        hidden_widths=(12, 8),
        latent_dim=3,
        architecture_family=family,
        low_rank_dim=4,
        residual_blocks=1,
        dropout=0.0,
        epochs=1,
        batch_size=2,
        device=DEVICE_CPU,
    )
    model = build_autoencoder(16, config)
    x = torch.randn(5, 16)
    reconstruction, z, _, _ = model(x)
    assert reconstruction.shape == (5, 16)
    assert z.shape == (5, 3)


def test_legacy_dense_checkpoint_config_remains_compatible(tmp_path: Path) -> None:
    from archcon.data.training import _config_compatible

    config = TrainingConfig(
        hidden_widths=(6, 4),
        latent_dim=2,
        activation="ReLU",
        loss_name=LOSS_MSE,
        architecture_family=ARCH_DENSE,
        low_rank_dim=64,
        residual_blocks=1,
    )
    legacy = {
        "hidden_widths": (6, 4),
        "latent_dim": 2,
        "activation": "ReLU",
        "loss_name": LOSS_MSE,
    }
    assert _config_compatible(legacy, config)


def test_validation_metrics_use_distinct_colors() -> None:
    from archcon.data.training import TrainingUpdate, plot_validation_metrics

    update = TrainingUpdate(
        run_id="colors",
        status="ok",
        running=False,
        done=False,
        val_epoch=[1, 2],
        val_mse=[1.5, 1.4],
        val_r2=[0.70, 0.72],
    )
    fig = plot_validation_metrics(update)
    assert len(fig.axes) == 2
    assert fig.axes[0].lines[0].get_color() != fig.axes[1].lines[0].get_color()


def test_training_history_contains_epoch_means_not_only_running_curve(tmp_path: Path) -> None:
    matrix = np.random.default_rng(10).normal(size=(20, 8)).astype(np.float32)
    final = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            _tiny_config(LOSS_MSE, epochs=2),
            tmp_path / "epoch-history",
            method="test",
        )
    )[-1]
    assert final.train_epoch == [1, 2]
    assert len(final.train_epoch_loss) == 2
    assert np.isfinite(final.train_epoch_loss).all()


def test_mmd_validation_exposes_latent_regularizer(tmp_path: Path) -> None:
    matrix = np.random.default_rng(11).normal(size=(16, 7)).astype(np.float32)
    config = TrainingConfig(
        hidden_widths=(5,),
        latent_dim=3,
        dropout=0.0,
        loss_name=LOSS_MMD,
        mmd_weight=0.1,
        epochs=1,
        batch_size=4,
        device=DEVICE_CPU,
        early_stopping_patience=2,
        lr_patience=1,
    )
    final = list(
        train_autoencoder_stream(
            matrix,
            np.arange(12),
            np.arange(12, 16),
            config,
            tmp_path / "mmd",
            method="test",
        )
    )[-1]
    assert final.aux_label == "MMD² to N(0,I) · unweighted"
    assert len(final.val_aux) == 1
    assert np.isfinite(final.val_aux[0])


def test_beta_vae_warmup_changes_effective_kl_weight() -> None:
    from archcon.data.training import _effective_kl_beta

    config = TrainingConfig(loss_name=LOSS_DVIB, kl_beta=0.2, kl_warmup_epochs=10)
    assert _effective_kl_beta(config, 1) == pytest.approx(0.02)
    assert _effective_kl_beta(config, 5) == pytest.approx(0.1)
    assert _effective_kl_beta(config, 10) == pytest.approx(0.2)
    assert _effective_kl_beta(config, 20) == pytest.approx(0.2)


def test_validation_plot_adds_regularizer_panel_when_available() -> None:
    from archcon.data.training import TrainingUpdate, plot_validation_metrics

    update = TrainingUpdate(
        run_id="mmd-plot",
        status="ok",
        running=False,
        done=False,
        val_epoch=[1, 2, 3],
        val_mse=[1.5, 1.4, 1.3],
        val_r2=[0.70, 0.72, 0.74],
        val_aux=[0.3, 0.2, 0.15],
        aux_label="MMD² to N(0,I) · unweighted",
    )
    fig = plot_validation_metrics(update)
    assert len(fig.axes) == 3
    colors = [axis.lines[0].get_color() for axis in fig.axes]
    assert len(set(colors)) == 3


def test_pytorch_audit_code_uses_same_residual_stage_widths() -> None:
    from archcon.data.pretraining import ARCH_RESIDUAL
    from archcon.data.training import model_execution_markdown, pytorch_model_code

    config = TrainingConfig(
        hidden_widths=(12, 8),
        latent_dim=3,
        architecture_family=ARCH_RESIDUAL,
        residual_blocks=2,
        dropout=0.0,
        device=DEVICE_CPU,
    )
    model = build_autoencoder(16, config)
    code = pytorch_model_code(16, config)
    plan_text = model_execution_markdown(16, config)

    assert model.execution_plan.encoder_stages[1].out_features == 8
    assert "nn.Linear(12, 8)" in code
    assert "ResidualMLPBlock(8, dropout=0.0, gated=False, expansion=1)" in code
    assert code.index("nn.Linear(12, 8)") < code.index(
        "ResidualMLPBlock(8, dropout=0.0, gated=False, expansion=1)"
    )
    assert "Linear(12→8)" in plan_text
    assert "residual block(LayerNorm; 8→8→8) ×2" in plan_text


def test_pytorch_audit_code_records_compile_step() -> None:
    from archcon.data.training import pytorch_model_code

    config = TrainingConfig(
        hidden_widths=(6, 4),
        latent_dim=2,
        compile_model=True,
        device=DEVICE_CPU,
    )
    assert "model = torch.compile(model)" in pytorch_model_code(8, config)

def test_row_major_cache_stays_with_data_store_when_models_are_elsewhere(tmp_path: Path) -> None:
    from archcon.data.training import _existing_row_major_cache

    store = tmp_path / "data" / "GEO_NUMPY_STORE"
    store.mkdir(parents=True)
    matrix_path = store / "rma_global.npy"
    np.save(matrix_path, np.asfortranarray(np.arange(24, dtype=np.float32).reshape(6, 4)))
    matrix = np.load(matrix_path, mmap_mode="r")
    assert matrix.flags.f_contiguous and not matrix.flags.c_contiguous

    cached, cache_path = _existing_row_major_cache(matrix, tmp_path / "launch" / "models")

    assert cached is None
    assert cache_path == (tmp_path / "data" / "training_cache" / "rma_global_row_major.npy").resolve()



def test_training_history_uses_log_y_scale() -> None:
    from archcon.data.training import TrainingUpdate, plot_training_history

    update = TrainingUpdate(
        run_id="log",
        status="ok",
        running=False,
        done=False,
        train_epoch=[1, 2],
        train_epoch_loss=[1.0, 0.01],
        val_epoch=[1, 2],
        val_loss=[1.2, 0.02],
    )
    fig = plot_training_history(update)
    try:
        assert fig.axes[0].get_yscale() == "log"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_stadniuk_family_defaults_to_no_normalization() -> None:
    from archcon.data.pretraining import ARCH_STADNIUK

    config = TrainingConfig(
        hidden_widths=(12, 6),
        latent_dim=3,
        architecture_family=ARCH_STADNIUK,
        activation="ReLU",
        dropout=0.1,
        device=DEVICE_CPU,
    )
    model = build_autoencoder(16, config)
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    assert batch_norms == []


def test_stadniuk_batchnorm_variant_normalizes_hidden_activations() -> None:
    from archcon.data.pretraining import ARCH_STADNIUK

    config = TrainingConfig(
        hidden_widths=(12, 6),
        latent_dim=3,
        architecture_family=ARCH_STADNIUK,
        activation="ReLU",
        dropout=0.1,
        stadniuk_batch_norm=True,
        device=DEVICE_CPU,
    )
    model = build_autoencoder(16, config)
    batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    # Two encoder hidden layers + two mirrored decoder hidden layers; latent/output stay unnormalized.
    assert len(batch_norms) == 4
    assert not isinstance(model.to_latent, torch.nn.BatchNorm1d)
    assert not isinstance(model.to_output.projection, torch.nn.BatchNorm1d)


def test_resnet_ln_uses_layernorm_and_requested_ffn_expansion() -> None:
    from archcon.data.pretraining import ARCH_RESNET_LN

    config = TrainingConfig(
        hidden_widths=(12, 6),
        latent_dim=3,
        architecture_family=ARCH_RESNET_LN,
        activation="GELU",
        residual_blocks=1,
        residual_expansion=4,
        dropout=0.0,
        device=DEVICE_CPU,
    )
    model = build_autoencoder(16, config)
    blocks = [module for module in model.modules() if module.__class__.__name__ == "ResidualMLPBlock"]
    assert len(blocks) == 4
    assert all(isinstance(block.norm, torch.nn.LayerNorm) for block in blocks)
    assert blocks[0].fc1.in_features == 12
    assert blocks[0].fc1.out_features == 48
    assert blocks[0].fc2.in_features == 48
    assert blocks[0].fc2.out_features == 12


def test_cosine_schedule_decreases_deterministically(tmp_path: Path) -> None:
    from archcon.data.training import LR_SCHEDULE_COSINE

    matrix = np.random.default_rng(12).normal(size=(20, 8)).astype(np.float32)
    config = _tiny_config(LOSS_MSE, epochs=4)
    config = TrainingConfig(**{**config.__dict__, "lr_schedule": LR_SCHEDULE_COSINE})
    final = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            config,
            tmp_path / "models",
            method="cosine-test",
        )
    )[-1]
    assert len(final.learning_rates) == 4
    assert final.learning_rates[0] == pytest.approx(1e-3)
    assert all(left > right for left, right in zip(final.learning_rates, final.learning_rates[1:]))


def test_ikem_diagnostic_does_not_change_geo_training_or_selection(tmp_path: Path) -> None:
    from archcon.data.training import LR_SCHEDULE_COSINE

    rng = np.random.default_rng(41)
    matrix = rng.normal(size=(24, 8)).astype(np.float32)
    ikem = rng.normal(loc=5.0, scale=3.0, size=(5, 8)).astype(np.float32)
    config = _tiny_config(LOSS_MSE, epochs=3)
    config = TrainingConfig(**{**config.__dict__, "lr_schedule": LR_SCHEDULE_COSINE})

    without = list(
        train_autoencoder_stream(
            matrix,
            np.arange(18),
            np.arange(18, 24),
            config,
            tmp_path / "without",
            method="geo",
        )
    )[-1]
    with_ikem = list(
        train_autoencoder_stream(
            matrix,
            np.arange(18),
            np.arange(18, 24),
            config,
            tmp_path / "with",
            method="geo",
            evaluation_matrix=ikem,
            evaluation_label="IKEM",
        )
    )[-1]

    assert np.allclose(without.val_loss, with_ikem.val_loss, rtol=0, atol=1e-7)
    assert len(with_ikem.ikem_mse) == len(with_ikem.val_epoch)
    assert len(with_ikem.ikem_r2) == len(with_ikem.val_epoch)
    assert Path(without.best_checkpoint).name == Path(with_ikem.best_checkpoint).name == "best.pt"


def test_best_validation_epoch_is_circled_on_each_metric_plot() -> None:
    from archcon.data.training import TrainingUpdate, plot_validation_metrics

    update = TrainingUpdate(
        run_id="circle",
        status="done",
        running=False,
        done=True,
        val_epoch=[1, 2, 3],
        val_loss=[0.9, 0.4, 0.6],
        val_mse=[1.0, 0.7, 0.8],
        val_r2=[0.2, 0.6, 0.5],
        ikem_mse=[2.0, 1.7, 1.5],
        ikem_r2=[-0.2, 0.1, 0.3],
    )
    fig = plot_validation_metrics(update)
    assert len(fig.axes) == 2
    # GEO best + IKEM diagnostic circle are both scatter collections on each metric axis.
    assert len(fig.axes[0].collections) >= 2
    assert len(fig.axes[1].collections) >= 2


def test_convergence_waits_for_lr_floor_and_uses_relative_difference(tmp_path: Path) -> None:
    matrix = np.random.default_rng(23).normal(size=(20, 8)).astype(np.float32)
    config = TrainingConfig(
        hidden_widths=(4,),
        latent_dim=2,
        dropout=0.0,
        epochs=20,
        batch_size=4,
        learning_rate=1e-3,
        min_learning_rate=1e-6,
        lr_decay_epochs=1,
        convergence_tolerance=1e9,
        convergence_window=2,
        early_stopping_patience=0,
        device=DEVICE_CPU,
        ui_update_batches=4,
        latent_pca_every_epochs=1,
        seed=19,
    )
    final = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            config,
            tmp_path / "convergence",
            method="test",
        )
    )[-1]
    assert final.done
    assert final.val_epoch[-1] == 3
    assert "converged at minimum learning rate" in final.status
    metadata = inspect_checkpoint(final.latest_checkpoint)
    assert metadata["config"]["early_stopping_patience"] == 0
    assert metadata["config"]["lr_decay_epochs"] == 1
    assert metadata["config"]["convergence_window"] == 2


def test_explicit_run_directory_receives_epoch_checkpoints_directly(tmp_path: Path) -> None:
    matrix = np.random.default_rng(61).normal(size=(20, 8)).astype(np.float32)
    run_dir = tmp_path / "persistent" / "run_0240"
    final = list(
        train_autoencoder_stream(
            matrix,
            np.arange(16),
            np.arange(16, 20),
            _tiny_config(LOSS_MSE, epochs=2),
            tmp_path / "persistent",
            method="unit-test-rma",
            run_directory=run_dir,
        )
    )[-1]
    assert final.done
    assert Path(final.latest_checkpoint).parent == run_dir.resolve()
    assert Path(final.best_checkpoint).parent == run_dir.resolve()
    assert (run_dir / "latest.pt").is_file()
    assert not list(run_dir.glob("geo_ae_*"))

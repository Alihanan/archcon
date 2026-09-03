"""Streaming GEO autoencoder pretraining backend.

The module is import-safe when PyTorch is not installed.  The web UI can still
open and explain the model; actual training is enabled by the optional
``archcon[training]`` dependency.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import shutil
import threading
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np

from .._compat import strict_zip
from .pretraining import (
    ARCH_DENSE,
    ARCH_RESNET_LN,
    ARCH_STADNIUK,
    ARCH_GATED_RESIDUAL,
    ARCH_LOW_RANK,
    ARCH_RESIDUAL,
    ARCHITECTURE_FAMILIES,
    LOSS_COSINE,
    LOSS_DVIB,
    LOSS_HUBER,
    LOSS_MMD,
    LOSS_MASKED,
    LOSS_MSE,
    LOSS_SOFTMAX,
    AutoencoderExecutionPlan,
    HiddenStagePlan,
    autoencoder_execution_plan,
    normalize_loss_name,
)

try:  # optional heavy dependency
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised on installations without torch
    torch = None
    nn = None
    F = None


DEVICE_AUTO = "Auto"
DEVICE_CPU = "CPU"
DEVICE_CUDA = "CUDA"
DEVICE_MPS = "Apple MPS"
DEVICE_OPTIONS = [DEVICE_AUTO, DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS]

OPTIMIZER_ADAM = "Adam"
OPTIMIZER_ADAMW = "AdamW"
OPTIMIZER_OPTIONS = [OPTIMIZER_ADAM, OPTIMIZER_ADAMW]

LR_SCHEDULE_PLATEAU = "ReduceLROnPlateau (validation-driven)"
LR_SCHEDULE_COSINE = "Cosine annealing (deterministic)"
LR_SCHEDULE_CONSTANT = "Constant learning rate"
LR_SCHEDULE_OPTIONS = [LR_SCHEDULE_COSINE, LR_SCHEDULE_PLATEAU, LR_SCHEDULE_CONSTANT]

PRECISION_AUTO = "Auto mixed precision"
PRECISION_FP32 = "Float32"
PRECISION_BF16 = "bfloat16"
PRECISION_FP16 = "float16"
PRECISION_OPTIONS = [PRECISION_AUTO, PRECISION_FP32, PRECISION_BF16, PRECISION_FP16]

CHECKPOINT_WEIGHTS = "Load weights only"
CHECKPOINT_RESUME = "Resume training state"
CHECKPOINT_MODES = [CHECKPOINT_WEIGHTS, CHECKPOINT_RESUME]

_CHECKPOINT_FORMAT = 1
_STOP_EVENTS: dict[str, threading.Event] = {}
_STOP_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one reconstruction-only GEO pretraining run."""

    hidden_widths: tuple[int, ...] = (256, 64)
    latent_dim: int = 3
    activation: str = "ReLU"
    dropout: float = 0.1
    weight_decay: float = 0.0
    l2_lambda: float = 0.0
    architecture_family: str = ARCH_DENSE
    low_rank_dim: int = 64
    residual_blocks: int = 1
    residual_expansion: int = 1
    stadniuk_batch_norm: bool = False
    loss_name: str = LOSS_MSE
    epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = OPTIMIZER_ADAM
    lr_schedule: str = LR_SCHEDULE_COSINE
    lr_decay_epochs: int = 500
    convergence_tolerance: float = 1e-5
    convergence_window: int = 5
    early_stopping_patience: int = 0  # legacy field; ignored since 0.5.4
    lr_patience: int = 7  # legacy ReduceLROnPlateau support
    lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    gradient_clip: float = 1.0
    seed: int = 42
    device: str = DEVICE_AUTO
    precision: str = PRECISION_AUTO
    compile_model: bool = False
    background_prefetch: bool = True
    deterministic: bool = True
    validation_every_epochs: int = 1
    ui_update_batches: int = 10
    latent_pca_every_epochs: int = 5
    latent_pca_max_samples: int = 2000
    huber_delta: float = 1.0
    cosine_weight: float = 0.25
    kl_beta: float = 1e-3
    kl_warmup_epochs: int = 10
    mmd_weight: float = 0.1
    mask_fraction: float = 0.15
    softmax_temperature: float = 1.0  # legacy checkpoint compatibility

    def validate(self, input_dim: int) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if not self.hidden_widths or any(int(value) < 1 for value in self.hidden_widths):
            raise ValueError("At least one positive hidden width is required.")
        if int(self.latent_dim) < 1:
            raise ValueError("latent_dim must be positive.")
        if self.architecture_family not in ARCHITECTURE_FAMILIES:
            raise ValueError(f"Unknown architecture family: {self.architecture_family}")
        if int(self.low_rank_dim) < 1:
            raise ValueError("low_rank_dim must be positive.")
        if int(self.residual_blocks) < 0 or int(self.residual_blocks) > 8:
            raise ValueError("residual_blocks must be between 0 and 8.")
        if int(self.residual_expansion) < 1 or int(self.residual_expansion) > 8:
            raise ValueError("residual_expansion must be between 1 and 8.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if float(self.weight_decay) < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if float(self.l2_lambda) < 0.0:
            raise ValueError("l2_lambda cannot be negative.")
        if int(self.epochs) < 1:
            raise ValueError("epochs must be positive.")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be positive.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if int(self.lr_decay_epochs) < 1:
            raise ValueError("lr_decay_epochs must be positive.")
        if float(self.convergence_tolerance) < 0.0:
            raise ValueError("convergence_tolerance cannot be negative.")
        if int(self.convergence_window) < 1:
            raise ValueError("convergence_window must be positive.")
        if int(self.early_stopping_patience) < 0:
            raise ValueError("legacy early_stopping_patience cannot be negative.")
        if int(self.lr_patience) < 1:
            raise ValueError("lr_patience must be positive.")
        if not 0.0 < float(self.lr_factor) < 1.0:
            raise ValueError("lr_factor must be in (0, 1).")
        if float(self.gradient_clip) < 0.0:
            raise ValueError("gradient_clip cannot be negative.")
        if self.precision not in PRECISION_OPTIONS:
            raise ValueError(f"Unknown precision mode: {self.precision}")
        if int(self.validation_every_epochs) < 1:
            raise ValueError("validation_every_epochs must be positive.")
        if int(self.ui_update_batches) < 1:
            raise ValueError("ui_update_batches must be positive.")
        if int(self.latent_pca_every_epochs) < 1:
            raise ValueError("latent_pca_every_epochs must be positive.")
        if int(self.latent_pca_max_samples) < 2:
            raise ValueError("latent_pca_max_samples must be at least two.")
        if float(self.huber_delta) <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if float(self.cosine_weight) < 0.0:
            raise ValueError("cosine_weight cannot be negative.")
        if float(self.kl_beta) < 0.0:
            raise ValueError("kl_beta cannot be negative.")
        if int(self.kl_warmup_epochs) < 0:
            raise ValueError("kl_warmup_epochs cannot be negative.")
        if float(self.mmd_weight) < 0.0:
            raise ValueError("mmd_weight cannot be negative.")
        if not 0.0 < float(self.mask_fraction) < 1.0:
            raise ValueError("mask_fraction must be in (0, 1).")
        if float(self.softmax_temperature) <= 0.0:
            raise ValueError("softmax_temperature must be positive.")
        if self.loss_name not in {
            LOSS_MSE,
            LOSS_HUBER,
            LOSS_COSINE,
            LOSS_DVIB,
            LOSS_MMD,
            LOSS_MASKED,
            LOSS_SOFTMAX,
        }:
            raise ValueError(f"Unknown reconstruction loss: {self.loss_name}")
        if self.optimizer not in OPTIMIZER_OPTIONS:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.lr_schedule not in LR_SCHEDULE_OPTIONS:
            raise ValueError(f"Unknown learning-rate schedule: {self.lr_schedule}")


@dataclass
class TrainingUpdate:
    """Small serializable snapshot emitted while training is running."""

    run_id: str
    status: str
    running: bool
    done: bool
    stopped: bool = False
    epoch: int = 0
    batch: int = 0
    n_batches: int = 0
    train_x: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_epoch: list[int] = field(default_factory=list)
    train_epoch_loss: list[float] = field(default_factory=list)
    val_epoch: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_r2: list[float] = field(default_factory=list)
    val_aux: list[float] = field(default_factory=list)
    ikem_mse: list[float] = field(default_factory=list)
    ikem_r2: list[float] = field(default_factory=list)
    aux_label: str | None = None
    learning_rates: list[float] = field(default_factory=list)
    latent_pca: np.ndarray | None = None
    latent_pca_epoch: int | None = None
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None


@dataclass
class _History:
    train_x: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_epoch: list[int] = field(default_factory=list)
    train_epoch_loss: list[float] = field(default_factory=list)
    val_epoch: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_r2: list[float] = field(default_factory=list)
    val_aux: list[float] = field(default_factory=list)
    ikem_mse: list[float] = field(default_factory=list)
    ikem_r2: list[float] = field(default_factory=list)
    aux_label: str | None = None
    learning_rates: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "train_x": list(self.train_x),
            "train_loss": list(self.train_loss),
            "train_epoch": list(self.train_epoch),
            "train_epoch_loss": list(self.train_epoch_loss),
            "val_epoch": list(self.val_epoch),
            "val_loss": list(self.val_loss),
            "val_mse": list(self.val_mse),
            "val_r2": list(self.val_r2),
            "val_aux": list(self.val_aux),
            "ikem_mse": list(self.ikem_mse),
            "ikem_r2": list(self.ikem_r2),
            "aux_label": self.aux_label,
            "learning_rates": list(self.learning_rates),
        }

    @classmethod
    def from_dict(cls, value: object) -> "_History":
        if not isinstance(value, dict):
            return cls()
        return cls(
            train_x=[float(x) for x in value.get("train_x", [])],
            train_loss=[float(x) for x in value.get("train_loss", [])],
            train_epoch=[int(x) for x in value.get("train_epoch", [])],
            train_epoch_loss=[float(x) for x in value.get("train_epoch_loss", [])],
            val_epoch=[int(x) for x in value.get("val_epoch", [])],
            val_loss=[float(x) for x in value.get("val_loss", [])],
            val_mse=[float(x) for x in value.get("val_mse", [])],
            val_r2=[float(x) for x in value.get("val_r2", [])],
            val_aux=[float(x) for x in value.get("val_aux", [])],
            ikem_mse=[float(x) for x in value.get("ikem_mse", [])],
            ikem_r2=[float(x) for x in value.get("ikem_r2", [])],
            aux_label=(str(value.get("aux_label")) if value.get("aux_label") else None),
            learning_rates=[float(x) for x in value.get("learning_rates", [])],
        )


def training_backend_status() -> str:
    """Return a concise user-facing PyTorch/device status."""
    if torch is None:
        return (
            "⚠️ **Training backend is not installed.** Install the optional dependency with "
            "`python -m pip install -e \".[training]\"`. The rest of ArchCon still works."
        )

    devices = ["CPU"]
    if torch.cuda.is_available():
        devices.append(f"CUDA ({torch.cuda.get_device_name(0)})")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("Apple MPS")
    return f"✅ **PyTorch {torch.__version__}** · available: {' · '.join(devices)}"


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for training. Install ArchCon with the training extra: "
            'python -m pip install -e ".[training]"'
        )


def _activation(name: str):
    _require_torch()
    mapping = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "ELU": nn.ELU,
        "Tanh": nn.Tanh,
        "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
    }
    try:
        factory = mapping[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc
    return factory()


def _hidden_stack(widths: list[int], activation: str, dropout: float):
    _require_torch()
    layers: list[nn.Module] = []
    for left, right in strict_zip(widths[:-1], widths[1:]):
        layers.append(nn.Linear(int(left), int(right)))
        layers.append(_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


if nn is not None:

    class FactorizedLinear(nn.Module):
        """Low-rank linear map used to reduce the huge 42,917-D edge cost."""

        def __init__(self, in_features: int, out_features: int, rank: int) -> None:
            super().__init__()
            rank = max(1, min(int(rank), int(in_features), int(out_features)))
            self.rank = rank
            self.down = nn.Linear(int(in_features), rank, bias=False)
            self.up = nn.Linear(rank, int(out_features), bias=True)

        def forward(self, x):
            return self.up(self.down(x))


    class ResidualMLPBlock(nn.Module):
        def __init__(
            self,
            width: int,
            activation: str,
            dropout: float,
            *,
            gated: bool,
            expansion: int = 1,
        ) -> None:
            super().__init__()
            width = int(width)
            expansion = max(1, int(expansion))
            self.norm = nn.LayerNorm(width)
            self.gated = bool(gated)
            self.expansion = expansion
            hidden = width * (2 if gated else expansion)
            self.fc1 = nn.Linear(width, hidden)
            self.fc2 = nn.Linear(width if gated else width * expansion, width)
            self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
            self.activation_name = str(activation)

        def forward(self, x):
            residual = x
            h = self.norm(x)
            h = self.fc1(h)
            if self.gated:
                value, gate = h.chunk(2, dim=-1)
                h = _activation(self.activation_name)(value) * torch.sigmoid(gate)
            else:
                h = _activation(self.activation_name)(h)
            h = self.dropout(h)
            h = self.fc2(h)
            h = self.dropout(h)
            return residual + h


    class FeatureStack(nn.Module):
        """Encoder/decoder hidden stack built from the shared execution plan."""

        def __init__(
            self,
            stages: tuple[HiddenStagePlan, ...],
            activation: str,
            dropout: float,
            low_rank_dim: int,
        ) -> None:
            super().__init__()
            modules: list[nn.Module] = []
            for stage in stages:
                if stage.factorized_projection:
                    modules.append(
                        FactorizedLinear(stage.in_features, stage.out_features, low_rank_dim)
                    )
                else:
                    modules.append(nn.Linear(stage.in_features, stage.out_features))
                modules.append(_activation(activation))
                # Stadniuk's legacy pretraining class places BatchNormalization
                # after the Dense activation and before Dropout.  ResNet uses
                # pre-LayerNorm inside each residual block instead.
                if stage.normalization == "BatchNorm":
                    modules.append(nn.BatchNorm1d(stage.out_features))
                if dropout > 0:
                    modules.append(nn.Dropout(float(dropout)))
                modules.extend(
                    ResidualMLPBlock(
                        stage.out_features,
                        activation,
                        dropout,
                        gated=stage.gated_residual,
                        expansion=stage.residual_expansion,
                    )
                    for _ in range(stage.residual_blocks)
                )
            self.net = nn.Sequential(*modules)

        def forward(self, x):
            return self.net(x)


    class OutputProjection(nn.Module):
        def __init__(
            self,
            in_features: int,
            output_dim: int,
            architecture_family: str,
            low_rank_dim: int,
        ) -> None:
            super().__init__()
            if architecture_family == ARCH_LOW_RANK:
                self.projection = FactorizedLinear(in_features, output_dim, low_rank_dim)
            else:
                self.projection = nn.Linear(int(in_features), int(output_dim))

        def forward(self, x):
            return self.projection(x)


    class DeterministicAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_widths: tuple[int, ...],
            latent_dim: int,
            activation: str,
            dropout: float,
            architecture_family: str = ARCH_DENSE,
            low_rank_dim: int = 64,
            residual_blocks: int = 1,
            residual_expansion: int = 1,
            stadniuk_batch_norm: bool = False,
        ) -> None:
            super().__init__()
            plan = autoencoder_execution_plan(
                input_dim,
                hidden_widths,
                latent_dim,
                architecture_family,
                low_rank_dim=low_rank_dim,
                residual_blocks=residual_blocks,
                residual_expansion=residual_expansion,
                stadniuk_batch_norm=stadniuk_batch_norm,
            )
            self.execution_plan = plan
            self.encoder_hidden = FeatureStack(
                plan.encoder_stages, activation, dropout, plan.low_rank_dim
            )
            self.to_latent = nn.Linear(plan.hidden_widths[-1], plan.latent_dim)
            self.decoder_hidden = FeatureStack(
                plan.decoder_stages, activation, dropout, plan.low_rank_dim
            )
            self.to_output = OutputProjection(
                plan.hidden_widths[0],
                plan.input_dim,
                plan.architecture_family,
                plan.low_rank_dim,
            )

        def encode(self, x):
            return self.to_latent(self.encoder_hidden(x))

        def decode(self, z):
            return self.to_output(self.decoder_hidden(z))

        def forward(self, x, *, sample: bool = True):  # noqa: ARG002
            z = self.encode(x)
            return self.decode(z), z, None, None


    class VariationalAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_widths: tuple[int, ...],
            latent_dim: int,
            activation: str,
            dropout: float,
            architecture_family: str = ARCH_DENSE,
            low_rank_dim: int = 64,
            residual_blocks: int = 1,
            residual_expansion: int = 1,
            stadniuk_batch_norm: bool = False,
        ) -> None:
            super().__init__()
            plan = autoencoder_execution_plan(
                input_dim,
                hidden_widths,
                latent_dim,
                architecture_family,
                low_rank_dim=low_rank_dim,
                residual_blocks=residual_blocks,
                residual_expansion=residual_expansion,
                stadniuk_batch_norm=stadniuk_batch_norm,
            )
            self.execution_plan = plan
            self.encoder_hidden = FeatureStack(
                plan.encoder_stages, activation, dropout, plan.low_rank_dim
            )
            self.to_mu = nn.Linear(plan.hidden_widths[-1], plan.latent_dim)
            self.to_logvar = nn.Linear(plan.hidden_widths[-1], plan.latent_dim)
            self.decoder_hidden = FeatureStack(
                plan.decoder_stages, activation, dropout, plan.low_rank_dim
            )
            self.to_output = OutputProjection(
                plan.hidden_widths[0],
                plan.input_dim,
                plan.architecture_family,
                plan.low_rank_dim,
            )

        def encode_distribution(self, x):
            hidden = self.encoder_hidden(x)
            return self.to_mu(hidden), self.to_logvar(hidden)

        def encode(self, x):
            mu, _ = self.encode_distribution(x)
            return mu

        def decode(self, z):
            return self.to_output(self.decoder_hidden(z))

        def forward(self, x, *, sample: bool = True):
            mu, logvar = self.encode_distribution(x)
            if sample:
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std
            else:
                z = mu
            return self.decode(z), z, mu, logvar

else:  # import-safe placeholders
    FactorizedLinear = None
    ResidualMLPBlock = None
    FeatureStack = None
    OutputProjection = None
    DeterministicAutoencoder = None
    VariationalAutoencoder = None


def build_autoencoder(input_dim: int, config: TrainingConfig):
    """Construct a deterministic AE or VAE according to the selected objective."""
    _require_torch()
    config.validate(input_dim)
    cls = VariationalAutoencoder if config.loss_name == LOSS_DVIB else DeterministicAutoencoder
    return cls(
        input_dim=input_dim,
        hidden_widths=config.hidden_widths,
        latent_dim=config.latent_dim,
        activation=config.activation,
        dropout=config.dropout,
        architecture_family=config.architecture_family,
        low_rank_dim=config.low_rank_dim,
        residual_blocks=config.residual_blocks,
        residual_expansion=config.residual_expansion,
        stadniuk_batch_norm=config.stadniuk_batch_norm,
    )


def _activation_code(name: str) -> str:
    mapping = {
        "ReLU": "nn.ReLU()",
        "GELU": "nn.GELU()",
        "SiLU": "nn.SiLU()",
        "ELU": "nn.ELU()",
        "Tanh": "nn.Tanh()",
        "LeakyReLU": "nn.LeakyReLU(negative_slope=0.01)",
    }
    try:
        return mapping[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def model_execution_markdown(input_dim: int, config: TrainingConfig) -> str:
    """Human-readable plan generated from the same structure used by the builder."""
    config.validate(input_dim)
    plan = autoencoder_execution_plan(
        input_dim,
        config.hidden_widths,
        config.latent_dim,
        config.architecture_family,
        low_rank_dim=config.low_rank_dim,
        residual_blocks=config.residual_blocks,
        residual_expansion=config.residual_expansion,
        stadniuk_batch_norm=config.stadniuk_batch_norm,
    )

    def describe(stage: HiddenStagePlan) -> str:
        projection = (
            f"FactorizedLinear({stage.in_features:,}→{stage.out_features:,}, "
            f"rank={plan.low_rank_dim})"
            if stage.factorized_projection
            else f"Linear({stage.in_features:,}→{stage.out_features:,})"
        )
        parts = [projection, config.activation]
        if stage.normalization == "BatchNorm":
            parts.append("BatchNorm")
        if config.dropout > 0:
            parts.append(f"Dropout({config.dropout:g})")
        if stage.residual_blocks:
            kind = "gated residual" if stage.gated_residual else "residual"
            parts.append(
                f"{kind} block(LayerNorm; {stage.out_features:,}→"
                f"{stage.out_features * stage.residual_expansion:,}→{stage.out_features:,}) "
                f"×{stage.residual_blocks}"
            )
        return " → ".join(parts)

    lines = [
        "**Executable architecture plan** — this is the pure-data plan consumed by the "
        "PyTorch model builder and also used to generate the audit code below.",
        "",
        "**Encoder**",
    ]
    lines.extend(f"{stage.index}. {describe(stage)}" for stage in plan.encoder_stages)
    if config.loss_name == LOSS_DVIB:
        lines.extend(
            [
                f"- latent μ head: `Linear({plan.hidden_widths[-1]:,}→{plan.latent_dim:,})`",
                f"- latent logσ² head: `Linear({plan.hidden_widths[-1]:,}→{plan.latent_dim:,})`",
                "- sample: `z = μ + exp(0.5·logσ²)·ε` during training",
            ]
        )
    else:
        lines.append(
            f"- latent head: `Linear({plan.hidden_widths[-1]:,}→{plan.latent_dim:,})`"
        )
    lines.extend(["", "**Decoder**"])
    lines.extend(f"{stage.index}. {describe(stage)}" for stage in plan.decoder_stages)
    output_projection = (
        f"FactorizedLinear({plan.hidden_widths[0]:,}→{plan.input_dim:,}, rank={plan.low_rank_dim})"
        if plan.output_factorized
        else f"Linear({plan.hidden_widths[0]:,}→{plan.input_dim:,})"
    )
    lines.append(f"- output head: `{output_projection}`")
    lines.extend(
        [
            "",
            "For residual families, every shortcut is **local and same-width**: "
            "`x → F(x) → +x`. A width change such as `512→256` happens in the "
            "projection *before* the `256→256` residual unit. There is no illegal "
            "direct addition of a 512-vector to a 256-vector.",
        ]
    )
    return "\n".join(lines)


def pytorch_model_code(
    input_dim: int,
    config: TrainingConfig,
    *,
    instantiate: bool = True,
) -> str:
    """Render an auditable PyTorch-equivalent model from the shared execution plan.

    The returned text is intentionally explicit rather than clever.  The training
    backend does not ``exec`` this string; both this string and the actual modules
    are generated from ``autoencoder_execution_plan()``, so the displayed audit
    view and the model construction follow the same structural source of truth.
    """
    config.validate(input_dim)
    plan: AutoencoderExecutionPlan = autoencoder_execution_plan(
        input_dim,
        config.hidden_widths,
        config.latent_dim,
        config.architecture_family,
        low_rank_dim=config.low_rank_dim,
        residual_blocks=config.residual_blocks,
        residual_expansion=config.residual_expansion,
        stadniuk_batch_norm=config.stadniuk_batch_norm,
    )
    activation = _activation_code(config.activation)
    residual_family = config.architecture_family in {ARCH_RESNET_LN, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL}
    low_rank_family = config.architecture_family == ARCH_LOW_RANK

    lines = [
        "# ArchCon architecture audit view",
        "# Generated from the SAME AutoencoderExecutionPlan used by build_autoencoder().",
        f"# Objective: {config.loss_name}",
        "import torch",
        "from torch import nn",
        "",
    ]

    if low_rank_family:
        lines.extend(
            [
                "class FactorizedLinear(nn.Module):",
                "    def __init__(self, in_features, out_features, rank):",
                "        super().__init__()",
                "        rank = max(1, min(int(rank), int(in_features), int(out_features)))",
                "        self.down = nn.Linear(in_features, rank, bias=False)",
                "        self.up = nn.Linear(rank, out_features, bias=True)",
                "",
                "    def forward(self, x):",
                "        return self.up(self.down(x))",
                "",
            ]
        )

    if residual_family:
        lines.extend(
            [
                "class ResidualMLPBlock(nn.Module):",
                "    def __init__(self, width, dropout=0.0, gated=False, expansion=1):",
                "        super().__init__()",
                "        self.norm = nn.LayerNorm(width)",
                "        self.gated = gated",
                "        self.fc1 = nn.Linear(width, width * (2 if gated else expansion))",
                "        self.fc2 = nn.Linear(width if gated else width * expansion, width)",
                "        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()",
                f"        self.activation = {activation}",
                "",
                "    def forward(self, x):",
                "        residual = x",
                "        h = self.fc1(self.norm(x))",
                "        if self.gated:",
                "            value, gate = h.chunk(2, dim=-1)",
                "            h = self.activation(value) * torch.sigmoid(gate)",
                "        else:",
                "            h = self.activation(h)",
                "        h = self.dropout(h)",
                "        h = self.dropout(self.fc2(h))",
                "        return residual + h",
                "",
            ]
        )

    lines.extend(
        [
            "class FeatureStack(nn.Module):",
            "    def __init__(self, *modules):",
            "        super().__init__()",
            "        self.net = nn.Sequential(*modules)",
            "",
            "    def forward(self, x):",
            "        return self.net(x)",
            "",
            "class OutputProjection(nn.Module):",
            "    def __init__(self, projection):",
            "        super().__init__()",
            "        self.projection = projection",
            "",
            "    def forward(self, x):",
            "        return self.projection(x)",
            "",
        ]
    )

    def module_lines(stages: tuple[HiddenStagePlan, ...], indent: str) -> list[str]:
        rendered: list[str] = []
        for stage in stages:
            if stage.factorized_projection:
                rendered.append(
                    f"{indent}FactorizedLinear({stage.in_features}, {stage.out_features}, "
                    f"rank={plan.low_rank_dim}),"
                )
            else:
                rendered.append(
                    f"{indent}nn.Linear({stage.in_features}, {stage.out_features}),"
                )
            rendered.append(f"{indent}{activation},")
            if stage.normalization == "BatchNorm":
                rendered.append(f"{indent}nn.BatchNorm1d({stage.out_features}),")
            if config.dropout > 0:
                rendered.append(f"{indent}nn.Dropout({config.dropout!r}),")
            for _ in range(stage.residual_blocks):
                rendered.append(
                    f"{indent}ResidualMLPBlock({stage.out_features}, "
                    f"dropout={config.dropout!r}, gated={stage.gated_residual}, "
                    f"expansion={stage.residual_expansion}),"
                )
        return rendered

    class_name = "ArchConVariationalAutoencoder" if config.loss_name == LOSS_DVIB else "ArchConAutoencoder"
    lines.extend(
        [
            f"class {class_name}(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            "        self.encoder_hidden = FeatureStack(",
        ]
    )
    lines.extend(module_lines(plan.encoder_stages, "            "))
    lines.append("        )")
    if config.loss_name == LOSS_DVIB:
        lines.extend(
            [
                f"        self.to_mu = nn.Linear({plan.hidden_widths[-1]}, {plan.latent_dim})",
                f"        self.to_logvar = nn.Linear({plan.hidden_widths[-1]}, {plan.latent_dim})",
            ]
        )
    else:
        lines.append(
            f"        self.to_latent = nn.Linear({plan.hidden_widths[-1]}, {plan.latent_dim})"
        )
    lines.append("        self.decoder_hidden = FeatureStack(")
    lines.extend(module_lines(plan.decoder_stages, "            "))
    lines.append("        )")
    if plan.output_factorized:
        lines.append(
            f"        self.to_output = OutputProjection(FactorizedLinear({plan.hidden_widths[0]}, "
            f"{plan.input_dim}, rank={plan.low_rank_dim}))"
        )
    else:
        lines.append(
            f"        self.to_output = OutputProjection(nn.Linear({plan.hidden_widths[0]}, {plan.input_dim}))"
        )

    if config.loss_name == LOSS_DVIB:
        lines.extend(
            [
                "",
                "    def forward(self, x, sample=True):",
                "        hidden = self.encoder_hidden(x)",
                "        mu, logvar = self.to_mu(hidden), self.to_logvar(hidden)",
                "        if sample:",
                "            std = torch.exp(0.5 * logvar)",
                "            z = mu + torch.randn_like(std) * std",
                "        else:",
                "            z = mu",
                "        reconstruction = self.to_output(self.decoder_hidden(z))",
                "        return reconstruction, z, mu, logvar",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "    def forward(self, x, sample=True):  # noqa: ARG002",
                "        z = self.to_latent(self.encoder_hidden(x))",
                "        reconstruction = self.to_output(self.decoder_hidden(z))",
                "        return reconstruction, z, None, None",
            ]
        )

    if instantiate:
        lines.extend(["", f"model = {class_name}()"])
        if config.compile_model:
            lines.extend(
                [
                    "# The real training backend applies the same optional compilation step:",
                    "model = torch.compile(model)",
                ]
            )
        else:
            lines.append("# torch.compile is disabled for this configuration.")
    return "\n".join(lines)


def resolve_device(choice: str):
    _require_torch()
    choice = str(choice)
    if choice == DEVICE_CPU:
        return torch.device("cpu")
    if choice == DEVICE_CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but PyTorch cannot see a CUDA device.")
        return torch.device("cuda")
    if choice == DEVICE_MPS:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Apple MPS was selected but is not available.")
        return torch.device("mps")
    if choice != DEVICE_AUTO:
        raise ValueError(f"Unknown device selection: {choice}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int, deterministic: bool = True) -> None:
    _require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:  # pragma: no cover - old torch fallback
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _optimizer(model, config: TrainingConfig):
    _require_torch()
    cls = torch.optim.Adam if config.optimizer == OPTIMIZER_ADAM else torch.optim.AdamW
    return cls(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )




def _scheduler(optimizer, config: TrainingConfig):
    _require_torch()
    if config.lr_schedule == LR_SCHEDULE_COSINE:
        # One-way cosine decay: reach eta_min after lr_decay_epochs, then stay
        # at the floor. Unlike CosineAnnealingLR, this cannot rise again when
        # max epochs is intentionally much larger than the decay horizon.
        decay_epochs = max(1, int(config.lr_decay_epochs))
        base_lr = float(config.learning_rate)
        min_lr = float(config.min_learning_rate)
        min_factor = min(1.0, max(0.0, min_lr / base_lr))

        def lr_lambda(step: int) -> float:
            progress = min(max(float(step), 0.0), float(decay_epochs)) / float(decay_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_factor + (1.0 - min_factor) * cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if config.lr_schedule == LR_SCHEDULE_CONSTANT:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if config.lr_schedule == LR_SCHEDULE_PLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.lr_factor),
            patience=int(config.lr_patience),
            min_lr=float(config.min_learning_rate),
        )
    raise ValueError(f"Unknown learning-rate schedule: {config.lr_schedule}")


def _explicit_l2_penalty(model, config: TrainingConfig):
    """Keras-style kernel L2, excluding biases, normalization and output head.

    Stadniuk's legacy pretraining class attached ``keras.regularizers.l2`` to
    hidden Dense kernels and the latent Dense layer, but not the final decoder
    output.  Keeping this separate from optimizer weight decay makes the grid
    parameter scientifically explicit and reproducible across Adam/AdamW.
    """
    _require_torch()
    coefficient = float(config.l2_lambda)
    if coefficient <= 0.0:
        return torch.zeros((), device=next(model.parameters()).device)
    total = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("to_output.projection") or name == "to_output":
            continue
        value = torch.sum(module.weight * module.weight)
        total = value if total is None else total + value
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return coefficient * total


def _effective_kl_beta(config: TrainingConfig, epoch: int | None) -> float:
    beta = float(config.kl_beta)
    warmup = int(config.kl_warmup_epochs)
    if warmup <= 0 or epoch is None:
        return beta
    return beta * min(1.0, max(float(epoch), 0.0) / float(warmup))


def _imq_kernel(x, y, scale: float):
    """Inverse-multiquadratic kernel used by WAE/InfoVAE-style MMD."""
    dim = max(int(x.shape[1]), 1)
    c = 2.0 * float(dim) * float(scale)
    distance2 = torch.cdist(x, y, p=2).pow(2)
    return c / (c + distance2 + 1e-8)


def _mmd_to_standard_normal(z):
    """Unbiased-ish multiscale MMD estimate against N(0, I)."""
    z = z.float()
    n = int(z.shape[0])
    if n < 2:
        return torch.zeros((), device=z.device, dtype=z.dtype)
    prior = torch.randn_like(z)
    total = torch.zeros((), device=z.device, dtype=z.dtype)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        k_xx = _imq_kernel(z, z, scale)
        k_yy = _imq_kernel(prior, prior, scale)
        k_xy = _imq_kernel(z, prior, scale)
        within_x = k_xx.masked_select(~eye).mean()
        within_y = k_yy.masked_select(~eye).mean()
        total = total + within_x + within_y - 2.0 * k_xy.mean()
    return total / 5.0


def _objective(
    x,
    reconstruction,
    z,
    mu,
    logvar,
    config: TrainingConfig,
    *,
    epoch: int | None = None,
    reconstruction_mask=None,
):
    """Return (selected objective, plain MSE diagnostic, latent/auxiliary term)."""
    _require_torch()
    squared_error = (reconstruction - x) ** 2
    mse = torch.mean(squared_error)

    if config.loss_name == LOSS_MSE:
        return mse, mse, torch.zeros((), device=x.device)

    if config.loss_name == LOSS_HUBER:
        huber = F.huber_loss(
            reconstruction,
            x,
            reduction="mean",
            delta=float(config.huber_delta),
        )
        return huber, mse, torch.zeros((), device=x.device)

    if config.loss_name == LOSS_COSINE:
        cosine = 1.0 - F.cosine_similarity(reconstruction, x, dim=1, eps=1e-8).mean()
        return mse + float(config.cosine_weight) * cosine, mse, cosine

    if config.loss_name == LOSS_DVIB:
        if mu is None or logvar is None:
            raise RuntimeError("β-VAE objective requires variational encoder outputs.")
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        beta = _effective_kl_beta(config, epoch)
        return mse + beta * kl, mse, kl

    if config.loss_name == LOSS_MMD:
        mmd = _mmd_to_standard_normal(z)
        return mse + float(config.mmd_weight) * mmd, mse, mmd

    if config.loss_name == LOSS_MASKED:
        if reconstruction_mask is None:
            raise RuntimeError("Masked reconstruction requires a probe mask.")
        masked_error = squared_error.masked_select(reconstruction_mask)
        if masked_error.numel() == 0:
            raise RuntimeError("Masked reconstruction received an empty probe mask.")
        masked_mse = masked_error.mean()
        return masked_mse, mse, torch.zeros((), device=x.device)

    temperature = float(config.softmax_temperature)
    with torch.no_grad():
        target = torch.softmax(x / temperature, dim=1)
    log_prediction = torch.log_softmax(reconstruction / temperature, dim=1)
    ce = -(target * log_prediction).sum(dim=1).mean()
    return ce, mse, torch.zeros((), device=x.device)


def _masked_reconstruction_input(
    x,
    fraction: float,
    *,
    numpy_rng: np.random.Generator | None = None,
):
    """Return a corrupted input and Boolean mask for masked-value prediction.

    RMA/log2 values in ArchCon are positive, so zero is an unambiguous mask
    sentinel.  Validation may pass a NumPy RNG so the same masks are reused
    across epochs; training uses PyTorch RNG and therefore receives fresh masks.
    """
    _require_torch()
    probability = float(fraction)
    if numpy_rng is None:
        mask = torch.rand(x.shape, device=x.device) < probability
    else:
        mask_np = numpy_rng.random(tuple(int(v) for v in x.shape)) < probability
        mask = torch.from_numpy(mask_np).to(device=x.device)
    if not bool(mask.any()):
        mask = mask.clone()
        mask.reshape(-1)[0] = True
    corrupted = x.masked_fill(mask, 0.0)
    return corrupted, mask


def _load_batch_numpy(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.array(matrix[rows, :], dtype=np.float32, copy=True, order="C")


def _tensor_from_numpy(batch_np: np.ndarray, device):
    tensor = torch.from_numpy(batch_np)
    if device.type == "cuda":
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            pass
        return tensor.to(device=device, non_blocking=True)
    return tensor.to(device=device)


def _iter_numpy_batches(
    matrix: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    *,
    background_prefetch: bool,
):
    """Yield one batch while at most one next memmap read is in flight."""
    chunks = [rows[start : start + int(batch_size)] for start in range(0, len(rows), int(batch_size))]
    if not background_prefetch or len(chunks) <= 1:
        for chunk in chunks:
            yield chunk, _load_batch_numpy(matrix, chunk)
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="archcon-batch-prefetch") as pool:
        future = pool.submit(_load_batch_numpy, matrix, chunks[0])
        for index, chunk in enumerate(chunks):
            batch_np = future.result()
            if index + 1 < len(chunks):
                future = pool.submit(_load_batch_numpy, matrix, chunks[index + 1])
            yield chunk, batch_np


def _autocast_settings(device, precision: str):
    """Return (enabled, dtype, needs_grad_scaler) for the chosen device."""
    if precision == PRECISION_FP32:
        return False, None, False
    if precision == PRECISION_BF16:
        return True, torch.bfloat16, False
    if precision == PRECISION_FP16:
        return True, torch.float16, device.type == "cuda"
    if precision != PRECISION_AUTO:
        raise ValueError(f"Unknown precision mode: {precision}")

    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True
    if device.type == "mps":
        return True, torch.float16, False
    # CPU bfloat16 varies greatly by hardware; keep float32 as the safe default.
    return False, None, False


def _autocast_context(device, enabled: bool, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _make_grad_scaler(device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    amp = getattr(torch, "amp", None)
    if amp is not None and hasattr(amp, "GradScaler"):
        try:
            return amp.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True)


def _register_run(run_id: str) -> threading.Event:
    event = threading.Event()
    with _STOP_LOCK:
        _STOP_EVENTS[run_id] = event
    return event


def request_training_stop(run_id: str | None) -> bool:
    """Request cooperative cancellation of one running training stream."""
    if not run_id:
        return False
    with _STOP_LOCK:
        event = _STOP_EVENTS.get(str(run_id))
    if event is None:
        return False
    event.set()
    return True


def _clear_run(run_id: str) -> None:
    with _STOP_LOCK:
        _STOP_EVENTS.pop(run_id, None)


def _safe_torch_load(path: str | Path):
    _require_torch()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch
        return torch.load(source, map_location="cpu")


def inspect_checkpoint(path: str | Path) -> dict[str, object]:
    """Return safe metadata from an ArchCon checkpoint."""
    checkpoint = _safe_torch_load(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("archcon_checkpoint_format") != _CHECKPOINT_FORMAT:
        raise ValueError("This is not a supported ArchCon autoencoder checkpoint.")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config is malformed.")
    return {
        "path": str(Path(path).expanduser().resolve()),
        "epoch": int(checkpoint.get("epoch", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "input_dim": int(checkpoint.get("input_dim", 0)),
        "method": str(checkpoint.get("method", "unknown")),
        "n_train": len(checkpoint.get("train_rows", [])),
        "n_validation": len(checkpoint.get("validation_rows", [])),
        "config": config,
    }


def _checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    input_dim: int,
    method: str,
    epoch: int,
    best_val_loss: float,
    history: _History,
    run_id: str,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
) -> dict[str, object]:
    return {
        "archcon_checkpoint_format": _CHECKPOINT_FORMAT,
        "created_unix": time.time(),
        "run_id": run_id,
        "input_dim": int(input_dim),
        "method": str(method),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": asdict(config),
        "history": history.to_dict(),
        "train_rows": [int(value) for value in train_rows],
        "validation_rows": [int(value) for value in validation_rows],
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)
    return str(path)


def _config_compatible(checkpoint_config: dict[str, object], config: TrainingConfig) -> bool:
    """Compare architecture-critical fields, including legacy 0.3.x checkpoints."""
    current = asdict(config)
    defaults: dict[str, object] = {
        "architecture_family": ARCH_DENSE,
        "low_rank_dim": 64,
        "residual_blocks": 1,
        "residual_expansion": 1,
        "stadniuk_batch_norm": False,
    }
    keys = (
        "hidden_widths",
        "latent_dim",
        "activation",
        "loss_name",
        "architecture_family",
        "low_rank_dim",
        "residual_blocks",
        "residual_expansion",
        "stadniuk_batch_norm",
    )
    for key in keys:
        left = checkpoint_config.get(key, defaults.get(key))
        right = current.get(key)
        if key == "loss_name":
            left = normalize_loss_name(str(left))
            right = normalize_loss_name(str(right))
        if key == "hidden_widths":
            left = tuple(left or ())
            right = tuple(right or ())
        if left != right:
            return False
    return True


def _load_checkpoint_into_run(
    path: str | Path,
    mode: str,
    model,
    optimizer,
    scheduler,
    config: TrainingConfig,
    input_dim: int,
    method: str,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
) -> tuple[int, float, _History]:
    checkpoint = _safe_torch_load(path)
    if checkpoint.get("archcon_checkpoint_format") != _CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format.")
    if int(checkpoint.get("input_dim", -1)) != int(input_dim):
        raise ValueError("Checkpoint input dimension does not match this GEO store.")
    checkpoint_config = checkpoint.get("config", {})
    if not _config_compatible(checkpoint_config, config):
        raise ValueError(
            "Checkpoint architecture/loss does not match the current controls. "
            "Load the checkpoint in the UI first to restore its settings."
        )
    model.load_state_dict(checkpoint["model_state"])
    if mode == CHECKPOINT_WEIGHTS:
        return 0, float("inf"), _History()
    if mode != CHECKPOINT_RESUME:
        raise ValueError(f"Unknown checkpoint mode: {mode}")
    if str(checkpoint.get("method", "")) != str(method):
        raise ValueError(
            "Exact resume requires the same stage-02 preprocessing matrix as the checkpoint. "
            f"Checkpoint: {checkpoint.get('method')}; current: {method}."
        )
    checkpoint_schedule = str(checkpoint_config.get("lr_schedule", LR_SCHEDULE_PLATEAU))
    if checkpoint_schedule != str(config.lr_schedule):
        raise ValueError(
            "Exact resume requires the same learning-rate scheduler as the checkpoint. "
            f"Checkpoint: {checkpoint_schedule}; current: {config.lr_schedule}."
        )
    checkpoint_train = np.asarray(checkpoint.get("train_rows", []), dtype=np.int64)
    checkpoint_validation = np.asarray(
        checkpoint.get("validation_rows", []), dtype=np.int64
    )
    if not np.array_equal(checkpoint_train, train_rows) or not np.array_equal(
        checkpoint_validation, validation_rows
    ):
        raise ValueError(
            "Exact resume requires the same train/validation split stored in the checkpoint. "
            "Use 'Load weights only' to initialize a new run on a different split."
        )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return (
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        _History.from_dict(checkpoint.get("history", {})),
    )



def _auxiliary_label(loss_name: str) -> str | None:
    if loss_name == LOSS_DVIB:
        return "KL divergence · unweighted"
    if loss_name == LOSS_MMD:
        return "MMD² to N(0,I) · unweighted"
    if loss_name == LOSS_COSINE:
        return "cosine distance"
    return None

def _validation_pass(
    model,
    matrix: np.ndarray,
    rows: np.ndarray,
    config: TrainingConfig,
    device,
    *,
    collect_latent: bool,
    autocast_enabled: bool = False,
    autocast_dtype=None,
    epoch: int | None = None,
    regularization_model=None,
    objective_function=None,
    l2_penalty_function=None,
) -> tuple[float, float, float, float, np.ndarray | None]:
    _require_torch()
    model.eval()
    objective_total = 0.0
    auxiliary_total = 0.0
    objective_weight = 0
    squared_error = 0.0
    value_count = 0
    sum_y = 0.0
    sum_y2 = 0.0
    latent_parts: list[np.ndarray] = []
    mask_rng = (
        np.random.default_rng(int(config.seed) + 104_729)
        if config.loss_name == LOSS_MASKED
        else None
    )

    with torch.no_grad():
        for _, batch_np in _iter_numpy_batches(
            matrix,
            rows,
            int(config.batch_size),
            background_prefetch=bool(config.background_prefetch),
        ):
            x = _tensor_from_numpy(batch_np, device)
            model_input = x
            reconstruction_mask = None
            if config.loss_name == LOSS_MASKED:
                model_input, reconstruction_mask = _masked_reconstruction_input(
                    x, float(config.mask_fraction), numpy_rng=mask_rng
                )
            with _autocast_context(device, autocast_enabled, autocast_dtype):
                reconstruction, z, mu, logvar = model(model_input, sample=False)
                objective_call = objective_function or _objective
                objective, _, auxiliary = objective_call(
                    x,
                    reconstruction,
                    z,
                    mu,
                    logvar,
                    config,
                    epoch=epoch,
                    reconstruction_mask=reconstruction_mask,
                )
                if float(config.l2_lambda) > 0.0:
                    penalty_call = l2_penalty_function or _explicit_l2_penalty
                    objective = objective + penalty_call(
                        regularization_model if regularization_model is not None else model,
                        config,
                    )
                if config.loss_name == LOSS_MASKED:
                    clean_reconstruction, clean_z, _, _ = model(x, sample=False)
                else:
                    clean_reconstruction, clean_z = reconstruction, z
            batch_size = int(x.shape[0])
            objective_total += float(objective.detach().cpu()) * batch_size
            auxiliary_total += float(auxiliary.detach().float().cpu()) * batch_size
            objective_weight += batch_size

            # Keep validation MSE/R² comparable across objectives: for masked
            # pretraining they are measured on clean input, while the selected
            # objective above remains masked-position prediction error.
            residual = clean_reconstruction - x
            squared_error += float(torch.sum(residual * residual).detach().cpu())
            sum_y += float(torch.sum(x).detach().cpu())
            sum_y2 += float(torch.sum(x * x).detach().cpu())
            value_count += int(x.numel())
            if collect_latent:
                latent_parts.append(
                    clean_z.detach().cpu().numpy().astype(np.float32, copy=False)
                )

    val_objective = objective_total / max(objective_weight, 1)
    val_auxiliary = auxiliary_total / max(objective_weight, 1)
    mse = squared_error / max(value_count, 1)
    denominator = sum_y2 - (sum_y * sum_y / max(value_count, 1))
    r2 = 1.0 - squared_error / denominator if denominator > 0 else float("nan")
    latents = np.concatenate(latent_parts, axis=0) if latent_parts else None
    return val_objective, mse, r2, val_auxiliary, latents



def _clean_reconstruction_metrics(
    model,
    matrix: np.ndarray,
    config: TrainingConfig,
    device,
    *,
    autocast_enabled: bool = False,
    autocast_dtype=None,
) -> tuple[float, float]:
    """Compute clean-input MSE/R² for an evaluation-only matrix.

    This is used for supervised-dataset monitoring.  It never contributes to convergence checks,
    learning-rate scheduling, or best-checkpoint selection.
    """
    _require_torch()
    model.eval()
    squared_error = 0.0
    value_count = 0
    sum_y = 0.0
    sum_y2 = 0.0
    rows = np.arange(int(matrix.shape[0]), dtype=np.int64)
    with torch.no_grad():
        for _, batch_np in _iter_numpy_batches(
            matrix,
            rows,
            int(config.batch_size),
            background_prefetch=bool(config.background_prefetch),
        ):
            x = _tensor_from_numpy(batch_np, device)
            with _autocast_context(device, autocast_enabled, autocast_dtype):
                reconstruction, _, _, _ = model(x, sample=False)
            residual = reconstruction - x
            squared_error += float(torch.sum(residual * residual).detach().cpu())
            sum_y += float(torch.sum(x).detach().cpu())
            sum_y2 += float(torch.sum(x * x).detach().cpu())
            value_count += int(x.numel())
    mse = squared_error / max(value_count, 1)
    denominator = sum_y2 - (sum_y * sum_y / max(value_count, 1))
    r2 = 1.0 - squared_error / denominator if denominator > 0 else float("nan")
    return mse, r2

def _compute_latent_pca(latents: np.ndarray, max_samples: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    values = np.asarray(latents, dtype=np.float32)
    if len(values) > int(max_samples):
        positions = np.unique(
            np.linspace(0, len(values) - 1, int(max_samples), dtype=np.int64)
        )
        values = values[positions]
    if values.shape[1] == 1:
        return np.column_stack([values[:, 0], np.zeros(len(values), dtype=np.float32)])
    coordinates = PCA(n_components=2, svd_solver="full").fit_transform(values)
    return coordinates.astype(np.float32, copy=False)


def _snapshot(
    *,
    run_id: str,
    status: str,
    running: bool,
    done: bool,
    history: _History,
    epoch: int,
    batch: int,
    n_batches: int,
    latent_pca: np.ndarray | None,
    latent_pca_epoch: int | None,
    latest_checkpoint: str | None,
    best_checkpoint: str | None,
    stopped: bool = False,
) -> TrainingUpdate:
    return TrainingUpdate(
        run_id=run_id,
        status=status,
        running=running,
        done=done,
        stopped=stopped,
        epoch=int(epoch),
        batch=int(batch),
        n_batches=int(n_batches),
        train_x=list(history.train_x),
        train_loss=list(history.train_loss),
        train_epoch=list(history.train_epoch),
        train_epoch_loss=list(history.train_epoch_loss),
        val_epoch=list(history.val_epoch),
        val_loss=list(history.val_loss),
        val_mse=list(history.val_mse),
        val_r2=list(history.val_r2),
        val_aux=list(history.val_aux),
        ikem_mse=list(history.ikem_mse),
        ikem_r2=list(history.ikem_r2),
        aux_label=history.aux_label,
        learning_rates=list(history.learning_rates),
        latent_pca=latent_pca,
        latent_pca_epoch=latent_pca_epoch,
        latest_checkpoint=latest_checkpoint,
        best_checkpoint=best_checkpoint,
    )



@contextmanager
def _row_major_cache_build_lock(cache_path: Path):
    """Serialize creation of a shared row-major cache across PBS/local processes."""
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _existing_row_major_cache(matrix: np.ndarray, output_root: str | Path):
    """Return a valid row-major training cache, or its destination path."""
    if bool(matrix.flags.c_contiguous):
        return matrix, None

    source_name = "expression"
    source_path = getattr(matrix, "filename", None)
    if source_path:
        source_path = Path(source_path).expanduser().resolve()
        source_name = source_path.stem
        # Canonical stores live at data/<STORE>/<matrix>.npy. Keep the potentially
        # multi-GB row-major cache beside the data stores even though checkpoints
        # are saved in the launch-directory ./models workspace.
        cache_dir = source_path.parent.parent / "training_cache"
    else:
        # Fallback for synthetic/in-memory matrices used outside the canonical store.
        cache_dir = Path(output_root).expanduser().resolve().parent / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source_name}_row_major.npy"
    complete_path = cache_path.with_suffix(cache_path.suffix + ".complete")

    if cache_path.is_file() and complete_path.is_file():
        try:
            cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            source_newer = (
                source_path is not None
                and source_path.is_file()
                and source_path.stat().st_mtime > complete_path.stat().st_mtime
            )
            if (
                cached.shape == matrix.shape
                and cached.dtype == np.dtype("float32")
                and bool(cached.flags.c_contiguous)
                and not source_newer
            ):
                return cached, cache_path
        except Exception:
            pass
        cache_path.unlink(missing_ok=True)
        complete_path.unlink(missing_ok=True)

    return None, cache_path



def train_autoencoder_stream(
    matrix: np.ndarray,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    config: TrainingConfig,
    output_root: str | Path,
    *,
    method: str,
    checkpoint_path: str | Path | None = None,
    checkpoint_mode: str = CHECKPOINT_WEIGHTS,
    evaluation_matrix: np.ndarray | None = None,
    evaluation_label: str = "Supervised dataset",
    model_factory=None,
    objective_function=None,
    optimizer_factory=None,
    scheduler_factory=None,
    l2_penalty_function=None,
    run_directory: str | Path | None = None,
):
    """Yield live training snapshots while reading expression batches on demand.

    The expensive GEO matrix remains memory-mapped. Validation is evaluated at
    the configured epoch cadence. When requested, validation latent coordinates are handed to a
    one-worker background executor for PCA so training can continue while the
    two-dimensional diagnostic is computed. When ``run_directory`` is supplied,
    every epoch checkpoint is atomically written directly into that exact
    directory rather than a generated child directory.
    """
    _require_torch()
    if matrix.ndim != 2:
        raise ValueError("Training matrix must be two-dimensional.")
    input_dim = int(matrix.shape[1])
    config.validate(input_dim)
    if evaluation_matrix is not None:
        if evaluation_matrix.ndim != 2 or int(evaluation_matrix.shape[1]) != input_dim:
            raise ValueError(
                f"{evaluation_label} evaluation matrix must have the same {input_dim:,} probes "
                "as the GEO training matrix."
            )

    train_rows = np.asarray(train_rows, dtype=np.int64)
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    if len(train_rows) < 1 or len(validation_rows) < 1:
        raise ValueError("Both train and validation sets must contain samples.")
    all_rows = np.concatenate([train_rows, validation_rows])
    if all_rows.min() < 0 or all_rows.max() >= matrix.shape[0]:
        raise ValueError("Split row index is outside the selected expression matrix.")
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError("Train/validation rows overlap or contain duplicates.")

    run_id = uuid.uuid4().hex[:12]
    stop_event = _register_run(run_id)
    if run_directory is None:
        output_dir = Path(output_root).expanduser().resolve() / f"geo_ae_{run_id}"
    else:
        output_dir = Path(run_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = _History(aux_label=_auxiliary_label(config.loss_name))
    start_epoch = 0
    best_val_loss = float("inf")
    latest_checkpoint: str | None = None
    best_checkpoint: str | None = None
    best_epoch = 0
    convergence_deltas: list[float] = []
    converged = False
    last_relative_delta: float | None = None
    n_batches = int(np.ceil(len(train_rows) / int(config.batch_size)))
    latest_pca: np.ndarray | None = None
    latest_pca_epoch: int | None = None
    pca_future: Future | None = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archcon-latent-pca")
    device = None

    try:
        cached_matrix, cache_path = _existing_row_major_cache(matrix, output_root)
        if cached_matrix is None:
            with _row_major_cache_build_lock(cache_path):
                # Another local/PBS process may have built the shared cache while this
                # run waited for the lock, so always re-check before doing multi-GB I/O.
                cached_matrix, cache_path = _existing_row_major_cache(matrix, output_root)
                if cached_matrix is not None:
                    matrix = cached_matrix
                else:
                    required = int(matrix.size) * np.dtype("float32").itemsize
                    free = shutil.disk_usage(cache_path.parent).free
                    if free < int(required * 1.05):
                        raise OSError(
                            "The selected matrix is column-major and needs a one-time row-major "
                            f"training cache (~{required / 1024**3:.2f} GiB), but there is not "
                            f"enough free disk space in {cache_path.parent}."
                        )
                    partial = cache_path.with_suffix(cache_path.suffix + ".part")
                    partial.unlink(missing_ok=True)
                    cache_path.with_suffix(cache_path.suffix + ".complete").unlink(
                        missing_ok=True
                    )
                    destination = np.lib.format.open_memmap(
                        partial,
                        mode="w+",
                        dtype=np.float32,
                        shape=matrix.shape,
                        fortran_order=False,
                    )
                    block_columns = 2048
                    total_blocks = int(np.ceil(matrix.shape[1] / block_columns))
                    for block_index, column_start in enumerate(
                        range(0, matrix.shape[1], block_columns), start=1
                    ):
                        if stop_event.is_set():
                            del destination
                            partial.unlink(missing_ok=True)
                            yield _snapshot(
                                run_id=run_id,
                                status=(
                                    "⏹ **Stopped before training** while preparing the "
                                    "row-major cache."
                                ),
                                running=False,
                                done=True,
                                stopped=True,
                                history=history,
                                epoch=0,
                                batch=0,
                                n_batches=n_batches,
                                latent_pca=None,
                                latent_pca_epoch=None,
                                latest_checkpoint=None,
                                best_checkpoint=None,
                            )
                            return
                        column_stop = min(column_start + block_columns, matrix.shape[1])
                        destination[:, column_start:column_stop] = matrix[
                            :, column_start:column_stop
                        ]
                        if block_index % 4 == 0 or block_index == total_blocks:
                            destination.flush()
                        yield _snapshot(
                            run_id=run_id,
                            status=(
                                "🧱 **Preparing one-time row-major training cache** for the "
                                f"selected column-major matrix · block {block_index}/{total_blocks}. "
                                "Future runs reuse this cache."
                            ),
                            running=True,
                            done=False,
                            history=history,
                            epoch=0,
                            batch=0,
                            n_batches=n_batches,
                            latent_pca=None,
                            latent_pca_epoch=None,
                            latest_checkpoint=None,
                            best_checkpoint=None,
                        )
                    destination.flush()
                    del destination
                    partial.replace(cache_path)
                    cache_path.with_suffix(cache_path.suffix + ".complete").write_text(
                        f"shape={matrix.shape}; dtype=float32; "
                        f"source={getattr(matrix, 'filename', '')}\n",
                        encoding="utf-8",
                    )
                    matrix = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        elif cache_path is not None:
            matrix = cached_matrix

        _seed_everything(config.seed, deterministic=bool(config.deterministic))
        device = resolve_device(config.device)
        base_model = (
            model_factory(input_dim, config)
            if model_factory is not None
            else build_autoencoder(input_dim, config)
        ).to(device)
        optimizer = (
            optimizer_factory(base_model, config)
            if optimizer_factory is not None
            else _optimizer(base_model, config)
        )
        scheduler = (
            scheduler_factory(optimizer, config)
            if scheduler_factory is not None
            else _scheduler(optimizer, config)
        )

        if checkpoint_path:
            start_epoch, best_val_loss, history = _load_checkpoint_into_run(
                checkpoint_path,
                checkpoint_mode,
                base_model,
                optimizer,
                scheduler,
                config,
                input_dim,
                method,
                train_rows,
                validation_rows,
            )
            history.aux_label = _auxiliary_label(config.loss_name)

        model = base_model
        compiled = False
        if bool(config.compile_model) and hasattr(torch, "compile"):
            try:
                model = torch.compile(base_model)
                compiled = True
            except Exception:
                model = base_model

        autocast_enabled, autocast_dtype, needs_scaler = _autocast_settings(
            device, config.precision
        )
        scaler = _make_grad_scaler(device, needs_scaler)
        precision_label = (
            str(autocast_dtype).replace("torch.", "") if autocast_enabled else "float32"
        )

        best_epoch = start_epoch
        rng = np.random.default_rng(int(config.seed) + start_epoch)

        source_note = f" · loaded {checkpoint_mode.lower()}" if checkpoint_path else ""
        speed_notes = [precision_label]
        if config.background_prefetch:
            speed_notes.append("1-batch async prefetch")
        if compiled:
            speed_notes.append("torch.compile")
        if not config.deterministic:
            speed_notes.append("fast non-deterministic kernels")
        yield _snapshot(
            run_id=run_id,
            status=(
                f"▶️ **Training started** on `{device}` · {len(train_rows):,} train / "
                f"{len(validation_rows):,} validation · **{config.architecture_family}** · "
                f"{config.loss_name}{source_note} · {' · '.join(speed_notes)}."
            ),
            running=True,
            done=False,
            history=history,
            epoch=start_epoch,
            batch=0,
            n_batches=n_batches,
            latent_pca=latest_pca,
            latent_pca_epoch=latest_pca_epoch,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )

        if start_epoch >= int(config.epochs):
            yield _snapshot(
                run_id=run_id,
                status=(
                    f"ℹ️ Checkpoint is already at epoch {start_epoch}, which is not below the "
                    f"requested total of {config.epochs} epochs. Increase the epoch limit to continue."
                ),
                running=False,
                done=True,
                history=history,
                epoch=start_epoch,
                batch=0,
                n_batches=n_batches,
                latent_pca=latest_pca,
                latent_pca_epoch=latest_pca_epoch,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
            )
            return

        for epoch in range(start_epoch + 1, int(config.epochs) + 1):
            if stop_event.is_set():
                break

            model.train()
            order = train_rows[rng.permutation(len(train_rows))]
            running_loss_sum = 0.0
            running_samples = 0
            ema_loss: float | None = None
            epoch_started = time.perf_counter()

            for batch_index, (_, batch_np) in enumerate(
                _iter_numpy_batches(
                    matrix,
                    order,
                    int(config.batch_size),
                    background_prefetch=bool(config.background_prefetch),
                ),
                start=1,
            ):
                if stop_event.is_set():
                    break
                x = _tensor_from_numpy(batch_np, device)
                model_input = x
                reconstruction_mask = None
                if config.loss_name == LOSS_MASKED:
                    model_input, reconstruction_mask = _masked_reconstruction_input(
                        x, float(config.mask_fraction)
                    )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, autocast_enabled, autocast_dtype):
                    reconstruction, z, mu, logvar = model(model_input, sample=True)
                    objective_call = objective_function or _objective
                    objective, _, _ = objective_call(
                        x,
                        reconstruction,
                        z,
                        mu,
                        logvar,
                        config,
                        epoch=epoch,
                        reconstruction_mask=reconstruction_mask,
                    )
                    if float(config.l2_lambda) > 0.0:
                        penalty_call = l2_penalty_function or _explicit_l2_penalty
                        objective = objective + penalty_call(base_model, config)
                if not torch.isfinite(objective):
                    raise FloatingPointError(
                        "Training loss became non-finite. Try a smaller learning rate or a "
                        "different normalization/loss."
                    )

                if scaler is not None:
                    scaler.scale(objective).backward()
                    if float(config.gradient_clip) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(), max_norm=float(config.gradient_clip)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    objective.backward()
                    if float(config.gradient_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(), max_norm=float(config.gradient_clip)
                        )
                    optimizer.step()

                current_batch_size = int(x.shape[0])
                batch_objective = float(objective.detach().float().cpu())
                running_loss_sum += batch_objective * current_batch_size
                running_samples += current_batch_size
                ema_loss = (
                    batch_objective
                    if ema_loss is None
                    else 0.90 * ema_loss + 0.10 * batch_objective
                )

                should_update = (
                    batch_index == 1
                    or batch_index == n_batches
                    or batch_index % int(config.ui_update_batches) == 0
                )
                if should_update:
                    fractional_epoch = (epoch - 1) + batch_index / max(n_batches, 1)
                    history.train_x.append(float(fractional_epoch))
                    history.train_loss.append(float(ema_loss))
                    if pca_future is not None and pca_future.done():
                        try:
                            latest_pca = pca_future.result()
                            latest_pca_epoch = history.val_epoch[-1] if history.val_epoch else None
                        except Exception:
                            pass
                        pca_future = None
                    lr = float(optimizer.param_groups[0]["lr"])
                    elapsed = max(time.perf_counter() - epoch_started, 1e-9)
                    throughput = running_samples / elapsed
                    yield _snapshot(
                        run_id=run_id,
                        status=(
                            f"🏃 **Epoch {epoch}/{config.epochs}** · batch "
                            f"{batch_index}/{n_batches} · running loss "
                            f"**{history.train_loss[-1]:.6g}** · lr `{lr:.3g}` · "
                            f"**{throughput:.1f} samples/s**"
                        ),
                        running=True,
                        done=False,
                        history=history,
                        epoch=epoch,
                        batch=batch_index,
                        n_batches=n_batches,
                        latent_pca=latest_pca,
                        latent_pca_epoch=latest_pca_epoch,
                        latest_checkpoint=latest_checkpoint,
                        best_checkpoint=best_checkpoint,
                    )

            if running_samples > 0:
                history.train_epoch.append(epoch)
                history.train_epoch_loss.append(running_loss_sum / running_samples)

            if stop_event.is_set():
                payload = _checkpoint_payload(
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    input_dim=input_dim,
                    method=method,
                    epoch=epoch - 1,
                    best_val_loss=best_val_loss,
                    history=history,
                    run_id=run_id,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                )
                latest_checkpoint = _save_checkpoint(output_dir / "stopped.pt", payload)
                break

            should_validate = (
                epoch == 1
                or epoch == int(config.epochs)
                or epoch % int(config.validation_every_epochs) == 0
            )
            val_loss = None
            val_mse = None
            val_r2 = None
            latents = None
            improved = False

            if should_validate:
                collect_latent = (
                    epoch == 1 or epoch % int(config.latent_pca_every_epochs) == 0
                )
                val_loss, val_mse, val_r2, val_aux, latents = _validation_pass(
                    model,
                    matrix,
                    validation_rows,
                    config,
                    device,
                    collect_latent=collect_latent,
                    autocast_enabled=autocast_enabled,
                    autocast_dtype=autocast_dtype,
                    epoch=epoch,
                    regularization_model=base_model,
                    objective_function=objective_function,
                    l2_penalty_function=l2_penalty_function,
                )
                if config.lr_schedule == LR_SCHEDULE_PLATEAU:
                    scheduler.step(val_loss)
                current_lr = float(optimizer.param_groups[0]["lr"])
                previous_val_loss = history.val_loss[-1] if history.val_loss else None
                history.val_epoch.append(epoch)
                history.val_loss.append(float(val_loss))
                history.val_mse.append(float(val_mse))
                history.val_r2.append(float(val_r2))
                history.val_aux.append(float(val_aux))
                history.learning_rates.append(current_lr)
                if evaluation_matrix is not None:
                    ikem_mse, ikem_r2 = _clean_reconstruction_metrics(
                        model,
                        evaluation_matrix,
                        config,
                        device,
                        autocast_enabled=autocast_enabled,
                        autocast_dtype=autocast_dtype,
                    )
                    history.ikem_mse.append(float(ikem_mse))
                    history.ikem_r2.append(float(ikem_r2))

                improved = val_loss < best_val_loss - 1e-12
                if improved:
                    best_val_loss = float(val_loss)
                    best_epoch = epoch

                # Convergence is a numerical criterion, not a patience rule:
                # compare successive validation objectives and only allow a stop
                # once the decayed learning rate is already at its floor.
                if previous_val_loss is not None:
                    denominator = max(abs(float(previous_val_loss)), 1e-12)
                    last_relative_delta = (
                        abs(float(val_loss) - float(previous_val_loss)) / denominator
                    )
                    convergence_deltas.append(float(last_relative_delta))
                    window = int(config.convergence_window)
                    if len(convergence_deltas) > window:
                        del convergence_deltas[:-window]
                else:
                    last_relative_delta = None

            if config.lr_schedule in {LR_SCHEDULE_COSINE, LR_SCHEDULE_CONSTANT}:
                scheduler.step()

            payload = _checkpoint_payload(
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                input_dim=input_dim,
                method=method,
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                run_id=run_id,
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
            latest_checkpoint = _save_checkpoint(output_dir / "latest.pt", payload)
            if improved:
                best_checkpoint = _save_checkpoint(output_dir / "best.pt", payload)

            if latents is not None and (pca_future is None or pca_future.done()):
                if pca_future is not None and pca_future.done():
                    try:
                        latest_pca = pca_future.result()
                        latest_pca_epoch = history.val_epoch[-2] if len(history.val_epoch) > 1 else 1
                    except Exception:
                        pass
                pca_future = executor.submit(
                    _compute_latent_pca,
                    latents,
                    int(config.latent_pca_max_samples),
                )

            epoch_seconds = max(time.perf_counter() - epoch_started, 1e-9)
            if should_validate:
                aux_text = ""
                if config.loss_name == LOSS_DVIB:
                    beta_eff = _effective_kl_beta(config, epoch)
                    aux_text = (
                        f" · KL **{val_aux:.6g}** · effective β **{beta_eff:.3g}**"
                    )
                elif config.loss_name == LOSS_MMD:
                    aux_text = (
                        f" · MMD² **{val_aux:.6g}** · λ **{config.mmd_weight:.3g}**"
                    )
                elif config.loss_name == LOSS_COSINE:
                    aux_text = (
                        f" · cosine distance **{val_aux:.6g}** · λ **{config.cosine_weight:.3g}**"
                    )
                elif config.loss_name == LOSS_MASKED:
                    aux_text = (
                        f" · mask **{100.0 * config.mask_fraction:.1f}%** · clean-input MSE shown separately"
                    )
                ikem_text = ""
                if evaluation_matrix is not None and history.ikem_mse:
                    ikem_text = (
                        f" · {evaluation_label} diagnostic MSE **{history.ikem_mse[-1]:.6g}** "
                        f"/ R² **{history.ikem_r2[-1]:.4f}** (not used for selection)"
                    )
                status = (
                    f"✅ **Epoch {epoch}/{config.epochs} complete** in {epoch_seconds:.1f}s · "
                    f"validation objective **{val_loss:.6g}** · MSE **{val_mse:.6g}** · "
                    f"R² **{val_r2:.4f}**{aux_text}{ikem_text} · best epoch **{best_epoch}**"
                    + (
                        f" · relative Δ **{last_relative_delta:.3g}**"
                        if last_relative_delta is not None
                        else ""
                    )
                )
            else:
                next_validation = min(
                    int(config.epochs),
                    epoch + (int(config.validation_every_epochs) - epoch % int(config.validation_every_epochs)),
                )
                status = (
                    f"✅ **Epoch {epoch}/{config.epochs} complete** in {epoch_seconds:.1f}s · "
                    f"validation skipped for speed; next scheduled at epoch **{next_validation}**."
                )

            yield _snapshot(
                run_id=run_id,
                status=status,
                running=True,
                done=False,
                history=history,
                epoch=epoch,
                batch=n_batches,
                n_batches=n_batches,
                latent_pca=latest_pca,
                latent_pca_epoch=latest_pca_epoch,
                latest_checkpoint=latest_checkpoint,
                best_checkpoint=best_checkpoint,
            )

            if should_validate and convergence_deltas:
                lr_used = float(current_lr)
                lr_floor = float(config.min_learning_rate)
                at_lr_floor = lr_used <= lr_floor * (1.0 + 1e-6) + 1e-15
                window = int(config.convergence_window)
                stable = (
                    len(convergence_deltas) >= window
                    and max(convergence_deltas[-window:])
                    <= float(config.convergence_tolerance)
                )
                if at_lr_floor and stable:
                    converged = True
                    break

        if pca_future is not None and pca_future.done():
            try:
                latest_pca = pca_future.result()
                latest_pca_epoch = history.val_epoch[-1] if history.val_epoch else None
            except Exception:
                pass

        if stop_event.is_set() and latest_checkpoint is None:
            stopped_epoch = history.val_epoch[-1] if history.val_epoch else start_epoch
            payload = _checkpoint_payload(
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                input_dim=input_dim,
                method=method,
                epoch=stopped_epoch,
                best_val_loss=best_val_loss,
                history=history,
                run_id=run_id,
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
            latest_checkpoint = _save_checkpoint(output_dir / "stopped.pt", payload)

        if stop_event.is_set():
            final_status = (
                "⏹ **Training stopped by user.** A recovery checkpoint was saved; "
                "controls are unlocked again."
            )
            stopped = True
        else:
            final_epoch = history.val_epoch[-1] if history.val_epoch else start_epoch
            reason = "converged at minimum learning rate" if converged else "maximum epoch limit"
            final_status = (
                f"🏁 **Training finished** at epoch {final_epoch} ({reason}). "
                f"Best validation objective: **{best_val_loss:.6g}** at epoch **{best_epoch}**."
            )
            stopped = False

        yield _snapshot(
            run_id=run_id,
            status=final_status,
            running=False,
            done=True,
            stopped=stopped,
            history=history,
            epoch=history.val_epoch[-1] if history.val_epoch else start_epoch,
            batch=0,
            n_batches=n_batches,
            latent_pca=latest_pca,
            latent_pca_epoch=latest_pca_epoch,
            latest_checkpoint=latest_checkpoint,
            best_checkpoint=best_checkpoint,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        _clear_run(run_id)
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()


def _best_validation_index(update: TrainingUpdate) -> int | None:
    if not update.val_epoch or not update.val_loss:
        return None
    values = np.asarray(update.val_loss, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    finite_indices = np.flatnonzero(finite)
    return int(finite_indices[np.argmin(values[finite])])


def plot_training_history(update: TrainingUpdate):
    """Plot comparable epoch objectives plus a faint within-epoch training EMA."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if update.train_x:
        ax.plot(
            update.train_x,
            update.train_loss,
            linewidth=1.2,
            alpha=0.35,
            label="train · batch EMA",
        )
    if update.train_epoch:
        ax.plot(
            update.train_epoch,
            update.train_epoch_loss,
            marker="o",
            linewidth=2.1,
            label="train · epoch mean",
        )
    if update.val_epoch:
        ax.plot(
            update.val_epoch,
            update.val_loss,
            marker="s",
            linewidth=2.1,
            label="validation · same objective",
        )
        best_index = _best_validation_index(update)
        if best_index is not None:
            ax.scatter(
                [update.val_epoch[best_index]],
                [update.val_loss[best_index]],
                s=190,
                facecolors="none",
                edgecolors="black",
                linewidths=2.2,
                label="best checkpoint epoch",
                zorder=6,
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("selected training objective · log scale")
    ax.set_title("Reconstruction / latent objective")
    ax.set_yscale("log", nonpositive="clip")
    ax.grid(alpha=0.18, which="both")
    if update.train_x or update.val_epoch:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_validation_metrics(update: TrainingUpdate):
    """Plot validation metrics plus evaluation-only supervised-dataset diagnostics."""
    show_aux = bool(update.aux_label and update.val_aux and any(np.isfinite(update.val_aux)))
    n_rows = 3 if show_aux else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.6, 2.65 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    mse_ax = axes[0]
    r2_ax = axes[1]

    if update.val_epoch:
        mse_color = "tab:blue"
        r2_color = "tab:orange"
        mse_ax.plot(
            update.val_epoch,
            update.val_mse,
            marker="o",
            linewidth=2.0,
            color=mse_color,
            label="GEO validation MSE · lower is better",
        )
        r2_ax.plot(
            update.val_epoch,
            update.val_r2,
            marker="s",
            linewidth=2.0,
            color=r2_color,
            label="GEO validation global R² · higher is better",
        )

        if len(update.ikem_mse) == len(update.val_epoch):
            mse_ax.plot(
                update.val_epoch,
                update.ikem_mse,
                linestyle="--",
                linewidth=1.8,
                label="Supervised MSE · diagnostic only",
            )
        if len(update.ikem_r2) == len(update.val_epoch):
            r2_ax.plot(
                update.val_epoch,
                update.ikem_r2,
                linestyle="--",
                linewidth=1.8,
                label="Supervised R² · diagnostic only",
            )

        mse_ax.set_ylabel("MSE", color=mse_color)
        mse_ax.tick_params(axis="y", labelcolor=mse_color)
        mse_ax.grid(alpha=0.18)
        mse_ax.legend(loc="best")
        r2_ax.set_ylabel("global R²", color=r2_color)
        r2_ax.tick_params(axis="y", labelcolor=r2_color)
        r2_ax.grid(alpha=0.18)
        r2_ax.legend(loc="best")

        best_index = _best_validation_index(update)
        if best_index is not None:
            epoch = update.val_epoch[best_index]
            for axis, values in ((mse_ax, update.val_mse), (r2_ax, update.val_r2)):
                axis.scatter(
                    [epoch],
                    [values[best_index]],
                    s=180,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2.0,
                    zorder=7,
                )
            if len(update.ikem_mse) == len(update.val_epoch):
                mse_ax.scatter(
                    [epoch], [update.ikem_mse[best_index]], s=150, facecolors="none",
                    edgecolors="black", linewidths=1.5, zorder=7
                )
            if len(update.ikem_r2) == len(update.val_epoch):
                r2_ax.scatter(
                    [epoch], [update.ikem_r2[best_index]], s=150, facecolors="none",
                    edgecolors="black", linewidths=1.5, zorder=7
                )

        if show_aux:
            aux_ax = axes[2]
            aux_color = "tab:purple"
            aux_ax.plot(
                update.val_epoch,
                update.val_aux,
                marker="^",
                linewidth=2.0,
                color=aux_color,
                label=f"{update.aux_label} · regularizer/aux term",
            )
            aux_ax.set_ylabel(update.aux_label, color=aux_color)
            aux_ax.tick_params(axis="y", labelcolor=aux_color)
            aux_ax.grid(alpha=0.18)
            aux_ax.legend(loc="best")
            if best_index is not None and best_index < len(update.val_aux):
                aux_ax.scatter(
                    [update.val_epoch[best_index]],
                    [update.val_aux[best_index]],
                    s=180,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2.0,
                    zorder=7,
                )

        axes[-1].set_xlabel("epoch")
        if len(update.val_epoch) < 3:
            mse_ax.text(
                0.02,
                0.93,
                "Only a few validation checks so far — straight segments are expected.",
                transform=mse_ax.transAxes,
                va="top",
                fontsize=9,
                alpha=0.72,
            )
    else:
        for ax in axes:
            ax.set_axis_off()
        mse_ax.text(
            0.5,
            0.5,
            "Validation metrics appear after the first validation pass",
            ha="center",
            va="center",
        )
    fig.suptitle(
        "Validation + supervised reconstruction diagnostic · circles mark selected best epoch"
    )
    fig.tight_layout()
    return fig


def plot_latent_pca(update: TrainingUpdate):
    """Plot the latest asynchronously-computed PCA of validation latent codes."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    if update.latent_pca is None or len(update.latent_pca) == 0:
        ax.text(
            0.5,
            0.5,
            "Validation latent PCA will appear after the first scheduled snapshot",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
    else:
        coords = np.asarray(update.latent_pca)
        ax.scatter(coords[:, 0], coords[:, 1], s=16, alpha=0.7)
        ax.set_xlabel("latent PC1")
        ax.set_ylabel("latent PC2")
        epoch_text = f" · epoch {update.latent_pca_epoch}" if update.latent_pca_epoch else ""
        ax.set_title(f"Validation latent space{epoch_text}")
        ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig

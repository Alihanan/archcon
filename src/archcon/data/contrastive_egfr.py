"""Fold-local reconstruction and soft-InfoNCE fine-tuning for eGFR.

The functions in this module always reload a molecular-pretraining checkpoint
and never alter the frozen CEL/RMA transformation.  Clinical variables define
only a soft neighbourhood distribution in latent space; eGFR values are not
used by the neural loss.  The downstream longitudinal mixed model is
responsible for selecting candidates using inner-validation eGFR RMSE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .downstream import ValidationCheckpoint, _model_from_record
from .pretraining import LOSS_MASKED, LOSS_MSE


CONTRASTIVE_NONE = "none"
CONTRASTIVE_SOFT_INFONCE = "soft_infonce"
CONTRASTIVE_MODES = (CONTRASTIVE_NONE, CONTRASTIVE_SOFT_INFONCE)
STADNIUK_CONTRASTIVE_COLUMNS = (
    "Donor_age",
    "AKI",
    "don_bio_cv",
    "don_bio_ah",
    "don_bio_ifta",
    "DCD",
    "ECD_1",
    "BMI_donor",
)


@dataclass(frozen=True)
class ContrastiveCandidate:
    """One inner-validation candidate; ``none`` ignores weight/temperature."""

    mode: str
    weight: float = 0.0
    temperature: float = 0.1

    def validate(self) -> None:
        if self.mode not in CONTRASTIVE_MODES:
            raise ValueError(f"Unknown contrastive mode: {self.mode}")
        if self.mode == CONTRASTIVE_NONE:
            if not math.isclose(float(self.weight), 0.0):
                raise ValueError("The reconstruction-only candidate must have weight zero.")
        elif float(self.weight) <= 0.0:
            raise ValueError("soft_infonce requires a positive contrastive weight.")
        if float(self.temperature) <= 0.0:
            raise ValueError("InfoNCE temperature must be positive.")

    @property
    def candidate_id(self) -> str:
        if self.mode == CONTRASTIVE_NONE:
            return CONTRASTIVE_NONE
        return (
            f"soft_infonce_w{_number_slug(self.weight)}"
            f"_t{_number_slug(self.temperature)}"
        )


@dataclass(frozen=True)
class FineTuneControls:
    """Fixed controls shared by all candidate losses."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    projection_dim: int = 16
    gradient_clip: float = 1.0
    deterministic: bool = True

    def validate(self) -> None:
        if int(self.epochs) < 1:
            raise ValueError("Fine-tuning epochs must be positive.")
        if int(self.batch_size) < 2:
            raise ValueError("Fine-tuning batch size must be at least two.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("Fine-tuning learning rate must be positive.")
        if float(self.weight_decay) < 0.0:
            raise ValueError("Fine-tuning weight decay cannot be negative.")
        if int(self.projection_dim) < 1:
            raise ValueError("Projection dimension must be positive.")
        if float(self.gradient_clip) < 0.0:
            raise ValueError("Gradient clipping cannot be negative.")


@dataclass(frozen=True)
class ClinicalGeometry:
    """Train-only clinical scaling and deterministic Gaussian bandwidth."""

    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    bandwidth: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.medians):
            raise ValueError("Clinical matrix has the wrong number of columns.")
        imputed = np.where(np.isnan(array), self.medians, array)
        transformed = (imputed - self.means) / self.scales
        if not np.isfinite(transformed).all():
            raise ValueError("Clinical geometry produced non-finite values.")
        return transformed.astype(np.float32)


@dataclass(frozen=True)
class FineTuneResult:
    """Latent features and an audit trail for one completed fine-tuning run."""

    z: np.ndarray
    reconstruction_history: tuple[float, ...]
    contrastive_history: tuple[float, ...]
    total_history: tuple[float, ...]
    bandwidth: float
    candidate: ContrastiveCandidate
    controls: FineTuneControls
    seed: int
    reconstruction_loss_name: str

    def metadata(self) -> dict[str, object]:
        return {
            "candidate": asdict(self.candidate),
            "candidate_id": self.candidate.candidate_id,
            "controls": asdict(self.controls),
            "seed": int(self.seed),
            "clinical_bandwidth": float(self.bandwidth),
            "reconstruction_history": list(self.reconstruction_history),
            "contrastive_history": list(self.contrastive_history),
            "total_history": list(self.total_history),
            "fine_tuning_uses_egfr_values": False,
            "fine_tuning_reconstruction_loss": self.reconstruction_loss_name,
            "projection_head_used_for_final_z": False,
        }


def _number_slug(value: float) -> str:
    text = format(float(value), ".12g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def contrastive_candidate_grid(
    weights: Iterable[float],
    temperatures: Iterable[float],
) -> tuple[ContrastiveCandidate, ...]:
    """Return reconstruction control plus the Cartesian soft-InfoNCE grid."""

    unique_weights = tuple(dict.fromkeys(float(value) for value in weights))
    unique_temperatures = tuple(dict.fromkeys(float(value) for value in temperatures))
    if not unique_weights or not unique_temperatures:
        raise ValueError("The soft-InfoNCE grid cannot be empty.")
    candidates = [ContrastiveCandidate(CONTRASTIVE_NONE, 0.0, 0.1)]
    candidates.extend(
        ContrastiveCandidate(CONTRASTIVE_SOFT_INFONCE, weight, temperature)
        for weight in unique_weights
        for temperature in unique_temperatures
    )
    for candidate in candidates:
        candidate.validate()
    return tuple(candidates)


def fit_clinical_geometry(reference: np.ndarray) -> ClinicalGeometry:
    """Fit imputation/scaling and median nonzero pairwise distance on train only."""

    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Clinical geometry needs at least two rows and one feature.")
    medians = np.nanmedian(values, axis=0)
    if np.isnan(medians).any():
        bad = np.flatnonzero(np.isnan(medians)).tolist()
        raise ValueError(f"Clinical features are completely missing: columns {bad}")
    imputed = np.where(np.isnan(values), medians, values)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (imputed - means) / scales
    difference = standardized[:, None, :] - standardized[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=2))
    positive = distances[np.triu_indices(len(standardized), k=1)]
    positive = positive[positive > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        bandwidth = 1.0
    return ClinicalGeometry(
        medians=medians,
        means=means,
        scales=scales,
        bandwidth=bandwidth,
    )


def soft_infonce_loss(
    latent,
    clinical,
    projection_head,
    *,
    temperature: float,
    bandwidth: float,
):
    """Cross-entropy from Gaussian clinical neighbours to latent similarities.

    For anchor ``i`` the target over every other sample is

    ``p(j|i) ∝ exp(-||c_i-c_j||² / (2 bandwidth²))``.

    The model distribution is the usual temperature-scaled cosine similarity
    over the projected latent vectors.  Unlike a hard positive/negative label,
    every off-diagonal pair receives a continuously valued target probability.
    """

    import torch
    from torch.nn import functional as F

    n = int(latent.shape[0])
    if n < 2:
        return latent.sum() * 0.0
    if float(temperature) <= 0.0 or float(bandwidth) <= 0.0:
        raise ValueError("InfoNCE temperature and clinical bandwidth must be positive.")
    projected = F.normalize(projection_head(latent).float(), dim=1, eps=1e-8)
    logits = projected @ projected.T / float(temperature)
    diagonal = torch.eye(n, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    log_probabilities = torch.log_softmax(logits, dim=1).masked_fill(diagonal, 0.0)

    clinical = clinical.float()
    squared_distance = torch.cdist(clinical, clinical, p=2).pow(2)
    similarities = torch.exp(
        -squared_distance / (2.0 * float(bandwidth) * float(bandwidth))
    ).masked_fill(diagonal, 0.0)
    # Extremely separated small batches can underflow.  A tiny off-diagonal
    # floor keeps the target a valid distribution without creating hard labels.
    similarities = torch.where(
        diagonal,
        torch.zeros_like(similarities),
        similarities.clamp_min(1e-12),
    )
    targets = similarities / similarities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(targets * log_probabilities).sum(dim=1).mean()


def _batch_indices(
    n_rows: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    order = rng.permutation(int(n_rows)).astype(np.int64, copy=False)
    batches = [order[start : start + int(batch_size)] for start in range(0, n_rows, batch_size)]
    # BatchNorm cannot update from a singleton.  Merge a final singleton into
    # the preceding batch while retaining exactly the same training membership.
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    return batches


def _encode(model, values: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
    import torch

    model.eval()
    result: list[np.ndarray] = []
    torch_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(values), int(batch_size)):
            block = np.ascontiguousarray(
                np.asarray(values[start : start + int(batch_size)], dtype=np.float32)
            )
            x = torch.from_numpy(block).to(torch_device)
            z = model.encode(x)
            result.append(z.detach().float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(result, axis=0), dtype=np.float32)


def fine_tune_checkpoint(
    record: ValidationCheckpoint,
    train_expression: np.ndarray,
    train_clinical: np.ndarray,
    clinical_reference: np.ndarray,
    encode_expression: np.ndarray,
    candidate: ContrastiveCandidate,
    controls: FineTuneControls,
    *,
    device: str,
    seed: int,
) -> FineTuneResult:
    """Reload, fine-tune, and encode without consulting validation/test outcomes."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("PyTorch is required; install archcon[training].") from exc

    candidate.validate()
    controls.validate()
    expression = np.ascontiguousarray(train_expression, dtype=np.float32)
    to_encode = np.ascontiguousarray(encode_expression, dtype=np.float32)
    if expression.ndim != 2 or to_encode.ndim != 2:
        raise ValueError("Fine-tuning expression inputs must be two-dimensional.")
    if expression.shape[1] != int(record.input_dim) or to_encode.shape[1] != int(record.input_dim):
        raise ValueError(
            f"Fine-tuning feature count differs from checkpoint input_dim={record.input_dim}."
        )
    if len(expression) != len(train_clinical):
        raise ValueError("Fine-tuning expression and clinical rows differ.")
    if len(expression) < 2:
        raise ValueError("Fine-tuning requires at least two samples.")
    if record.config.loss_name not in {LOSS_MSE, LOSS_MASKED}:
        raise ValueError(
            "Nested eGFR fine-tuning supports the final-paper MSE and masked-MSE "
            f"checkpoint objectives, not {record.config.loss_name!r}."
        )

    geometry = fit_clinical_geometry(clinical_reference)
    clinical = geometry.transform(train_clinical)
    torch_device = torch.device(device)
    model, loaded_input_dim = _model_from_record(record, device)
    if loaded_input_dim != expression.shape[1]:
        raise RuntimeError("Checkpoint input dimension changed while loading.")

    torch.manual_seed(int(seed) + 17)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 17)
    projection_head = None
    parameters = list(model.parameters())
    if candidate.mode == CONTRASTIVE_SOFT_INFONCE:
        projection_head = nn.Linear(
            int(record.latent_dim), int(controls.projection_dim), bias=True
        ).to(torch_device)
        parameters.extend(projection_head.parameters())

    # Reset the stochastic training stream after projection-head construction so
    # reconstruction batches and dropout masks match the ``none`` control.
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if controls.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older PyTorch
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    optimizer = torch.optim.Adam(
        parameters,
        lr=float(controls.learning_rate),
        weight_decay=float(controls.weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    reconstruction_history: list[float] = []
    contrastive_history: list[float] = []
    total_history: list[float] = []
    try:
        for _epoch in range(int(controls.epochs)):
            model.train()
            if projection_head is not None:
                projection_head.train()
            reconstruction_sum = 0.0
            contrastive_sum = 0.0
            total_sum = 0.0
            seen = 0
            for indices in _batch_indices(len(expression), controls.batch_size, rng):
                x = torch.from_numpy(expression[indices]).to(torch_device)
                clinical_batch = torch.from_numpy(clinical[indices]).to(torch_device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction_mask = None
                model_input = x
                if record.config.loss_name == LOSS_MASKED:
                    reconstruction_mask = (
                        torch.rand(x.shape, device=torch_device)
                        < float(record.config.mask_fraction)
                    )
                    if not bool(reconstruction_mask.any()):
                        reconstruction_mask = reconstruction_mask.clone()
                        reconstruction_mask.reshape(-1)[0] = True
                    model_input = x.masked_fill(reconstruction_mask, 0.0)
                reconstruction, latent, _, _ = model(model_input, sample=False)
                squared_error = (reconstruction.float() - x.float()).pow(2)
                if reconstruction_mask is None:
                    reconstruction_loss = squared_error.mean()
                else:
                    reconstruction_loss = squared_error.masked_select(
                        reconstruction_mask
                    ).mean()
                if projection_head is None:
                    contrastive_loss = reconstruction_loss.new_zeros(())
                else:
                    contrastive_loss = soft_infonce_loss(
                        latent,
                        clinical_batch,
                        projection_head,
                        temperature=float(candidate.temperature),
                        bandwidth=float(geometry.bandwidth),
                    )
                total_loss = reconstruction_loss + float(candidate.weight) * contrastive_loss
                total_loss.backward()
                if float(controls.gradient_clip) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        parameters, float(controls.gradient_clip)
                    )
                optimizer.step()
                count = len(indices)
                seen += count
                reconstruction_sum += float(reconstruction_loss.detach().cpu()) * count
                contrastive_sum += float(contrastive_loss.detach().cpu()) * count
                total_sum += float(total_loss.detach().cpu()) * count
            reconstruction_history.append(reconstruction_sum / seen)
            contrastive_history.append(contrastive_sum / seen)
            total_history.append(total_sum / seen)
        z = _encode(
            model,
            to_encode,
            device=device,
            batch_size=controls.batch_size,
        )
        del (
            x,
            model_input,
            clinical_batch,
            reconstruction_mask,
            reconstruction,
            latent,
            squared_error,
            reconstruction_loss,
            contrastive_loss,
            total_loss,
        )
    finally:
        del optimizer
        if projection_head is not None:
            del projection_head
        del model
        del parameters
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return FineTuneResult(
        z=z,
        reconstruction_history=tuple(reconstruction_history),
        contrastive_history=tuple(contrastive_history),
        total_history=tuple(total_history),
        bandwidth=float(geometry.bandwidth),
        candidate=candidate,
        controls=controls,
        seed=int(seed),
        reconstruction_loss_name=record.config.loss_name,
    )


def checkpoint_identity(record: ValidationCheckpoint) -> dict[str, object]:
    """Small stable identity used in resumable per-candidate cache metadata."""

    path = Path(record.path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "run": record.run,
        "method": record.method,
        "architecture": record.architecture,
        "latent_dim": int(record.latent_dim),
        "input_dim": int(record.input_dim),
        "molecular_selection_mse": record.molecular_selection_mse,
    }

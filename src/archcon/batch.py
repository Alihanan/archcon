"""Headless training and portable PBS sweep export for ArchCon.

The browser UI and batch runner intentionally share :class:`TrainingConfig` and
``train_autoencoder_stream``.  A configuration exported in the UI therefore
uses the same backend as an interactive run rather than reimplementing training
in a cluster-specific script.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from itertools import product
import csv
import json
import os
from pathlib import Path
import pprint
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

from ._compat import strict_zip
from .data.defaults import ProjectDataLayout, project_data_layout
from .data.geo_rma import GEO_RMA_METHODS, METHOD_GLOBAL_RMA, METHOD_PER_GSE_RMA
from .data.pretraining import (
    LOSS_COSINE,
    LOSS_DVIB,
    LOSS_HUBER,
    LOSS_MASKED,
    LOSS_MMD,
    LOSS_MSE,
)
from .data.training_sources import (
    METHOD_STADNIUK_RESCALED,
    TRAINING_PREPROCESSING_OPTIONS,
    create_shared_preprocessing_split,
    load_prepared_pretraining_source,
    load_prepared_split_rows,
    load_pretraining_source,
    prepare_pretraining_assets,
    split_rows_for_source,
)
from .data.training import (
    CHECKPOINT_RESUME,
    TrainingConfig,
    pytorch_model_code,
    train_autoencoder_stream,
)

RUN_CONFIG_FORMAT = 4
_SWEEP_MAX_RUNS = 10_000

_LOSS_ALIASES = {
    "mse": LOSS_MSE,
    "l2": LOSS_MSE,
    "huber": LOSS_HUBER,
    "smooth_l1": LOSS_HUBER,
    "cosine": LOSS_COSINE,
    "mse_cosine": LOSS_COSINE,
    "vae": LOSS_DVIB,
    "beta_vae": LOSS_DVIB,
    "dvib": LOSS_DVIB,
    "mmd": LOSS_MMD,
    "wae": LOSS_MMD,
    "masked": LOSS_MASKED,
    "masked_mse": LOSS_MASKED,
}

RECOMMENDED_COMPARISON_SWEEP = {
    "branches": [
        {
            "name": "stadniuk_mlp",
            "fixed": {
                "architecture_family": "Stadniuk MLP",
                "activation": "ReLU",
                "dropout": 0.1,
                "weight_decay": 0.0,
                "residual_blocks": 0,
                "residual_expansion": 1,
                "learning_rate": 0.001,
                "lr_schedule": "Cosine annealing (deterministic)",
                "optimizer": "Adam",
                "batch_size": 64,
                "epochs": 1000,
                "lr_decay_epochs": 500,
                "convergence_tolerance": 1e-5,
                "convergence_window": 5,
                "early_stopping_patience": 0,
                "gradient_clip": 1.0,
                "seed": 42,
                "device": "CPU",
                "precision": "Float32",
                "compile_model": False,
                "deterministic": True,
            },
            "grid": {
                "method": [
                    "Stadniuk rescaling",
                    "Per-dataset RMA",
                    "Global RMA",
                ],
                "hidden_widths": [
                    [256],
                    [256, 64],
                    [256, 128, 64],
                    [256, 192, 128, 64],
                    [256, 224, 192, 128, 64],
                ],
                "latent_dim": [3, 8, 16],
                "loss_name": ["mse", "masked"],
                "l2_lambda": [0.0, 1e-5, 1e-4],
                "stadniuk_batch_norm": [False, True],
            },
        },
        {
            "name": "resnet_ln",
            "fixed": {
                "architecture_family": "ResNet MLP + LayerNorm",
                "activation": "GELU",
                "dropout": 0.0,
                "weight_decay": 0.0,
                "l2_lambda": 0.0,
                "learning_rate": 0.001,
                "lr_schedule": "Cosine annealing (deterministic)",
                "optimizer": "Adam",
                "batch_size": 64,
                "epochs": 1000,
                "lr_decay_epochs": 500,
                "convergence_tolerance": 1e-5,
                "convergence_window": 5,
                "early_stopping_patience": 0,
                "gradient_clip": 1.0,
                "seed": 42,
                "device": "CPU",
                "precision": "Float32",
                "compile_model": False,
                "deterministic": True,
            },
            "grid": {
                "method": [
                    "Stadniuk rescaling",
                    "Per-dataset RMA",
                    "Global RMA",
                ],
                "hidden_widths": [
                    [256],
                    [256, 64],
                    [256, 128, 64],
                    [256, 192, 128, 64],
                    [256, 224, 192, 128, 64],
                ],
                "latent_dim": [3, 8, 16],
                "loss_name": ["mse", "masked"],
                "residual_blocks": [1, 2],
                "residual_expansion": [2, 4],
            },
        },
    ]
}


def recommended_comparison_grid_json() -> str:
    """Return the 900-run three-preprocessing Stadniuk-vs-ResNet comparison grid."""
    return json.dumps(RECOMMENDED_COMPARISON_SWEEP, indent=2, ensure_ascii=False)




def preprocessing_policy(method: str) -> str:
    """Describe what one preprocessing arm means for held-out GEO studies."""
    method = str(method)
    if method == METHOD_PER_GSE_RMA:
        return (
            "independent per-study RMA; held-out GSEs do not contribute to training-study "
            "normalization"
        )
    if method == METHOD_GLOBAL_RMA:
        return (
            "legacy-named Global RMA arm; train-reference quantile normalization fitted only "
            "on frozen GEO training rows from probe-set PM medians; not exact CEL-level RMA"
        )
    if method == METHOD_STADNIUK_RESCALED:
        return "thesis-inspired Stadniuk rescaling comparison on the shared split"
    return "unspecified preprocessing policy"

def training_config_dict(config: TrainingConfig) -> dict[str, Any]:
    """Return a JSON-friendly representation of one training configuration."""
    value = asdict(config)
    value["hidden_widths"] = list(config.hidden_widths)
    return value


def build_run_request(
    *,
    method: str,
    split_seed: int,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    training: TrainingConfig,
    split_file: str | None = None,
) -> dict[str, Any]:
    """Build a portable run request; filesystem locations are CLI overrides."""
    if method not in {*GEO_RMA_METHODS, *TRAINING_PREPROCESSING_OPTIONS}:
        raise ValueError(f"Unknown GEO preprocessing method: {method}")
    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    if float(train_fraction) <= 0.0 or float(validation_fraction) <= 0.0 or test_fraction <= 0.0:
        raise ValueError("train, validation, and test fractions must all be positive.")
    request: dict[str, Any] = {
        "archcon_run_config_format": RUN_CONFIG_FORMAT,
        "method": method,
        "split_seed": int(split_seed),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "split_unit": "connected source-GSE component + outcome-blind supervised sample",
        "test_policy": "held out from training, convergence checks, checkpoint selection, and sweep ranking",
        "molecular_pretraining_data": "public GEO plus supervised-dataset samples without eGFR",
        "supervised_outcome_policy": "samples with any eGFR remain completely excluded from molecular pretraining",
        "preprocessing_policy": preprocessing_policy(method),
        "training": training_config_dict(training),
    }
    if split_file:
        request["split_file"] = str(split_file)
    return request


def _training_config_from_mapping(value: object) -> TrainingConfig:
    if not isinstance(value, dict):
        raise ValueError("Run config field 'training' must be a JSON object.")
    allowed = {item.name for item in fields(TrainingConfig)}
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError("Unknown TrainingConfig field(s): " + ", ".join(unknown))
    normalized = dict(value)
    if "hidden_widths" in normalized:
        normalized["hidden_widths"] = tuple(int(x) for x in normalized["hidden_widths"])
    return TrainingConfig(**normalized)


def load_run_request(path: str | Path) -> tuple[dict[str, Any], TrainingConfig]:
    source = Path(path).expanduser().resolve()
    request = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("ArchCon run config must contain one JSON object.")
    if int(request.get("archcon_run_config_format", -1)) != RUN_CONFIG_FORMAT:
        raise ValueError("Unsupported ArchCon run-config format.")
    method = str(request.get("method", ""))
    if method not in {*GEO_RMA_METHODS, *TRAINING_PREPROCESSING_OPTIONS}:
        raise ValueError(f"Unknown GEO preprocessing method: {method}")
    config = _training_config_from_mapping(request.get("training", {}))
    return request, config


def _split_rows(split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "row_index_python" not in split.columns or "split" not in split.columns:
        raise ValueError("Split has no row_index_python/split columns.")
    labels = split["split"].astype(str).str.lower()
    rows = pd.to_numeric(split["row_index_python"], errors="raise")
    train_rows = rows.loc[labels.eq("train")].to_numpy(dtype=np.int64)
    validation_rows = rows.loc[labels.eq("validation")].to_numpy(dtype=np.int64)
    return train_rows, validation_rows


def run_training_request(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run one exported ArchCon configuration without starting Gradio."""
    cpu_threads = int(os.environ.get("ARCHCON_CPU_THREADS", "0") or 0)
    if cpu_threads > 0:
        try:
            import torch

            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
        except (ImportError, RuntimeError):
            # ImportError is handled later by the training backend; RuntimeError
            # can occur if another library initialized inter-op threading first.
            pass
    source = Path(config_path).expanduser().resolve()
    request, config = load_run_request(source)
    layout = project_data_layout(data_dir)
    method = str(request["method"])
    prepared_dir = request.get("prepared_dir")
    if prepared_dir:
        prepared_source = Path(str(prepared_dir)).expanduser()
        if not prepared_source.is_absolute():
            prepared_source = source.parent / prepared_source
        training_source = load_prepared_pretraining_source(layout, method, prepared_source)
        train_rows, validation_rows, test_rows = load_prepared_split_rows(prepared_source)
    else:
        # Legacy single-run compatibility only. Generated 0.5.8 sweeps always
        # contain prepared_dir and never execute this branch.
        training_source = load_pretraining_source(layout, method)
        split_file = request.get("split_file")
        if split_file:
            split_source = Path(str(split_file)).expanduser()
            if not split_source.is_absolute():
                split_source = source.parent / split_source
            split = pd.read_csv(split_source)
        else:
            split = create_shared_preprocessing_split(
                layout,
                int(request.get("split_seed", 42)),
                float(request.get("train_fraction", 0.9)),
                methods=[method],
                validation_fraction=float(request.get("validation_fraction", 0.05)),
            )
        train_rows, validation_rows, test_rows = split_rows_for_source(
            split, training_source, include_test=True
        )
    matrix = training_source.matrix
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else (Path.cwd().resolve() / "models")
    )
    root.mkdir(parents=True, exist_ok=True)

    final = None
    last_epoch = -1
    for update in train_autoencoder_stream(
        matrix,
        train_rows,
        validation_rows,
        config,
        root,
        method=method,
    ):
        final = update
        if update.epoch != last_epoch or update.done:
            print(update.status, flush=True)
            last_epoch = update.epoch
    if final is None:
        raise RuntimeError("Training backend returned no updates.")

    return write_run_summary(
        final,
        request=request,
        source_config=str(source),
        method=method,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        data_root=layout.root,
        output_root=root,
    )


def write_run_summary(
    final,
    *,
    request: dict[str, Any],
    source_config: str,
    method: str,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    test_rows: np.ndarray | None = None,
    data_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Persist the common request/summary artifacts for UI, JSON and Python jobs."""
    root = Path(output_root).expanduser().resolve()
    run_dir = None
    for checkpoint in (final.latest_checkpoint, final.best_checkpoint):
        if checkpoint:
            run_dir = Path(checkpoint).expanduser().resolve().parent
            break
    if run_dir is None:
        run_dir = root / f"geo_ae_{final.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

    request_copy = dict(request)
    request_copy["source_config"] = str(source_config)
    (run_dir / "run_request.json").write_text(
        json.dumps(request_copy, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    best_index = int(np.argmin(final.val_loss)) if final.val_loss else None
    summary = {
        "run_id": final.run_id,
        "status": final.status,
        "done": bool(final.done),
        "stopped": bool(final.stopped),
        "epoch": int(final.epoch),
        "latest_checkpoint": final.latest_checkpoint,
        "best_checkpoint": final.best_checkpoint,
        "method": method,
        "n_train": int(len(train_rows)),
        "n_validation": int(len(validation_rows)),
        "n_test": int(len(test_rows)) if test_rows is not None else 0,
        "geo_test_used": False,
        "ikem_diagnostic": False,
        "best_geo_validation_epoch": (
            int(final.val_epoch[best_index]) if best_index is not None else None
        ),
        "config_path": str(source_config),
        "data_dir": str(Path(data_root).expanduser().resolve()),
        "output_root": str(root),
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"ArchCon run summary: {run_dir / 'run_summary.json'}", flush=True)
    return summary


def _training_config_python(config: TrainingConfig) -> str:
    """Render every TrainingConfig field explicitly for an auditable job script."""
    lines = ["TRAINING_CONFIG = TrainingConfig("]
    for item in fields(TrainingConfig):
        value = getattr(config, item.name)
        lines.append(f"    {item.name}={value!r},")
    lines.append(")")
    return "\n".join(lines)


def _job_objective_python(config: TrainingConfig) -> str:
    """Render the selected objective as ordinary readable PyTorch code."""
    if config.loss_name == LOSS_MSE:
        body = [
            "    squared_error = (reconstruction - x) ** 2",
            "    mse = squared_error.mean()",
            "    return mse, mse, torch.zeros((), device=x.device)",
        ]
        helpers: list[str] = []
    elif config.loss_name == LOSS_MASKED:
        body = [
            "    squared_error = (reconstruction - x) ** 2",
            "    mse = squared_error.mean()  # clean/full MSE diagnostic",
            "    if reconstruction_mask is None:",
            "        raise RuntimeError('Masked MSE requires reconstruction_mask.')",
            "    masked_error = squared_error.masked_select(reconstruction_mask)",
            "    if masked_error.numel() == 0:",
            "        raise RuntimeError('Masked MSE received an empty mask.')",
            "    masked_mse = masked_error.mean()",
            "    return masked_mse, mse, torch.zeros((), device=x.device)",
        ]
        helpers = []
    elif config.loss_name == LOSS_HUBER:
        body = [
            "    mse = ((reconstruction - x) ** 2).mean()",
            f"    huber = F.huber_loss(reconstruction, x, reduction='mean', delta={float(config.huber_delta)!r})",
            "    return huber, mse, torch.zeros((), device=x.device)",
        ]
        helpers = []
    elif config.loss_name == LOSS_COSINE:
        body = [
            "    mse = ((reconstruction - x) ** 2).mean()",
            "    cosine = 1.0 - F.cosine_similarity(reconstruction, x, dim=1, eps=1e-8).mean()",
            f"    return mse + {float(config.cosine_weight)!r} * cosine, mse, cosine",
        ]
        helpers = []
    elif config.loss_name == LOSS_DVIB:
        body = [
            "    mse = ((reconstruction - x) ** 2).mean()",
            "    if mu is None or logvar is None:",
            "        raise RuntimeError('beta-VAE objective requires mu/logvar.')",
            "    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())",
            f"    warmup = {int(config.kl_warmup_epochs)}",
            f"    beta = {float(config.kl_beta)!r}",
            "    if warmup > 0 and epoch is not None:",
            "        beta *= min(1.0, max(float(epoch), 0.0) / float(warmup))",
            "    return mse + beta * kl, mse, kl",
        ]
        helpers = []
    elif config.loss_name == LOSS_MMD:
        helpers = [
            "def _imq_kernel(x, y, scale):",
            "    dim = max(int(x.shape[1]), 1)",
            "    c = 2.0 * float(dim) * float(scale)",
            "    distance2 = torch.cdist(x, y, p=2).pow(2)",
            "    return c / (c + distance2 + 1e-8)",
            "",
            "def _mmd_to_standard_normal(z):",
            "    z = z.float()",
            "    n = int(z.shape[0])",
            "    if n < 2:",
            "        return torch.zeros((), device=z.device, dtype=z.dtype)",
            "    prior = torch.randn_like(z)",
            "    total = torch.zeros((), device=z.device, dtype=z.dtype)",
            "    eye = torch.eye(n, device=z.device, dtype=torch.bool)",
            "    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):",
            "        k_xx = _imq_kernel(z, z, scale)",
            "        k_yy = _imq_kernel(prior, prior, scale)",
            "        k_xy = _imq_kernel(z, prior, scale)",
            "        total += k_xx.masked_select(~eye).mean() + k_yy.masked_select(~eye).mean() - 2.0 * k_xy.mean()",
            "    return total / 5.0",
            "",
        ]
        body = [
            "    mse = ((reconstruction - x) ** 2).mean()",
            "    mmd = _mmd_to_standard_normal(z)",
            f"    return mse + {float(config.mmd_weight)!r} * mmd, mse, mmd",
        ]
    else:  # legacy softmax objective
        body = [
            f"    temperature = {float(config.softmax_temperature)!r}",
            "    mse = ((reconstruction - x) ** 2).mean()",
            "    with torch.no_grad():",
            "        target = torch.softmax(x / temperature, dim=1)",
            "    log_prediction = torch.log_softmax(reconstruction / temperature, dim=1)",
            "    ce = -(target * log_prediction).sum(dim=1).mean()",
            "    return ce, mse, torch.zeros((), device=x.device)",
        ]
        helpers = []

    signature = [
        "def job_objective(",
        "    x, reconstruction, z, mu, logvar, config, *, epoch=None, reconstruction_mask=None",
        "):",
    ]
    return "\n".join([*helpers, *signature, *body])


def _optimizer_scheduler_python(config: TrainingConfig) -> str:
    optimizer_cls = "torch.optim.Adam" if config.optimizer == "Adam" else "torch.optim.AdamW"
    if config.lr_schedule == "Cosine annealing (deterministic)":
        scheduler = [
            "def build_scheduler(optimizer, config):",
            f"    decay_epochs = {int(config.lr_decay_epochs)}",
            f"    base_lr = {float(config.learning_rate)!r}",
            f"    min_lr = {float(config.min_learning_rate)!r}",
            "    min_factor = min(1.0, max(0.0, min_lr / base_lr))",
            "    def lr_lambda(step):",
            "        progress = min(max(float(step), 0.0), float(decay_epochs)) / float(decay_epochs)",
            "        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))",
            "        return min_factor + (1.0 - min_factor) * cosine",
            "    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)",
        ]
    elif config.lr_schedule == "Constant learning rate":
        scheduler = [
            "def build_scheduler(optimizer, config):",
            "    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)",
        ]
    else:
        scheduler = [
            "def build_scheduler(optimizer, config):",
            "    return torch.optim.lr_scheduler.ReduceLROnPlateau(",
            f"        optimizer, mode='min', factor={float(config.lr_factor)!r},",
            f"        patience={int(config.lr_patience)}, min_lr={float(config.min_learning_rate)!r}",
            "    )",
        ]
    lines = [
        "def build_optimizer(model, config):",
        f"    return {optimizer_cls}(",
        f"        model.parameters(), lr={float(config.learning_rate)!r}, weight_decay={float(config.weight_decay)!r}",
        "    )",
        "",
        *scheduler,
        "",
        "def job_l2_penalty(model, config):",
        f"    coefficient = {float(config.l2_lambda)!r}",
        "    if coefficient <= 0.0:",
        "        return torch.zeros((), device=next(model.parameters()).device)",
        "    total = None",
        "    for name, module in model.named_modules():",
        "        if not isinstance(module, nn.Linear):",
        "            continue",
        "        if name.endswith('to_output.projection') or name == 'to_output':",
        "            continue",
        "        value = torch.sum(module.weight * module.weight)",
        "        total = value if total is None else total + value",
        "    if total is None:",
        "        return torch.zeros((), device=next(model.parameters()).device)",
        "    return coefficient * total",
    ]
    return "\n".join(lines)


def generated_job_python(
    *,
    index: int,
    branch_name: str,
    request: dict[str, Any],
    expected_input_dim: int = 42_917,
    default_data_dir: str = "$HOME/ArchCon/data",
) -> str:
    """Create one readable executable Python program for a sweep configuration."""
    config = _training_config_from_mapping(request["training"])
    model_source = pytorch_model_code(expected_input_dim, config, instantiate=False)
    class_name = (
        "ArchConVariationalAutoencoder" if config.loss_name == LOSS_DVIB else "ArchConAutoencoder"
    )
    request_literal = pprint.pformat(request, sort_dicts=False, width=100)
    header = [
        "#!/usr/bin/env python3",
        '"""ArchCon generated MetaCentrum job.\n',
        f"Array index: {index}",
        f"Branch: {branch_name}",
        f"Preprocessing: {request['method']}",
        f"Preprocessing policy: {preprocessing_policy(str(request['method']))}",
        f"Architecture: {config.architecture_family}",
        f"Hidden widths: {list(config.hidden_widths)}",
        f"Latent dimension: {config.latent_dim}",
        f"Loss: {config.loss_name}",
        f"BatchNorm (Stadniuk): {config.stadniuk_batch_norm}",
        f"Residual blocks/stage: {config.residual_blocks}",
        f"Residual expansion: {config.residual_expansion}",
        f"L2 lambda: {config.l2_lambda}",
        f"Optimizer: {config.optimizer}",
        f"Initial LR: {config.learning_rate}",
        f"LR schedule: {config.lr_schedule}",
        f"LR decay epochs: {config.lr_decay_epochs}",
        f"Convergence: max relative validation Δ <= {config.convergence_tolerance} over {config.convergence_window} validations, only at LR floor",
        f"Max epochs: {config.epochs}",
        f"Seed: {config.seed}",
        "Split: frozen once at sweep generation · GEO connected GSEs + supervised no-eGFR samples · approximately 90/5/5",
        "Test: held out from neural training/ranking and not evaluated by sweep jobs",
        "Supervised samples with eGFR: completely excluded from molecular pretraining",
        '"""',
    ]
    imports = [
        "from __future__ import annotations",
        "",
        "import argparse",
        "import math",
        "import os",
        "from pathlib import Path",
        "",
        "import numpy as np",
        "import torch",
        "from torch import nn",
        "from torch.nn import functional as F",
        "",
        "from archcon.batch import write_run_summary",
        "from archcon.data.defaults import project_data_layout",
        "from archcon.data.training import CHECKPOINT_RESUME, TrainingConfig, train_autoencoder_stream",
        "from archcon.data.training_sources import (",
        "    load_prepared_pretraining_source,",
        "    load_prepared_split_rows,",
        ")",
        "",
        f"JOB_INDEX = {int(index)}",
        f"BRANCH = {branch_name!r}",
        f"EXPECTED_INPUT_DIM = {int(expected_input_dim)}",
        f"DEFAULT_DATA_DIR = {default_data_dir!r}",
        f"RUN_REQUEST = {request_literal}",
        "",
        _training_config_python(config),
        "",
    ]
    # pytorch_model_code includes imports at the top. They are harmless but noisy in a
    # full job script, so retain only the definitions after its first blank line.
    model_lines = model_source.splitlines()
    while model_lines and (
        model_lines[0].startswith("#") or model_lines[0].startswith("import ") or not model_lines[0]
    ):
        model_lines.pop(0)
    model_block = "\n".join(model_lines)
    factories = [
        "",
        "def build_job_model(input_dim, config):",
        "    if int(input_dim) != EXPECTED_INPUT_DIM:",
        "        raise ValueError(",
        "            f'Generated job expects {EXPECTED_INPUT_DIM} probes, got {input_dim}.'",
        "        )",
        "    if config != TRAINING_CONFIG:",
        "        raise ValueError('Runtime TrainingConfig differs from the generated job constants.')",
        f"    return {class_name}()",
        "",
        _job_objective_python(config),
        "",
        _optimizer_scheduler_python(config),
        "",
    ]
    main = [
        "def _expanded_path(value):",
        "    return Path(os.path.expandvars(str(value))).expanduser().resolve()",
        "",
        "def main():",
        "    parser = argparse.ArgumentParser(description=__doc__)",
        "    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)",
        "    parser.add_argument('--output-root', default=str(Path(__file__).resolve().parent.parent / 'results'))",
        "    parser.add_argument('--run-directory', default=None)",
        "    parser.add_argument('--resume-checkpoint', default=None)",
        "    args = parser.parse_args()",
        "",
        "    cpu_threads = int(os.environ.get('ARCHCON_CPU_THREADS', '0') or 0)",
        "    if cpu_threads > 0:",
        "        torch.set_num_threads(cpu_threads)",
        "        try:",
        "            torch.set_num_interop_threads(1)",
        "        except RuntimeError:",
        "            pass",
        "",
        "    data_dir = _expanded_path(args.data_dir)",
        "    output_root = _expanded_path(args.output_root)",
        "    output_root.mkdir(parents=True, exist_ok=True)",
        "    run_directory = _expanded_path(args.run_directory) if args.run_directory else None",
        "    if run_directory is not None:",
        "        run_directory.mkdir(parents=True, exist_ok=True)",
        "    resume_checkpoint = _expanded_path(args.resume_checkpoint) if args.resume_checkpoint else None",
        "    layout = project_data_layout(data_dir)",
        "    method = str(RUN_REQUEST['method'])",
        "",
        "    # 0.5.8 batch policy: preprocessing identities, 42,917-probe alignment,",
        "    # supervised eligibility, and final split rows were frozen once when",
        "    # the sweep was generated. A job only mmaps those prepared artifacts.",
        "    prepared_rel = RUN_REQUEST.get('prepared_dir')",
        "    if not prepared_rel:",
        "        raise RuntimeError('Sweep has no frozen prepared_dir; regenerate it with ArchCon 0.5.8.')",
        "    prepared_dir = (Path(__file__).resolve().parent / str(prepared_rel)).resolve()",
        "    training_source = load_prepared_pretraining_source(layout, method, prepared_dir)",
        "    train_rows, validation_rows, test_rows = load_prepared_split_rows(prepared_dir)",
        "    if int(training_source.matrix.shape[1]) != EXPECTED_INPUT_DIM:",
        "        raise ValueError(",
        "            f'{method} has {training_source.matrix.shape[1]} probes; expected {EXPECTED_INPUT_DIM}.'",
        "        )",
        "",
        "    print(f'Job {JOB_INDEX:04d} · {BRANCH}')",
        "    print(f'Preprocessing: {method}')",
        "    print(f'Train/validation/test: {len(train_rows)}/{len(validation_rows)}/{len(test_rows)}')",
        "    print('Test policy: BLINDED during training and sweep selection')",
        "    print('Supervised outcome policy: any sample with eGFR is excluded from molecular pretraining')",
        "    print(TRAINING_CONFIG)",
        "",
        "    final = None",
        "    last_epoch = -1",
        "    for update in train_autoencoder_stream(",
        "        training_source.matrix,",
        "        train_rows,",
        "        validation_rows,",
        "        TRAINING_CONFIG,",
        "        output_root,",
        "        method=method,",
        "        model_factory=build_job_model,",
        "        objective_function=job_objective,",
        "        optimizer_factory=build_optimizer,",
        "        scheduler_factory=build_scheduler,",
        "        l2_penalty_function=job_l2_penalty,",
        "        run_directory=run_directory,",
        "        checkpoint_path=resume_checkpoint,",
        "        checkpoint_mode=CHECKPOINT_RESUME if resume_checkpoint else 'Load weights only',",
        "    ):",
        "        final = update",
        "        if update.epoch != last_epoch or update.done:",
        "            print(update.status, flush=True)",
        "            last_epoch = update.epoch",
        "    if final is None:",
        "        raise RuntimeError('Training backend returned no updates.')",
        "",
        "    write_run_summary(",
        "        final,",
        "        request=RUN_REQUEST,",
        "        source_config=str(Path(__file__).resolve()),",
        "        method=method,",
        "        train_rows=np.asarray(train_rows, dtype=np.int64),",
        "        validation_rows=np.asarray(validation_rows, dtype=np.int64),",
        "        test_rows=np.asarray(test_rows, dtype=np.int64),",
        "        data_root=layout.root,",
        "        output_root=output_root,",
        "    )",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ]
    return "\n".join([*header, "", *imports, model_block, *factories, *main])


def _normalized_grid_value(key: str, value: Any) -> Any:
    if key == "loss_name" and isinstance(value, str):
        return _LOSS_ALIASES.get(value.strip().lower(), value)
    if key == "hidden_widths":
        if isinstance(value, str):
            return [int(part.strip()) for part in value.split(",") if part.strip()]
        return [int(part) for part in value]
    return value


def parse_sweep_grid(text: str | None) -> dict[str, list[Any]]:
    """Parse a JSON Cartesian grid over TrainingConfig fields and split settings."""
    raw = json.loads(text or "{}")
    if not isinstance(raw, dict):
        raise ValueError("Sweep grid must be a JSON object.")
    training_fields = {item.name for item in fields(TrainingConfig)}
    allowed = training_fields | {"method", "split_seed", "train_fraction", "validation_fraction"}
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError("Unknown sweep field(s): " + ", ".join(unknown))
    grid: dict[str, list[Any]] = {}
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep field '{key}' must be a non-empty JSON list.")
        grid[key] = [_normalized_grid_value(key, value) for value in values]
    return grid


def _allowed_sweep_fields() -> set[str]:
    return {item.name for item in fields(TrainingConfig)} | {
        "method",
        "split_seed",
        "train_fraction",
        "validation_fraction",
    }


def _normalize_override_mapping(raw: object, *, list_values: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Sweep branch fixed/grid entries must be JSON objects.")
    allowed = _allowed_sweep_fields()
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError("Unknown sweep field(s): " + ", ".join(unknown))
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if list_values:
            if not isinstance(value, list) or not value:
                raise ValueError(f"Sweep field '{key}' must be a non-empty JSON list.")
            result[key] = [_normalized_grid_value(key, item) for item in value]
        else:
            result[key] = _normalized_grid_value(key, value)
    return result


def _expand_sweep_variants(grid_text: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Expand legacy flat grids or architecture-specific branch grids."""
    raw = json.loads(grid_text or "{}")
    if not isinstance(raw, dict):
        raise ValueError("Sweep grid must be a JSON object.")

    if "branches" not in raw:
        grid = parse_sweep_grid(grid_text)
        keys = list(grid)
        combos = list(product(*(grid[key] for key in keys))) if keys else [()]
        return [("grid", dict(strict_zip(keys, values))) for values in combos]

    if set(raw) != {"branches"}:
        raise ValueError("A branch sweep may contain only the top-level 'branches' field.")
    branches = raw["branches"]
    if not isinstance(branches, list) or not branches:
        raise ValueError("'branches' must be a non-empty JSON list.")

    variants: list[tuple[str, dict[str, Any]]] = []
    seen_names: set[str] = set()
    for branch_index, branch in enumerate(branches, start=1):
        if not isinstance(branch, dict):
            raise ValueError("Each sweep branch must be a JSON object.")
        unknown_branch = sorted(set(branch).difference({"name", "fixed", "grid"}))
        if unknown_branch:
            raise ValueError("Unknown branch field(s): " + ", ".join(unknown_branch))
        name = str(branch.get("name") or f"branch_{branch_index}")
        if name in seen_names:
            raise ValueError(f"Duplicate sweep branch name: {name}")
        seen_names.add(name)
        fixed = _normalize_override_mapping(branch.get("fixed", {}), list_values=False)
        grid = _normalize_override_mapping(branch.get("grid", {}), list_values=True)
        overlap = sorted(set(fixed).intersection(grid))
        if overlap:
            raise ValueError(
                f"Branch '{name}' defines fields as both fixed and grid values: "
                + ", ".join(overlap)
            )
        keys = list(grid)
        combos = list(product(*(grid[key] for key in keys))) if keys else [()]
        for values in combos:
            overrides = dict(fixed)
            overrides.update(dict(strict_zip(keys, values)))
            variants.append((name, overrides))
    return variants



def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or "archcon-sweep"


def _apply_grid_value(request: dict[str, Any], key: str, value: Any) -> None:
    if key in {"method", "split_seed", "train_fraction", "validation_fraction"}:
        request[key] = value
        if key == "method":
            request["preprocessing_policy"] = preprocessing_policy(str(value))
    else:
        request["training"][key] = value


def _shell_default(value: str) -> str:
    """Return a shell expression used inside an already double-quoted default."""
    return value.replace('"', '\\"')


def generate_sweep_bundle(
    *,
    base_request: dict[str, Any],
    grid_text: str,
    destination_root: str | Path,
    sweep_name: str,
    project_dir: str = "$HOME/ArchCon",
    data_dir: str = "$HOME/ArchCon/data",
    python_executable: str = "$HOME/ArchCon/.venv/bin/python",
    ncpus: int = 1,
    memory: str = "10gb",
    scratch: str = "4gb",
    walltime: str = "48:00:00",
    ngpus: int = 0,
    gpu_memory: str = "12gb",
    split_frame: pd.DataFrame | None = None,
    data_layout: ProjectDataLayout | None = None,
) -> dict[str, Any]:
    """Write readable Python jobs, audit JSON, and a MetaCentrum PBS array."""
    variants = _expand_sweep_variants(grid_text)
    if len(variants) > _SWEEP_MAX_RUNS:
        raise ValueError(
            f"Sweep expands to {len(variants):,} runs; maximum is {_SWEEP_MAX_RUNS:,}."
        )

    root = Path(destination_root).expanduser().resolve() / _safe_name(sweep_name)
    if root.exists():
        shutil.rmtree(root)
    configs_dir = root / "configs"
    jobs_dir = root / "jobs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    portable_base = json.loads(json.dumps(base_request))
    split_varies = any(
        key in {"split_seed", "train_fraction", "validation_fraction"}
        for _, overrides in variants
        for key in overrides
    )
    if split_frame is not None and not split_frame.empty and not split_varies:
        split_path = root / "split.csv"
        split_frame.to_csv(split_path, index=False)
        portable_base["split_file"] = "../split.csv"

        # Freeze every data-dependent choice once.  Generated jobs must never
        # repeat probe alignment, outcome filtering, sample mapping, or splitting.
        resolved_layout = data_layout
        if resolved_layout is None:
            expanded_data = Path(os.path.expandvars(str(data_dir))).expanduser()
            if expanded_data.is_absolute() and expanded_data.exists():
                resolved_layout = project_data_layout(expanded_data)
        has_identity_split = "sample_key" in split_frame.columns
        if has_identity_split:
            if resolved_layout is None:
                raise ValueError(
                    "A frozen identity-level split requires data_layout (or an existing absolute "
                    "data_dir) so ArchCon can prepare final row/probe mappings once before jobs run."
                )
            method_values = {str(base_request.get("method", METHOD_PER_GSE_RMA))}
            for _, overrides in variants:
                if "method" in overrides:
                    method_values.add(str(overrides["method"]))
            prepared_root = prepare_pretraining_assets(
                resolved_layout,
                split_frame,
                root / "prepared",
                methods=tuple(
                    method for method in TRAINING_PREPROCESSING_OPTIONS if method in method_values
                ),
            )
            portable_base["prepared_dir"] = "../prepared"
            portable_base["prepared_policy"] = (
                "Frozen once during sweep generation; jobs only read saved row/probe mappings."
            )

    manifest_rows: list[dict[str, Any]] = []
    manifest_keys = sorted({key for _, overrides in variants for key in overrides})
    for index, (branch_name, overrides) in enumerate(variants, start=1):
        request = json.loads(json.dumps(portable_base))
        for key, value in overrides.items():
            _apply_grid_value(request, key, value)
        if "train_fraction" in request or "validation_fraction" in request:
            request["test_fraction"] = float(
                1.0
                - float(request.get("train_fraction", 0.90))
                - float(request.get("validation_fraction", 0.05))
            )
            if request["test_fraction"] <= 0.0:
                raise ValueError("Sweep override leaves no held-out GEO test fraction.")
        config_name = f"run_{index:04d}.json"
        (configs_dir / config_name).write_text(
            json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        job_name = f"run_{index:04d}.py"
        job_path = jobs_dir / job_name
        job_path.write_text(
            generated_job_python(
                index=index,
                branch_name=branch_name,
                request=request,
                default_data_dir=data_dir,
            ),
            encoding="utf-8",
        )
        job_path.chmod(0o755)
        row: dict[str, Any] = {
            "array_index": index,
            "branch": branch_name,
            "config": f"configs/{config_name}",
            "script": f"jobs/{job_name}",
            "preprocessing_policy": request.get("preprocessing_policy", ""),
        }
        for key in manifest_keys:
            if key not in overrides:
                row[key] = ""
                continue
            value = request[key] if key in {"method", "split_seed", "train_fraction", "validation_fraction"} else request["training"][key]
            row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        manifest_rows.append(row)

    manifest_path = root / "manifest.csv"
    fieldnames = list(manifest_rows[0]) if manifest_rows else ["array_index", "config"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    resources = [f"ncpus={int(ncpus)}", f"mem={memory}"]
    if scratch.strip():
        resources.append(f"scratch_local={scratch.strip()}")
    if int(ngpus) > 0:
        resources.append(f"ngpus={int(ngpus)}")
        if gpu_memory.strip():
            resources.append(f"gpu_mem={gpu_memory.strip()}")
    select_line = ":".join(resources)

    run_script = f"""#!/usr/bin/env bash
#PBS -N {_safe_name(sweep_name)[:40]}
#PBS -l select=1:{select_line}
#PBS -l walltime={walltime}
#PBS -j oe

set -euo pipefail

: "${{PBS_ARRAY_INDEX:?This script must be submitted as a PBS job array.}}"
: "${{ARCHCON_SWEEP_DIR:?submit.sh sets ARCHCON_SWEEP_DIR to the copied sweep directory.}}"

PROJECT_DIR="${{ARCHCON_PROJECT_DIR:-{_shell_default(project_dir)}}}"
DATA_DIR="${{ARCHCON_DATA_DIR:-{_shell_default(data_dir)}}}"
PYTHON_BIN="${{ARCHCON_PYTHON:-{_shell_default(python_executable)}}}"
RESULT_ROOT="${{ARCHCON_RESULT_ROOT:-$ARCHCON_SWEEP_DIR/results}}"
RUN_NAME=$(printf "run_%04d" "$PBS_ARRAY_INDEX")
CONFIG="$ARCHCON_SWEEP_DIR/configs/${{RUN_NAME}}.json"
JOB="$ARCHCON_SWEEP_DIR/jobs/${{RUN_NAME}}.py"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

mkdir -p "$RESULT_ROOT" "$RUN_DIR"

export OMP_NUM_THREADS="{int(ncpus)}"
export MKL_NUM_THREADS="{int(ncpus)}"
export OPENBLAS_NUM_THREADS="{int(ncpus)}"
export NUMEXPR_NUM_THREADS="{int(ncpus)}"
export ARCHCON_CPU_THREADS="{int(ncpus)}"
if [[ -n "${{SCRATCHDIR:-}}" ]]; then
    export TMPDIR="$SCRATCHDIR"
    export TEMP="$SCRATCHDIR"
    export TMP="$SCRATCHDIR"
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCHDIR/torchinductor"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
fi
cd "$PROJECT_DIR"

echo "ArchCon array index: $PBS_ARRAY_INDEX"
echo "Run name: $RUN_NAME"
echo "Python job: $JOB"
echo "Config: $CONFIG"
echo "Data: $DATA_DIR"
echo "Persistent result: $RUN_DIR"
echo "Checkpoint policy: latest.pt is atomically written here after every completed epoch"

if [[ -f "$RUN_DIR/run_summary.json" ]]; then
    if "$PYTHON_BIN" - "$RUN_DIR/run_summary.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if summary.get("done") else 1)
PY
    then
        echo "Already completed: $RUN_NAME"
        exit 0
    fi
fi

RESUME_ARGS=()
if [[ -f "$RUN_DIR/latest.pt" ]]; then
    echo "Interrupted checkpoint found; resuming from $RUN_DIR/latest.pt"
    RESUME_ARGS=(--resume-checkpoint "$RUN_DIR/latest.pt")
fi

"$PYTHON_BIN" "$JOB" \
    --data-dir "$DATA_DIR" \
    --output-root "$RESULT_ROOT" \
    --run-directory "$RUN_DIR" \
    "${{RESUME_ARGS[@]}}"

if [[ -f "$RUN_DIR/run_summary.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/run_summary.json" "$RUN_DIR" "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path
summary_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2]).resolve()
result_root = Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["output_root"] = str(result_root)
summary["result_directory"] = str(run_dir)
summary["array_index"] = int(run_dir.name.split("_")[-1])
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
PY
fi

echo "Completed: $RUN_NAME"
echo "Results stored in: $RUN_DIR"
"""
    run_path = root / "run_array.pbs.sh"
    run_path.write_text(run_script, encoding="utf-8")
    run_path.chmod(0o755)

    submit_script = f"""#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
if [[ ! -f "$HERE/prepared/prepared.json" ]]; then
    echo "Missing frozen prepared/prepared.json. Regenerate this sweep with ArchCon 0.5.8 before submitting." >&2
    exit 2
fi
for name in train_rows.npy validation_rows.npy test_rows.npy supervised_no_egfr_common.npy; do
    if [[ ! -f "$HERE/prepared/$name" ]]; then
        echo "Missing frozen prepared/$name. Refusing to submit." >&2
        exit 2
    fi
done
qsub -J 1-{len(variants)} -v ARCHCON_SWEEP_DIR="$HERE" run_array.pbs.sh
"""
    submit_path = root / "submit.sh"
    submit_path.write_text(submit_script, encoding="utf-8")
    submit_path.chmod(0o755)

    readme = f"""ArchCon MetaCentrum sweep
==========================
Runs: {len(variants)}

After copying this whole directory to MetaCentrum:
  1. Edit run_array.pbs.sh defaults if your server paths/resources differ.
  2. Ensure ArchCon is installed in the selected Python environment.
  3. Run: ./submit.sh
  4. Monitor: qstat -t

Each PBS_ARRAY_INDEX executes one human-readable jobs/run_XXXX.py program. The matching
configs/run_XXXX.json file is kept as a machine-readable record of the same setup.
Every generated Python program contains its exact PyTorch architecture, selected loss,
optimizer/scheduler, L2 penalty and all TrainingConfig values. The `prepared/` directory is the
only source of batch split/mapping decisions: train_rows.npy, validation_rows.npy, test_rows.npy,
method-specific GEO row maps, and the already probe-aligned supervised-no-eGFR matrix are written
once during sweep generation. Jobs only read those files; they never reclassify outcomes, align
probes, remap samples, or resplit data. Outcome-bearing samples are excluded. The three GEO preprocessing arms are Stadniuk rescaling, independent per-study RMA,
and the legacy-named Global RMA arm, which requires a train-reference matrix tied to this
sweep's frozen split. Results are written under results/
unless ARCHCON_RESULT_ROOT is overridden. Crucially, run_XXXX/ is created before training and
latest.pt is atomically written there after every completed epoch; best.pt is updated immediately
on validation improvement. Resubmitting an interrupted array element resumes from latest.pt.
If PBS provides SCRATCHDIR, only temporary Python/Torch compilation files are redirected there.

For an interactive ArchCon browser session, request an interactive PBS job and run
`archcon --host 127.0.0.1 --port 7860 --no-browser` on the allocated compute node.
Use MetaCentrum Open OnDemand/Interactive Desktop or an SSH tunnel from your local
workstation; do not train directly on a shared frontend.
"""
    (root / "README.txt").write_text(readme, encoding="utf-8")

    return {
        "root": str(root),
        "count": len(variants),
        "manifest": str(manifest_path),
        "jobs_dir": str(jobs_dir),
        "run_script": str(run_path),
        "submit_script": str(submit_path),
    }

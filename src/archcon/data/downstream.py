"""Frozen kidney embeddings and leakage-safe molecular eGFR evaluation.

Configurations are first selected separately inside every preprocessing by
architecture group using pooled molecular validation+test reconstruction MSE.
The six resulting encoders are compared with nested donor-grouped eGFR CV: inner
folds select the encoder and outer folds evaluate the complete selection rule.
Outcome data are joined only after every frozen z has been computed. Molecular
and clinical scaling, clinical imputation, and PCA are fitted inside each fold.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

from .defaults import ProjectDataLayout
from .loading import normalize_sample_id
from .supervised import EGFR_COLUMNS, classify_supervised_samples
from .training import TrainingConfig, build_autoencoder
from .training_sources import (
    load_ikem_source,
    load_prepared_pretraining_source,
    load_prepared_split_rows,
)

TIME_LABELS = {
    "egfr_7d": "7d",
    "egfr_3m": "3m",
    "egfr_6m": "6m",
    "egfr_12m": "12m",
}
TIME_LEVELS = ("7d", "3m", "6m", "12m")
CLINICAL_COLUMNS = ("KDRI_8", "don_patient_age", "Cold_ischemia_hours")
CLINICAL_LABELS = {
    "KDRI_8": "KDRI",
    "don_patient_age": "Donor age",
    "Cold_ischemia_hours": "Cold ischemia",
}


@dataclass(frozen=True)
class ValidationCheckpoint:
    """One readable ``best.pt`` checkpoint and its comparable validation score."""

    run: str
    path: Path
    checkpoint: dict
    method: str
    architecture: str
    latent_dim: int
    validation_mse: float
    validation_r2: float
    validation_objective: float
    best_epoch: int
    config: TrainingConfig
    molecular_validation_mse: float | None = None
    molecular_test_mse: float | None = None
    molecular_selection_mse: float | None = None
    molecular_selection_r2: float | None = None
    n_molecular_validation: int = 0
    n_molecular_test: int = 0


@dataclass(frozen=True)
class EmbeddingResult:
    """Frozen latent representation for one selected checkpoint."""

    model_id: str
    label: str
    record: ValidationCheckpoint
    z: np.ndarray
    samples: pd.DataFrame
    reconstruction_mse: float


def load_checkpoint(path: str | Path) -> dict:
    """Load a trusted ArchCon checkpoint on CPU without modifying it."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "PyTorch is required for checkpoint evaluation; install archcon[training]."
        ) from exc

    source = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {source}")
    return checkpoint


def training_config_from_checkpoint(checkpoint: dict) -> TrainingConfig:
    """Restore known configuration fields while tolerating future metadata."""

    raw = dict(checkpoint.get("config", {}))
    if "hidden_widths" in raw:
        raw["hidden_widths"] = tuple(int(value) for value in raw["hidden_widths"])
    allowed = {item.name for item in fields(TrainingConfig)}
    return TrainingConfig(**{key: value for key, value in raw.items() if key in allowed})


def _checkpoint_validation_metrics(checkpoint: dict) -> tuple[float, float, float, int]:
    history = checkpoint.get("history", {})
    if not isinstance(history, dict):
        raise TypeError("Malformed checkpoint history.")
    val_mse = list(history.get("val_mse", []))
    if not val_mse:
        raise ValueError("Checkpoint contains no validation MSE.")
    index = len(val_mse) - 1
    val_r2 = list(history.get("val_r2", []))
    val_loss = list(history.get("val_loss", []))
    val_epoch = list(history.get("val_epoch", []))
    return (
        float(val_mse[index]),
        float(val_r2[index]) if index < len(val_r2) else float("nan"),
        (
            float(val_loss[index])
            if index < len(val_loss)
            else float(checkpoint.get("best_val_loss", float("nan")))
        ),
        (
            int(val_epoch[index])
            if index < len(val_epoch)
            else int(checkpoint.get("epoch", 0))
        ),
    )


def scan_validation_checkpoints(results_root: str | Path) -> tuple[list[ValidationCheckpoint], list[str]]:
    """Rank readable ``best.pt`` files by clean validation MSE.

    Returns both records and warning strings so callers can report incomplete or
    corrupt runs without treating them as scientific results.
    """

    records: list[ValidationCheckpoint] = []
    warnings: list[str] = []
    root = Path(results_root).expanduser().resolve()
    for run_dir in sorted(root.glob("run_*")):
        path = run_dir / "best.pt"
        if not path.is_file():
            continue
        try:
            checkpoint = load_checkpoint(path)
            mse, r2, objective, epoch = _checkpoint_validation_metrics(checkpoint)
            config = training_config_from_checkpoint(checkpoint)
            records.append(
                ValidationCheckpoint(
                    run=run_dir.name,
                    path=path,
                    checkpoint=checkpoint,
                    method=str(checkpoint.get("method", "unknown")),
                    architecture=str(config.architecture_family),
                    latent_dim=int(config.latent_dim),
                    validation_mse=mse,
                    validation_r2=r2,
                    validation_objective=objective,
                    best_epoch=epoch,
                    config=config,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one corrupt run must not stop the scan
            warnings.append(f"Skipped {path}: {exc}")
    records.sort(key=lambda record: record.validation_mse)
    return records, warnings


def _clean_reconstruction_totals(
    model,
    matrix,
    rows: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[float, int, float, float]:
    """Return SSE, value count, sum(y), and sum(y^2) on clean inputs."""

    import torch

    squared_error = 0.0
    value_count = 0
    sum_y = 0.0
    sum_y2 = 0.0
    torch_device = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = np.asarray(rows[start : start + int(batch_size)], dtype=np.int64)
            values = np.ascontiguousarray(
                np.asarray(matrix[batch_rows, :], dtype=np.float32)
            )
            x = torch.from_numpy(values).to(torch_device)
            reconstruction, _, _, _ = model(x, sample=False)
            residual = reconstruction.float() - x.float()
            squared_error += float(torch.sum(residual * residual).cpu())
            value_count += int(x.numel())
            sum_y += float(torch.sum(x).cpu())
            sum_y2 += float(torch.sum(x * x).cpu())
    return squared_error, value_count, sum_y, sum_y2


def _target_totals(matrix, rows: np.ndarray, *, batch_size: int) -> tuple[int, float, float]:
    """Compute target count and moments without running a neural network."""

    value_count = 0
    sum_y = 0.0
    sum_y2 = 0.0
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = np.asarray(rows[start : start + int(batch_size)], dtype=np.int64)
        values = np.asarray(matrix[batch_rows, :], dtype=np.float64)
        value_count += int(values.size)
        sum_y += float(values.sum())
        sum_y2 += float(np.square(values).sum())
    return value_count, sum_y, sum_y2


def score_molecular_selection_records(
    records: list[ValidationCheckpoint],
    layout: ProjectDataLayout,
    prepared_root: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    progress=None,
) -> list[ValidationCheckpoint]:
    """Score every validation-selected checkpoint on pooled validation+test rows.

    ``best.pt`` still fixes the epoch without looking at the molecular test rows.
    At this later comparison stage validation and test rows deliberately become
    one molecular selection set. Their SSEs are pooled, rather than averaging
    two MSEs with unequal sample counts.
    """

    if not records:
        raise ValueError("No readable validation checkpoints were found.")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("PyTorch is required; install archcon[training].") from exc

    prepared = Path(prepared_root).expanduser().resolve()
    _, validation_rows, test_rows = load_prepared_split_rows(prepared)
    source_cache: dict[str, object] = {}
    validation_target_cache: dict[str, tuple[int, float, float]] = {}
    scored: list[ValidationCheckpoint] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        if progress is not None:
            progress(index, total, record)
        source = source_cache.get(record.method)
        if source is None:
            source = load_prepared_pretraining_source(layout, record.method, prepared)
            source_cache[record.method] = source
            validation_target_cache[record.method] = _target_totals(
                source.matrix, validation_rows, batch_size=batch_size
            )
        input_dim = int(record.checkpoint.get("input_dim", 0))
        if input_dim < 1 or int(source.matrix.shape[1]) != input_dim:
            raise ValueError(
                f"Prepared source/checkpoint feature mismatch for {record.run}: "
                f"{source.matrix.shape[1]} vs {input_dim}."
            )
        model = build_autoencoder(input_dim, record.config)
        model.load_state_dict(record.checkpoint["model_state"], strict=True)
        model.to(torch.device(device))
        model.eval()
        test = _clean_reconstruction_totals(
            model,
            source.matrix,
            test_rows,
            device=device,
            batch_size=batch_size,
        )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        validation_count, validation_sum, validation_sum2 = validation_target_cache[
            record.method
        ]
        validation_sse = float(record.validation_mse) * validation_count
        combined_sse = validation_sse + test[0]
        combined_count = validation_count + test[1]
        combined_sum = validation_sum + test[2]
        combined_sum2 = validation_sum2 + test[3]
        denominator = combined_sum2 - combined_sum * combined_sum / max(combined_count, 1)
        combined_r2 = (
            1.0 - combined_sse / denominator if denominator > 0.0 else float("nan")
        )
        scored.append(
            replace(
                record,
                molecular_validation_mse=record.validation_mse,
                molecular_test_mse=test[0] / max(test[1], 1),
                molecular_selection_mse=combined_sse / max(combined_count, 1),
                molecular_selection_r2=combined_r2,
                n_molecular_validation=len(validation_rows),
                n_molecular_test=len(test_rows),
            )
        )
    return sorted(
        scored,
        key=lambda record: (
            record.method,
            record.architecture,
            float(record.molecular_selection_mse),
        ),
    )


def select_molecular_group_winners(
    records: list[ValidationCheckpoint],
    *,
    expected_groups: int | None = 6,
) -> list[tuple[str, str, ValidationCheckpoint]]:
    """Select one pooled validation+test winner per preprocessing×architecture."""

    if not records:
        raise ValueError("No molecular selection scores were supplied.")
    unscored = [record.run for record in records if record.molecular_selection_mse is None]
    if unscored:
        raise ValueError(f"Checkpoints lack molecular selection scores: {unscored[:5]}")
    groups: dict[tuple[str, str], list[ValidationCheckpoint]] = {}
    for record in records:
        groups.setdefault((record.method, record.architecture), []).append(record)
    if expected_groups is not None and len(groups) != int(expected_groups):
        readable = ", ".join(f"{method} × {architecture}" for method, architecture in groups)
        raise ValueError(
            f"Expected {expected_groups} preprocessing×architecture groups, found {len(groups)}: "
            f"{readable}. Wait for at least one completed run in every sweep group."
        )

    selected: list[tuple[str, str, ValidationCheckpoint]] = []
    for method, architecture in sorted(groups):
        record = min(
            groups[(method, architecture)],
            key=lambda candidate: float(candidate.molecular_selection_mse),
        )
        model_id = f"candidate_{_slug(method)}_{_slug(architecture)}_z"
        label = f"{method} × {architecture} z"
        selected.append((model_id, label, record))
    return selected


def save_molecular_selection(
    records: list[ValidationCheckpoint],
    selected: list[tuple[str, str, ValidationCheckpoint]],
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Persist all pooled scores and the six group winners."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run": record.run,
            "checkpoint": str(record.path),
            "preprocessing": record.method,
            "architecture": record.architecture,
            "latent_dim": record.latent_dim,
            "best_epoch": record.best_epoch,
            "checkpoint_validation_mse": record.validation_mse,
            "selection_validation_mse": record.molecular_validation_mse,
            "test_mse": record.molecular_test_mse,
            "pooled_validation_test_mse": record.molecular_selection_mse,
            "pooled_validation_test_r2": record.molecular_selection_r2,
            "n_validation_samples": record.n_molecular_validation,
            "n_test_samples": record.n_molecular_test,
        }
        for record in records
    ]
    scores = pd.DataFrame(rows).sort_values(
        ["preprocessing", "architecture", "pooled_validation_test_mse"]
    )
    score_path = root / "molecular_selection_scores.csv"
    scores.to_csv(score_path, index=False)
    selected_runs = {record.run: model_id for model_id, _, record in selected}
    winners = scores.loc[scores["run"].isin(selected_runs)].copy()
    winners.insert(0, "candidate_id", winners["run"].map(selected_runs))
    winner_path = root / "molecular_group_winners.csv"
    winners.to_csv(winner_path, index=False)
    return score_path, winner_path


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return result or "model"


def select_embedding_checkpoints(
    records: list[ValidationCheckpoint],
    *,
    include_non_stadniuk: bool = True,
    include_preprocessing_winners: bool = False,
) -> list[tuple[str, str, ValidationCheckpoint]]:
    """Select comparison encoders using validation metrics only."""

    if not records:
        raise ValueError("No readable validation checkpoints were found.")
    selected: list[tuple[str, str, ValidationCheckpoint]] = [
        ("winner_z", "Validation-selected winner z", records[0])
    ]
    if include_non_stadniuk:
        candidates = [record for record in records if record.architecture != "Stadniuk MLP"]
        if candidates:
            selected.append(
                ("best_non_stadniuk_z", "Best validation-selected non-Stadniuk z", candidates[0])
            )
        # If the overall winner is itself non-Stadniuk, retain a genuine
        # architecture-family comparator rather than silently deduplicating the
        # only requested baseline.
        if records[0].architecture != "Stadniuk MLP":
            stadniuk = [record for record in records if record.architecture == "Stadniuk MLP"]
            if stadniuk:
                selected.append(
                    ("best_stadniuk_z", "Best validation-selected Stadniuk z", stadniuk[0])
                )
    if include_preprocessing_winners:
        methods = list(dict.fromkeys(record.method for record in records))
        for method in methods:
            record = min(
                (candidate for candidate in records if candidate.method == method),
                key=lambda candidate: candidate.validation_mse,
            )
            selected.append(
                (f"best_{_slug(method)}_z", f"Best z - {method}", record)
            )

    # The overall winner may also be the best non-Stadniuk or arm-specific model.
    # Deduplicate checkpoints while retaining the most informative first label.
    result: list[tuple[str, str, ValidationCheckpoint]] = []
    seen: set[Path] = set()
    for model_id, label, record in selected:
        resolved = record.path.resolve()
        if resolved not in seen:
            result.append((model_id, label, record))
            seen.add(resolved)
    return result


def _prepared_probe_ids(prepared_root: str | Path, expected: int) -> tuple[str, ...] | None:
    path = Path(prepared_root).expanduser().resolve() / "probe_index.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "probe_id" not in frame.columns:
        raise ValueError(f"Prepared probe index lacks probe_id: {path}")
    values = tuple(frame["probe_id"].astype(str))
    if len(values) != int(expected):
        raise ValueError(
            f"Prepared probe index has {len(values):,} rows but checkpoint expects {expected:,}."
        )
    if len(set(values)) != len(values):
        raise ValueError("Prepared probe index contains duplicate probe IDs.")
    return values


def canonical_ikem_columns(
    layout: ProjectDataLayout,
    prepared_root: str | Path,
    input_dim: int,
) -> tuple[object, np.ndarray, pd.DataFrame]:
    """Return the supervised source and its mapping into checkpoint feature order."""

    source = load_ikem_source(layout)
    if source is None:
        raise FileNotFoundError("No supervised expression store was found.")
    target_ids = _prepared_probe_ids(prepared_root, input_dim)
    if target_ids is None or source.probe_ids is None:
        if source.n_probes != int(input_dim):
            raise ValueError(
                "Cannot align supervised expression without both probe indexes: "
                f"{source.n_probes:,} source probes vs {input_dim:,} checkpoint features."
            )
        columns = np.arange(input_dim, dtype=np.int64)
    else:
        if len(set(source.probe_ids)) != len(source.probe_ids):
            raise ValueError("Supervised probe index contains duplicate probe IDs.")
        lookup = {probe: index for index, probe in enumerate(source.probe_ids)}
        missing = [probe for probe in target_ids if probe not in lookup]
        if missing:
            raise ValueError(
                f"Supervised expression is missing {len(missing):,} checkpoint probes; "
                f"examples: {missing[:5]}."
            )
        columns = np.asarray([lookup[probe] for probe in target_ids], dtype=np.int64)

    raw_ids = source.sample_index[source.sample_id_column].astype(str)
    sample_ids = raw_ids.map(normalize_sample_id)
    if sample_ids.duplicated().any():
        duplicates = sample_ids[sample_ids.duplicated(keep=False)].head(5).tolist()
        raise ValueError(f"Supervised expression contains duplicate normalized IDs: {duplicates}")
    status = classify_supervised_samples(layout, sample_ids)
    samples = pd.DataFrame(
        {
            "source_row_index": source.sample_index["row_index_python"].astype(np.int64),
            "sample_id": sample_ids,
            "donor_id": sample_ids.map(lambda value: str(value).split("_", 1)[0]),
            "has_egfr": status.table["has_egfr"].astype(bool).to_numpy(),
        }
    )
    return source, columns, samples


def aligned_ikem_matrix(
    layout: ProjectDataLayout,
    prepared_root: str | Path,
    input_dim: int,
    *,
    rows: Iterable[int] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Materialize only the small supervised cohort in canonical probe order."""

    source, columns, samples = canonical_ikem_columns(layout, prepared_root, input_dim)
    row_values = (
        np.arange(source.n_samples, dtype=np.int64)
        if rows is None
        else np.asarray(list(rows), dtype=np.int64)
    )
    matrix = np.asarray(source.matrix[np.ix_(row_values, columns)], dtype=np.float32)
    return np.ascontiguousarray(matrix), samples.iloc[row_values].reset_index(drop=True)


def extract_checkpoint_embedding(
    record: ValidationCheckpoint,
    model_id: str,
    label: str,
    layout: ProjectDataLayout,
    prepared_root: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingResult:
    """Freeze one selected encoder and apply it to all supervised samples."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("PyTorch is required; install archcon[training].") from exc

    input_dim = int(record.checkpoint.get("input_dim", 0))
    if input_dim < 1:
        raise ValueError(f"Checkpoint {record.path} has invalid input_dim.")
    source, columns, samples = canonical_ikem_columns(layout, prepared_root, input_dim)
    model = build_autoencoder(input_dim, record.config)
    model.load_state_dict(record.checkpoint["model_state"], strict=True)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    z = np.empty((source.n_samples, record.latent_dim), dtype=np.float32)
    total_sse = 0.0
    total_values = 0
    with torch.inference_mode():
        for start in range(0, source.n_samples, int(batch_size)):
            stop = min(start + int(batch_size), source.n_samples)
            rows = np.arange(start, stop, dtype=np.int64)
            values = np.asarray(source.matrix[np.ix_(rows, columns)], dtype=np.float32)
            x = torch.from_numpy(np.ascontiguousarray(values)).to(torch_device)
            reconstruction, latent, _, _ = model(x, sample=False)
            z[start:stop] = latent.detach().cpu().numpy().astype(np.float32, copy=False)
            total_sse += float(((reconstruction.float() - x.float()) ** 2).sum().item())
            total_values += int(values.size)
    return EmbeddingResult(
        model_id=model_id,
        label=label,
        record=record,
        z=z,
        samples=samples,
        reconstruction_mse=total_sse / max(total_values, 1),
    )


def save_embeddings(
    embeddings: list[EmbeddingResult],
    output_root: str | Path,
) -> tuple[Path, list[str]]:
    """Save exact arrays, convenient tables, and provenance metadata."""

    if not embeddings:
        raise ValueError("At least one embedding is required.")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    tables: list[pd.DataFrame] = []
    metadata: dict[str, object] = {"format": 1, "models": []}
    reference_ids = embeddings[0].samples["sample_id"].astype(str).tolist()

    for result in embeddings:
        if result.samples["sample_id"].astype(str).tolist() != reference_ids:
            raise ValueError("Selected embeddings do not share the same supervised sample order.")
        np.save(root / f"{result.model_id}.npy", result.z, allow_pickle=False)
        table = result.samples.copy()
        table.insert(0, "model_id", result.model_id)
        table.insert(1, "model_label", result.label)
        for index in range(result.z.shape[1]):
            table[f"z{index + 1}"] = result.z[:, index]
        tables.append(table)
        metadata["models"].append(
            {
                "model_id": result.model_id,
                "label": result.label,
                "run": result.record.run,
                "checkpoint": str(result.record.path),
                "method": result.record.method,
                "architecture": result.record.architecture,
                "latent_dim": result.record.latent_dim,
                "validation_mse": result.record.validation_mse,
                "validation_r2": result.record.validation_r2,
                "molecular_validation_mse": result.record.molecular_validation_mse,
                "molecular_test_mse": result.record.molecular_test_mse,
                "molecular_selection_mse": result.record.molecular_selection_mse,
                "molecular_selection_r2": result.record.molecular_selection_r2,
                "best_epoch": result.record.best_epoch,
                "reconstruction_mse_supervised_all": result.reconstruction_mse,
                "config": asdict(result.record.config),
                "array": f"{result.model_id}.npy",
            }
        )

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(root / "embeddings.csv", index=False)
    np.savez_compressed(
        root / "embeddings.npz",
        **{result.model_id: result.z for result in embeddings},
        sample_id=np.asarray(reference_ids, dtype=str),
    )
    try:
        combined.to_parquet(root / "embeddings.parquet", index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.append(f"Parquet was not written ({exc}); CSV and NPZ/NPY outputs are complete.")
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return root, warnings


def load_egfr_wide(layout: ProjectDataLayout, sample_ids: Iterable[str]) -> pd.DataFrame:
    """Load one outcome row per molecular sample and keep canonical time columns."""

    frame = pd.read_excel(layout.egfr_table).copy()
    candidates = ("patient", "Patient", "Sample_ID", "sample_id", "id", "ID")
    id_column = next((column for column in candidates if column in frame.columns), None)
    if id_column is None:
        raise ValueError("Could not identify patient/sample ID in the eGFR table.")
    frame["patient"] = frame[id_column].map(normalize_sample_id)
    if frame["patient"].duplicated().any():
        duplicates = frame.loc[frame["patient"].duplicated(keep=False), "patient"].head(5)
        raise ValueError(f"eGFR table contains duplicate patient rows: {duplicates.tolist()}")
    known = [column for column in EGFR_COLUMNS if column in frame.columns]
    if not known:
        raise ValueError("eGFR table has none of the canonical longitudinal outcome columns.")
    wanted = {str(value) for value in sample_ids}
    frame = frame.loc[frame["patient"].isin(wanted)].copy()
    frame = frame.dropna(subset=known, how="all").reset_index(drop=True)
    frame["donor"] = frame["patient"].map(lambda value: str(value).split("_", 1)[0])
    if frame.empty:
        raise ValueError("No molecular samples with eGFR values were matched.")
    return frame


def egfr_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical eGFR columns to one row per patient and time point."""

    id_columns = [column for column in frame.columns if column not in EGFR_COLUMNS]
    known = [column for column in EGFR_COLUMNS if column in frame.columns]
    result = frame.melt(
        id_vars=id_columns,
        value_vars=known,
        var_name="time_column",
        value_name="egfr",
    ).dropna(subset=["egfr"])
    result["time"] = result["time_column"].map(TIME_LABELS)
    result["time"] = pd.Categorical(result["time"], categories=TIME_LEVELS, ordered=True)
    return result.drop(columns=["time_column"]).reset_index(drop=True)


def repeated_donor_folds(
    patients: pd.DataFrame,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    stratify_column: str | None = "KDRI_8",
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Create repeated patient-disjoint folds grouped by donor identity."""

    if patients["patient"].duplicated().any():
        raise ValueError("Fold input must contain one row per patient.")
    donors = patients["donor"].astype(str).to_numpy()
    unique_donors = np.unique(donors)
    if len(unique_donors) < int(n_splits):
        raise ValueError(
            f"Need at least {n_splits} donor groups, found {len(unique_donors)}."
        )
    folds: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(int(n_repeats)):
        random_state = int(seed) + repeat
        use_stratified = stratify_column is not None and stratify_column in patients.columns
        if use_stratified:
            values = pd.to_numeric(patients[stratify_column], errors="coerce")
            if values.notna().sum() < int(n_splits):
                use_stratified = False
            else:
                values = values.fillna(values.median())
                strata = pd.qcut(values, q=4, labels=False, duplicates="drop")
                use_stratified = int(pd.Series(strata).nunique()) >= 2
                if use_stratified:
                    stratum_donors = pd.DataFrame(
                        {"stratum": np.asarray(strata), "donor": donors}
                    ).groupby("stratum")["donor"].nunique()
                    use_stratified = bool((stratum_donors >= int(n_splits)).all())
        if use_stratified:
            splitter = StratifiedGroupKFold(
                n_splits=int(n_splits), shuffle=True, random_state=random_state
            )
            split_indices = splitter.split(patients, strata, groups=donors)
        else:
            rng = np.random.default_rng(random_state)
            donor_parts = np.array_split(rng.permutation(unique_donors), int(n_splits))
            split_indices = []
            all_indices = np.arange(len(patients), dtype=np.int64)
            for part in donor_parts:
                test_mask = np.isin(donors, part)
                split_indices.append((all_indices[~test_mask], all_indices[test_mask]))
        for fold, (train_index, test_index) in enumerate(split_indices):
            train_index = np.asarray(train_index, dtype=np.int64)
            test_index = np.asarray(test_index, dtype=np.int64)
            if set(donors[train_index]).intersection(donors[test_index]):
                raise RuntimeError("Donor leakage detected while creating CV folds.")
            folds.append((repeat, fold, train_index, test_index))
    return folds


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(train, dtype=np.float64).mean(axis=0)
    std = np.asarray(train, dtype=np.float64).std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (
        ((np.asarray(train, dtype=np.float64) - mean) / std).astype(np.float32),
        ((np.asarray(test, dtype=np.float64) - mean) / std).astype(np.float32),
    )


def _impute_standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute and standardize without using held-out-fold information."""

    train_values = np.asarray(train, dtype=np.float64)
    test_values = np.asarray(test, dtype=np.float64)
    medians = np.nanmedian(train_values, axis=0)
    if np.isnan(medians).any():
        bad = np.flatnonzero(np.isnan(medians)).tolist()
        raise ValueError(
            "Clinical columns contain no observed training values in this fold: "
            f"column positions {bad}"
        )
    train_values = np.where(np.isnan(train_values), medians, train_values)
    test_values = np.where(np.isnan(test_values), medians, test_values)
    return _standardize_train_test(train_values, test_values)


def prepare_mixed_model_design(
    egfr_wide: pd.DataFrame,
    embeddings: list[EmbeddingResult],
    aligned_expression: np.ndarray,
    expression_samples: pd.DataFrame,
    output_root: str | Path,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    stratify_column: str | None = "KDRI_8",
    include_pca: bool = True,
    include_clinical: bool = True,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build fold-specific, standardized train/test rows for R ``lme4``.

    Clinical predictors are median-imputed and standardized using the training
    patients in each fold. Their model terms interact with categorical time.
    """

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    patients = egfr_wide.reset_index(drop=True).copy()
    long = egfr_long(patients)
    sample_lookup = {value: index for index, value in enumerate(expression_samples["sample_id"])}
    missing = [value for value in patients["patient"] if value not in sample_lookup]
    if missing:
        raise ValueError(f"eGFR patients are missing molecular rows: {missing[:5]}")
    expression_rows = np.asarray([sample_lookup[value] for value in patients["patient"]], dtype=np.int64)
    expression = np.asarray(aligned_expression[expression_rows], dtype=np.float32)

    if include_clinical:
        clinical_columns = CLINICAL_COLUMNS
        missing_clinical = [column for column in clinical_columns if column not in patients]
        if missing_clinical:
            raise ValueError(
                "eGFR data are missing required clinical columns: "
                f"{missing_clinical}. Use include_clinical=False to run molecular-only models."
            )
        clinical = patients.loc[:, list(clinical_columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
    else:
        clinical = np.empty((len(patients), 0), dtype=np.float64)

    embedding_matrices: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {"time_only": "Time only"}
    for result in embeddings:
        lookup = {value: index for index, value in enumerate(result.samples["sample_id"])}
        embedding_matrices[result.model_id] = np.asarray(
            [result.z[lookup[value]] for value in patients["patient"]], dtype=np.float32
        )
        labels[result.model_id] = result.label

    fold_definitions = repeated_donor_folds(
        patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        stratify_column=stratify_column,
    )
    rows: list[pd.DataFrame] = []
    specifications: list[dict[str, object]] = []

    def append_model(
        model_id: str,
        label: str,
        repeat: int,
        fold: int,
        train_index: np.ndarray,
        test_index: np.ndarray,
        train_main_features: np.ndarray,
        test_main_features: np.ndarray,
        train_interaction_features: np.ndarray | None = None,
        test_interaction_features: np.ndarray | None = None,
        feature_description: str = "",
    ) -> None:
        if train_interaction_features is None:
            train_interaction_features = np.empty((len(train_index), 0), dtype=np.float32)
        if test_interaction_features is None:
            test_interaction_features = np.empty((len(test_index), 0), dtype=np.float32)
        n_main_features = int(train_main_features.shape[1])
        n_time_interaction_features = int(train_interaction_features.shape[1])
        n_features = n_main_features + n_time_interaction_features
        train_features = np.column_stack(
            (train_main_features, train_interaction_features)
        ).astype(np.float32, copy=False)
        test_features = np.column_stack(
            (test_main_features, test_interaction_features)
        ).astype(np.float32, copy=False)
        specifications.append(
            {
                "model_id": model_id,
                "model_label": label,
                "repeat": repeat,
                "fold": fold,
                "n_features": n_features,
                "n_main_features": n_main_features,
                "n_time_interaction_features": n_time_interaction_features,
                "feature_description": feature_description,
            }
        )
        for partition, indices, features in (
            ("train", train_index, train_features),
            ("test", test_index, test_features),
        ):
            patient_ids = patients.iloc[indices]["patient"].astype(str).tolist()
            subset = long.loc[long["patient"].isin(patient_ids), ["patient", "donor", "time", "egfr"]].copy()
            feature_lookup = {patients.iloc[index]["patient"]: features[position] for position, index in enumerate(indices)}
            subset.insert(0, "partition", partition)
            subset.insert(0, "fold", fold)
            subset.insert(0, "repeat", repeat)
            subset.insert(0, "model_id", model_id)
            for feature_index in range(n_features):
                subset[f"x{feature_index + 1}"] = [
                    float(feature_lookup[patient][feature_index]) for patient in subset["patient"]
                ]
            rows.append(subset)

    for repeat, fold, train_index, test_index in fold_definitions:
        empty_train = np.empty((len(train_index), 0), dtype=np.float32)
        empty_test = np.empty((len(test_index), 0), dtype=np.float32)
        append_model(
            "time_only",
            labels["time_only"],
            repeat,
            fold,
            train_index,
            test_index,
            empty_train,
            empty_test,
        )
        standardized_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model_id, matrix in embedding_matrices.items():
            train_features, test_features = _standardize_train_test(
                matrix[train_index], matrix[test_index]
            )
            standardized_embeddings[model_id] = (train_features, test_features)
            append_model(
                model_id,
                labels[model_id],
                repeat,
                fold,
                train_index,
                test_index,
                train_features,
                test_features,
                feature_description=f"{model_id} latent components",
            )
        train_pca: np.ndarray | None = None
        test_pca: np.ndarray | None = None
        if include_pca:
            winner_dim = embeddings[0].z.shape[1]
            n_components = min(int(winner_dim), len(train_index) - 1, expression.shape[1])
            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed + repeat)
            train_pca = pca.fit_transform(expression[train_index])
            test_pca = pca.transform(expression[test_index])
            train_pca, test_pca = _standardize_train_test(train_pca, test_pca)
            labels["pca"] = f"Fold-fitted PCA ({n_components} components)"
            append_model(
                "pca",
                labels["pca"],
                repeat,
                fold,
                train_index,
                test_index,
                train_pca,
                test_pca,
                feature_description="fold-fitted PCA components",
            )

        if include_clinical:
            train_clinical, test_clinical = _impute_standardize_train_test(
                clinical[train_index], clinical[test_index]
            )
            clinical_indices = {column: index for index, column in enumerate(clinical_columns)}

            for column, model_id in (
                ("don_patient_age", "clinical_age"),
                ("KDRI_8", "clinical_kdri"),
                ("Cold_ischemia_hours", "clinical_cold_ischemia"),
            ):
                index = clinical_indices[column]
                append_model(
                    model_id,
                    f"{CLINICAL_LABELS[column]} × time",
                    repeat,
                    fold,
                    train_index,
                    test_index,
                    empty_train,
                    empty_test,
                    train_clinical[:, [index]],
                    test_clinical[:, [index]],
                    feature_description=column,
                )

            append_model(
                "clinical_full",
                "Clinical (KDRI + donor age + cold ischemia) × time",
                repeat,
                fold,
                train_index,
                test_index,
                empty_train,
                empty_test,
                train_clinical,
                test_clinical,
                feature_description=" + ".join(clinical_columns),
            )

            winner_train, winner_test = standardized_embeddings["winner_z"]
            kdri_index = clinical_indices["KDRI_8"]
            append_model(
                "winner_z_kdri",
                "Winner z + KDRI × time",
                repeat,
                fold,
                train_index,
                test_index,
                winner_train,
                winner_test,
                train_clinical[:, [kdri_index]],
                test_clinical[:, [kdri_index]],
                feature_description="winner_z + KDRI_8",
            )
            append_model(
                "winner_z_clinical",
                "Winner z + full clinical × time",
                repeat,
                fold,
                train_index,
                test_index,
                winner_train,
                winner_test,
                train_clinical,
                test_clinical,
                feature_description="winner_z + " + " + ".join(clinical_columns),
            )

            if include_pca and train_pca is not None and test_pca is not None:
                append_model(
                    "pca_kdri",
                    "PCA + KDRI × time",
                    repeat,
                    fold,
                    train_index,
                    test_index,
                    train_pca,
                    test_pca,
                    train_clinical[:, [kdri_index]],
                    test_clinical[:, [kdri_index]],
                    feature_description="PCA + KDRI_8",
                )
                append_model(
                    "pca_clinical",
                    "PCA + full clinical × time",
                    repeat,
                    fold,
                    train_index,
                    test_index,
                    train_pca,
                    test_pca,
                    train_clinical,
                    test_clinical,
                    feature_description="PCA + " + " + ".join(clinical_columns),
                )

    design = pd.concat(rows, ignore_index=True, sort=False)
    specs = pd.DataFrame(specifications).drop_duplicates().reset_index(drop=True)
    design_path = root / "mixed_model_design.csv"
    specs_path = root / "mixed_model_specs.csv"
    design.to_csv(design_path, index=False)
    specs.to_csv(specs_path, index=False)
    return design_path, specs_path, specs


def prepare_nested_mixed_model_design(
    egfr_wide: pd.DataFrame,
    embeddings: list[EmbeddingResult],
    aligned_expression: np.ndarray,
    expression_samples: pd.DataFrame,
    output_root: str | Path,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    inner_splits: int = 5,
    seed: int = 0,
    stratify_column: str | None = "KDRI_8",
    include_pca: bool = True,
    include_clinical: bool = True,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build inner-selection and outer-evaluation fits for nested donor CV.

    Inner folds compare only the six molecular z candidates. Outer folds contain
    every candidate plus common PCA and clinical baselines; after R finishes,
    :func:`finalize_nested_selection` retains the candidate chosen without using
    that outer fold's outcomes.
    """

    if not embeddings:
        raise ValueError("Nested eGFR evaluation requires at least one candidate encoder.")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    patients = egfr_wide.reset_index(drop=True).copy()
    long = egfr_long(patients)
    sample_lookup = {
        value: index for index, value in enumerate(expression_samples["sample_id"])
    }
    missing = [value for value in patients["patient"] if value not in sample_lookup]
    if missing:
        raise ValueError(f"eGFR patients are missing molecular rows: {missing[:5]}")
    expression_rows = np.asarray(
        [sample_lookup[value] for value in patients["patient"]], dtype=np.int64
    )
    expression = np.asarray(aligned_expression[expression_rows], dtype=np.float32)

    if include_clinical:
        missing_clinical = [column for column in CLINICAL_COLUMNS if column not in patients]
        if missing_clinical:
            raise ValueError(
                "eGFR data are missing required clinical columns: "
                f"{missing_clinical}. Use include_clinical=False for molecular-only models."
            )
        clinical = patients.loc[:, list(CLINICAL_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
    else:
        clinical = np.empty((len(patients), 0), dtype=np.float64)

    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {"time_only": "Time only"}
    latent_dims: dict[str, int] = {}
    for result in embeddings:
        lookup = {value: index for index, value in enumerate(result.samples["sample_id"])}
        matrices[result.model_id] = np.asarray(
            [result.z[lookup[value]] for value in patients["patient"]], dtype=np.float32
        )
        labels[result.model_id] = result.label
        latent_dims[result.model_id] = int(result.z.shape[1])

    rows: list[pd.DataFrame] = []
    specifications: list[dict[str, object]] = []

    def append_fit(
        *,
        fit_id: str,
        stage: str,
        model_id: str,
        model_label: str,
        candidate_id: str,
        outer_repeat: int,
        outer_fold: int,
        inner_fold: int,
        train_index: np.ndarray,
        test_index: np.ndarray,
        train_main: np.ndarray,
        test_main: np.ndarray,
        train_interactions: np.ndarray | None = None,
        test_interactions: np.ndarray | None = None,
        feature_description: str = "",
    ) -> None:
        if train_interactions is None:
            train_interactions = np.empty((len(train_index), 0), dtype=np.float32)
        if test_interactions is None:
            test_interactions = np.empty((len(test_index), 0), dtype=np.float32)
        n_main = int(train_main.shape[1])
        n_interactions = int(train_interactions.shape[1])
        train_features = np.column_stack((train_main, train_interactions)).astype(
            np.float32, copy=False
        )
        test_features = np.column_stack((test_main, test_interactions)).astype(
            np.float32, copy=False
        )
        specifications.append(
            {
                "fit_id": fit_id,
                "stage": stage,
                "model_id": model_id,
                "model_label": model_label,
                "candidate_id": candidate_id,
                "repeat": outer_repeat,
                "fold": outer_fold,
                "inner_fold": inner_fold,
                "n_features": n_main + n_interactions,
                "n_main_features": n_main,
                "n_time_interaction_features": n_interactions,
                "feature_description": feature_description,
            }
        )
        for partition, indices, features in (
            ("train", train_index, train_features),
            ("test", test_index, test_features),
        ):
            patient_ids = patients.iloc[indices]["patient"].astype(str).tolist()
            subset = long.loc[
                long["patient"].isin(patient_ids), ["patient", "donor", "time", "egfr"]
            ].copy()
            feature_lookup = {
                patients.iloc[index]["patient"]: features[position]
                for position, index in enumerate(indices)
            }
            subset.insert(0, "partition", partition)
            subset.insert(0, "fit_id", fit_id)
            for feature_index in range(features.shape[1]):
                subset[f"x{feature_index + 1}"] = [
                    float(feature_lookup[patient][feature_index])
                    for patient in subset["patient"]
                ]
            rows.append(subset)

    outer_folds = repeated_donor_folds(
        patients,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        stratify_column=stratify_column,
    )
    for outer_repeat, outer_fold, outer_train, outer_test in outer_folds:
        outer_patients = patients.iloc[outer_train].reset_index(drop=True)
        inner_folds = repeated_donor_folds(
            outer_patients,
            n_splits=inner_splits,
            n_repeats=1,
            seed=seed + 10_000 + outer_repeat * 101 + outer_fold,
            stratify_column=stratify_column,
        )
        for _, inner_fold, relative_train, relative_test in inner_folds:
            inner_train = outer_train[relative_train]
            inner_test = outer_train[relative_test]
            for candidate_id, matrix in matrices.items():
                train_z, test_z = _standardize_train_test(
                    matrix[inner_train], matrix[inner_test]
                )
                append_fit(
                    fit_id=(
                        f"inner_r{outer_repeat}_f{outer_fold}_i{inner_fold}_{candidate_id}"
                    ),
                    stage="inner_selection",
                    model_id=candidate_id,
                    model_label=labels[candidate_id],
                    candidate_id=candidate_id,
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    train_index=inner_train,
                    test_index=inner_test,
                    train_main=train_z,
                    test_main=test_z,
                    feature_description=f"{candidate_id} latent components",
                )

        empty_train = np.empty((len(outer_train), 0), dtype=np.float32)
        empty_test = np.empty((len(outer_test), 0), dtype=np.float32)
        append_fit(
            fit_id=f"outer_r{outer_repeat}_f{outer_fold}_time_only",
            stage="outer_evaluation",
            model_id="time_only",
            model_label=labels["time_only"],
            candidate_id="",
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            inner_fold=-1,
            train_index=outer_train,
            test_index=outer_test,
            train_main=empty_train,
            test_main=empty_test,
        )

        standardized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for candidate_id, matrix in matrices.items():
            train_z, test_z = _standardize_train_test(
                matrix[outer_train], matrix[outer_test]
            )
            standardized[candidate_id] = (train_z, test_z)
            append_fit(
                fit_id=f"outer_r{outer_repeat}_f{outer_fold}_{candidate_id}",
                stage="outer_evaluation",
                model_id=candidate_id,
                model_label=labels[candidate_id],
                candidate_id=candidate_id,
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                inner_fold=-1,
                train_index=outer_train,
                test_index=outer_test,
                train_main=train_z,
                test_main=test_z,
                feature_description=f"{candidate_id} latent components",
            )

        pca_features: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if include_pca:
            for dimension in sorted(set(latent_dims.values())):
                n_components = min(dimension, len(outer_train) - 1, expression.shape[1])
                pca = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=seed + outer_repeat,
                )
                train_pca = pca.fit_transform(expression[outer_train])
                test_pca = pca.transform(expression[outer_test])
                train_pca, test_pca = _standardize_train_test(train_pca, test_pca)
                pca_features[dimension] = (train_pca, test_pca)
                append_fit(
                    fit_id=f"outer_r{outer_repeat}_f{outer_fold}_pca_d{dimension}",
                    stage="outer_evaluation",
                    model_id=f"pca_d{dimension}",
                    model_label=f"Fold-fitted PCA ({dimension} components)",
                    candidate_id="",
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    inner_fold=-1,
                    train_index=outer_train,
                    test_index=outer_test,
                    train_main=train_pca,
                    test_main=test_pca,
                    feature_description=f"fold-fitted PCA ({dimension})",
                )

        if include_clinical:
            train_clinical, test_clinical = _impute_standardize_train_test(
                clinical[outer_train], clinical[outer_test]
            )
            clinical_indices = {column: index for index, column in enumerate(CLINICAL_COLUMNS)}
            for column, model_id in (
                ("don_patient_age", "clinical_age"),
                ("KDRI_8", "clinical_kdri"),
                ("Cold_ischemia_hours", "clinical_cold_ischemia"),
            ):
                index = clinical_indices[column]
                append_fit(
                    fit_id=f"outer_r{outer_repeat}_f{outer_fold}_{model_id}",
                    stage="outer_evaluation",
                    model_id=model_id,
                    model_label=f"{CLINICAL_LABELS[column]} × time",
                    candidate_id="",
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    inner_fold=-1,
                    train_index=outer_train,
                    test_index=outer_test,
                    train_main=empty_train,
                    test_main=empty_test,
                    train_interactions=train_clinical[:, [index]],
                    test_interactions=test_clinical[:, [index]],
                    feature_description=column,
                )
            append_fit(
                fit_id=f"outer_r{outer_repeat}_f{outer_fold}_clinical_full",
                stage="outer_evaluation",
                model_id="clinical_full",
                model_label="Clinical (KDRI + donor age + cold ischemia) × time",
                candidate_id="",
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                inner_fold=-1,
                train_index=outer_train,
                test_index=outer_test,
                train_main=empty_train,
                test_main=empty_test,
                train_interactions=train_clinical,
                test_interactions=test_clinical,
                feature_description=" + ".join(CLINICAL_COLUMNS),
            )
            kdri_index = clinical_indices["KDRI_8"]
            for candidate_id, (train_z, test_z) in standardized.items():
                for suffix, interaction_train, interaction_test, description in (
                    (
                        "kdri",
                        train_clinical[:, [kdri_index]],
                        test_clinical[:, [kdri_index]],
                        "KDRI_8",
                    ),
                    ("clinical", train_clinical, test_clinical, " + ".join(CLINICAL_COLUMNS)),
                ):
                    append_fit(
                        fit_id=(
                            f"outer_r{outer_repeat}_f{outer_fold}_{candidate_id}_{suffix}"
                        ),
                        stage="outer_evaluation",
                        model_id=f"{candidate_id}_{suffix}",
                        model_label=f"{labels[candidate_id]} + {description} × time",
                        candidate_id=candidate_id,
                        outer_repeat=outer_repeat,
                        outer_fold=outer_fold,
                        inner_fold=-1,
                        train_index=outer_train,
                        test_index=outer_test,
                        train_main=train_z,
                        test_main=test_z,
                        train_interactions=interaction_train,
                        test_interactions=interaction_test,
                        feature_description=f"{candidate_id} + {description}",
                    )
            for dimension, (train_pca, test_pca) in pca_features.items():
                for suffix, interaction_train, interaction_test, description in (
                    (
                        "kdri",
                        train_clinical[:, [kdri_index]],
                        test_clinical[:, [kdri_index]],
                        "KDRI_8",
                    ),
                    ("clinical", train_clinical, test_clinical, " + ".join(CLINICAL_COLUMNS)),
                ):
                    append_fit(
                        fit_id=f"outer_r{outer_repeat}_f{outer_fold}_pca_d{dimension}_{suffix}",
                        stage="outer_evaluation",
                        model_id=f"pca_d{dimension}_{suffix}",
                        model_label=f"PCA ({dimension}) + {description} × time",
                        candidate_id="",
                        outer_repeat=outer_repeat,
                        outer_fold=outer_fold,
                        inner_fold=-1,
                        train_index=outer_train,
                        test_index=outer_test,
                        train_main=train_pca,
                        test_main=test_pca,
                        train_interactions=interaction_train,
                        test_interactions=interaction_test,
                        feature_description=f"PCA ({dimension}) + {description}",
                    )

    design = pd.concat(rows, ignore_index=True, sort=False)
    specs = pd.DataFrame(specifications)
    if specs["fit_id"].duplicated().any():
        raise RuntimeError("Nested mixed-model fit IDs are not unique.")
    design_path = root / "nested_mixed_model_design.csv"
    specs_path = root / "nested_mixed_model_specs.csv"
    design.to_csv(design_path, index=False)
    specs.to_csv(specs_path, index=False)
    return design_path, specs_path, specs


def run_lme4_benchmark(
    design_path: str | Path,
    specs_path: str | Path,
    r_script: str | Path,
    output_root: str | Path,
    *,
    rscript: str | Path = "Rscript",
    output_prefix: str = "",
) -> tuple[Path, Path]:
    """Fit every fold/model with R ``lme4`` in one reproducible call."""

    root = Path(output_root).expanduser().resolve()
    metrics_path = root / f"{output_prefix}fold_metrics.csv"
    predictions_path = root / f"{output_prefix}oof_predictions.csv"
    command = [
        str(rscript),
        str(Path(r_script).expanduser().resolve()),
        str(Path(design_path).expanduser().resolve()),
        str(Path(specs_path).expanduser().resolve()),
        str(metrics_path),
        str(predictions_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Rscript executable was not found: {rscript}. Use --rscript with an absolute path."
        ) from exc
    (root / f"{output_prefix}lme4_stdout.txt").write_text(result.stdout, encoding="utf-8")
    stderr_path = root / f"{output_prefix}lme4_stderr.txt"
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"R mixed-model benchmark failed with exit code {result.returncode}. "
            f"See {stderr_path}."
        )
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("R completed without producing the expected result files.")
    return metrics_path, predictions_path


def finalize_nested_selection(
    metrics_path: str | Path,
    predictions_path: str | Path,
    embeddings: list[EmbeddingResult],
    output_root: str | Path,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Choose candidates in inner CV and assemble unbiased outer predictions."""

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    required = {
        "stage",
        "model_id",
        "candidate_id",
        "repeat",
        "fold",
        "inner_fold",
        "rmse",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Nested mixed-model metrics are missing columns: {sorted(missing)}")
    candidate_metadata = {
        result.model_id: {
            "run": result.record.run,
            "preprocessing": result.record.method,
            "architecture": result.record.architecture,
            "latent_dim": int(result.z.shape[1]),
            "molecular_selection_mse": result.record.molecular_selection_mse,
        }
        for result in embeddings
    }
    candidate_ids = set(candidate_metadata)
    inner = metrics.loc[
        metrics["stage"].eq("inner_selection") & metrics["model_id"].isin(candidate_ids)
    ].copy()
    if inner.empty:
        raise ValueError("Nested benchmark produced no inner candidate-selection metrics.")
    inner_scores = (
        inner.groupby(["repeat", "fold", "model_id", "model_label"], as_index=False)
        .agg(mean_inner_rmse=("rmse", "mean"), inner_folds=("rmse", "count"))
        .sort_values(["repeat", "fold", "mean_inner_rmse", "model_id"])
        .reset_index(drop=True)
    )
    selections = inner_scores.groupby(["repeat", "fold"], as_index=False).first()
    selections = selections.rename(
        columns={"model_id": "selected_candidate_id", "model_label": "selected_candidate_label"}
    )
    for column in (
        "run",
        "preprocessing",
        "architecture",
        "latent_dim",
        "molecular_selection_mse",
    ):
        selections[column] = selections["selected_candidate_id"].map(
            lambda value, name=column: candidate_metadata[value][name]
        )

    outer = metrics.loc[metrics["stage"].eq("outer_evaluation")].copy()
    outer_predictions = predictions.loc[predictions["stage"].eq("outer_evaluation")].copy()
    keys = ["repeat", "fold"]
    deployment_ranking = (
        outer.loc[outer["model_id"].isin(candidate_ids)]
        .groupby(["model_id", "model_label"], as_index=False)
        .agg(mean_cv_rmse=("rmse", "mean"), sd_cv_rmse=("rmse", "std"), n_folds=("rmse", "count"))
        .sort_values(["mean_cv_rmse", "model_id"])
        .reset_index(drop=True)
    )
    frequency = selections["selected_candidate_id"].value_counts()
    deployment_ranking["nested_selection_count"] = deployment_ranking["model_id"].map(
        frequency
    ).fillna(0).astype(int)
    deployment_ranking["nested_selection_fraction"] = (
        deployment_ranking["nested_selection_count"] / len(selections)
    )
    for column in ("run", "preprocessing", "architecture", "latent_dim", "molecular_selection_mse"):
        deployment_ranking[column] = deployment_ranking["model_id"].map(
            lambda value, name=column: candidate_metadata[value][name]
        )

    final_metrics: list[pd.DataFrame] = []
    final_predictions: list[pd.DataFrame] = []
    baseline_ids = {
        "time_only",
        "clinical_age",
        "clinical_kdri",
        "clinical_cold_ischemia",
        "clinical_full",
    }
    final_metrics.append(outer.loc[outer["model_id"].isin(baseline_ids)].copy())
    final_predictions.append(
        outer_predictions.loc[outer_predictions["model_id"].isin(baseline_ids)].copy()
    )

    def append_selected(
        source_id,
        target_id: str,
        target_label: str,
    ) -> None:
        metric_parts: list[pd.DataFrame] = []
        prediction_parts: list[pd.DataFrame] = []
        for selection in selections.itertuples(index=False):
            wanted = source_id(selection)
            mask = (
                outer["repeat"].eq(selection.repeat)
                & outer["fold"].eq(selection.fold)
                & outer["model_id"].eq(wanted)
            )
            part = outer.loc[mask].copy()
            if len(part) != 1:
                raise ValueError(
                    f"Expected one outer metric for {wanted} in repeat {selection.repeat}, "
                    f"fold {selection.fold}; found {len(part)}."
                )
            part["selected_candidate_id"] = selection.selected_candidate_id
            part["model_id"] = target_id
            part["model_label"] = target_label
            metric_parts.append(part)
            prediction_mask = (
                outer_predictions["repeat"].eq(selection.repeat)
                & outer_predictions["fold"].eq(selection.fold)
                & outer_predictions["model_id"].eq(wanted)
            )
            prediction_part = outer_predictions.loc[prediction_mask].copy()
            if prediction_part.empty:
                raise ValueError(
                    f"Missing outer predictions for {wanted} in repeat {selection.repeat}, "
                    f"fold {selection.fold}."
                )
            prediction_part["selected_candidate_id"] = selection.selected_candidate_id
            prediction_part["model_id"] = target_id
            prediction_part["model_label"] = target_label
            prediction_parts.append(prediction_part)
        final_metrics.append(pd.concat(metric_parts, ignore_index=True))
        final_predictions.append(pd.concat(prediction_parts, ignore_index=True))

    append_selected(
        lambda selection: selection.selected_candidate_id,
        "winner_z",
        "Nested-CV-selected z",
    )
    available_outer = set(outer["model_id"])
    if any(f"{candidate}_kdri" in available_outer for candidate in candidate_ids):
        append_selected(
            lambda selection: f"{selection.selected_candidate_id}_kdri",
            "winner_z_kdri",
            "Nested-CV-selected z + KDRI × time",
        )
        append_selected(
            lambda selection: f"{selection.selected_candidate_id}_clinical",
            "winner_z_clinical",
            "Nested-CV-selected z + full clinical × time",
        )
    if any(model_id.startswith("pca_d") for model_id in available_outer):
        append_selected(
            lambda selection: f"pca_d{int(selection.latent_dim)}",
            "pca",
            "Fold-fitted PCA matched to selected z dimension",
        )
        if any(model_id.endswith("_kdri") for model_id in available_outer):
            append_selected(
                lambda selection: f"pca_d{int(selection.latent_dim)}_kdri",
                "pca_kdri",
                "Matched-dimension PCA + KDRI × time",
            )
            append_selected(
                lambda selection: f"pca_d{int(selection.latent_dim)}_clinical",
                "pca_clinical",
                "Matched-dimension PCA + full clinical × time",
            )

    final_metric_frame = pd.concat(final_metrics, ignore_index=True, sort=False)
    final_prediction_frame = pd.concat(final_predictions, ignore_index=True, sort=False)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    inner_scores.to_csv(root / "nested_inner_candidate_scores.csv", index=False)
    selections.to_csv(root / "nested_outer_selections.csv", index=False)
    deployment_ranking.to_csv(root / "candidate_cv_ranking.csv", index=False)
    final_metrics_path = root / "fold_metrics.csv"
    final_predictions_path = root / "oof_predictions.csv"
    final_metric_frame.to_csv(final_metrics_path, index=False)
    final_prediction_frame.to_csv(final_predictions_path, index=False)
    return final_metrics_path, final_predictions_path, selections, deployment_ranking


def summarize_mixed_model_results(
    metrics_path: str | Path,
    predictions_path: str | Path,
    output_root: str | Path,
    *,
    winner_model_id: str = "winner_z",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create matched-fold time, winner, and clinical incremental summaries."""

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    required = {"model_id", "model_label", "repeat", "fold", "rmse", "mae"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Mixed-model metrics are missing columns: {sorted(missing)}")
    key = ["repeat", "fold"]
    reference = metrics.loc[metrics["model_id"].eq("time_only"), key + ["rmse"]].rename(
        columns={"rmse": "rmse_time_only"}
    )
    merged = metrics.merge(reference, on=key, how="left", validate="many_to_one")
    merged["delta_vs_time_only"] = merged["rmse_time_only"] - merged["rmse"]

    pooled = (
        predictions.assign(squared_error=lambda value: (value["egfr"] - value["prediction"]) ** 2)
        .groupby("model_id", as_index=False)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "pooled_mse"})
    )
    pooled["pooled_rmse"] = np.sqrt(pooled["pooled_mse"])
    summary = (
        merged.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_delta_vs_time=("delta_vs_time_only", "mean"),
            sd_delta_vs_time=("delta_vs_time_only", "std"),
            positive_folds_vs_time=("delta_vs_time_only", lambda value: float((value > 0).mean())),
            n_folds=("rmse", "count"),
        )
        .merge(pooled[["model_id", "pooled_rmse"]], on="model_id", how="left")
    )
    summary["se_delta_vs_time"] = summary["sd_delta_vs_time"] / np.sqrt(summary["n_folds"])
    summary["lcb_delta_vs_time"] = (
        summary["mean_delta_vs_time"] - summary["se_delta_vs_time"]
    )
    summary = summary.sort_values("mean_rmse").reset_index(drop=True)

    winner = metrics.loc[metrics["model_id"].eq(winner_model_id), key + ["rmse"]].rename(
        columns={"rmse": "winner_rmse"}
    )
    comparisons = metrics.loc[~metrics["model_id"].eq(winner_model_id)].merge(
        winner, on=key, how="left", validate="many_to_one"
    )
    comparisons["winner_gain"] = comparisons["rmse"] - comparisons["winner_rmse"]
    pairwise = (
        comparisons.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            mean_winner_gain=("winner_gain", "mean"),
            sd_winner_gain=("winner_gain", "std"),
            winner_better_fraction=("winner_gain", lambda value: float((value > 0).mean())),
            n_folds=("winner_gain", "count"),
        )
        .sort_values("mean_winner_gain", ascending=False)
        .reset_index(drop=True)
    )
    pairwise["se_winner_gain"] = pairwise["sd_winner_gain"] / np.sqrt(pairwise["n_folds"])
    pairwise["lcb_winner_gain"] = pairwise["mean_winner_gain"] - pairwise["se_winner_gain"]

    clinical_comparisons = (
        ("winner_z_kdri", "clinical_kdri", "Winner z beyond KDRI"),
        ("pca_kdri", "clinical_kdri", "PCA beyond KDRI"),
        ("winner_z_clinical", "clinical_full", "Winner z beyond full clinical"),
        ("pca_clinical", "clinical_full", "PCA beyond full clinical"),
        ("winner_z_clinical", "pca_clinical", "Winner z vs PCA beyond full clinical"),
    )
    incremental_rows: list[dict[str, object]] = []
    available = set(metrics["model_id"])
    for augmented_id, baseline_id, comparison_label in clinical_comparisons:
        if augmented_id not in available or baseline_id not in available:
            continue
        augmented = metrics.loc[
            metrics["model_id"].eq(augmented_id), key + ["rmse"]
        ].rename(columns={"rmse": "augmented_rmse"})
        baseline = metrics.loc[
            metrics["model_id"].eq(baseline_id), key + ["rmse"]
        ].rename(columns={"rmse": "baseline_rmse"})
        matched = baseline.merge(augmented, on=key, validate="one_to_one")
        gain = matched["baseline_rmse"] - matched["augmented_rmse"]
        n_folds = len(gain)
        sd_gain = float(gain.std(ddof=1)) if n_folds > 1 else float("nan")
        se_gain = sd_gain / np.sqrt(n_folds) if n_folds else float("nan")
        incremental_rows.append(
            {
                "comparison_label": comparison_label,
                "augmented_model_id": augmented_id,
                "baseline_model_id": baseline_id,
                "mean_gain": float(gain.mean()),
                "sd_gain": sd_gain,
                "se_gain": se_gain,
                "lcb_gain": float(gain.mean()) - se_gain,
                "augmented_better_fraction": float((gain > 0).mean()),
                "n_folds": n_folds,
            }
        )
    incremental_columns = [
        "comparison_label",
        "augmented_model_id",
        "baseline_model_id",
        "mean_gain",
        "sd_gain",
        "se_gain",
        "lcb_gain",
        "augmented_better_fraction",
        "n_folds",
    ]
    clinical_incremental = pd.DataFrame(incremental_rows, columns=incremental_columns)
    if not clinical_incremental.empty:
        clinical_incremental = clinical_incremental.sort_values(
            "mean_gain", ascending=False
        ).reset_index(drop=True)

    root = Path(output_root).expanduser().resolve()
    merged.to_csv(root / "fold_metrics_with_deltas.csv", index=False)
    summary.to_csv(root / "summary.csv", index=False)
    pairwise.to_csv(root / "winner_pairwise_summary.csv", index=False)
    clinical_incremental.to_csv(root / "clinical_incremental_summary.csv", index=False)
    return summary, pairwise, clinical_incremental

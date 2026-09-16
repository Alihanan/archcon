"""Utilities for molecular pretraining split design and latent autoencoder planning.

This module provides reproducible GSE-disjoint train/validation/test splits for
the canonical unique-GSM GEO store plus configurable autoencoder architecture
presets, parameter estimates, loss descriptions, and paper-style diagrams used
by the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .._compat import strict_zip
from .geo_rma import GeoExpressionStore, load_geo_expression_store

LOSS_MSE = "MSE / L2 reconstruction (thesis baseline)"
LOSS_HUBER = "Smooth L1 / Huber reconstruction"
LOSS_COSINE = "MSE + profile cosine reconstruction"
LOSS_DVIB = "β-VAE · MSE + KL latent prior"
LOSS_MMD = "WAE / InfoVAE · MSE + MMD latent prior"
LOSS_MASKED = "Masked denoising reconstruction · masked MSE"
LOSS_SOFTMAX = "Softmax distribution reconstruction (legacy experimental)"

LOSS_OPTIONS = [
    LOSS_MSE,
    LOSS_HUBER,
    LOSS_COSINE,
    LOSS_DVIB,
    LOSS_MMD,
    LOSS_MASKED,
]


LEGACY_LOSS_ALIASES = {
    "MSE / L2 reconstruction (thesis default)": LOSS_MSE,
    "Gaussian likelihood + KL bottleneck (DVIB/VAE-style)": LOSS_DVIB,
    "Softmax cross-entropy reconstruction (experimental)": LOSS_SOFTMAX,
    "Softmax distribution reconstruction (experimental)": LOSS_SOFTMAX,
}


def normalize_loss_name(value: str) -> str:
    """Map 0.4.1 checkpoint/UI labels to the current objective names."""
    name = str(value)
    return LEGACY_LOSS_ALIASES.get(name, name)
ACTIVATION_OPTIONS = ["ReLU", "GELU", "SiLU", "ELU", "Tanh", "LeakyReLU"]

ARCH_STADNIUK = "Stadniuk MLP"
ARCH_RESNET_LN = "ResNet MLP + LayerNorm"

# Legacy 0.4.x architecture labels remain loadable for checkpoint compatibility,
# but only the two research families below are offered for new experiments.
ARCH_DENSE = "Dense MLP"
ARCH_RESIDUAL = "Residual MLP"
ARCH_GATED_RESIDUAL = "Gated residual MLP"
ARCH_LOW_RANK = "Low-rank input/output MLP"
ARCHITECTURE_OPTIONS = [ARCH_STADNIUK, ARCH_RESNET_LN]
ARCHITECTURE_FAMILIES = [
    *ARCHITECTURE_OPTIONS,
    ARCH_DENSE,
    ARCH_RESIDUAL,
    ARCH_GATED_RESIDUAL,
    ARCH_LOW_RANK,
]


@dataclass(frozen=True)
class HiddenStagePlan:
    """One width-changing hidden stage in the executable MLP plan.

    Residual units, when enabled, always operate *after* the projection at the
    stage output width.  Keeping this as a small pure-data object gives the UI
    diagram, the PyTorch audit view, and the model builder one structural source
    of truth.
    """

    role: str
    index: int
    in_features: int
    out_features: int
    factorized_projection: bool
    residual_blocks: int
    gated_residual: bool
    normalization: str
    residual_expansion: int


@dataclass(frozen=True)
class AutoencoderExecutionPlan:
    """Pure-data architecture plan shared by visualization and PyTorch build."""

    input_dim: int
    hidden_widths: tuple[int, ...]
    latent_dim: int
    architecture_family: str
    low_rank_dim: int
    encoder_stages: tuple[HiddenStagePlan, ...]
    decoder_stages: tuple[HiddenStagePlan, ...]
    output_factorized: bool


def autoencoder_execution_plan(
    input_dim: int,
    hidden_widths: list[int] | tuple[int, ...],
    latent_dim: int,
    architecture_family: str = ARCH_DENSE,
    *,
    low_rank_dim: int = 64,
    residual_blocks: int = 1,
    residual_expansion: int = 1,
    stadniuk_batch_norm: bool = False,
) -> AutoencoderExecutionPlan:
    """Create the canonical executable stage plan for one autoencoder.

    The plan deliberately distinguishes a width-changing projection from the
    same-width residual blocks that follow it.  For example, a 512→256 stage is
    ``Linear(512, 256)`` followed by residual units of width 256; an identity
    shortcut never attempts to add a 512-vector directly to a 256-vector.
    """
    widths = tuple(int(value) for value in hidden_widths)
    if int(input_dim) < 1 or int(latent_dim) < 1 or not widths:
        raise ValueError("Positive input, hidden, and latent dimensions are required.")
    if any(value < 1 for value in widths):
        raise ValueError("Hidden widths must be positive.")
    if architecture_family not in ARCHITECTURE_FAMILIES:
        raise ValueError(f"Unknown architecture family: {architecture_family}")

    residual_family = architecture_family in {ARCH_RESNET_LN, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL}
    n_residual = max(0, int(residual_blocks)) if residual_family else 0
    gated = architecture_family == ARCH_GATED_RESIDUAL
    normalization = (
        "BatchNorm"
        if architecture_family == ARCH_STADNIUK and bool(stadniuk_batch_norm)
        else "LayerNorm"
        if residual_family
        else "None"
    )
    expansion = max(1, int(residual_expansion)) if residual_family else 1

    encoder_dims = (int(input_dim), *widths)
    encoder_stages = tuple(
        HiddenStagePlan(
            role="encoder",
            index=index + 1,
            in_features=int(left),
            out_features=int(right),
            factorized_projection=(architecture_family == ARCH_LOW_RANK and index == 0),
            residual_blocks=n_residual,
            gated_residual=gated,
            normalization=normalization,
            residual_expansion=expansion,
        )
        for index, (left, right) in enumerate(
            strict_zip(encoder_dims[:-1], encoder_dims[1:])
        )
    )

    decoder_widths = tuple(reversed(widths))
    decoder_dims = (int(latent_dim), *decoder_widths)
    decoder_stages = tuple(
        HiddenStagePlan(
            role="decoder",
            index=index + 1,
            in_features=int(left),
            out_features=int(right),
            factorized_projection=False,
            residual_blocks=n_residual,
            gated_residual=gated,
            normalization=normalization,
            residual_expansion=expansion,
        )
        for index, (left, right) in enumerate(
            strict_zip(decoder_dims[:-1], decoder_dims[1:])
        )
    )

    return AutoencoderExecutionPlan(
        input_dim=int(input_dim),
        hidden_widths=widths,
        latent_dim=int(latent_dim),
        architecture_family=architecture_family,
        low_rank_dim=max(1, int(low_rank_dim)),
        encoder_stages=encoder_stages,
        decoder_stages=decoder_stages,
        output_factorized=architecture_family == ARCH_LOW_RANK,
    )


@dataclass(frozen=True)
class ArchitecturePreset:
    """Small, explicit architecture preset for GEO reconstruction pretraining."""

    key: str
    label: str
    family: str
    hidden_widths: tuple[int, ...]
    latent_dim: int
    activation: str
    dropout: float
    weight_decay: float
    l2_lambda: float
    low_rank_dim: int
    residual_blocks: int
    residual_expansion: int
    stadniuk_batch_norm: bool
    batch_size: int
    learning_rate: float
    note: str


ARCHITECTURE_PRESETS: dict[str, ArchitecturePreset] = {
    "stadniuk": ArchitecturePreset(
        key="stadniuk",
        label="📘 Stadniuk MLP · 256→64",
        family=ARCH_STADNIUK,
        hidden_widths=(256, 64),
        latent_dim=3,
        activation="ReLU",
        dropout=0.10,
        weight_decay=0.0,
        l2_lambda=1e-5,
        low_rank_dim=64,
        residual_blocks=0,
        residual_expansion=1,
        stadniuk_batch_norm=False,
        batch_size=64,
        learning_rate=1e-3,
        note=(
            "Faithful final Stadniuk-style dense encoder/decoder with ReLU and no hidden "
            "normalization. The comparison sweep also tests the legacy/experimental BatchNorm variant."
        ),
    ),
    "resnet_ln": ArchitecturePreset(
        key="resnet_ln",
        label="↪ ResNet-LN · 256→128→64",
        family=ARCH_RESNET_LN,
        hidden_widths=(256, 128, 64),
        latent_dim=8,
        activation="GELU",
        dropout=0.00,
        weight_decay=0.0,
        l2_lambda=0.0,
        low_rank_dim=64,
        residual_blocks=1,
        residual_expansion=2,
        stadniuk_batch_norm=False,
        batch_size=64,
        learning_rate=1e-3,
        note=(
            "Single MLP with pre-LayerNorm same-width residual FFN blocks. The comparison "
            "grid varies stage depth, blocks per stage, and FFN expansion."
        ),
    ),
}
ARCHITECTURE_PRESET_OPTIONS = [preset.label for preset in ARCHITECTURE_PRESETS.values()]


def architecture_preset_from_label(label: str) -> ArchitecturePreset:
    for preset in ARCHITECTURE_PRESETS.values():
        if preset.label == str(label):
            return preset
    raise ValueError(f"Unknown architecture preset: {label}")


def parse_hidden_widths(value: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse user-entered comma/arrow-separated hidden widths."""
    if isinstance(value, (list, tuple)):
        widths = [int(item) for item in value]
    else:
        cleaned = str(value).replace("→", ",").replace("->", ",").replace(";", ",")
        widths = [int(part.strip()) for part in cleaned.split(",") if part.strip()]
    if not widths or len(widths) > 8 or any(width < 2 or width > 8192 for width in widths):
        raise ValueError("Hidden widths must contain 1–8 integers between 2 and 8192.")
    return widths


def architecture_research_markdown() -> str:
    return """
### 🧭 Two-model comparison

ArchCon now keeps the architecture experiment deliberately narrow: **Stadniuk-style dense MLP** versus **ResNet-LN**. The Stadniuk branch preserves the conventional dense/ReLU autoencoder shape and explicitly tests both the final no-normalization form and a BatchNorm variant. The ResNet branch asks whether same-width residual FFN blocks and LayerNorm improve optimization at comparable outer widths. Legacy low-rank/gated 0.4.x checkpoints remain loadable but are no longer primary new-run choices.
"""


def _geo_url(accession: str) -> str:
    accession = str(accession).strip().upper()
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={quote(accession)}"


def _canonical_row_column(frame: pd.DataFrame) -> str | None:
    for column in ("global_row_python", "row_index_python", "sample_row_python"):
        if column in frame.columns:
            return column
    return None


def geo_pretraining_metadata(store: GeoExpressionStore) -> pd.DataFrame:
    """Return one metadata row per unique GSM, enriched with GEO links."""
    frame = store.sample_index.copy()
    row_column = _canonical_row_column(frame)
    if row_column is None:
        frame.insert(0, "row_index_python", np.arange(len(frame), dtype=np.int64))
        row_column = "row_index_python"

    wanted = [
        row_column,
        "GSM",
        "GSE",
        "canonical_GSE",
        "source_GSE",
        "cel_source_GSE",
        "is_multi_series_gsm",
        "series_relation",
        "selection_reason",
    ]
    result = frame[[column for column in wanted if column in frame.columns]].copy()
    result = result.rename(columns={row_column: "row_index_python"})
    result["row_index_python"] = pd.to_numeric(
        result["row_index_python"], errors="raise"
    ).astype(np.int64)

    if "GSM" in result.columns:
        result["GEO sample"] = result["GSM"].astype(str).map(
            lambda value: f"[{value}]({_geo_url(value)})"
        )

    gse_column = "GSE" if "GSE" in result.columns else "canonical_GSE"
    if gse_column in result.columns:
        result["GEO Series"] = result[gse_column].astype(str).map(
            lambda value: f"[{value}]({_geo_url(value)})"
        )

    return result.sort_values("row_index_python").reset_index(drop=True)


def _source_gse_components(
    store: GeoExpressionStore,
    eligible_gsms: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    """Map GSMs to leakage-safe connected components of source GEO Series.

    GEO occasionally exposes one physical GSM through multiple related Series
    (for example a SubSeries and its SuperSeries).  Splitting by the canonical
    GSE column alone can therefore place two aliases of the same study structure
    on opposite sides of a split.  We instead build connected components of the
    bipartite GSM↔source-GSE graph and assign the whole component together.
    """
    eligible = None if eligible_gsms is None else {str(value) for value in eligible_gsms}
    occurrences = store.source_occurrences.copy()
    if "GSM" not in occurrences.columns or "source_GSE" not in occurrences.columns:
        raise ValueError("GEO source occurrence metadata must contain GSM and source_GSE columns.")
    occurrences["GSM"] = occurrences["GSM"].astype(str)
    occurrences["source_GSE"] = occurrences["source_GSE"].astype(str)
    if eligible is not None:
        occurrences = occurrences[occurrences["GSM"].isin(eligible)]

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    gsm_to_gses: dict[str, list[str]] = {}
    for gsm, frame in occurrences.groupby("GSM", sort=False):
        gses = sorted({str(value) for value in frame["source_GSE"].dropna() if str(value)})
        if not gses:
            continue
        gsm_to_gses[str(gsm)] = gses
        first = gses[0]
        find(first)
        for other in gses[1:]:
            union(first, other)

    metadata = geo_pretraining_metadata(store)
    canonical_gse_column = next(
        (name for name in ("GSE", "canonical_GSE", "source_GSE") if name in metadata.columns),
        None,
    )
    for _, row in metadata.iterrows():
        gsm = str(row["GSM"])
        if eligible is not None and gsm not in eligible:
            continue
        if gsm in gsm_to_gses:
            continue
        fallback = None
        if canonical_gse_column is not None and pd.notna(row[canonical_gse_column]):
            fallback = str(row[canonical_gse_column])
        if not fallback:
            fallback = f"UNGROUPED::{gsm}"
        gsm_to_gses[gsm] = [fallback]
        find(fallback)

    root_to_gses: dict[str, set[str]] = {}
    for gse in list(parent):
        root_to_gses.setdefault(find(gse), set()).add(gse)
    component_gses = {
        root: tuple(sorted(values, key=lambda value: value))
        for root, values in root_to_gses.items()
    }

    gsm_component: dict[str, str] = {}
    for gsm, gses in gsm_to_gses.items():
        root = find(gses[0])
        gsm_component[gsm] = root
    return gsm_component, component_gses


def _balanced_component_assignment(
    component_sizes: dict[str, int],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    attempts: int = 2048,
) -> dict[str, str]:
    """Assign whole study components near requested sample fractions.

    The holdouts are small and GSE sizes are uneven, so we search many
    deterministic randomized greedy subsets for validation and test, then leave
    every remaining study component in training.  This guarantees that the
    large training target cannot starve a small holdout merely because its ideal
    sample count is below the size of one GSE.
    """
    test_fraction = 1.0 - float(train_fraction) - float(validation_fraction)
    if min(float(train_fraction), float(validation_fraction), test_fraction) <= 0.0:
        raise ValueError("train/validation/test fractions must all be positive.")
    if len(component_sizes) < 3:
        raise ValueError("At least three GSE components are required for train/validation/test.")

    total_samples = int(sum(component_sizes.values()))
    val_target = float(validation_fraction) * total_samples
    test_target = float(test_fraction) * total_samples
    components = list(component_sizes)
    rng = np.random.default_rng(int(seed))
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")

    def choose_subset(names: list[str], target: float) -> list[str]:
        order = list(names)
        rng.shuffle(order)
        selected: list[str] = []
        current = 0
        for name in order:
            candidate = current + int(component_sizes[name])
            if abs(candidate - target) < abs(current - target):
                selected.append(name)
                current = candidate
        if not selected:
            # If every GSE is larger than the target, choose the least-oversized
            # study rather than leaving the partition empty.
            selected = [
                min(
                    names,
                    key=lambda name: (abs(component_sizes[name] - target), rng.random()),
                )
            ]
        return selected

    for _ in range(max(128, int(attempts))):
        validation = choose_subset(components, val_target)
        remaining = [name for name in components if name not in set(validation)]
        if len(remaining) < 2:
            continue
        test = choose_subset(remaining, test_target)
        test_set = set(test)
        validation_set = set(validation)
        train = [name for name in remaining if name not in test_set]
        if not train:
            continue

        counts = {
            "train": sum(component_sizes[name] for name in train),
            "validation": sum(component_sizes[name] for name in validation),
            "test": sum(component_sizes[name] for name in test),
        }
        targets = {
            "train": float(train_fraction),
            "validation": float(validation_fraction),
            "test": float(test_fraction),
        }
        sample_error = sum(
            abs(counts[label] / total_samples - targets[label])
            for label in ("train", "validation", "test")
        )
        # Prefer more independent holdout studies if the sample balance is tied.
        diversity_penalty = 1.0 / len(validation) + 1.0 / len(test)
        score = sample_error + 0.002 * diversity_penalty
        if score < best_score:
            best_score = score
            assignment = {name: "train" for name in train}
            assignment.update({name: "validation" for name in validation_set})
            assignment.update({name: "test" for name in test_set})
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not construct a non-empty GSE-disjoint train/validation/test split.")
    return best_assignment


def create_train_validation_split(
    store: GeoExpressionStore,
    seed: int,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    eligible_gsms: set[str] | None = None,
) -> pd.DataFrame:
    """Create one deterministic GSE-disjoint train/validation/test split.

    The public name is kept for 0.4/0.5 API compatibility, but the semantics are
    now three-way and study-grouped.  With the default 90% training fraction,
    validation and test are each 5% of canonical GSMs approximately; exact counts
    depend on whole-GSE component sizes.
    """
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    test_fraction = 1.0 - train_fraction - validation_fraction
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0).")
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation and test fractions must both be positive.")

    metadata = geo_pretraining_metadata(store)
    if eligible_gsms is not None:
        eligible = {str(value) for value in eligible_gsms}
        metadata = metadata[metadata["GSM"].astype(str).isin(eligible)].copy()
    if len(metadata) < 3:
        raise ValueError("At least three unique GSMs are required for a three-way split.")

    gsm_component, component_gses = _source_gse_components(
        store,
        set(metadata["GSM"].astype(str)),
    )
    metadata["split_group"] = metadata["GSM"].astype(str).map(gsm_component)
    if metadata["split_group"].isna().any():
        raise ValueError("Could not assign every canonical GSM to a GEO study component.")

    component_sizes = metadata.groupby("split_group")["GSM"].nunique().astype(int).to_dict()
    assignment = _balanced_component_assignment(
        component_sizes,
        seed=int(seed),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    metadata["split"] = metadata["split_group"].map(assignment)
    metadata["seed"] = int(seed)
    metadata["split_seed"] = int(seed)
    metadata["train_fraction"] = train_fraction
    metadata["validation_fraction"] = validation_fraction
    metadata["test_fraction"] = test_fraction
    metadata["split_group_gses"] = metadata["split_group"].map(
        lambda key: ";".join(component_gses.get(str(key), (str(key),)))
    )
    metadata["split_group_n_gses"] = metadata["split_group"].map(
        lambda key: len(component_gses.get(str(key), (str(key),)))
    )
    metadata["split_group_n_samples"] = metadata["split_group"].map(component_sizes).astype(int)

    preferred = [
        "row_index_python",
        "GSM",
        "split",
        "split_group",
        "split_group_gses",
        "split_group_n_gses",
        "split_group_n_samples",
        "seed",
        "split_seed",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
    ]
    remaining = [column for column in metadata.columns if column not in preferred]
    return metadata[[*preferred, *remaining]].sort_values("row_index_python").reset_index(drop=True)


def validate_loaded_split(store: GeoExpressionStore, path: str | Path) -> pd.DataFrame:
    """Read and validate a saved GSE-disjoint train/validation/test split."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Split file not found: {source}")

    frame = pd.read_json(source) if source.suffix.lower() == ".json" else pd.read_csv(source)
    required = {"GSM", "split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Split file is missing required columns: " + ", ".join(missing))
    if frame["GSM"].astype(str).duplicated().any():
        raise ValueError("Split file contains duplicate GSM rows.")

    split_values = set(frame["split"].astype(str).str.lower().unique())
    expected_labels = {"train", "validation", "test"}
    if split_values != expected_labels:
        raise ValueError(
            "A strict GEO split must contain train, validation, and test assignments. "
            f"Observed labels: {sorted(split_values)}"
        )

    expected = geo_pretraining_metadata(store)
    expected_gsms = set(expected["GSM"].astype(str))
    supplied_gsms = set(frame["GSM"].astype(str))
    if supplied_gsms != expected_gsms:
        missing_gsms = sorted(expected_gsms.difference(supplied_gsms))[:10]
        extra_gsms = sorted(supplied_gsms.difference(expected_gsms))[:10]
        raise ValueError(
            "Split GSM set does not match the canonical GEO store. "
            f"Missing examples: {missing_gsms}; extra examples: {extra_gsms}."
        )

    assignment = frame.copy()
    assignment["GSM"] = assignment["GSM"].astype(str)
    assignment["split"] = assignment["split"].astype(str).str.lower()
    gsm_component, component_gses = _source_gse_components(store, expected_gsms)
    assignment["_expected_group"] = assignment["GSM"].map(gsm_component)
    group_nlabels = assignment.groupby("_expected_group")["split"].nunique()
    bad_groups = group_nlabels[group_nlabels > 1]
    if not bad_groups.empty:
        preview = ", ".join(str(value) for value in bad_groups.index[:5])
        raise ValueError(
            "Loaded split leaks related GEO Series across partitions; connected GSE "
            f"components with multiple labels include: {preview}."
        )

    keep_columns = [
        column
        for column in assignment.columns
        if column not in expected.columns or column in {"GSM", "split"}
    ]
    assignment = assignment[["GSM", *[c for c in keep_columns if c != "GSM"]]]
    merged = expected.merge(assignment, on="GSM", how="left", validate="one_to_one")
    if "split_group" not in merged.columns:
        merged["split_group"] = merged["GSM"].map(gsm_component)
    if "split_group_gses" not in merged.columns:
        merged["split_group_gses"] = merged["split_group"].map(
            lambda key: ";".join(component_gses.get(str(key), (str(key),)))
        )
    return merged.sort_values("row_index_python").reset_index(drop=True)


def split_summary_markdown(split: pd.DataFrame, source: str = "generated") -> str:
    counts = split["split"].astype(str).str.lower().value_counts()
    n_train = int(counts.get("train", 0))
    n_validation = int(counts.get("validation", 0))
    n_test = int(counts.get("test", 0))
    n_total = len(split)
    seed = "—"
    if "seed" in split.columns and split["seed"].notna().any():
        seed = str(int(float(split["seed"].dropna().iloc[0])))
    n_groups = int(split["split_group"].nunique()) if "split_group" in split.columns else 0
    n_val_groups = (
        int(split.loc[split["split"].astype(str).str.lower().eq("validation"), "split_group"].nunique())
        if "split_group" in split.columns
        else 0
    )
    n_test_groups = (
        int(split.loc[split["split"].astype(str).str.lower().eq("test"), "split_group"].nunique())
        if "split_group" in split.columns
        else 0
    )
    source_kind = split.get("source_kind", pd.Series("", index=split.index))
    dataset_role = split.get("dataset_role", pd.Series("", index=split.index))
    supervised_mask = source_kind.astype(str).str.casefold().eq("ikem") | (
        dataset_role.astype(str)
        .str.contains(r"IKEM|supervised dataset", case=False, na=False, regex=True)
    )
    n_supervised = int(supervised_mask.sum())
    n_geo = int(n_total - n_supervised)
    total_label = "molecular samples" if n_supervised else "unique GSMs"
    extra = ""
    if n_supervised:
        sup_counts = split.loc[supervised_mask, "split"].astype(str).str.lower().value_counts()
        extra = (
            f"\n\nThe molecular pool contains **{n_geo:,} public GEO samples** plus "
            f"**{n_supervised:,} donor-clean IKEM samples without eGFR** "
            f"({int(sup_counts.get('train', 0))} train / "
            f"{int(sup_counts.get('validation', 0))} validation / "
            f"{int(sup_counts.get('test', 0))} test). IKEM is split by donor, has no "
            "molecular-test rows, and excludes every biopsy from a donor with measured eGFR."
        )

    return f"""
<div class="metric-row">
  <div class="metric"><div class="value">{n_total:,}</div><div class="label">{total_label}</div></div>
  <div class="metric"><div class="value">{n_train:,}</div><div class="label">train · {100*n_train/max(n_total,1):.1f}%</div></div>
  <div class="metric"><div class="value">{n_validation:,}</div><div class="label">validation · {100*n_validation/max(n_total,1):.1f}%</div></div>
  <div class="metric"><div class="value">{n_test:,}</div><div class="label">test · {100*n_test/max(n_total,1):.1f}%</div></div>
  <div class="metric"><div class="value">{seed}</div><div class="label">seed</div></div>
</div>

**Split source:** {source}. Public GEO assignment is by **connected source-GSE component**, not by individual GSM. Related SubSeries/SuperSeries stay together. The GEO validation and test sets contain {n_val_groups:,} and {n_test_groups:,} independent study components respectively ({n_groups:,} total). Donor-clean IKEM rows use the separately frozen 24-train / 6-validation donor split. GEO test rows are held out from gradient updates, convergence checks, checkpoint selection, and the hyperparameter sweep itself.{extra}
"""


def save_split_csv(split: pd.DataFrame, store_root: str | Path) -> Path:
    root = Path(store_root).expanduser().resolve()
    output_dir = root.parent / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = "loaded"
    if "seed" in split.columns and split["seed"].notna().any():
        seed = str(int(float(split["seed"].dropna().iloc[0])))
    path = output_dir / f"geo_train_validation_test_gse_seed_{seed}.csv"
    split.to_csv(path, index=False)
    return path


def default_hidden_widths(layer_count: int) -> list[int]:
    """Return a simple taper with Stadniuk's 256→64 architecture as default."""
    schedules = {
        1: [64],
        2: [256, 64],
        3: [512, 256, 64],
        4: [1024, 512, 256, 64],
        5: [2048, 1024, 512, 256, 64],
    }
    count = int(layer_count)
    if count not in schedules:
        raise ValueError("Hidden layer count must be between 1 and 5.")
    return schedules[count]


def loss_theory_markdown(loss_name: str) -> str:
    """Explain what the selected objective does to reconstruction and latent geometry."""
    if loss_name == LOSS_MSE:
        return r"""
### MSE / L2 · deterministic reconstruction baseline

$$
\mathcal L_{\mathrm{MSE}}
=\frac{1}{np}\sum_{i=1}^{n}\sum_{j=1}^{p}(x_{ij}-\hat x_{ij})^2,
\qquad
\hat x=D_\theta(E_\phi(x)).
$$

**What it does to latent space:** nothing directly. The encoder may arrange $z$ in any geometry that helps reconstruction. This is the cleanest reproduction baseline because Stadniuk used MSE for GEO pretraining.

**Use when:** you want a deterministic AE and care primarily about absolute RMA reconstruction.
"""

    if loss_name == LOSS_HUBER:
        return r"""
### Smooth L1 / Huber · robust deterministic reconstruction

For an error $e=x-\hat x$, Huber loss is quadratic near zero and linear for large errors:

$$
\ell_\delta(e)=
\begin{cases}
\frac{1}{2}e^2,& |e|<\delta,\\
\delta\left(|e|-\frac{\delta}{2}\right),&\text{otherwise.}
\end{cases}
$$

**Latent effect:** like MSE, it does not impose a prior on $z$. It simply stops a relatively small number of very large reconstruction errors from dominating every update.

This is a useful modern tabular-AE baseline when expression outliers or heterogeneous GEO studies make pure MSE unstable.
"""

    if loss_name == LOSS_COSINE:
        return r"""
### MSE + profile cosine · absolute values + expression-pattern shape

$$
\mathcal L =
\mathcal L_{\mathrm{MSE}}
+\lambda_{\cos}
\frac1n\sum_i
\left[
1-\frac{x_i^\top\hat x_i}
{\lVert x_i\rVert_2\lVert\hat x_i\rVert_2+\epsilon}
\right].
$$

MSE preserves absolute RMA values. The cosine term additionally rewards reconstruction of the **shape of a sample's whole expression profile**.

**Latent effect:** still no explicit latent prior. It encourages the bottleneck to preserve directions/patterns that may be biologically useful even when absolute offsets differ.
"""

    if loss_name == LOSS_DVIB:
        return r"""
### β-VAE · reconstruction + KL regularization of each latent posterior

The encoder predicts a distribution instead of one point:

$$
q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}\sigma_\phi^2(x)),
$$

and training minimizes

$$
\mathcal L_{\beta\text{-VAE}}
=
\mathcal L_{\mathrm{MSE}}
+
\beta(t)
D_{\mathrm{KL}}\!\left(
q_\phi(z\mid x)\Vert\mathcal N(0,I)
\right).
$$

**This is the important distinction:** KL is **not another reconstruction loss**. MSE tells the decoder what information to preserve; KL actively shapes latent space by pulling every sample posterior toward a common smooth prior.

A KL warm-up is available because applying full KL pressure from epoch 1 can cause posterior collapse. Larger $\beta$ gives a smoother, more regular latent space but can worsen reconstruction.
"""

    if loss_name == LOSS_MMD:
        return r"""
### WAE / InfoVAE-style · reconstruction + MMD on the aggregate latent distribution

A deterministic encoder is kept, but its batch of latent codes is matched to samples from a standard-normal prior:

$$
\mathcal L =
\mathcal L_{\mathrm{MSE}}
+\lambda_{\mathrm{MMD}}\,
\operatorname{MMD}^2
\big(q_Z(z),\,\mathcal N(0,I)\big).
$$

ArchCon uses a multiscale inverse-multiquadratic kernel for the MMD estimate.

**Latent effect:** unlike β-VAE KL, this regularizes the **aggregate cloud of latent codes**, not every individual $q(z\mid x)$. It can therefore encourage a smooth, fillable latent space while often preserving more sample-specific information.

This is a good alternative when KL regularization feels too restrictive or collapses dimensions.
"""

    if loss_name == LOSS_MASKED:
        return r"""
### Masked denoising reconstruction · self-supervised gene-value prediction

A random fraction of probe values is hidden from the encoder, while the decoder is trained to recover the original values **only at those masked positions**:

$$
\tilde x = M(x),
\qquad
\mathcal L_{\mathrm{mask}}
=\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}
(x_{ij}-\hat x_{ij})^2.
$$

Here $\Omega$ is the set of masked probe positions. ArchCon uses **0 as an explicit mask sentinel**; this is safely outside the observed positive RMA/log2 range used by this project. Validation still reports ordinary clean-input MSE and $R^2$ separately, so the representation can be compared with the thesis baseline.

**Why this is different from plain reconstruction:** the network cannot simply pass every observed probe value through the bottleneck. It must infer hidden expression values from the remaining profile, encouraging the encoder to learn cross-probe dependencies.

This is an adaptation of masked-value / denoising objectives used in modern transcriptomic representation learning (for example scFoundation, CellFM, scLong, RegFormer and scMAE). Those models differ substantially in architecture and data type; ArchCon borrows the **pretext task**, not their full model.

**Use when:** you want a stronger self-supervised GEO pretraining task while keeping a continuous-value reconstruction target appropriate for RMA microarrays. The mask fraction should be treated as a hyperparameter and compared against ordinary MSE on the same split.
"""

    return r"""
### Legacy softmax reconstruction

This objective is retained only so old checkpoints can still be interpreted. It is no longer offered for new runs because converting RMA expression profiles to a probe-wise softmax distribution changes the statistical target without a strong transcriptomic justification.
"""


def autoencoder_parameter_count(
    input_dim: int,
    hidden_widths: list[int],
    latent_dim: int,
) -> int:
    encoder = [int(input_dim), *[int(v) for v in hidden_widths], int(latent_dim)]
    decoder = [int(latent_dim), *[int(v) for v in reversed(hidden_widths)], int(input_dim)]
    total = 0
    for left, right in strict_zip(encoder[:-1], encoder[1:]):
        total += left * right + right
    for left, right in strict_zip(decoder[:-1], decoder[1:]):
        total += left * right + right
    return total


def autoencoder_parameter_count_advanced(
    input_dim: int,
    hidden_widths: list[int],
    latent_dim: int,
    architecture_family: str = ARCH_DENSE,
    *,
    low_rank_dim: int = 64,
    residual_blocks: int = 1,
    residual_expansion: int = 1,
    stadniuk_batch_norm: bool = False,
) -> int:
    """Approximate trainable parameter count for the selectable AE families."""
    widths = [int(value) for value in hidden_widths]
    if architecture_family == ARCH_LOW_RANK:
        rank = max(1, min(int(low_rank_dim), int(input_dim), widths[0]))
        first = input_dim * rank + rank + rank * widths[0] + widths[0]
        last = widths[0] * rank + rank + rank * input_dim + input_dim
        middle_encoder = 0
        for left, right in strict_zip(widths[:-1], widths[1:]):
            middle_encoder += left * right + right
        latent = widths[-1] * latent_dim + latent_dim
        middle_decoder = 0
        reversed_widths = list(reversed(widths))
        for left, right in strict_zip(reversed_widths[:-1], reversed_widths[1:]):
            middle_decoder += left * right + right
        return first + last + middle_encoder + middle_decoder + 2 * latent

    base = autoencoder_parameter_count(input_dim, widths, latent_dim)
    if architecture_family == ARCH_STADNIUK and bool(stadniuk_batch_norm):
        # BatchNorm gamma/beta after every encoder/decoder hidden activation.
        base += 4 * sum(widths)
    if architecture_family in {ARCH_RESNET_LN, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL}:
        gated = architecture_family == ARCH_GATED_RESIDUAL
        expansion = max(1, int(residual_expansion))
        residual = 0
        for width in widths:
            if gated:
                # Legacy GLU branch remains supported for old checkpoints.
                per_block = 2 * width + 3 * width * width + 2 * width
            else:
                hidden = expansion * width
                per_block = (
                    2 * width  # LayerNorm gamma/beta
                    + width * hidden + hidden
                    + hidden * width + width
                )
            residual += int(residual_blocks) * per_block
        base += 2 * residual
    return base


def architecture_summary_markdown(
    input_dim: int,
    hidden_widths: list[int],
    latent_dim: int,
    activation: str,
    loss_name: str,
    dropout: float,
    l2_weight: float,
    architecture_family: str = ARCH_DENSE,
    low_rank_dim: int = 64,
    residual_blocks: int = 1,
    residual_expansion: int = 1,
    stadniuk_batch_norm: bool = False,
) -> str:
    encoder = " → ".join(str(value) for value in [input_dim, *hidden_widths, latent_dim])
    decoder = " → ".join(str(value) for value in [latent_dim, *reversed(hidden_widths), input_dim])
    params = autoencoder_parameter_count_advanced(
        input_dim,
        hidden_widths,
        latent_dim,
        architecture_family,
        low_rank_dim=low_rank_dim,
        residual_blocks=residual_blocks,
        residual_expansion=residual_expansion,
        stadniuk_batch_norm=stadniuk_batch_norm,
    )
    extra = ""
    if architecture_family == ARCH_LOW_RANK:
        extra = f" · **input/output rank:** `{int(low_rank_dim)}`"
    elif architecture_family in {ARCH_RESNET_LN, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL}:
        extra = (
            f" · **residual blocks/stage:** `{int(residual_blocks)}`"
            f" · **FFN expansion:** `{int(residual_expansion)}×`"
        )
    elif architecture_family == ARCH_STADNIUK:
        extra = (
            " · **hidden normalization:** `BatchNorm`"
            if bool(stadniuk_batch_norm)
            else " · **hidden normalization:** `None`"
        )
    return f"""
### {architecture_family}

**Encoder:** `{encoder}`  
**Decoder:** `{decoder}`  
**Activation:** `{activation}` · **latent activation:** linear · **dropout:** `{dropout:g}`  
**Explicit L2 λ:** `{l2_weight:g}`{extra} · **objective:** **{loss_name}**  
**Approximate parameters:** **{params:,}**

The primary experiment is intentionally limited to the Stadniuk-style MLP (with or without BatchNorm) and the LayerNorm ResNet MLP; legacy 0.4.x architecture families are retained only for checkpoint compatibility.
"""


def autoencoder_architecture_svg(
    input_dim: int,
    hidden_widths: list[int],
    latent_dim: int,
    activation: str,
    loss_name: str = LOSS_MSE,
    architecture_family: str = ARCH_DENSE,
    low_rank_dim: int = 64,
    residual_blocks: int = 1,
    residual_expansion: int = 1,
    stadniuk_batch_norm: bool = False,
) -> str:
    """Return a large horizontally-scrollable SVG architecture diagram.

    Residual families are drawn as the model is actually implemented: each
    width-changing projection creates a hidden representation first, then one
    or more same-width residual blocks operate on that representation. Every
    residual unit therefore has its own explicit identity path from the input
    of F(x) to the '+' merge. This avoids implying that a skip crosses an
    unrelated dimension-changing layer.
    """
    dimensions = [input_dim, *hidden_widths, latent_dim, *reversed(hidden_widths), input_dim]
    names = ["Input"]
    names.extend(f"Encoder {index + 1}" for index in range(len(hidden_widths)))
    names.append("Latent")
    names.extend(f"Decoder {index + 1}" for index in range(len(hidden_widths)))
    names.append("Reconstruction")

    node_width = 178
    base_gap = 94
    margin = 90
    height = 650
    center_y = 350
    residual_units = max(0, int(residual_blocks))
    latent_index = len(hidden_widths) + 1
    residual_family = architecture_family in {ARCH_RESNET_LN, ARCH_RESIDUAL, ARCH_GATED_RESIDUAL}
    residual_indices = set(range(1, 1 + len(hidden_widths))) | set(
        range(latent_index + 1, latent_index + 1 + len(hidden_widths))
    )

    # Reserve real horizontal space for every residual unit. The diagram may
    # become wide by design; the surrounding UI provides horizontal scrolling.
    edge_gaps: list[float] = []
    for index in range(len(dimensions) - 1):
        if residual_family and residual_units and index in residual_indices:
            edge_gaps.append(base_gap + residual_units * 118)
        else:
            edge_gaps.append(base_gap)

    x_positions = [float(margin)]
    for edge_gap in edge_gaps:
        x_positions.append(x_positions[-1] + node_width + edge_gap)
    width = max(1580, int(x_positions[-1] + node_width + margin))

    log_dims = np.log10(np.maximum(np.asarray(dimensions, dtype=float), 2.0))
    min_log = float(log_dims.min())
    max_log = float(log_dims.max())
    span = max(max_log - min_log, 1e-9)
    heights = 135 + 230 * (log_dims - min_log) / span

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="11" markerHeight="11" refX="9" refY="3.5" orient="auto"><path d="M0,0 L0,7 L10,3.5 z" fill="currentColor"/></marker></defs>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:currentColor}.node{fill:transparent;stroke:currentColor;stroke-width:2.5}.latent{stroke-width:4}.arrow{stroke:currentColor;stroke-width:2.4;fill:none;marker-end:url(#arrow)}.skip{stroke:currentColor;stroke-width:2;fill:none;stroke-dasharray:7 5;opacity:.68}.resunit{fill:transparent;stroke:currentColor;stroke-width:2}.merge{fill:var(--background-fill-primary,white);stroke:currentColor;stroke-width:2}.badge{fill:transparent;stroke:currentColor;stroke-width:1.5}</style>',
    ]

    positions: list[tuple[float, float, float]] = []
    for index, (dimension, name, node_height) in enumerate(
        strict_zip(dimensions, names, heights)
    ):
        x = x_positions[index]
        y = center_y - node_height / 2
        positions.append((x, y, node_height))
        css_class = "node latent" if index == latent_index else "node"
        elements.append(
            f'<rect class="{css_class}" x="{x:.1f}" y="{y:.1f}" width="{node_width}" height="{node_height:.1f}" rx="16"/>'
        )
        elements.append(
            f'<text x="{x + node_width / 2:.1f}" y="{center_y - 5:.1f}" text-anchor="middle" font-size="29" font-weight="750">{dimension:,}</text>'
        )
        elements.append(
            f'<text x="{x + node_width / 2:.1f}" y="{center_y + node_height / 2 + 42:.1f}" text-anchor="middle" font-size="22" font-weight="650">{name}</text>'
        )

    # Draw data-flow edges after nodes so residual paths can be represented as
    # actual bypasses around F(x), rather than decorative arcs inside a box.
    for index in range(len(dimensions) - 1):
        left_x = positions[index][0] + node_width
        right_x = positions[index + 1][0]
        if residual_family and residual_units and index in residual_indices:
            cursor = left_x + 10
            stage_dim = int(dimensions[index])
            for block_index in range(residual_units):
                unit_x = cursor + 30
                unit_w = 72
                unit_h = 66
                unit_y = center_y - unit_h / 2
                merge_x = unit_x + unit_w + 30
                skip_y = center_y - 96

                elements.append(
                    f'<line class="arrow" x1="{cursor:.1f}" y1="{center_y:.1f}" x2="{unit_x - 8:.1f}" y2="{center_y:.1f}"/>'
                )
                elements.append(
                    f'<rect class="resunit" x="{unit_x:.1f}" y="{unit_y:.1f}" width="{unit_w}" height="{unit_h}" rx="12"/>'
                )
                unit_name = "GLU" if architecture_family == ARCH_GATED_RESIDUAL else "F(x)"
                elements.append(
                    f'<text x="{unit_x + unit_w / 2:.1f}" y="{center_y - 5:.1f}" text-anchor="middle" font-size="18" font-weight="700">{unit_name}</text>'
                )
                elements.append(
                    f'<text x="{unit_x + unit_w / 2:.1f}" y="{center_y + 18:.1f}" text-anchor="middle" font-size="13" opacity=".72">LN · {stage_dim}→{int(residual_expansion) * stage_dim}→{stage_dim}</text>'
                )
                elements.append(
                    f'<line class="arrow" x1="{unit_x + unit_w + 7:.1f}" y1="{center_y:.1f}" x2="{merge_x - 13:.1f}" y2="{center_y:.1f}"/>'
                )
                elements.append(
                    f'<path class="skip" d="M {cursor:.1f} {center_y:.1f} C {cursor:.1f} {skip_y:.1f}, {merge_x:.1f} {skip_y:.1f}, {merge_x:.1f} {center_y - 13:.1f}"/>'
                )
                elements.append(
                    f'<text x="{(cursor + merge_x) / 2:.1f}" y="{skip_y - 10:.1f}" text-anchor="middle" font-size="13" opacity=".7">identity skip · block {block_index + 1}</text>'
                )
                elements.append(
                    f'<circle class="merge" cx="{merge_x:.1f}" cy="{center_y:.1f}" r="13"/>'
                )
                elements.append(
                    f'<text x="{merge_x:.1f}" y="{center_y + 6:.1f}" text-anchor="middle" font-size="20" font-weight="750">+</text>'
                )
                cursor = merge_x + 18

            elements.append(
                f'<line class="arrow" x1="{cursor:.1f}" y1="{center_y:.1f}" x2="{right_x - 12:.1f}" y2="{center_y:.1f}"/>'
            )
            projection_x = (cursor + right_x) / 2
        else:
            elements.append(
                f'<line class="arrow" x1="{left_x + 12:.1f}" y1="{center_y:.1f}" x2="{right_x - 12:.1f}" y2="{center_y:.1f}"/>'
            )
            projection_x = (left_x + right_x) / 2

        projection_label = f"Linear {int(dimensions[index]):,}→{int(dimensions[index + 1]):,}"
        if architecture_family == ARCH_STADNIUK and index != latent_index - 1 and index != len(dimensions) - 2:
            projection_label += " · ReLU"
            if bool(stadniuk_batch_norm):
                projection_label += " · BN"
        if architecture_family == ARCH_LOW_RANK and index in {0, len(dimensions) - 2}:
            projection_label = (
                f"factorized {int(dimensions[index]):,}→{int(dimensions[index + 1]):,} "
                f"· rank {int(low_rank_dim)}"
            )
        elements.append(
            f'<text x="{projection_x:.1f}" y="{center_y + 76:.1f}" text-anchor="middle" font-size="12.5" opacity=".68">{projection_label}</text>'
        )

    if architecture_family == ARCH_LOW_RANK:
        first_x = (positions[0][0] + node_width + positions[1][0]) / 2
        last_x = (positions[-2][0] + node_width + positions[-1][0]) / 2
        for x, label in (
            (first_x, f"rank {int(low_rank_dim)}"),
            (last_x, f"rank {int(low_rank_dim)}"),
        ):
            elements.append(
                f'<rect class="badge" x="{x - 50:.1f}" y="{center_y - 78:.1f}" width="100" height="34" rx="10"/>'
            )
            elements.append(
                f'<text x="{x:.1f}" y="{center_y - 55:.1f}" text-anchor="middle" font-size="16" font-weight="650">{label}</text>'
            )

    latent_x = positions[latent_index][0] + node_width / 2
    right_x = positions[-1][0] + node_width
    elements.append(
        f'<text x="{(margin + latent_x) / 2:.1f}" y="55" text-anchor="middle" font-size="29" font-weight="760">Encoder · {activation}</text>'
    )
    elements.append(
        f'<text x="{(latent_x + right_x) / 2:.1f}" y="55" text-anchor="middle" font-size="29" font-weight="760">Decoder · {activation}</text>'
    )
    elements.append(
        f'<text x="{width / 2:.1f}" y="82" text-anchor="middle" font-size="21" font-weight="650">{architecture_family}</text>'
    )
    if residual_family:
        elements.append(
            f'<text x="{width / 2:.1f}" y="112" text-anchor="middle" font-size="14" opacity=".72">rectangles are representation widths · projection arrows change width · each dashed skip is local x + F(x)</text>'
        )
    latent_label = "μ / logσ² → z" if loss_name == LOSS_DVIB else "linear code z"
    elements.append(
        f'<text x="{latent_x:.1f}" y="132" text-anchor="middle" font-size="21" font-weight="650">{latent_label}</text>'
    )
    elements.append("</svg>")
    return '<div class="architecture-scroll-shell">' + "".join(elements) + "</div>"


def plot_autoencoder_architecture(
    input_dim: int,
    hidden_widths: list[int],
    latent_dim: int,
    activation: str,
):
    """Draw a paper-style schematic of a symmetric dense autoencoder."""
    dimensions = [input_dim, *hidden_widths, latent_dim, *reversed(hidden_widths), input_dim]
    names = ["Input"]
    names.extend(f"Enc {index + 1}" for index in range(len(hidden_widths)))
    names.append("Latent")
    names.extend(f"Dec {index + 1}" for index in range(len(hidden_widths)))
    names.append("Reconstruction")

    fig_width = max(10.0, 1.45 * len(dimensions))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    ax.set_xlim(-0.7, len(dimensions) - 0.3)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    log_dims = np.log10(np.maximum(np.asarray(dimensions, dtype=float), 2.0))
    min_log = float(log_dims.min())
    max_log = float(log_dims.max())
    span = max(max_log - min_log, 1e-9)
    heights = 0.42 + 0.88 * (log_dims - min_log) / span

    for index, (dimension, name, height) in enumerate(strict_zip(dimensions, names, heights)):
        left = index - 0.36
        bottom = -height / 2
        rectangle = plt.Rectangle(
            (left, bottom),
            0.72,
            height,
            fill=False,
            linewidth=1.7,
        )
        ax.add_patch(rectangle)
        ax.text(index, 0.05, f"{dimension:,}", ha="center", va="center", fontsize=9)
        ax.text(index, -0.72, name, ha="center", va="top", fontsize=9)

        if index < len(dimensions) - 1:
            ax.annotate(
                "",
                xy=(index + 0.61, 0),
                xytext=(index + 0.39, 0),
                arrowprops={"arrowstyle": "->", "linewidth": 1.3},
            )

    latent_index = len(hidden_widths) + 1
    ax.text(
        latent_index,
        0.78,
        "linear latent code z",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.text(
        0.5 * (latent_index - 1),
        0.92,
        f"encoder · {activation}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        0.5 * (latent_index + len(dimensions) - 1),
        0.92,
        f"decoder · {activation}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax.set_title("GEO pretraining autoencoder · architecture preview", pad=12)
    fig.tight_layout()
    return fig


def open_store_for_split(path: str | Path) -> GeoExpressionStore:
    """Small named wrapper used by the web callback layer."""
    return load_geo_expression_store(path)

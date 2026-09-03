"""Data loading, alignment, preprocessing bridges, and QC plots for ArchCon."""

from .defaults import (
    ProjectDataLayout,
    data_directory_status,
    detected_default_paths,
    project_data_layout,
    resolve_data_directory,
)
from .geo_rma import GeoExpressionStore, load_geo_expression_store
from .training import TrainingConfig, train_autoencoder_stream, training_backend_status
from .loading import (
    DataWorkspace,
    ExpressionData,
    align_workspace,
    load_expression_matrix,
    load_table,
    normalize_sample_id,
)

__all__ = [
    "DataWorkspace",
    "ProjectDataLayout",
    "ExpressionData",
    "GeoExpressionStore",
    "TrainingConfig",
    "align_workspace",
    "data_directory_status",
    "detected_default_paths",
    "load_expression_matrix",
    "load_geo_expression_store",
    "load_table",
    "normalize_sample_id",
    "project_data_layout",
    "train_autoencoder_stream",
    "training_backend_status",
    "resolve_data_directory",
]

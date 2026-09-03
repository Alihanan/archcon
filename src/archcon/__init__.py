"""Public package interface for ArchCon."""

from importlib.metadata import PackageNotFoundError, version

from .data import (
    DataWorkspace,
    ExpressionData,
    GeoExpressionStore,
    TrainingConfig,
    load_expression_matrix,
    load_geo_expression_store,
    normalize_sample_id,
    train_autoencoder_stream,
    training_backend_status,
)

try:
    __version__ = version("archcon")
except PackageNotFoundError:
    __version__ = "0+unknown"


def __getattr__(name: str):
    """Load the Gradio application only when the browser API is requested."""
    if name in {"build_app", "start"}:
        from .app import build_app, start

        return {"build_app": build_app, "start": start}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataWorkspace",
    "ExpressionData",
    "GeoExpressionStore",
    "TrainingConfig",
    "__version__",
    "build_app",
    "load_expression_matrix",
    "load_geo_expression_store",
    "normalize_sample_id",
    "start",
    "train_autoencoder_stream",
    "training_backend_status",
]

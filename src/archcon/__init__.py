"""Public package interface for ArchCon."""

from importlib.metadata import PackageNotFoundError, version

from .app import build_app, start
from .io import read_text_file

try:
    __version__ = version("archcon")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "build_app",
    "read_text_file",
    "start",
]

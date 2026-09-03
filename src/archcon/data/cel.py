"""Optional bridge to the thesis' R/affy CEL → RMA preprocessing."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pandas as pd

from .loading import ExpressionData, load_expression_matrix

R_INSTALL_URL = "https://cran.r-project.org/"
AFFY_URL = "https://bioconductor.org/packages/affy/"
BIOCMANAGER_URL = "https://bioconductor.org/install/"


def discover_cel_files(directory: str | Path) -> list[Path]:
    """Return deterministic CEL/CEL.gz paths from a local directory."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"CEL directory not found: {path}")
    cel_files = [
        item
        for item in path.iterdir()
        if item.is_file()
        and (item.name.lower().endswith(".cel") or item.name.lower().endswith(".cel.gz"))
    ]
    return sorted(cel_files, key=lambda item: item.name.lower())


def _run_rscript_expression(rscript: str, expression: str) -> subprocess.CompletedProcess[str]:
    """Run a fixed R expression through Rscript without invoking a shell."""
    return subprocess.run(
        [rscript, "--vanilla", "-e", expression],
        capture_output=True,
        text=True,
        check=False,
    )


def rma_environment_status() -> dict[str, Any]:
    """Return a detailed, non-mutating status report for the optional RMA backend."""
    rscript = shutil.which("Rscript")
    status: dict[str, Any] = {
        "ready": False,
        "rscript": rscript,
        "r_version": None,
        "biocmanager": False,
        "affy": False,
        "message": "",
    }

    if rscript is None:
        status["message"] = (
            "Rscript was not found on PATH. ArchCon itself still works; only raw CEL → RMA "
            "preprocessing is unavailable until R is installed."
        )
        return status

    version_result = subprocess.run(
        [rscript, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version_text = (version_result.stdout or version_result.stderr or "").strip()
    if version_text:
        status["r_version"] = version_text.splitlines()[0]

    package_probe = (
        'cat(ifelse(requireNamespace("BiocManager", quietly=TRUE), "1", "0"), "\\n"); '
        'cat(ifelse(requireNamespace("affy", quietly=TRUE), "1", "0"), "\\n")'
    )
    result = _run_rscript_expression(rscript, package_probe)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "R package check failed.").strip()
        status["message"] = f"R was found, but the package check failed: {details}"
        return status

    values = [line.strip() for line in result.stdout.splitlines() if line.strip() in {"0", "1"}]
    if len(values) >= 2:
        status["biocmanager"] = values[-2] == "1"
        status["affy"] = values[-1] == "1"

    status["ready"] = bool(status["affy"])
    if status["ready"]:
        status["message"] = (
            "RMA backend is ready. ArchCon can call Bioconductor affy for CEL → RMA."
        )
    elif status["biocmanager"]:
        status["message"] = "R and BiocManager are available, but Bioconductor affy is missing."
    else:
        status["message"] = "R is available, but BiocManager and/or Bioconductor affy are missing."
    return status


def check_rma_environment() -> tuple[bool, str]:
    """Check whether Rscript and the Bioconductor affy package are available."""
    status = rma_environment_status()
    return bool(status["ready"]), str(status["message"])


def install_rma_dependencies() -> tuple[bool, str]:
    """Try to install BiocManager and affy into the user's configured R library.

    This intentionally does *not* install R itself and never invokes sudo, apt,
    brew, or another system package manager. The action is only run when the
    user explicitly clicks the installation button in the web interface.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        message = (
            "R is not installed or Rscript is not on PATH. Install R first, then retry. "
            f"Official installer: {R_INSTALL_URL}"
        )
        return False, message

    install_expression = """
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    stop("BiocManager could not be installed")
}
BiocManager::install("affy", ask = FALSE, update = FALSE)
if (!requireNamespace("affy", quietly = TRUE)) {
    stop("affy is still unavailable after installation")
}
cat("ARCHCON_RMA_BACKEND_READY\\n")
""".strip()

    result = _run_rscript_expression(rscript, install_expression)
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    # Keep UI output useful without dumping thousands of compilation lines.
    lines = combined.splitlines()
    tail = "\n".join(lines[-80:]) if lines else "No R output was produced."

    if result.returncode != 0:
        message = (
            "R dependency installation failed. This can happen when the R library is not writable, "
            "a compiler/system dependency is missing, or the network is unavailable.\n\n"
            f"Last R output:\n{tail}"
        )
        return False, message

    ready, message = check_rma_environment()
    if not ready:
        return (
            False,
            f"The installer finished, but the backend check still failed: {message}\n\n{tail}",
        )

    return True, f"Bioconductor affy is installed and the RMA backend is ready.\n\n{tail}"


def run_rma(
    cel_files: list[str | Path],
    *,
    output_directory: str | Path | None = None,
) -> tuple[ExpressionData, pd.DataFrame, Path]:
    """Run RMA jointly on CEL files and return normalized data plus raw QC sample."""
    if not cel_files:
        raise ValueError("No CEL files were selected.")

    ready, message = check_rma_environment()
    if not ready:
        raise RuntimeError(message)

    resolved = [Path(path).expanduser().resolve() for path in cel_files]
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CEL file not found: {missing[0]}")

    if output_directory is None:
        out_dir = Path(tempfile.mkdtemp(prefix="archcon-rma-"))
    else:
        out_dir = Path(output_directory).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    expression_csv = out_dir / "rma_expression_matrix.csv"
    raw_sample_csv = out_dir / "raw_cel_qc_sample.csv"

    r_script_resource = files("archcon.assets").joinpath("rma_preprocess.R")
    with as_file(r_script_resource) as r_script:
        command = [
            shutil.which("Rscript") or "Rscript",
            "--vanilla",
            str(r_script),
            str(expression_csv),
            str(raw_sample_csv),
            *[str(path) for path in resolved],
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Unknown R error").strip()
        raise RuntimeError(f"RMA preprocessing failed:\n{details}")

    expression = load_expression_matrix(expression_csv, orientation="samples_rows")
    raw_sample = pd.read_csv(raw_sample_csv)
    return expression, raw_sample, out_dir

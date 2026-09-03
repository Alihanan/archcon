from unittest.mock import patch

from archcon.data.cel import install_rma_dependencies, rma_environment_status


def test_rma_status_when_r_is_missing() -> None:
    with patch("archcon.data.cel.shutil.which", return_value=None):
        status = rma_environment_status()

    assert status["ready"] is False
    assert status["rscript"] is None
    assert "only raw CEL" in status["message"]


def test_rma_installer_does_not_try_system_install_without_r() -> None:
    with patch("archcon.data.cel.shutil.which", return_value=None):
        success, message = install_rma_dependencies()

    assert success is False
    assert "Install R first" in message
    assert "cran.r-project.org" in message

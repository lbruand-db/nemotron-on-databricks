import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def page_elements_code() -> Path:
    """Local checkout of the NVIDIA page-elements-v3 *code* (no weights needed).

    Set NEMOTRON_PE3_CODE to a directory containing `nemotron_page_elements_v3/`, e.g. created with
    `hf download nvidia/nemotron-page-elements-v3 --include 'nemotron_page_elements_v3/*.py'
    'nemotron_page_elements_v3/**/*.py' --local-dir /tmp/pe3`.
    """
    path = os.environ.get("NEMOTRON_PE3_CODE")
    if not path or not (Path(path) / "nemotron_page_elements_v3").is_dir():
        pytest.skip("NEMOTRON_PE3_CODE not set")
    return Path(path)

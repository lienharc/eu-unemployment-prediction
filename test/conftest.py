from pathlib import Path

import pytest


@pytest.fixture
def project_dir() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_dir: Path) -> Path:
    return project_dir / "data"


@pytest.fixture
def module_dir(project_dir: Path) -> Path:
    return project_dir / "model"

"""Rend les modules de scripts/ importables dans les tests, comme le fait run.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline_config import load_config  # noqa: E402


@pytest.fixture
def config():
    """Config réelle du projet : les tests portent sur les défauts livrés."""
    return load_config(PROJECT_ROOT / "config.yaml")

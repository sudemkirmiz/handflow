"""Test-wide safety: no test may write into the real project folder."""

import pytest


@pytest.fixture(autouse=True)
def isolated_output_dir(tmp_path, monkeypatch):
    """Redirect PNG exports to a per-test temporary folder."""
    import config

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")

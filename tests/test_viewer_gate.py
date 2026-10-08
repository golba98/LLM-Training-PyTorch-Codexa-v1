"""Operator launch attachment must fail before creating training artifacts."""

from pathlib import Path

import pytest

from llm_training.viewer import attached_viewer


def test_missing_graphical_viewer_prevents_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="visible Kitty"):
        with attached_viewer(metrics_path=tmp_path / "metrics.jsonl", checkpoint_path=tmp_path / "latest.pt", total_steps=1, title="fixture"):
            pytest.fail("Training must not be reached.")
    assert not list(tmp_path.iterdir())

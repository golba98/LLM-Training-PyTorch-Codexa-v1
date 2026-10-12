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


def test_supplied_viewer_command_must_attach(tmp_path, monkeypatch):
    import json
    import sys
    from llm_training import conversational_training as chat
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/usr/bin/" + name)
    status = tmp_path / "metrics" / "progress.json"
    status.parent.mkdir()
    attach = ("import sys, time; args = sys.argv[1:]; "
              "open(args[args.index('--marker') + 1], 'w').write('{}'); time.sleep(30)")
    monkeypatch.setenv(chat.VIEWER_COMMAND_ENVIRONMENT, json.dumps([sys.executable, "-c", attach]))
    process = chat.start_chat_viewer(status, title="fixture")
    try:
        assert process.poll() is None and status.with_suffix(".attached").exists()
        assert "--status" in process.args and str(status.resolve()) in process.args
    finally:
        process.kill()
        process.wait()
    monkeypatch.setenv(chat.VIEWER_COMMAND_ENVIRONMENT, json.dumps([sys.executable, "-c", "raise SystemExit(1)"]))
    with pytest.raises(RuntimeError, match="did not attach"):
        chat.start_chat_viewer(status, title="fixture")
    monkeypatch.setenv(chat.VIEWER_COMMAND_ENVIRONMENT, '"kitty"')
    with pytest.raises(ValueError, match="JSON list"):
        chat.start_chat_viewer(status, title="fixture")

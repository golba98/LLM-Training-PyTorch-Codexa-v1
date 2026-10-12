"""Find sibling packages during source development; installed use needs none."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for repo in sorted(ROOT.parent.glob("LLM-*")):
    if repo.name != "LLM-From-Scratch" and (repo / "src").is_dir():
        sys.path.insert(0, str(repo / "src"))


@pytest.fixture(autouse=True)
def explicit_training_environment():
    """These tests train tiny CPU models through the public trainer functions. In the Codexa
    workspace an interpreter startup guard wraps those functions and requires an explicit
    environment selection; select development explicitly. Nothing is faked: the guard then
    validates the real workspace (pinned, committed, source-first) and fails closed otherwise.
    Outside the workspace (no guard installed) the variable is inert. A private MonkeyPatch keeps
    the selection in force when a test calls ``monkeypatch.undo()`` on its own patches."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CODEXA_TRAINING_ENVIRONMENT", "development")
        guard = sys.modules.get("training_environment_guard")
        if guard is not None:
            patch.setattr(guard, "_selected", None)
        yield

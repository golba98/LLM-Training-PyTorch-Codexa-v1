"""Find sibling packages during source development; installed use needs none."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for repo in sorted(ROOT.parent.glob("LLM-*")):
    if repo.name != "LLM-From-Scratch" and (repo / "src").is_dir():
        sys.path.insert(0, str(repo / "src"))

"""Summarize one or more Codexa training metrics files."""

import argparse
import json
from pathlib import Path
import sys



from llm_training.reporting import summarize_training_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summaries = [
        summarize_training_metrics(path).to_dict()
        for path in arguments.metrics
    ]
    rendered = json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

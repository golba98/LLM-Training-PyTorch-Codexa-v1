"""Summarize one flushed Codexa GPU telemetry JSONL file."""

import argparse
import json
from pathlib import Path
import sys



from llm_training.hardware_monitor import summarize_gpu_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    summary = summarize_gpu_metrics(arguments.metrics).to_dict()
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

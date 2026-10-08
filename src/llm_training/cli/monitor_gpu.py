"""Record flushed NVIDIA GPU telemetry as local JSONL."""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
import time



from llm_training.hardware_monitor import query_gpu


def monitor_gpu(
    output_path: Path,
    *,
    interval_seconds: float,
    maximum_samples: int | None,
    overwrite: bool,
    append: bool,
    query: Callable[[], dict[str, object]] = query_gpu,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")
    if maximum_samples is not None and maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive when supplied.")
    if overwrite and append:
        raise ValueError("overwrite and append cannot both be enabled.")
    if output_path.exists() and not overwrite and not append:
        raise FileExistsError(f"Refusing to overwrite telemetry: {output_path}")
    if append and not output_path.is_file():
        raise FileNotFoundError(f"Cannot append missing telemetry: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else ("w" if overwrite else "x")
    samples = 0
    with output_path.open(mode, encoding="utf-8") as output_file:
        while maximum_samples is None or samples < maximum_samples:
            record = query()
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            output_file.flush()
            samples += 1
            if maximum_samples is None or samples < maximum_samples:
                sleep(interval_seconds)
    return samples


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--append", action="store_true")
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    try:
        samples = monitor_gpu(
            arguments.output,
            interval_seconds=arguments.interval_seconds,
            maximum_samples=arguments.max_samples,
            overwrite=arguments.overwrite,
            append=arguments.append,
        )
    except KeyboardInterrupt:
        print("GPU monitoring interrupted.", file=sys.stderr)
        return
    print(f"GPU samples recorded: {samples}")


if __name__ == "__main__":
    main()

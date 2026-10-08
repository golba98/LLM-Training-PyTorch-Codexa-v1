"""Strict summaries for Codexa JSONL training metrics."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics


@dataclass(frozen=True)
class TrainingRunSummary:
    run_name: str
    run_id: str
    optimizer_steps: int
    micro_steps: int
    tokens_seen: int
    first_training_loss: float
    final_training_loss: float
    minimum_training_loss: float
    best_validation_loss: float | None
    best_validation_step: int | None
    validation_evaluations: int
    mean_tokens_per_second: float
    median_tokens_per_second: float
    total_step_time_seconds: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_metrics(path: str | Path) -> list[dict[str, object]]:
    """Load and validate one non-empty metrics JSONL file."""

    metrics_path = Path(path)
    records: list[dict[str, object]] = []
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{metrics_path}:{line_number}: malformed JSON "
                    f"({error.msg})"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{metrics_path}:{line_number}: metric must be an object."
                )
            for key, value in record.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(
                        f"{metrics_path}:{line_number}: {key} is non-finite."
                    )
            records.append(record)
    if not records:
        raise ValueError("Metrics JSONL must not be empty.")
    return records


def summarize_training_metrics(
    path: str | Path,
) -> TrainingRunSummary:
    """Calculate stable aggregate metrics from one training run."""

    records = load_metrics(path)
    required = {
        "run_name",
        "run_id",
        "optimizer_step",
        "micro_step",
        "total_tokens_seen",
        "training_loss",
        "validation_loss",
        "tokens_per_second",
        "step_time_seconds",
        "peak_allocated_vram_bytes",
        "peak_reserved_vram_bytes",
    }
    for index, record in enumerate(records, start=1):
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"Metrics record {index} is missing: {', '.join(sorted(missing))}."
            )
    run_names = {record["run_name"] for record in records}
    run_ids = {record["run_id"] for record in records}
    if len(run_names) != 1 or len(run_ids) != 1:
        raise ValueError("Metrics JSONL contains multiple runs.")

    validations = [
        (int(record["optimizer_step"]), float(record["validation_loss"]))
        for record in records
        if record["validation_loss"] is not None
    ]
    best = min(validations, key=lambda item: item[1]) if validations else None
    final = records[-1]
    return TrainingRunSummary(
        run_name=str(final["run_name"]),
        run_id=str(final["run_id"]),
        optimizer_steps=int(final["optimizer_step"]),
        micro_steps=int(final["micro_step"]),
        tokens_seen=int(final["total_tokens_seen"]),
        first_training_loss=float(records[0]["training_loss"]),
        final_training_loss=float(final["training_loss"]),
        minimum_training_loss=min(
            float(record["training_loss"]) for record in records
        ),
        best_validation_loss=None if best is None else best[1],
        best_validation_step=None if best is None else best[0],
        validation_evaluations=len(validations),
        mean_tokens_per_second=statistics.mean(
            float(record["tokens_per_second"]) for record in records
        ),
        median_tokens_per_second=statistics.median(
            float(record["tokens_per_second"]) for record in records
        ),
        total_step_time_seconds=sum(
            float(record["step_time_seconds"]) for record in records
        ),
        peak_allocated_vram_bytes=max(
            int(record["peak_allocated_vram_bytes"]) for record in records
        ),
        peak_reserved_vram_bytes=max(
            int(record["peak_reserved_vram_bytes"]) for record in records
        ),
    )

"""JSON-serializable NVIDIA GPU monitoring and summary helpers."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess


GPU_QUERY_FIELDS = (
    "name",
    "temperature.gpu",
    "power.draw",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "clocks.sm",
    "clocks.mem",
)


@dataclass(frozen=True)
class GpuTelemetrySummary:
    sample_count: int
    start_timestamp_utc: str
    end_timestamp_utc: str
    minimum_temperature_celsius: int
    maximum_temperature_celsius: int
    mean_temperature_celsius: float
    mean_power_draw_watts: float
    maximum_power_draw_watts: float
    mean_utilization_percent: float
    maximum_memory_used_mib: int
    minimum_sm_clock_mhz: int
    maximum_sm_clock_mhz: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_nvidia_smi_row(row: str) -> dict[str, object]:
    """Parse one no-header, no-units NVIDIA-SMI query row."""

    values = [value.strip() for value in row.strip().split(",")]
    if len(values) != len(GPU_QUERY_FIELDS):
        raise ValueError(
            f"Expected {len(GPU_QUERY_FIELDS)} GPU fields, got {len(values)}."
        )
    try:
        return {
            "timestamp_utc": utc_timestamp(),
            "gpu_name": values[0],
            "temperature_celsius": int(values[1]),
            "power_draw_watts": float(values[2]),
            "memory_used_mib": int(values[3]),
            "memory_total_mib": int(values[4]),
            "utilization_percent": int(values[5]),
            "sm_clock_mhz": int(values[6]),
            "memory_clock_mhz": int(values[7]),
        }
    except ValueError as error:
        raise ValueError(f"Malformed NVIDIA-SMI row: {row!r}") from error


def query_gpu() -> dict[str, object]:
    """Query the first visible NVIDIA GPU."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(GPU_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one visible NVIDIA GPU, got {len(rows)}."
        )
    return parse_nvidia_smi_row(rows[0])


def summarize_gpu_metrics(path: str | Path) -> GpuTelemetrySummary:
    """Load one strict telemetry JSONL file and calculate stable aggregates."""

    metrics_path = Path(path)
    records: list[dict[str, object]] = []
    required = {
        "timestamp_utc",
        "temperature_celsius",
        "power_draw_watts",
        "memory_used_mib",
        "utilization_percent",
        "sm_clock_mhz",
    }
    with metrics_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{metrics_path}:{line_number}: malformed JSON."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{metrics_path}:{line_number}: metric must be an object."
                )
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"{metrics_path}:{line_number}: missing fields: "
                    + ", ".join(sorted(missing))
                )
            for key in required - {"timestamp_utc"}:
                value = record[key]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"{metrics_path}:{line_number}: {key} must be finite."
                    )
            if not isinstance(record["timestamp_utc"], str):
                raise ValueError(
                    f"{metrics_path}:{line_number}: timestamp must be a string."
                )
            records.append(record)
    if not records:
        raise ValueError("GPU telemetry JSONL must not be empty.")

    temperatures = [int(record["temperature_celsius"]) for record in records]
    powers = [float(record["power_draw_watts"]) for record in records]
    utilizations = [
        float(record["utilization_percent"]) for record in records
    ]
    memories = [int(record["memory_used_mib"]) for record in records]
    sm_clocks = [int(record["sm_clock_mhz"]) for record in records]
    return GpuTelemetrySummary(
        sample_count=len(records),
        start_timestamp_utc=str(records[0]["timestamp_utc"]),
        end_timestamp_utc=str(records[-1]["timestamp_utc"]),
        minimum_temperature_celsius=min(temperatures),
        maximum_temperature_celsius=max(temperatures),
        mean_temperature_celsius=statistics.mean(temperatures),
        mean_power_draw_watts=statistics.mean(powers),
        maximum_power_draw_watts=max(powers),
        mean_utilization_percent=statistics.mean(utilizations),
        maximum_memory_used_mib=max(memories),
        minimum_sm_clock_mhz=min(sm_clocks),
        maximum_sm_clock_mhz=max(sm_clocks),
    )

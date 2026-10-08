"""Typed checks and storage estimates for long training-run preflight."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "fail"}:
            raise ValueError("Preflight status must be pass, warning, or fail.")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointStorageEstimate:
    checkpoint_bytes: int
    milestone_count: int
    retained_checkpoint_count: int
    projected_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_checkpoint_storage(
    *,
    parameter_count: int,
    max_steps: int,
    checkpoint_interval: int,
) -> CheckpointStorageEstimate:
    """Estimate AdamW checkpoint retention plus serialization overhead."""

    for name, value in (
        ("parameter_count", parameter_count),
        ("max_steps", max_steps),
        ("checkpoint_interval", checkpoint_interval),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    checkpoint_bytes = parameter_count * 12 + 1024 * 1024
    milestone_count = max_steps // checkpoint_interval
    if max_steps % checkpoint_interval:
        milestone_count += 1
    retained_count = milestone_count + 3
    return CheckpointStorageEstimate(
        checkpoint_bytes=checkpoint_bytes,
        milestone_count=milestone_count,
        retained_checkpoint_count=retained_count,
        projected_bytes=checkpoint_bytes * retained_count,
    )


def independent_filesystems(source: Path, destination: Path) -> bool:
    """Return whether existing source and destination use different devices."""

    if not source.exists() or not destination.exists():
        raise FileNotFoundError("Both filesystem paths must already exist.")
    return source.stat().st_dev != destination.stat().st_dev


def disk_capacity_check(
    path: Path,
    *,
    required_bytes: int,
    name: str = "disk_capacity",
) -> PreflightCheck:
    if required_bytes <= 0:
        raise ValueError("required_bytes must be positive.")
    if not name:
        raise ValueError("name must be non-empty.")
    if not path.exists():
        return PreflightCheck(
            name,
            "fail",
            f"Path does not exist: {path}",
        )
    free = shutil.disk_usage(path).free
    status = "pass" if free >= required_bytes else "fail"
    return PreflightCheck(
        name,
        status,
        f"free={free} required={required_bytes} path={path}",
    )


def preflight_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

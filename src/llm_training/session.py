"""Persistent session budgets and signal requests, independent of CUDA."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any


def atomic_document(path: Path, value: Any) -> None:
    """Durably replace an owned JSON document and sync its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class SessionBudget:
    """Original deadline survives restarts; active enforcement is monotonic."""

    seconds: float
    started_utc: float
    deadline_utc: float
    deadline_monotonic: float
    reserve_seconds: float

    @classmethod
    def start(cls, seconds: float, reserve_seconds: float) -> 'SessionBudget':
        if not 0 < reserve_seconds < seconds <= 36000:
            raise ValueError('Budget must be <=36000 seconds and exceed finalization reserve.')
        wall, monotonic = time.time(), time.monotonic()
        return cls(seconds, wall, wall + seconds, monotonic + seconds, reserve_seconds)

    @classmethod
    def resume(cls, document: dict) -> 'SessionBudget':
        wall, monotonic = time.time(), time.monotonic()
        if wall < document['last_recorded_utc'] - 2:
            raise ValueError('Clock moved backwards; explicit budget reconciliation is required.')
        remaining = min(document['deadline_utc'] - wall,
                        document['remaining_seconds'] - max(0, wall - document['last_recorded_utc']))
        boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if document.get('boot_id') == boot_id and 'deadline_monotonic' in document:
            remaining = min(remaining, document['deadline_monotonic'] - monotonic)
        if remaining <= 0:
            raise TimeoutError('The original session budget has expired.')
        return cls(document['seconds'], document['started_utc'], document['deadline_utc'],
                   monotonic + remaining, document['reserve_seconds'])

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())

    @property
    def training_deadline(self) -> float:
        return self.deadline_monotonic - self.reserve_seconds

    def document(self) -> dict:
        """Snapshot elapsed time, including startup and downtime."""
        remaining = self.remaining
        return {'seconds': self.seconds, 'started_utc': self.started_utc,
                'deadline_utc': self.deadline_utc, 'reserve_seconds': self.reserve_seconds,
                'remaining_seconds': remaining, 'elapsed_seconds': self.seconds - remaining,
                'last_recorded_utc': time.time(), 'deadline_monotonic': self.deadline_monotonic,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


class StopRequest:
    """Signal handlers only request stopping; they never serialize tensors."""

    def __init__(self) -> None:
        self.signal_number: int | None = None
        self.previous: dict[int, Any] = {}

    def __call__(self) -> bool:
        return self.signal_number is not None

    def __enter__(self) -> 'StopRequest':
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self._request)
        return self

    def _request(self, number: int, frame: Any) -> None:
        self.signal_number = number

    def __exit__(self, *args: Any) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)

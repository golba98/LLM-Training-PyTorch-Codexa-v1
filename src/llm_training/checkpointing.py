"""Atomic checkpoint persistence and retention for Codexa training."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from llm_training.config import ProjectConfig
from llm_training.training import TrainingState


from llm_architecture.checkpointing import (CHECKPOINT_FORMAT_VERSION, InferenceCheckpoint, file_sha256, verify_checkpoint_checksum, _validate_training_state, load_model_checkpoint)


@dataclass(frozen=True)
class SchedulerState:
    """Inputs needed to reproduce the stateless learning-rate schedule."""

    warmup_steps: int
    max_steps: int
    peak_learning_rate: float
    minimum_learning_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Metadata restored from a validated checkpoint."""

    path: Path
    state: TrainingState
    scheduler: SchedulerState
    run_name: str
    run_id: str
    tokenizer_reference: str | None
    tokenizer_sha256: str | None
    resume_context: dict[str, object] | None






def capture_random_states() -> dict[str, object]:
    """Capture Python, NumPy, CPU, and available CUDA RNG states."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_random_states(states: Mapping[str, object]) -> None:
    """Restore all RNG states saved by :func:`capture_random_states`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(states) != required:
        raise ValueError("Checkpoint RNG state has invalid keys.")
    random.setstate(states["python"])  # type: ignore[arg-type]
    np.random.set_state(states["numpy"])  # type: ignore[arg-type]
    torch_cpu = states["torch_cpu"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise ValueError("Checkpoint CPU RNG state must be a tensor.")
    torch.set_rng_state(torch_cpu.cpu())
    torch_cuda = states["torch_cuda"]
    if torch_cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if not isinstance(torch_cuda, list) or not all(
            isinstance(item, torch.Tensor) for item in torch_cuda
        ):
            raise ValueError("Checkpoint CUDA RNG state is invalid.")
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(text)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_checksum(path: Path) -> None:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    _atomic_text_write(f"{file_sha256(path)}  {path.name}\n", checksum_path)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary_path)
        with temporary_path.open("rb") as copied_file:
            os.fsync(copied_file.fileno())
        temporary_path.replace(destination)
        _write_checksum(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise




def _validate_scheduler_state(value: object) -> SchedulerState:
    if not isinstance(value, dict):
        raise ValueError("Checkpoint scheduler state must be an object.")
    try:
        state = SchedulerState(
            warmup_steps=value["warmup_steps"],
            max_steps=value["max_steps"],
            peak_learning_rate=value["peak_learning_rate"],
            minimum_learning_rate=value["minimum_learning_rate"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint scheduler state is incomplete.") from error
    if state.warmup_steps < 0 or state.max_steps <= state.warmup_steps:
        raise ValueError("Checkpoint scheduler step settings are invalid.")
    if (
        not math.isfinite(state.peak_learning_rate)
        or state.peak_learning_rate <= 0
        or not math.isfinite(state.minimum_learning_rate)
        or state.minimum_learning_rate < 0
        or state.minimum_learning_rate > state.peak_learning_rate
    ):
        raise ValueError("Checkpoint scheduler learning rates are invalid.")
    return state




def build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    state: TrainingState,
    scheduler: SchedulerState,
    config: ProjectConfig,
    run_name: str,
    run_id: str,
    tokenizer_reference: str | None,
    tokenizer_sha256: str | None,
    resume_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a complete checkpoint payload at an optimizer boundary."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": (
            None if scaler is None else scaler.state_dict()
        ),
        "training_state": state.to_dict(),
        "scheduler_state": scheduler.to_dict(),
        "random_states": capture_random_states(),
        "config": {
            "model": asdict(config.model),
            "training": asdict(config.training),
        },
        "run_name": run_name,
        "run_id": run_id,
        "tokenizer_reference": tokenizer_reference,
        "tokenizer_sha256": tokenizer_sha256,
        "resume_context": (
            None if resume_context is None else dict(resume_context)
        ),
    }


class CheckpointManager:
    """Save atomic latest, previous, best, and milestone checkpoints."""

    def __init__(self, root: str | Path, run_name: str) -> None:
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be a non-empty path-safe name.")
        self.run_dir = Path(root) / run_name
        self.milestone_dir = self.run_dir / "milestones"
        self.latest_path = self.run_dir / "latest.pt"
        self.previous_path = self.run_dir / "previous.pt"
        self.best_path = self.run_dir / "best.pt"

    def save(
        self,
        payload: Mapping[str, object],
        *,
        is_best: bool = False,
        milestone: bool = False,
    ) -> Path:
        started = time.perf_counter()
        state = _validate_training_state(payload.get("training_state"))
        if self.latest_path.exists():
            verify_checkpoint_checksum(self.latest_path)
            _copy_checkpoint(self.latest_path, self.previous_path)
        _atomic_torch_save(dict(payload), self.latest_path)
        _write_checksum(self.latest_path)
        checksum = verify_checkpoint_checksum(self.latest_path)
        duration = time.perf_counter() - started
        completion_path = self.latest_path.with_suffix(
            self.latest_path.suffix + ".complete.json"
        )
        _atomic_text_write(
            json.dumps(
                {
                    "complete": True,
                    "checkpoint": self.latest_path.name,
                    "sha256": checksum,
                    "bytes": self.latest_path.stat().st_size,
                    "optimizer_step": state.optimizer_step,
                    "write_duration_seconds": duration,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            completion_path,
        )
        if is_best:
            _copy_checkpoint(self.latest_path, self.best_path)
        if milestone:
            milestone_path = (
                self.milestone_dir
                / f"step_{state.optimizer_step:09d}.pt"
            )
            _copy_checkpoint(self.latest_path, milestone_path)
        return self.latest_path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler | None,
    expected_config: ProjectConfig | None = None,
    expected_data_manifest_sha256: str | None = None,
    restore_rng: bool = True,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Validate and restore a trusted local checkpoint.

    PyTorch checkpoints use pickle internally. Only load checkpoint files from
    trusted sources.
    """

    checkpoint_path = Path(path)
    verify_checkpoint_checksum(checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(
            f"Failed to deserialize checkpoint {checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {payload.get('format_version')!r}."
        )
    config_value = payload.get("config")
    if not isinstance(config_value, dict):
        raise ValueError("Checkpoint configuration is missing.")
    if expected_config is not None:
        expected = {
            "model": asdict(expected_config.model),
            "training": asdict(expected_config.training),
        }
        if config_value != expected:
            raise ValueError(
                "Checkpoint configuration does not match the requested run."
            )

    try:
        model_state = payload["model_state_dict"]
        optimizer_state = payload["optimizer_state_dict"]
        scaler_state = payload["grad_scaler_state_dict"]
        random_states = payload["random_states"]
    except KeyError as error:
        raise ValueError(
            f"Checkpoint is missing required field {error.args[0]!r}."
        ) from error
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint model or optimizer state is invalid.")
    if not isinstance(random_states, dict):
        raise ValueError("Checkpoint RNG state is invalid.")

    checkpoint_uses_scaler = scaler_state is not None
    if checkpoint_uses_scaler != (scaler is not None):
        raise ValueError(
            "Checkpoint GradScaler policy does not match current precision."
        )
    state = _validate_training_state(payload.get("training_state"))
    scheduler = _validate_scheduler_state(payload.get("scheduler_state"))
    run_name = payload.get("run_name")
    run_id = payload.get("run_id")
    if not isinstance(run_name, str) or not isinstance(run_id, str):
        raise ValueError("Checkpoint run identity is invalid.")
    tokenizer_reference = payload.get("tokenizer_reference")
    tokenizer_sha256 = payload.get("tokenizer_sha256")
    resume_context = payload.get("resume_context")
    if tokenizer_reference is not None and not isinstance(
        tokenizer_reference,
        str,
    ):
        raise ValueError("Checkpoint tokenizer reference is invalid.")
    if tokenizer_sha256 is not None and not isinstance(tokenizer_sha256, str):
        raise ValueError("Checkpoint tokenizer checksum is invalid.")
    if resume_context is not None and not isinstance(resume_context, dict):
        raise ValueError("Checkpoint resume context is invalid.")
    if expected_data_manifest_sha256 is not None:
        if resume_context is None or resume_context.get(
            "data_manifest_sha256"
        ) != expected_data_manifest_sha256:
            raise ValueError(
                "Checkpoint data-manifest checksum does not match this run."
            )
    if scaler is not None and not isinstance(scaler_state, dict):
        raise ValueError("Checkpoint GradScaler state is invalid.")

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scaler is not None:
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_random_states(random_states)
    return LoadedCheckpoint(
        path=checkpoint_path,
        state=state,
        scheduler=scheduler,
        run_name=run_name,
        run_id=run_id,
        tokenizer_reference=tokenizer_reference,
        tokenizer_sha256=tokenizer_sha256,
        resume_context=resume_context,
    )



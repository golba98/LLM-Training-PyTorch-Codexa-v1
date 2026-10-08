"""Deterministic mixed-precision training utilities."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any
import uuid

from llm_architecture.state import TrainingState
from llm_architecture.runtime import (PrecisionPolicy, resolve_device, resolve_precision, autocast_dtype, requires_grad_scaler, autocast_context, create_grad_scaler)

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def utc_timestamp() -> str:
    """Return an ISO-8601 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def _json_value(value: object, field_name: str) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite; got {value!r}.")
    return value




@dataclass(frozen=True)
class TrainingMetrics:
    """Metrics emitted after one completed optimizer update."""

    run_name: str
    run_id: str
    micro_step: int
    optimizer_step: int
    completed_epochs: int
    training_loss: float
    validation_loss: float | None
    learning_rate: float
    tokens_processed: int
    total_tokens_seen: int
    tokens_per_second: float
    step_time_seconds: float
    forward_backward_seconds: float
    optimizer_update_seconds: float
    gradient_norm: float
    allocated_vram_bytes: int | None
    reserved_vram_bytes: int | None
    peak_allocated_vram_bytes: int | None
    peak_reserved_vram_bytes: int | None
    device: str
    precision: str
    timestamp_utc: str

    def to_dict(self) -> dict[str, object]:
        values = {
            name: _json_value(value, name)
            for name, value in asdict(self).items()
        }
        values["validation_perplexity"] = (
            None
            if self.validation_loss is None
            else math.exp(min(self.validation_loss, 80.0))
        )
        return values




def set_deterministic_seed(seed: int) -> torch.Generator:
    """Seed all local random generators.

    Exact bitwise reproducibility is not guaranteed across different GPUs,
    drivers, CUDA versions, or PyTorch versions.
    """

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator














def create_adamw_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    epsilon: float = 1e-8,
    optimizer_name: str = "adamw",
) -> Optimizer:
    """Construct deduplicated AdamW decay and no-decay groups.

    ``adamw8bit`` uses bitsandbytes' 8-bit optimizer state and is intended
    for large CUDA models whose full-precision Adam state does not fit.
    """

    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be a positive finite number.")
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be a non-negative finite number.")
    if optimizer_name not in {"adamw", "adamw8bit"}:
        raise ValueError(
            "optimizer_name must be 'adamw' or 'adamw8bit'; "
            f"got {optimizer_name!r}."
        )

    groups: dict[int, tuple[nn.Parameter, bool, list[str]]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )
        identity = id(parameter)
        if identity in groups:
            existing_parameter, existing_no_decay, aliases = groups[identity]
            if existing_parameter is not parameter:
                raise ValueError("Parameter identity collision detected.")
            aliases.append(name)
            groups[identity] = (
                parameter,
                existing_no_decay and no_decay,
                aliases,
            )
        else:
            groups[identity] = (parameter, no_decay, [name])

    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for parameter, no_decay, _aliases in groups.values():
        target = no_decay_parameters if no_decay else decay_parameters
        target.append(parameter)

    grouped_ids = [
        id(parameter)
        for parameter in [*decay_parameters, *no_decay_parameters]
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("Duplicate parameters found across optimizer groups.")
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if set(grouped_ids) != trainable_ids:
        raise ValueError(
            "Optimizer groups must contain every trainable parameter exactly once."
        )

    optimizer_class: type[Optimizer]
    if optimizer_name == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as error:
            raise RuntimeError(
                "adamw8bit requires the bitsandbytes package."
            ) from error
        optimizer_class = AdamW8bit
    else:
        optimizer_class = AdamW

    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": betas,
        "eps": epsilon,
    }
    if optimizer_name == "adamw":
        optimizer_kwargs["foreach"] = False

    return optimizer_class(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
                "group_name": "decay",
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
                "group_name": "no_decay",
            },
        ],
        **optimizer_kwargs,
    )


def cosine_learning_rate(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Return the LR assigned before optimizer update ``step``."""

    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"step must be a non-negative integer; got {step!r}.")
    if (
        not isinstance(warmup_steps, int)
        or isinstance(warmup_steps, bool)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be a non-negative integer.")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be less than max_steps.")
    if peak_learning_rate <= 0 or not math.isfinite(peak_learning_rate):
        raise ValueError("peak_learning_rate must be positive and finite.")
    if minimum_learning_rate < 0 or not math.isfinite(minimum_learning_rate):
        raise ValueError("minimum_learning_rate must be non-negative and finite.")
    if minimum_learning_rate > peak_learning_rate:
        raise ValueError(
            "minimum_learning_rate must not exceed peak_learning_rate."
        )

    if step >= max_steps:
        return float(minimum_learning_rate)
    if warmup_steps > 0 and step <= warmup_steps:
        return float(peak_learning_rate * step / warmup_steps)
    if warmup_steps == 0 and step == 0:
        return float(peak_learning_rate)

    decay_span = max_steps - warmup_steps
    progress = (step - warmup_steps) / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    learning_rate = minimum_learning_rate + (
        peak_learning_rate - minimum_learning_rate
    ) * cosine
    return float(max(minimum_learning_rate, learning_rate))


class CyclingDataIterator:
    """Cycle a DataLoader with reproducible epoch-dependent shuffling."""

    def __init__(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        seed: int,
        completed_epochs: int = 0,
        batches_in_epoch: int = 0,
    ) -> None:
        if len(loader) == 0:
            raise ValueError("Training DataLoader must not be empty.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        if completed_epochs < 0:
            raise ValueError("completed_epochs must be non-negative.")
        if batches_in_epoch < 0 or batches_in_epoch >= len(loader):
            raise ValueError(
                "batches_in_epoch must identify a batch inside the epoch."
            )
        self.loader = loader
        self.seed = seed
        self.completed_epochs = completed_epochs
        self._batches_in_epoch = 0
        self._iterator = self._new_iterator()
        for _ in range(batches_in_epoch):
            next(self._iterator)
            self._batches_in_epoch += 1

    @property
    def batches_in_epoch(self) -> int:
        return self._batches_in_epoch

    def _new_iterator(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        generator = getattr(self.loader, "generator", None)
        if generator is not None:
            generator.manual_seed(self.seed + self.completed_epochs)
        return iter(self.loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.completed_epochs += 1
            self._batches_in_epoch = 0
            self._iterator = self._new_iterator()
            batch = next(self._iterator)
        self._batches_in_epoch += 1
        if self._batches_in_epoch == len(self.loader):
            self.completed_epochs += 1
            self._batches_in_epoch = 0
            self._iterator = self._new_iterator()
        return batch


def _finite_loss(loss: torch.Tensor, description: str) -> float:
    value = float(loss.detach().float().item())
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite {description}: {value!r}.")
    return value


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    max_batches: int | None = None,
    non_blocking: bool = False,
) -> tuple[float, int]:
    """Return target-token-weighted validation loss and token count."""

    if max_batches is not None and (
        not isinstance(max_batches, int)
        or isinstance(max_batches, bool)
        or max_batches <= 0
    ):
        raise ValueError("max_batches must be a positive integer when supplied.")
    if len(data_loader) == 0:
        raise ValueError("Validation DataLoader must not be empty.")

    was_training = model.training
    total_weighted_loss = 0.0
    total_tokens = 0
    try:
        model.eval()
        for batch_index, (input_ids, labels) in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_ids = input_ids.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with autocast_context(device, precision):
                _logits, loss = model(input_ids, labels)
            if loss is None:
                raise RuntimeError("Model did not return a validation loss.")
            loss_value = _finite_loss(loss, "validation loss")
            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                continue
            total_weighted_loss += loss_value * token_count
            total_tokens += token_count
        if total_tokens == 0:
            raise ValueError("Validation DataLoader produced no target tokens.")
        mean_loss = total_weighted_loss / total_tokens
        if not math.isfinite(mean_loss):
            raise FloatingPointError("Final validation loss is non-finite.")
        return mean_loss, total_tokens
    finally:
        model.train(was_training)


def _cuda_memory(device: torch.device) -> tuple[int, int, int, int]:
    if device.type != "cuda":
        return 0, 0, 0, 0
    return (
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_reserved(device),
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    max_steps: int,
    gradient_accumulation_steps: int,
    gradient_clip: float,
    warmup_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float | None = None,
    seed: int = 42,
    state: TrainingState | None = None,
    scaler: torch.amp.GradScaler | None = None,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    evaluation_interval: int | None = None,
    max_validation_batches: int | None = None,
    run_name: str = "training",
    run_id: str | None = None,
    logger: "JsonlRunLogger | None" = None,
    on_optimizer_step: (
        Callable[[TrainingState, TrainingMetrics], None] | None
    ) = None,
    max_micro_steps: int | None = None,
    progress: bool = False,
    deadline_monotonic: float | None = None,
) -> tuple[TrainingState, list[TrainingMetrics]]:
    """Train until exactly ``max_steps`` optimizer updates are complete."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if gradient_clip <= 0 or not math.isfinite(gradient_clip):
        raise ValueError("gradient_clip must be positive and finite.")
    if evaluation_interval is not None and evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive when supplied.")
    if validation_loader is not None and evaluation_interval is None:
        raise ValueError(
            "evaluation_interval is required when validation is enabled."
        )
    if max_micro_steps is not None and max_micro_steps < 0:
        raise ValueError("max_micro_steps must be non-negative.")
    if precision.uses_grad_scaler and scaler is None:
        raise ValueError("FP16 training requires a GradScaler.")
    if not precision.uses_grad_scaler and scaler is not None:
        raise ValueError("GradScaler must only be used for FP16 training.")

    minimum_lr = (
        peak_learning_rate * 0.1
        if minimum_learning_rate is None
        else minimum_learning_rate
    )
    run_state = TrainingState() if state is None else state
    resolved_run_id = str(uuid.uuid4()) if run_id is None else run_id
    cycling = CyclingDataIterator(
        train_loader,
        seed=seed,
        completed_epochs=run_state.completed_epochs,
        batches_in_epoch=run_state.batches_in_epoch,
    )
    model.train()
    model.to(device)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metrics: list[TrainingMetrics] = []
    initial_micro_step = run_state.micro_step
    attempted_micro_steps = 0
    progress_bar = (
        tqdm(
            total=max_steps,
            initial=run_state.optimizer_step,
            desc=run_name,
            unit="step",
            dynamic_ncols=True,
        )
        if progress
        else None
    )
    while run_state.optimizer_step < max_steps:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            break
        if (
            max_micro_steps is not None
            and attempted_micro_steps + gradient_accumulation_steps
            > max_micro_steps
        ):
            break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        step_loss_sum = 0.0
        step_tokens = 0
        start_micro_step = run_state.micro_step

        for _ in range(gradient_accumulation_steps):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                optimizer.zero_grad(set_to_none=True)
                run_state.micro_step = start_micro_step
                run_state.tokens_seen -= step_tokens
                raise TimeoutError("Training deadline reached during accumulation.")
            input_ids, labels = cycling.next()
            non_blocking = device.type == "cuda" and bool(
                getattr(train_loader, "pin_memory", False)
            )
            input_ids = input_ids.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            with autocast_context(device, precision):
                _logits, raw_loss = model(input_ids, labels)
            if raw_loss is None:
                raise RuntimeError("Model did not return a training loss.")
            raw_loss_value = _finite_loss(raw_loss, "training loss")
            scaled_loss = raw_loss / gradient_accumulation_steps
            _finite_loss(scaled_loss, "scaled training loss")
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()

            token_count = int((labels != -100).sum().item())
            if token_count == 0:
                raise ValueError("Training batch contains no target tokens.")
            step_loss_sum += raw_loss_value * token_count
            step_tokens += token_count
            run_state.micro_step += 1
            run_state.tokens_seen += token_count
            attempted_micro_steps += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_backward_seconds = time.perf_counter() - start_time
        optimizer_start = time.perf_counter()

        run_state.completed_epochs = cycling.completed_epochs
        run_state.batches_in_epoch = cycling.batches_in_epoch
        learning_rate = cosine_learning_rate(
            run_state.optimizer_step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            peak_learning_rate=peak_learning_rate,
            minimum_learning_rate=minimum_lr,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        if scaler is not None:
            scaler.unscale_(optimizer)
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip,
        )
        gradient_norm = float(gradient_norm_tensor.detach().float().item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm: {gradient_norm!r}."
            )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        run_state.optimizer_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_update_seconds = time.perf_counter() - optimizer_start
        elapsed = time.perf_counter() - start_time
        if elapsed <= 0:
            raise RuntimeError("Measured optimizer-step duration was not positive.")

        validation_loss: float | None = None
        should_validate = (
            validation_loader is not None
            and evaluation_interval is not None
            and run_state.optimizer_step % evaluation_interval == 0
        )
        if should_validate:
            validation_loss, _validation_tokens = evaluate(
                model,
                validation_loader,
                device=device,
                precision=precision,
                max_batches=max_validation_batches,
                non_blocking=device.type == "cuda"
                and bool(getattr(validation_loader, "pin_memory", False)),
            )
            if (
                run_state.best_validation_loss is None
                or validation_loss < run_state.best_validation_loss
            ):
                run_state.best_validation_loss = validation_loss

        allocated, reserved, peak_allocated, peak_reserved = _cuda_memory(device)
        training_loss = step_loss_sum / step_tokens
        record = TrainingMetrics(
            run_name=run_name,
            run_id=resolved_run_id,
            micro_step=run_state.micro_step,
            optimizer_step=run_state.optimizer_step,
            completed_epochs=run_state.completed_epochs,
            training_loss=training_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            tokens_processed=step_tokens,
            total_tokens_seen=run_state.tokens_seen,
            tokens_per_second=step_tokens / elapsed,
            step_time_seconds=elapsed,
            forward_backward_seconds=forward_backward_seconds,
            optimizer_update_seconds=optimizer_update_seconds,
            gradient_norm=gradient_norm,
            allocated_vram_bytes=allocated,
            reserved_vram_bytes=reserved,
            peak_allocated_vram_bytes=peak_allocated,
            peak_reserved_vram_bytes=peak_reserved,
            device=str(device),
            precision=precision.name,
            timestamp_utc=utc_timestamp(),
        )
        if record.micro_step - start_micro_step != gradient_accumulation_steps:
            raise RuntimeError("Optimizer update used an incomplete accumulation.")
        metrics.append(record)
        if logger is not None:
            logger.write_metrics(record)
        if on_optimizer_step is not None:
            on_optimizer_step(run_state, record)
        if progress_bar is not None:
            postfix: dict[str, str] = {
                "loss": f"{record.training_loss:.4f}",
                "lr": f"{record.learning_rate:.2e}",
                "tok/s": f"{record.tokens_per_second:.0f}",
            }
            if validation_loss is not None:
                postfix["val"] = f"{validation_loss:.4f}"
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

    if run_state.micro_step - initial_micro_step != attempted_micro_steps:
        raise RuntimeError("Training micro-step accounting mismatch.")
    if progress_bar is not None:
        progress_bar.close()
    return run_state, metrics


class JsonlRunLogger:
    """Own a run directory and flush every JSONL metrics record."""

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        *,
        overwrite: bool = False,
        resume: bool = False,
        expected_run_id: str | None = None,
        expected_optimizer_step: int | None = None,
    ) -> None:
        if not run_name or Path(run_name).name != run_name:
            raise ValueError("run_name must be a non-empty path-safe name.")
        self.run_dir = Path(log_dir) / run_name
        self.metrics_path = self.run_dir / "train_metrics.jsonl"
        self.metadata_path = self.run_dir / "run_metadata.json"
        if overwrite and resume:
            raise ValueError("overwrite and resume cannot both be enabled.")
        if resume:
            if not self.metrics_path.is_file() or not self.metadata_path.is_file():
                raise FileNotFoundError(
                    f"Cannot resume missing run logs in {self.run_dir}."
                )
            metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict) or (
                expected_run_id is not None
                and metadata.get("run_id") != expected_run_id
            ):
                raise ValueError(
                    "Existing run metadata does not match the checkpoint."
                )
            if expected_optimizer_step is not None:
                self._truncate_metrics_for_resume(
                    expected_run_id=expected_run_id,
                    expected_optimizer_step=expected_optimizer_step,
                )
            mode = "a"
        elif (
            self.metrics_path.exists() or self.metadata_path.exists()
        ) and not overwrite:
            raise FileExistsError(
                f"Run {run_name!r} already exists in {self.run_dir}."
            )
        else:
            mode = "w" if overwrite else "x"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self.metrics_path.open(
            mode,
            encoding="utf-8",
        )

    def _truncate_metrics_for_resume(
        self,
        *,
        expected_run_id: str | None,
        expected_optimizer_step: int,
    ) -> None:
        if (
            not isinstance(expected_optimizer_step, int)
            or isinstance(expected_optimizer_step, bool)
            or expected_optimizer_step < 0
        ):
            raise ValueError(
                "expected_optimizer_step must be a non-negative integer."
            )
        retained: list[str] = []
        previous_step = 0
        dropped = False
        with self.metrics_path.open("r", encoding="utf-8") as metrics_file:
            for line_number, line in enumerate(metrics_file, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{self.metrics_path}:{line_number}: malformed JSON."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{self.metrics_path}:{line_number}: metric must be "
                        "an object."
                    )
                run_id = record.get("run_id")
                step = record.get("optimizer_step")
                if expected_run_id is not None and run_id != expected_run_id:
                    raise ValueError(
                        "Metrics run identity does not match the checkpoint."
                    )
                if (
                    not isinstance(step, int)
                    or isinstance(step, bool)
                    or step <= previous_step
                ):
                    raise ValueError(
                        "Metrics optimizer steps must be strictly increasing."
                    )
                previous_step = step
                if step <= expected_optimizer_step:
                    retained.append(line)
                else:
                    dropped = True
        retained_step = (
            0
            if not retained
            else int(json.loads(retained[-1])["optimizer_step"])
        )
        if retained_step != expected_optimizer_step:
            raise ValueError(
                "Metrics do not contain the checkpoint optimizer-step tail."
            )
        if not dropped:
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.run_dir,
            prefix=".train_metrics.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                output_file.writelines(retained)
                output_file.flush()
                os.fsync(output_file.fileno())
            temporary_path.replace(self.metrics_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def write_metadata(self, metadata: Mapping[str, object]) -> None:
        serialized = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.run_dir,
            prefix=".run_metadata.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                output_file.write(serialized)
                output_file.flush()
                os.fsync(output_file.fileno())
            temporary_path.replace(self.metadata_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def write_metrics(self, metrics: TrainingMetrics | Mapping[str, object]) -> None:
        record = (
            metrics.to_dict()
            if isinstance(metrics, TrainingMetrics)
            else dict(metrics)
        )
        self._metrics_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        self._metrics_file.flush()

    def flush(self) -> None:
        self._metrics_file.flush()

    def close(self) -> None:
        if not self._metrics_file.closed:
            self._metrics_file.flush()
            self._metrics_file.close()

    def __enter__(self) -> "JsonlRunLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


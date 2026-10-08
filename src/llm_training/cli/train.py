"""Train Codexa from a prepared memory-mapped token dataset."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import platform
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import torch



from llm_training.config import load_config
from llm_training.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
    load_checkpoint,
    load_model_checkpoint,
)
from llm_architecture.model import LanguageModel, count_parameters
from llm_data.token_data import (
    MemmapTokenDataset,
    create_token_dataloader,
    file_sha256,
)
from llm_training.training import (
    JsonlRunLogger,
    create_adamw_optimizer,
    create_grad_scaler,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
    TrainingState,
    utc_timestamp,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic mixed-precision language-model training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-token-file", type=Path, required=True)
    parser.add_argument("--validation-token-file", type=Path, required=True)
    parser.add_argument("--train-token-offset", type=int, default=0)
    parser.add_argument("--train-token-count", type=int)
    parser.add_argument("--validation-token-offset", type=int, default=0)
    parser.add_argument("--validation-token-count", type=int)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("config", "fp32", "fp16", "bf16"),
        default="config",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default="training")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Trusted local checkpoint to resume, such as latest.pt.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Trusted checkpoint whose model weights initialize a new run.",
    )
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute Transformer activations to reduce training VRAM.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adamw8bit"),
        default="adamw",
        help="Optimizer state format; adamw8bit reduces CUDA memory.",
    )
    parser.add_argument("--overwrite-log", action="store_true")
    return parser


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed token manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Token manifest must contain a JSON object.")
    return value


def _required_mapping(
    manifest: dict[str, object],
    key: str,
) -> dict[str, object]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Token manifest field {key!r} must be an object.")
    return value


def _validate_token_file(
    path: Path,
    *,
    dtype: np.dtype,
    expected_checksum: object,
    model_vocab_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Token file does not exist: {path}")
    if path.stat().st_size % dtype.itemsize != 0:
        raise ValueError(f"Token file ends with a partial {dtype.name} element.")
    if not isinstance(expected_checksum, str):
        raise ValueError("Token manifest checksum must be a string.")
    actual_checksum = file_sha256(path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Token file checksum mismatch for {path}: "
            f"expected {expected_checksum}, got {actual_checksum}."
        )
    tokens = np.memmap(path, mode="r", dtype=dtype)
    if len(tokens) > 0 and int(tokens.max()) >= model_vocab_size:
        raise ValueError(
            f"Token file {path} contains an ID outside model vocabulary "
            f"size {model_vocab_size}."
        )


def validate_manifest(
    manifest: dict[str, object],
    *,
    train_token_file: Path,
    validation_token_file: Path | None,
    context_length: int,
    model_vocab_size: int,
) -> np.dtype:
    try:
        dtype = np.dtype(manifest["dtype"])
    except (KeyError, TypeError) as error:
        raise ValueError("Token manifest has an invalid dtype.") from error
    if dtype not in {np.dtype(np.uint16), np.dtype(np.uint32)}:
        raise ValueError("Token manifest dtype must be uint16 or uint32.")
    if manifest.get("context_length") != context_length:
        raise ValueError(
            "Token-data context length does not match the model configuration."
        )
    tokenizer_vocab_size = manifest.get("tokenizer_actual_vocab_size")
    if (
        not isinstance(tokenizer_vocab_size, int)
        or tokenizer_vocab_size <= 0
        or tokenizer_vocab_size > model_vocab_size
    ):
        raise ValueError(
            "Tokenizer vocabulary in the manifest is invalid for the model."
        )
    if manifest.get("model_vocab_size") != model_vocab_size:
        raise ValueError(
            "Token-data model vocabulary does not match configuration."
        )
    checksums = _required_mapping(manifest, "output_checksums")
    _validate_token_file(
        train_token_file,
        dtype=dtype,
        expected_checksum=checksums.get("train"),
        model_vocab_size=model_vocab_size,
    )
    if validation_token_file is not None:
        _validate_token_file(
            validation_token_file,
            dtype=dtype,
            expected_checksum=checksums.get("validation"),
            model_vocab_size=model_vocab_size,
        )
    return dtype


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _read_run_metadata(logger: JsonlRunLogger) -> dict[str, object]:
    value = json.loads(logger.metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Run metadata must contain a JSON object.")
    return value


def _record_resume(
    logger: JsonlRunLogger,
    *,
    timestamp: str,
    checkpoint: Path,
    state: TrainingState,
) -> None:
    metadata = _read_run_metadata(logger)
    events = metadata.get("resume_events", [])
    if not isinstance(events, list):
        raise ValueError("Run metadata resume_events must be an array.")
    previous_stop = metadata.get("last_stop_timestamp_utc")
    downtime_basis: str | None = None
    if not isinstance(previous_stop, str):
        last_record: dict[str, object] | None = None
        with logger.metrics_path.open("r", encoding="utf-8") as metrics_file:
            for line_number, line in enumerate(metrics_file, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{logger.metrics_path}:{line_number}: malformed "
                        "metrics JSON."
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{logger.metrics_path}:{line_number}: metric must "
                        "be an object."
                    )
                last_record = value
        if last_record is not None and isinstance(
            last_record.get("timestamp_utc"), str
        ):
            previous_stop = last_record["timestamp_utc"]
            downtime_basis = "last_metric"
    downtime_seconds: float | None = None
    if isinstance(previous_stop, str):
        try:
            downtime_seconds = max(
                0.0,
                (
                    datetime.fromisoformat(timestamp)
                    - datetime.fromisoformat(previous_stop)
                ).total_seconds(),
            )
        except ValueError as error:
            raise ValueError(
                "Run metadata has an invalid last stop timestamp."
            ) from error
        if downtime_basis is None:
            downtime_basis = "last_stop"
    events.append(
        {
            "resumed_at_utc": timestamp,
            "resume_checkpoint": str(checkpoint),
            "optimizer_step": state.optimizer_step,
            "micro_step": state.micro_step,
            "tokens_seen": state.tokens_seen,
            "downtime_seconds": downtime_seconds,
            "downtime_basis": downtime_basis,
        }
    )
    metadata["resume_events"] = events
    metadata["last_resume_timestamp_utc"] = timestamp
    metadata["run_status"] = "running"
    logger.write_metadata(metadata)


def _record_stop(
    logger: JsonlRunLogger,
    *,
    timestamp: str,
    state: TrainingState,
    status: str,
) -> None:
    metadata = _read_run_metadata(logger)
    metadata.update(
        {
            "run_status": status,
            "last_stop_timestamp_utc": timestamp,
            "completed_optimizer_steps": state.optimizer_step,
            "completed_micro_steps": state.micro_step,
            "tokens_seen": state.tokens_seen,
        }
    )
    logger.write_metadata(metadata)


def run(arguments: argparse.Namespace) -> int:
    project_config = load_config(arguments.config)
    training_config = project_config.training
    if arguments.resume is not None and arguments.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive.")
    if arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if arguments.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive.")
    max_steps = (
        training_config.max_steps
        if arguments.max_steps is None
        else arguments.max_steps
    )
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    device = resolve_device(arguments.device)
    requested_precision = (
        training_config.precision
        if arguments.precision == "config"
        else arguments.precision
    )
    precision = resolve_precision(requested_precision, device)
    set_deterministic_seed(training_config.seed)

    manifest = _load_manifest(arguments.token_manifest)
    data_manifest_sha256 = file_sha256(arguments.token_manifest)
    config_sha256 = file_sha256(arguments.config)
    validation_path = (
        None if arguments.no_validation else arguments.validation_token_file
    )
    dtype = validate_manifest(
        manifest,
        train_token_file=arguments.train_token_file,
        validation_token_file=validation_path,
        context_length=project_config.model.context_length,
        model_vocab_size=project_config.model.vocab_size,
    )
    train_dataset = MemmapTokenDataset(
        arguments.train_token_file,
        dtype=dtype,
        context_length=project_config.model.context_length,
        model_vocab_size=project_config.model.vocab_size,
        token_offset=arguments.train_token_offset,
        token_count=arguments.train_token_count,
    )
    pin_memory = device.type == "cuda"
    train_loader = create_token_dataloader(
        train_dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=pin_memory,
        seed=training_config.seed,
    )

    validation_loader = None
    if validation_path is not None:
        validation_dataset = MemmapTokenDataset(
            validation_path,
            dtype=dtype,
            context_length=project_config.model.context_length,
            model_vocab_size=project_config.model.vocab_size,
            token_offset=arguments.validation_token_offset,
            token_count=arguments.validation_token_count,
        )
        validation_loader = create_token_dataloader(
            validation_dataset,
            batch_size=training_config.micro_batch_size,
            shuffle=False,
            num_workers=arguments.num_workers,
            pin_memory=pin_memory,
            seed=training_config.seed,
        )

    model = LanguageModel(project_config.model)
    model.set_gradient_checkpointing(arguments.gradient_checkpointing)
    total_parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)
    initialization = None
    if arguments.init_checkpoint is not None:
        initialization = load_model_checkpoint(
            arguments.init_checkpoint,
            model=model,
            map_location="cpu",
        )
        checkpoint_model = initialization.config.get("model")
        if checkpoint_model != asdict(project_config.model):
            raise ValueError(
                "Initialization checkpoint model configuration does not "
                "match the requested run."
            )
    model.to(device)
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        optimizer_name=arguments.optimizer,
    )
    scaler = create_grad_scaler(device, precision)
    minimum_learning_rate = training_config.learning_rate * 0.1
    scheduler_state = SchedulerState(
        warmup_steps=min(training_config.warmup_steps, max_steps - 1),
        max_steps=max_steps,
        peak_learning_rate=training_config.learning_rate,
        minimum_learning_rate=minimum_learning_rate,
    )
    state = TrainingState()
    run_id = str(uuid.uuid4())
    if arguments.resume is not None:
        if arguments.overwrite_log:
            raise ValueError("--resume cannot be combined with --overwrite-log.")
        loaded = load_checkpoint(
            arguments.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=project_config,
            expected_data_manifest_sha256=data_manifest_sha256,
            restore_rng=True,
            map_location=device,
        )
        if loaded.run_name != arguments.run_name:
            raise ValueError(
                f"Checkpoint run name {loaded.run_name!r} does not match "
                f"--run-name {arguments.run_name!r}."
            )
        if loaded.scheduler != scheduler_state:
            raise ValueError(
                "Checkpoint learning-rate schedule does not match this run."
            )
        state = loaded.state
        run_id = loaded.run_id
    start_timestamp = utc_timestamp()

    print(
        "Resolved configuration: "
        + json.dumps(
            {
                "model": asdict(project_config.model),
                "training": asdict(training_config),
            },
            sort_keys=True,
        )
    )
    print(f"Device: {device}")
    print(f"Precision: {precision.name}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    logger = JsonlRunLogger(
        arguments.log_dir,
        arguments.run_name,
        overwrite=arguments.overwrite_log,
        resume=arguments.resume is not None,
        expected_run_id=run_id,
        expected_optimizer_step=(
            state.optimizer_step if arguments.resume is not None else None
        ),
    )
    checkpoint_manager = CheckpointManager(
        arguments.checkpoint_dir,
        arguments.run_name,
    )
    tokenizer_reference = manifest.get("tokenizer_path")
    tokenizer_checksum = manifest.get("tokenizer_sha256")
    if tokenizer_reference is not None and not isinstance(
        tokenizer_reference,
        str,
    ):
        raise ValueError("Manifest tokenizer path must be a string.")
    if tokenizer_checksum is not None and not isinstance(tokenizer_checksum, str):
        raise ValueError("Manifest tokenizer checksum must be a string.")

    def save_checkpoint(
        current_state: TrainingState,
        metrics: object,
        *,
        force: bool = False,
    ) -> None:
        optimizer_step = current_state.optimizer_step
        interval_boundary = (
            optimizer_step % training_config.checkpoint_interval == 0
        )
        validation_loss = getattr(metrics, "validation_loss", None)
        is_best = (
            validation_loss is not None
            and current_state.best_validation_loss == validation_loss
        )
        if (
            not force
            and not interval_boundary
            and optimizer_step != max_steps
            and not is_best
        ):
            return
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=current_state,
            scheduler=scheduler_state,
            config=project_config,
            run_name=arguments.run_name,
            run_id=run_id,
            tokenizer_reference=tokenizer_reference,
            tokenizer_sha256=tokenizer_checksum,
            resume_context={
                "microstep_within_gradient_accumulation": (
                    current_state.micro_step
                    % training_config.gradient_accumulation_steps
                ),
                "data_manifest_sha256": data_manifest_sha256,
                "configuration_sha256": config_sha256,
                "git_revision": _git_commit(),
                "data_cursor": {
                    "completed_epochs": current_state.completed_epochs,
                    "batches_in_epoch": current_state.batches_in_epoch,
                },
                "mixture_sampler_state": manifest.get("mixture_sampler_state"),
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "operating_system": platform.platform(),
                    "gpu_model": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                },
            },
        )
        checkpoint_manager.save(
            payload,
            is_best=is_best,
            milestone=interval_boundary,
        )

    try:
        if arguments.resume is None:
            logger.write_metadata(
                {
                    "run_name": arguments.run_name,
                    "run_id": run_id,
                    "config_path": str(arguments.config),
                    "train_token_file_path": str(arguments.train_token_file),
                    "validation_token_file_path": (
                        None
                        if validation_path is None
                        else str(validation_path)
                    ),
                    "token_manifest_path": str(arguments.token_manifest),
                    "device": str(device),
                    "precision": precision.name,
                    "start_timestamp_utc": start_timestamp,
                    "requested_maximum_optimizer_steps": max_steps,
                    "gradient_accumulation_steps": (
                        training_config.gradient_accumulation_steps
                    ),
                    "micro_batch_size": training_config.micro_batch_size,
                    "context_length": project_config.model.context_length,
                    "total_parameter_count": total_parameters,
                    "trainable_parameter_count": trainable_parameters,
                    "gradient_checkpointing": arguments.gradient_checkpointing,
                    "optimizer": arguments.optimizer,
                    "git_commit": _git_commit(),
                    "command_line_arguments": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(arguments).items()
                    },
                    "model_config": asdict(project_config.model),
                    "training_config": asdict(training_config),
                    "checkpoint_directory": str(checkpoint_manager.run_dir),
                    "resume_checkpoint": None,
                    "initialization_checkpoint": (
                        None
                        if arguments.init_checkpoint is None
                        else str(arguments.init_checkpoint)
                    ),
                    "initialization_source_run": (
                        None if initialization is None else initialization.run_name
                    ),
                    "initialization_source_step": (
                        None
                        if initialization is None
                        else initialization.training_state.optimizer_step
                    ),
                    "resume_events": [],
                    "run_status": "running",
                }
            )
        else:
            _record_resume(
                logger,
                timestamp=start_timestamp,
                checkpoint=arguments.resume,
                state=state,
            )
        if state.optimizer_step >= max_steps:
            _record_stop(
                logger,
                timestamp=utc_timestamp(),
                state=state,
                status="completed",
            )
            print(
                f"Checkpoint already completed {state.optimizer_step} "
                f"optimizer steps; nothing to do."
            )
            return 0
        state, metrics = train_model(
            model,
            train_loader,
            optimizer,
            device=device,
            precision=precision,
            max_steps=max_steps,
            gradient_accumulation_steps=(
                training_config.gradient_accumulation_steps
            ),
            gradient_clip=training_config.gradient_clip,
            warmup_steps=scheduler_state.warmup_steps,
            peak_learning_rate=training_config.learning_rate,
            minimum_learning_rate=minimum_learning_rate,
            seed=training_config.seed,
            state=state,
            scaler=scaler,
            validation_loader=validation_loader,
            evaluation_interval=(
                None
                if validation_loader is None
                else training_config.evaluation_interval
            ),
            max_validation_batches=arguments.max_validation_batches,
            run_name=arguments.run_name,
            run_id=run_id,
            logger=logger,
            on_optimizer_step=save_checkpoint,
            progress=True,
        )
    except KeyboardInterrupt:
        logger.flush()
        at_boundary = (
            state.micro_step
            == state.optimizer_step
            * training_config.gradient_accumulation_steps
        )
        if state.optimizer_step > 0 and at_boundary:
            save_checkpoint(state, object(), force=True)
        _record_stop(
            logger,
            timestamp=utc_timestamp(),
            state=state,
            status="interrupted",
        )
        print(
            f"Interrupted after {state.optimizer_step} completed optimizer steps.",
            file=sys.stderr,
        )
        return 130
    except BaseException:
        _record_stop(
            logger,
            timestamp=utc_timestamp(),
            state=state,
            status="failed",
        )
        raise
    finally:
        logger.close()

    _record_stop(
        logger,
        timestamp=utc_timestamp(),
        state=state,
        status="completed",
    )
    final = metrics[-1]
    print(f"Completed optimizer steps: {state.optimizer_step}")
    print(f"Completed micro-steps: {state.micro_step}")
    print(f"Tokens processed: {state.tokens_seen}")
    print(f"Final loss: {final.training_loss:.6f}")
    print(f"Metrics: {logger.metrics_path}")
    print(f"Run metadata: {logger.metadata_path}")
    return 0


def main() -> None:
    arguments = build_argument_parser().parse_args()
    from llm_training.viewer import attached_viewer
    config = load_config(arguments.config)
    with attached_viewer(
        metrics_path=arguments.log_dir / arguments.run_name / "train_metrics.jsonl",
        checkpoint_path=arguments.checkpoint_dir / arguments.run_name / "latest.pt",
        total_steps=arguments.max_steps or config.training.max_steps,
        title="LLM base training",
    ):
        raise SystemExit(run(arguments))


if __name__ == "__main__":
    main()

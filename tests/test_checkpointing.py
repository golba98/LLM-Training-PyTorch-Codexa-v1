"""Executable tests for atomic checkpointing and deterministic resume."""

from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from llm_training.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
    capture_random_states,
    file_sha256,
    load_checkpoint,
    restore_random_states,
    verify_checkpoint_checksum,
)
from llm_training.config import ProjectConfig, TrainingConfig
from llm_architecture.model import LanguageModel, ModelConfig
from llm_training.training import (
    JsonlRunLogger,
    TrainingState,
    create_adamw_optimizer,
    resolve_precision,
    set_deterministic_seed,
    train_model,
)
from llm_data.token_data import create_token_dataloader
from torch.utils.data import TensorDataset


def assert_raises(
    exception_type: type[BaseException],
    callable_object: object,
    *args: object,
    **kwargs: object,
) -> BaseException:
    try:
        callable_object(*args, **kwargs)  # type: ignore[operator]
    except exception_type as error:
        return error
    raise AssertionError(f"Expected {exception_type.__name__}.")


def project_config() -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(
            vocab_size=32,
            context_length=4,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
            dropout=0.1,
        ),
        training=TrainingConfig(
            micro_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=1,
            max_steps=4,
            gradient_clip=1.0,
            precision="fp32",
            checkpoint_interval=2,
            evaluation_interval=2,
            seed=123,
        ),
    )


def data_loader(seed: int = 123) -> torch.utils.data.DataLoader:
    tokens = torch.arange(30, dtype=torch.long).reshape(6, 5) % 32
    dataset = TensorDataset(tokens[:, :-1], tokens[:, 1:])
    return create_token_dataloader(
        dataset,  # type: ignore[arg-type]
        batch_size=2,
        shuffle=True,
        seed=seed,
    )


def optimizer_for(model: LanguageModel) -> torch.optim.AdamW:
    return create_adamw_optimizer(
        model,
        learning_rate=1e-3,
        weight_decay=0.01,
    )


def scheduler_state() -> SchedulerState:
    return SchedulerState(
        warmup_steps=1,
        max_steps=4,
        peak_learning_rate=1e-3,
        minimum_learning_rate=1e-4,
    )


def train(
    model: LanguageModel,
    optimizer: torch.optim.AdamW,
    *,
    state: TrainingState | None = None,
    max_micro_steps: int | None = None,
    callback: object = None,
) -> tuple[TrainingState, list]:
    return train_model(
        model,
        data_loader(),
        optimizer,
        device=torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        max_steps=4,
        gradient_accumulation_steps=2,
        gradient_clip=1.0,
        warmup_steps=1,
        peak_learning_rate=1e-3,
        minimum_learning_rate=1e-4,
        seed=123,
        state=state,
        max_micro_steps=max_micro_steps,
        run_name="checkpoint-test",
        run_id="fixed-run-id",
        on_optimizer_step=callback,  # type: ignore[arg-type]
    )


def assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_rng_round_trip() -> None:
    set_deterministic_seed(5)
    states = capture_random_states()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    restore_random_states(states)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_save_load_retention_and_corruption() -> None:
    config = project_config()
    set_deterministic_seed(config.training.seed)
    model = LanguageModel(config.model)
    optimizer = optimizer_for(model)
    state = TrainingState()
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = CheckpointManager(temporary_directory, "run")
        for step in range(1, 4):
            state.optimizer_step = step
            state.micro_step = step * 2
            state.tokens_seen = step * 16
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scaler=None,
                state=state,
                scheduler=scheduler_state(),
                config=config,
                run_name="run",
                run_id="id",
                tokenizer_reference="tokenizer.json",
                tokenizer_sha256="a" * 64,
                resume_context={
                    "data_manifest_sha256": "b" * 64,
                    "configuration_sha256": "c" * 64,
                    "data_cursor": {"completed_epochs": 0, "batches_in_epoch": step},
                    "mixture_sampler_state": {"seed": 42, "cursor": step},
                },
            )
            manager.save(
                payload,
                is_best=step == 2,
                milestone=step == 2,
            )

        assert manager.latest_path.is_file()
        completion = json.loads(
            manager.latest_path.with_suffix(".pt.complete.json").read_text(
                encoding="utf-8"
            )
        )
        assert completion["complete"] is True
        assert completion["sha256"] == file_sha256(manager.latest_path)
        assert completion["optimizer_step"] == 3
        assert manager.previous_path.is_file()
        assert manager.best_path.is_file()
        milestone = manager.milestone_dir / "step_000000002.pt"
        assert milestone.is_file()
        for path in (
            manager.latest_path,
            manager.previous_path,
            manager.best_path,
            milestone,
        ):
            assert verify_checkpoint_checksum(path) == file_sha256(path)

        restored_model = LanguageModel(config.model)
        restored_optimizer = optimizer_for(restored_model)
        loaded = load_checkpoint(
            manager.latest_path,
            model=restored_model,
            optimizer=restored_optimizer,
            scaler=None,
            expected_config=config,
            expected_data_manifest_sha256="b" * 64,
        )
        assert loaded.state.optimizer_step == 3
        assert loaded.scheduler == scheduler_state()
        assert loaded.run_name == "run"
        assert loaded.run_id == "id"
        assert loaded.tokenizer_reference == "tokenizer.json"
        assert loaded.resume_context is not None
        assert loaded.resume_context["configuration_sha256"] == "c" * 64
        assert_raises(
            ValueError,
            load_checkpoint,
            manager.latest_path,
            model=LanguageModel(config.model),
            optimizer=optimizer_for(LanguageModel(config.model)),
            scaler=None,
            expected_config=config,
            expected_data_manifest_sha256="d" * 64,
        )
        for original, restored in zip(
            model.parameters(),
            restored_model.parameters(),
            strict=True,
        ):
            assert torch.equal(original, restored)

        mismatched = ProjectConfig(
            model=replace(config.model, hidden_size=32, num_heads=4),
            training=config.training,
        )
        assert_raises(
            ValueError,
            load_checkpoint,
            manager.latest_path,
            model=LanguageModel(mismatched.model),
            optimizer=optimizer_for(restored_model),
            scaler=None,
            expected_config=mismatched,
        )

        corrupted = Path(temporary_directory) / "corrupted.pt"
        corrupted.write_bytes(manager.latest_path.read_bytes())
        corrupted.with_suffix(".pt.sha256").write_text(
            manager.latest_path.with_suffix(".pt.sha256").read_text(
                encoding="utf-8"
            ).replace("latest.pt", "corrupted.pt"),
            encoding="utf-8",
        )
        with corrupted.open("r+b") as checkpoint_file:
            checkpoint_file.seek(16)
            original_byte = checkpoint_file.read(1)
            checkpoint_file.seek(16)
            checkpoint_file.write(bytes([original_byte[0] ^ 0xFF]))
        assert_raises(ValueError, verify_checkpoint_checksum, corrupted)


def test_exact_resume() -> tuple[TrainingState, list]:
    config = project_config()

    set_deterministic_seed(config.training.seed)
    uninterrupted_model = LanguageModel(config.model)
    uninterrupted_optimizer = optimizer_for(uninterrupted_model)
    uninterrupted_state, uninterrupted_metrics = train(
        uninterrupted_model,
        uninterrupted_optimizer,
    )

    set_deterministic_seed(config.training.seed)
    interrupted_model = LanguageModel(config.model)
    interrupted_optimizer = optimizer_for(interrupted_model)
    with tempfile.TemporaryDirectory() as temporary_directory:
        manager = CheckpointManager(temporary_directory, "resume")
        interrupted_state = TrainingState()
        partial_losses: list[float] = []

        def save_at_boundary(state: TrainingState, metrics: object) -> None:
            partial_losses.append(metrics.training_loss)  # type: ignore[attr-defined]
            payload = build_checkpoint_payload(
                model=interrupted_model,
                optimizer=interrupted_optimizer,
                scaler=None,
                state=state,
                scheduler=scheduler_state(),
                config=config,
                run_name="resume",
                run_id="fixed-run-id",
                tokenizer_reference=None,
                tokenizer_sha256=None,
            )
            manager.save(payload)
            if state.optimizer_step == 2:
                raise KeyboardInterrupt

        assert_raises(
            KeyboardInterrupt,
            train,
            interrupted_model,
            interrupted_optimizer,
            state=interrupted_state,
            callback=save_at_boundary,
        )
        assert interrupted_state.optimizer_step == 2
        assert interrupted_state.micro_step == 4
        assert interrupted_state.completed_epochs == 1
        assert interrupted_state.batches_in_epoch == 1
        assert manager.latest_path.is_file()
        assert manager.previous_path.is_file()

        resumed_model = LanguageModel(config.model)
        resumed_optimizer = optimizer_for(resumed_model)
        loaded = load_checkpoint(
            manager.latest_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scaler=None,
            expected_config=config,
        )
        resumed_state, resumed_metrics = train(
            resumed_model,
            resumed_optimizer,
            state=loaded.state,
        )

        assert resumed_state.to_dict() == uninterrupted_state.to_dict()
        assert partial_losses == [
            item.training_loss for item in uninterrupted_metrics[:2]
        ]
        assert [item.training_loss for item in resumed_metrics] == [
            item.training_loss for item in uninterrupted_metrics[2:]
        ]
        for uninterrupted, resumed in zip(
            uninterrupted_model.parameters(),
            resumed_model.parameters(),
            strict=True,
        ):
            assert torch.equal(uninterrupted, resumed)
        assert_nested_equal(
            uninterrupted_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        )
        return resumed_state, resumed_metrics


def test_resume_logger_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        logger = JsonlRunLogger(temporary_directory, "run")
        logger.write_metadata({"run_name": "run", "run_id": "id"})
        logger.write_metrics({"optimizer_step": 1, "run_id": "id"})
        logger.write_metrics({"optimizer_step": 2, "run_id": "id"})
        logger.close()

        resumed = JsonlRunLogger(
            temporary_directory,
            "run",
            resume=True,
            expected_run_id="id",
            expected_optimizer_step=1,
        )
        resumed.write_metrics({"optimizer_step": 2, "run_id": "id"})
        resumed.close()
        records = [
            json.loads(line)
            for line in resumed.metrics_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [record["optimizer_step"] for record in records] == [1, 2]
        assert_raises(
            ValueError,
            JsonlRunLogger,
            temporary_directory,
            "run",
            resume=True,
            expected_run_id="wrong",
        )


def main() -> None:
    test_rng_round_trip()
    test_save_load_retention_and_corruption()
    state, metrics = test_exact_resume()
    test_resume_logger_identity()
    print(f"Checkpoint format version: {CHECKPOINT_FORMAT_VERSION}")
    print(f"Resumed optimizer step: {state.optimizer_step}")
    print(f"Resumed micro-step: {state.micro_step}")
    print(f"Resumed tokens seen: {state.tokens_seen}")
    print(f"First resumed loss: {metrics[0].training_loss:.6f}")
    print(f"Final resumed loss: {metrics[-1].training_loss:.6f}")
    print("All checkpointing tests passed.")


if __name__ == "__main__":
    main()

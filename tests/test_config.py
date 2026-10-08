"""Executable tests for strict YAML configuration loading."""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
import torch

from llm_training.config import ProjectConfig, TrainingConfig, load_config
from llm_architecture.model import LanguageModel, ModelConfig, count_parameters


SMOKE_CONFIG_PATH = Path("configs/smoke.yaml")
EXPECTED_PARAMETER_COUNT = 17_406_336
EXPECTED_TIER_PARAMETER_COUNTS = {
    Path("configs/1b.yaml"): 934_356_480,
}


def assert_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str,
) -> None:
    """Assert that an operation raises an informative exception."""

    try:
        operation()
    except exception_type as error:
        assert message_fragment in str(error), (
            f"Expected {message_fragment!r} in error message, got {error!r}."
        )
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def load_temporary_text(content: str) -> ProjectConfig:
    """Write and load one temporary YAML configuration."""

    with TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(content, encoding="utf-8")
        return load_config(path)


def load_temporary_data(data: object) -> ProjectConfig:
    """Serialize and load one temporary configuration object."""

    return load_temporary_text(yaml.safe_dump(data, sort_keys=False))


def read_smoke_data() -> dict[str, Any]:
    """Read the checked-in smoke YAML as mutable test data."""

    data = yaml.safe_load(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_valid_smoke_config() -> ProjectConfig:
    """Load the real smoke configuration and verify every resolved value."""

    config = load_config(SMOKE_CONFIG_PATH)
    assert isinstance(config, ProjectConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)

    assert config.model == ModelConfig()
    assert config.training == TrainingConfig(
        micro_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=0.0003,
        weight_decay=0.1,
        warmup_steps=100,
        max_steps=2000,
        gradient_clip=1.0,
        precision="bf16",
        checkpoint_interval=250,
        evaluation_interval=100,
        seed=42,
    )

    model = LanguageModel(config.model)
    assert count_parameters(model) == EXPECTED_PARAMETER_COUNT
    return config


def test_file_and_yaml_errors() -> None:
    """Reject missing files, malformed YAML, and invalid top-level values."""

    assert_raises(
        FileNotFoundError,
        lambda: load_config("configs/does-not-exist.yaml"),
        "No such file or directory",
    )


def test_larger_configurations() -> None:
    """Load larger tiers and verify exact counts without allocating weights."""

    for path, expected_count in EXPECTED_TIER_PARAMETER_COUNTS.items():
        config = load_config(path)
        with torch.device("meta"):
            model = LanguageModel(config.model)
        assert count_parameters(model) == expected_count
    assert_raises(
        ValueError,
        lambda: load_temporary_text("model: [\n"),
        "Failed to parse YAML configuration",
    )
    assert_raises(
        ValueError,
        lambda: load_temporary_data(["model", "training"]),
        "configuration must be a mapping",
    )


def test_schema_errors() -> None:
    """Reject missing, unknown, or incorrectly shaped configuration data."""

    smoke_data = read_smoke_data()

    missing_section = deepcopy(smoke_data)
    del missing_section["training"]
    assert_raises(
        ValueError,
        lambda: load_temporary_data(missing_section),
        "missing keys: training",
    )

    unknown_section = deepcopy(smoke_data)
    unknown_section["tokenizer"] = {}
    assert_raises(
        ValueError,
        lambda: load_temporary_data(unknown_section),
        "unknown keys: 'tokenizer'",
    )

    invalid_section = deepcopy(smoke_data)
    invalid_section["training"] = []
    assert_raises(
        ValueError,
        lambda: load_temporary_data(invalid_section),
        "training section must be a mapping",
    )

    missing_model_key = deepcopy(smoke_data)
    del missing_model_key["model"]["vocab_size"]
    assert_raises(
        ValueError,
        lambda: load_temporary_data(missing_model_key),
        "missing keys: vocab_size",
    )

    unknown_training_key = deepcopy(smoke_data)
    unknown_training_key["training"]["optimizer"] = "adamw"
    assert_raises(
        ValueError,
        lambda: load_temporary_data(unknown_training_key),
        "unknown keys: 'optimizer'",
    )


def test_value_errors() -> None:
    """Reject invalid model geometry and invalid training values."""

    smoke_data = read_smoke_data()

    invalid_model = deepcopy(smoke_data)
    invalid_model["model"]["hidden_size"] = 383
    assert_raises(
        ValueError,
        lambda: load_temporary_data(invalid_model),
        "Invalid model configuration",
    )

    invalid_training_values = (
        ("micro_batch_size", 0, "must be a positive integer"),
        ("gradient_accumulation_steps", True, "must be a positive integer"),
        ("warmup_steps", -1, "must be a non-negative integer"),
        ("learning_rate", 0.0, "must be greater than 0.0"),
        ("weight_decay", -0.1, "must be at least 0.0"),
        ("gradient_clip", float("inf"), "must be a finite number"),
        ("precision", "fp8", "precision must be one of"),
        ("seed", True, "must be a non-negative integer"),
    )
    for name, value, message in invalid_training_values:
        invalid_config = deepcopy(smoke_data)
        invalid_config["training"][name] = value
        assert_raises(
            ValueError,
            lambda invalid_config=invalid_config: load_temporary_data(
                invalid_config
            ),
            message,
        )

    excessive_warmup = deepcopy(smoke_data)
    excessive_warmup["training"]["warmup_steps"] = 2001
    assert_raises(
        ValueError,
        lambda: load_temporary_data(excessive_warmup),
        "warmup_steps must not exceed max_steps",
    )


def main() -> None:
    """Run all configuration-loader tests."""

    config = test_valid_smoke_config()
    test_larger_configurations()
    test_full_training_token_budget()
    test_file_and_yaml_errors()
    test_schema_errors()
    test_value_errors()

    print(f"Loaded configuration: {SMOKE_CONFIG_PATH}")
    print(f"Model parameter count: {EXPECTED_PARAMETER_COUNT:,}")
    print(f"Training precision: {config.training.precision}")
    print("All configuration tests passed.")


if __name__ == "__main__":
    main()

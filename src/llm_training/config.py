"""Typed YAML configuration loading for Codexa training runs."""

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml

from llm_architecture.model import ModelConfig


@dataclass
class TrainingConfig:
    """Training settings loaded from a project configuration file."""

    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    gradient_clip: float
    precision: str
    checkpoint_interval: int
    evaluation_interval: int
    seed: int

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "checkpoint_interval",
            "evaluation_interval",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer; got {value!r}."
                )

        for name in ("warmup_steps", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"{name} must be a non-negative integer; got {value!r}."
                )

        if self.warmup_steps > self.max_steps:
            raise ValueError(
                "warmup_steps must not exceed max_steps; "
                f"got warmup_steps={self.warmup_steps} "
                f"and max_steps={self.max_steps}."
            )

        self.learning_rate = _validated_float(
            "learning_rate",
            self.learning_rate,
            minimum=0.0,
            inclusive=False,
        )
        self.weight_decay = _validated_float(
            "weight_decay",
            self.weight_decay,
            minimum=0.0,
            inclusive=True,
        )
        self.gradient_clip = _validated_float(
            "gradient_clip",
            self.gradient_clip,
            minimum=0.0,
            inclusive=False,
        )

        supported_precisions = {"fp32", "fp16", "bf16"}
        if (
            not isinstance(self.precision, str)
            or self.precision not in supported_precisions
        ):
            choices = ", ".join(sorted(supported_precisions))
            raise ValueError(
                f"precision must be one of {choices}; got {self.precision!r}."
            )


@dataclass
class ProjectConfig:
    """Complete model and training configuration."""

    model: ModelConfig
    training: TrainingConfig


def _validated_float(
    name: str,
    value: object,
    *,
    minimum: float,
    inclusive: bool,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number; got {value!r}.")

    valid_range = converted >= minimum if inclusive else converted > minimum
    if not valid_range:
        qualifier = "at least" if inclusive else "greater than"
        raise ValueError(
            f"{name} must be {qualifier} {minimum}; got {value!r}."
        )
    return converted


def _validate_mapping(value: object, section_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{section_name} must be a mapping; "
            f"got {type(value).__name__}."
        )
    return value


def _validate_keys(
    mapping: dict[str, Any],
    expected_keys: set[str],
    section_name: str,
) -> None:
    actual_keys = set(mapping)
    missing_keys = expected_keys - actual_keys
    unknown_keys = actual_keys - expected_keys
    if not missing_keys and not unknown_keys:
        return

    details: list[str] = []
    if missing_keys:
        details.append(f"missing keys: {', '.join(sorted(missing_keys))}")
    if unknown_keys:
        rendered = ", ".join(sorted(repr(key) for key in unknown_keys))
        details.append(f"unknown keys: {rendered}")
    raise ValueError(f"{section_name} has invalid keys ({'; '.join(details)}).")


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a complete project configuration from YAML."""

    config_path = Path(path)
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(
            f"Failed to parse YAML configuration {config_path}: {error}"
        ) from error

    top_level = _validate_mapping(raw_config, "configuration")
    _validate_keys(top_level, {"model", "training"}, "configuration")

    model_values = _validate_mapping(top_level["model"], "model section")
    training_values = _validate_mapping(
        top_level["training"],
        "training section",
    )
    _validate_keys(
        model_values,
        {field.name for field in fields(ModelConfig)},
        "model section",
    )
    _validate_keys(
        training_values,
        {field.name for field in fields(TrainingConfig)},
        "training section",
    )

    try:
        model_config = ModelConfig(**model_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid model configuration: {error}") from error

    try:
        training_config = TrainingConfig(**training_values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid training configuration: {error}") from error

    return ProjectConfig(model=model_config, training=training_config)

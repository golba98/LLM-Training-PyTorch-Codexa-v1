"""Public training API.

The implementation lives in :mod:`llm_training.training`; this module provides the
roadmap's stable ``llm_training.train`` import path without duplicating logic.
"""

from llm_training.training import (
    CyclingDataIterator,
    JsonlRunLogger,
    PrecisionPolicy,
    TrainingMetrics,
    TrainingState,
    autocast_context,
    cosine_learning_rate,
    create_adamw_optimizer,
    create_grad_scaler,
    evaluate,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
)


__all__ = [
    "CyclingDataIterator",
    "JsonlRunLogger",
    "PrecisionPolicy",
    "TrainingMetrics",
    "TrainingState",
    "autocast_context",
    "cosine_learning_rate",
    "create_adamw_optimizer",
    "create_grad_scaler",
    "evaluate",
    "resolve_device",
    "resolve_precision",
    "set_deterministic_seed",
    "train_model",
]

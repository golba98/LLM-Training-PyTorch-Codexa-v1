"""Run the first assistant-only conversational SFT pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - optional low-memory CUDA path
    bnb = None


from llm_training.config import load_config
from llm_training.checkpointing import file_sha256
from llm_architecture.model import LanguageModel
from llm_tokenizer.sft import ChatMessage, serialize_conversation, IGNORE_LABEL
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.training import (
    JsonlRunLogger,
    TrainingState,
    create_adamw_optimizer,
    create_grad_scaler,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
)


class FixedSFTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, records: list[tuple[list[int], list[int]]], context: int) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        for input_ids, labels in records:
            input_ids = input_ids[:context]
            labels = labels[:context]
            pad = context - len(input_ids)
            self.items.append((
                torch.tensor(input_ids + [0] * pad, dtype=torch.long),
                torch.tensor(labels + [IGNORE_LABEL] * pad, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.items[index]


def _records(paths: list[Path], tokenizer, context: int, limit_per_source: int | None) -> list[tuple[list[int], list[int]]]:
    output: list[tuple[list[int], list[int]]] = []
    for path in paths:
        source_count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                messages = [ChatMessage(item["role"], item["content"]) for item in row["messages"]]
                try:
                    serialized = serialize_conversation(messages, tokenizer, maximum_tokens=context)
                except ValueError:
                    continue
                shifted = list(serialized.labels[1:]) + [IGNORE_LABEL]
                output.append((list(serialized.input_ids), shifted))
                source_count += 1
                if limit_per_source is not None and source_count >= limit_per_source:
                    break
    return output


def _base_replay(path: Path, context: int, count: int, seed: int) -> FixedSFTDataset:
    tokens = np.memmap(path, mode="r", dtype=np.uint16)
    rng = random.Random(seed)
    records: list[tuple[list[int], list[int]]] = []
    maximum = len(tokens) - context - 1
    for _ in range(count):
        start = rng.randrange(maximum)
        window = np.asarray(tokens[start : start + context + 1], dtype=np.int64).tolist()
        records.append((window[:-1], window[1:]))
    return FixedSFTDataset(records, context)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/100m-native.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--validation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--base-token-file", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="codexa-100m-sft-pilot")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--train-limit", type=int, default=60000)
    parser.add_argument("--validation-limit", type=int, default=1000)
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--base-replay-ratio", type=float, default=0.1)
    parser.add_argument("--learning-rate-scale", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    args = parser.parse_args()
    from llm_training.viewer import attached_viewer
    with attached_viewer(metrics_path=args.log_dir / args.run_name / "train_metrics.jsonl",
                         checkpoint_path=args.checkpoint_dir / args.run_name / "latest.pt",
                         total_steps=args.max_steps, title="LLM conversational SFT"):
        _run(args)


def _run(args) -> None:
    """Execute one parsed run after the visible viewer has attached."""
    if not 0 <= args.base_replay_ratio <= 1:
        raise ValueError("--base-replay-ratio must be in [0, 1].")
    if args.learning_rate_scale <= 0:
        raise ValueError("--learning-rate-scale must be positive.")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup-steps must be non-negative and less than --max-steps.")
    config = load_config(args.config)
    set_deterministic_seed(config.training.seed)
    tokenizer = load_tokenizer(args.tokenizer)
    train_records = _records(args.train_jsonl, tokenizer, config.model.context_length, args.train_limit)
    validation_records = _records(args.validation_jsonl, tokenizer, config.model.context_length, args.validation_limit)
    replay_count = round(len(train_records) * args.base_replay_ratio)
    datasets = [FixedSFTDataset(train_records, config.model.context_length)]
    if replay_count:
        datasets.append(_base_replay(args.base_token_file, config.model.context_length, replay_count, config.training.seed))
    train_dataset = ConcatDataset(datasets)
    validation_dataset = FixedSFTDataset(validation_records, config.model.context_length)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(config.training.seed))
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    device = resolve_device("auto")
    precision = resolve_precision(config.training.precision, device)
    model = LanguageModel(config.model)
    # The 900M model plus BF16 activations exceeds a 16 GB card without
    # recomputation. This does not alter the model; it trades compute for
    # memory during SFT.
    model.set_gradient_checkpointing(True)
    resume_payload = None
    if args.resume_checkpoint is not None:
        resume_payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        payload = resume_payload
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from llm_training.checkpointing import verify_checkpoint_checksum
    verify_checkpoint_checksum(args.resume_checkpoint or args.checkpoint)
    if payload.get("tokenizer_sha256") not in (None, file_sha256(args.tokenizer)):
        raise ValueError("Tokenizer fingerprint mismatch.")
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("Tokenizer vocabulary mismatch.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if args.optimizer == "adamw8bit":
        if bnb is None or device.type != "cuda":
            raise RuntimeError("--optimizer adamw8bit requires bitsandbytes and CUDA.")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=config.training.learning_rate * args.learning_rate_scale,
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=config.training.learning_rate * args.learning_rate_scale,
            weight_decay=config.training.weight_decay,
        )
    scaler = create_grad_scaler(device, precision)
    resume_step = int(resume_payload.get("optimizer_step", 0)) if resume_payload else 0
    logger = JsonlRunLogger(
        args.log_dir,
        args.run_name,
        overwrite=resume_payload is None,
        resume=resume_payload is not None,
        expected_optimizer_step=resume_step if resume_payload is not None else None,
    )
    run_state = TrainingState(optimizer_step=resume_step)
    logger.write_metadata({
        "run_name": args.run_name, "run_status": "running", "config": asdict(config.model),
        "training_config": asdict(config.training), "max_steps": args.max_steps,
        "train_conversations": len(train_records), "validation_conversations": len(validation_records),
        "base_replay_examples": replay_count,
        "base_replay_ratio": args.base_replay_ratio,
        "learning_rate_scale": args.learning_rate_scale,
        "warmup_steps": args.warmup_steps,
        "source_checkpoint": str(args.checkpoint),
    })
    checkpoint_dir = args.checkpoint_dir / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(state, _metrics):
        if state.optimizer_step % config.training.checkpoint_interval == 0 or state.optimizer_step == args.max_steps:
            checkpoint_path = checkpoint_dir / "latest.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_step": state.optimizer_step,
                "tokenizer_reference": str(args.tokenizer),
                "tokenizer_sha256": file_sha256(args.tokenizer),
                "config": {
                    "model": asdict(config.model),
                    "training": asdict(config.training),
                },
            }, checkpoint_path)
            (checkpoint_path.with_suffix(".pt.sha256")).write_text(
                f"{file_sha256(checkpoint_path)}  {checkpoint_path.name}\n",
                encoding="utf-8",
            )

    try:
        train_model(model, train_loader, optimizer, device=device, precision=precision,
                    max_steps=args.max_steps, gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                    gradient_clip=config.training.gradient_clip, warmup_steps=args.warmup_steps,
                    peak_learning_rate=config.training.learning_rate * args.learning_rate_scale,
                    minimum_learning_rate=config.training.learning_rate * args.learning_rate_scale * 0.1,
                    seed=config.training.seed, state=run_state, scaler=scaler,
                    validation_loader=validation_loader, evaluation_interval=100, max_validation_batches=32,
                    run_name=args.run_name, logger=logger, on_optimizer_step=save, progress=True)
        logger.write_metadata({"run_name": args.run_name, "run_status": "completed", "max_steps": args.max_steps})
    finally:
        logger.close()


if __name__ == "__main__":
    main()
